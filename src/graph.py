"""
RAG Graph Pipeline for Financial Document Screening.

This module implements a LangGraph-based RAG system for answering questions about
J.P. Morgan's 2025 Outlook and Mid-Year Outlook documents. The system routes questions
to appropriate document sources and synthesizes answers with analyst-level reasoning.
"""
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from dotenv import load_dotenv
import sys
import re

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingest import get_forecast_retriever, get_midyear_retriever, get_combined_retriever

# Load .env.local from project root
load_dotenv(project_root / ".env.local")


class RAGState(TypedDict):
    """State structure for the RAG graph workflow."""
    question: str
    route: str
    documents: List[Document]
    answer: str


def router_node(state: RAGState) -> RAGState:
    """
    Route questions to appropriate document sources based on question content.
    
    Routes:
    - "forecast": Questions about Outlook 2025 / forecast documents
    - "midyear": Questions about Mid-Year Outlook 2025 / mid-year documents
    - "both": Questions requiring comparison between forecast and mid-year
    """
    question = state["question"].lower()
    
    # Check if question explicitly references a specific document
    if "according to outlook 2025" in question or "according to forecast" in question:
        route = "forecast"
        return {**state, "route": route}
    elif "according to mid-year" in question or "according to midyear" in question:
        route = "midyear"
        return {**state, "route": route}
    
    # Check if question mentions both forecast and mid-year
    has_forecast = any(word in question for word in ["forecast", "expected", "outlook", "start of 2025", "2025 forecast"])
    has_midyear = any(word in question for word in ["mid-year", "midyear", "played out", "by mid", "reality", "actual"])
    
    # If question mentions both forecast and mid-year, route to both
    if has_forecast and has_midyear:
        route = "both"
    elif has_midyear:
        route = "midyear"
    elif has_forecast:
        route = "forecast"
    else:
        route = "both"
    
    return {
        **state,
        "route": route
    }


def forecast_retriever_node(state: RAGState) -> RAGState:
    """
    Retrieve documents from forecast source only.
    
    Uses MMR (Maximum Marginal Relevance) retrieval with k=12 documents
    to balance relevance and diversity.
    """
    question = state["question"]
    retriever = get_forecast_retriever(k=12, use_mmr=True)
    documents = retriever.invoke(question)
    return {
        **state,
        "documents": documents
    }


def midyear_retriever_node(state: RAGState) -> RAGState:
    """
    Retrieve documents from mid-year source only.
    
    Uses MMR (Maximum Marginal Relevance) retrieval with k=12 documents
    to balance relevance and diversity.
    """
    question = state["question"]
    retriever = get_midyear_retriever(k=12, use_mmr=True)
    documents = retriever.invoke(question)
    return {
        **state,
        "documents": documents
    }


def both_retriever_node(state: RAGState) -> RAGState:
    """
    Retrieve documents from both forecast and mid-year sources.
    
    Uses MMR (Maximum Marginal Relevance) retrieval with k=12 documents
    to balance relevance and diversity across both document types.
    """
    question = state["question"]
    retriever = get_combined_retriever(k=12, use_mmr=True)
    documents = retriever.invoke(question)
    return {
        **state,
        "documents": documents
    }


def format_documents_for_prompt(documents: List[Document]) -> str:
    """
    Format retrieved documents with clear citations for the prompt.
    
    Each document is numbered and includes source, document name, and page number
    to enable proper citation in the generated answer.
    """
    if not documents:
        return "No documents were retrieved."
    
    formatted = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get('source', 'unknown')
        doc_name = doc.metadata.get('doc_name', 'unknown')
        page = doc.metadata.get('page', 'N/A')
        
        # Create citation header with document number and metadata
        citation = f"[{i}] Source: {source}, Document: {doc_name}"
        if page != 'N/A':
            citation += f", Page: {page}"
        
        formatted.append(f"{citation}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(formatted)


def truncate_table_cells(text: str, max_length: int = 60) -> str:
    """
    Post-process markdown tables to truncate cells that exceed max_length.
    
    Preserves table structure while ensuring cells are concise and readable.
    Truncates at semicolons first (if present), then at word boundaries.
    
    Args:
        text: Text containing markdown tables
        max_length: Maximum characters per table cell (default: 60)
    
    Returns:
        Text with truncated table cells
    """
    def truncate_cell(cell_content: str) -> str:
        """Truncate a single table cell if it's too long."""
        cell_content = cell_content.strip()
        if len(cell_content) <= max_length:
            return cell_content
        
        # Try to truncate at a semicolon if present
        if ';' in cell_content:
            parts = cell_content.split(';')
            truncated = parts[0].strip()
            for part in parts[1:]:
                if len(truncated + '; ' + part.strip()) <= max_length:
                    truncated += '; ' + part.strip()
                else:
                    break
            if len(truncated) <= max_length:
                return truncated
        
        # Otherwise truncate at word boundary
        words = cell_content.split()
        truncated = ""
        for word in words:
            if len(truncated + ' ' + word) <= max_length - 3:
                truncated += (' ' if truncated else '') + word
            else:
                break
        return truncated + '...' if truncated != cell_content else cell_content[:max_length-3] + '...'
    
    # Find all markdown tables and process them
    lines = text.split('\n')
    result = []
    
    for line in lines:
        # Check if this is a table row (contains | and is not a separator)
        if '|' in line and not line.strip().startswith('|--'):
            # Split by | and process each cell
            cells = line.split('|')
            processed_cells = []
            for i, cell in enumerate(cells):
                cell = cell.strip()
                if cell and i > 0 and i < len(cells) - 1:  # Skip first and last empty cells
                    processed_cells.append(truncate_cell(cell))
                else:
                    processed_cells.append(cell)
            # Reconstruct the line
            line = '|'.join(processed_cells)
        
        result.append(line)
    
    return '\n'.join(result)


def create_citation_reference_table(documents: List[Document]) -> str:
    """
    Create a reference table mapping citation numbers to document details.
    
    Generates a markdown table showing which citation number [N] corresponds
    to which source, document, and page. This is appended to answers for clarity.
    """
    if not documents:
        return ""
    
    # Create properly formatted markdown table with correct alignment
    lines = [
        "\n### Citation Reference",
        "",
        "| Citation | Source | Document | Page |",
        "|---------|--------|----------|------|"
    ]
    
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get('source', 'unknown')
        doc_name = doc.metadata.get('doc_name', 'unknown')
        # Truncate long document names for better readability
        if len(doc_name) > 40:
            doc_name = doc_name[:37] + "..."
        page = doc.metadata.get('page', 'N/A')
        lines.append(f"| [{i}] | {source} | {doc_name} | {page} |")
    
    return "\n".join(lines)


def synthesis_node(state: RAGState) -> RAGState:
    """
    Generate answer using retrieved documents with analyst-level reasoning.
    
    This is the core synthesis node that:
    1. Formats retrieved documents with citations
    2. Applies equity-focused analyst reasoning prompts
    3. Generates answers with proper grounding and citations
    4. Post-processes tables to ensure concise formatting
    5. Appends citation reference table
    
    The prompt includes strict rules for:
    - Distinguishing equity themes from macro themes
    - Identifying highlighted stocks vs illustrative mentions
    - Semantic precision (valuation concerns vs performance)
    - Cross-document synthesis for forecast vs mid-year comparisons
    - Table formatting (max 60 chars per cell)
    """
    question = state["question"]
    documents = state["documents"]
    
    # Handle empty documents case
    if not documents:
        answer = "I cannot answer this question because no relevant information was found in the retrieved documents."
        return {
            **state,
            "answer": answer
        }
    
    # Separate documents by source for better synthesis
    forecast_docs = [d for d in documents if d.metadata.get('source') == 'forecast']
    midyear_docs = [d for d in documents if d.metadata.get('source') == 'midyear']
    
    # Format documents with citations
    formatted_docs = format_documents_for_prompt(documents)
    
    # Enhanced prompt with equity-focused analyst-level reasoning
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior equity research analyst conducting a screening task. Your goal is to extract, synthesize, and compare information from financial documents with precision and analytical depth.

CRITICAL ANALYST RULES:

1. EQUITY THEMES vs MACRO THEMES (CRITICAL DISTINCTION):
   - EQUITY MARKET THEMES = specific equity investment themes that identify which stocks/sectors will perform
     Examples: "U.S. vs non-U.S. equity divergence", "Mega-cap tech concentration", "AI capex beneficiaries", 
              "Power/infrastructure equities", "Financials & dealmaking exposure", "Japanese equity outperformance"
   - MACRO/STRATEGY THEMES = broad investment strategy or macroeconomic themes (NOT equity-specific)
     Examples: "Understanding Election Impacts", "Evolving Investment Landscapes", "Renewing Portfolio Resilience"
   - When asked for "equity market themes", extract ONLY themes that directly relate to specific equity investments
   - DO NOT extract report structure, table of contents, or macro strategy themes as "equity market themes"
   - Look for themes that identify: specific sectors, geographic equity markets, market cap segments, or investment styles

2. HIGHLIGHTED STOCK CRITERIA (Investment Conviction vs Illustrative):
   - "HIGHLIGHTED" = mentioned with investment conviction, positive outlook, or as a key recommendation
     Indicators: "expected to perform well", "beneficiaries", "outperform", "attractive", "opportunity", 
                "strong growth", "favorable", "recommended", "highlighted", "focus on"
   - "ILLUSTRATIVE" = mentioned as example, case study, or passing reference without investment conviction
     Indicators: "for example", "such as", "illustrated by", "case study", "mentioned in context of"
   - When asked for "highlighted stocks", prioritize stocks with investment conviction
   - For illustrative mentions, either exclude them OR label them as "mentioned illustratively" with lower confidence
   - High-confidence highlighted stocks: Microsoft, Magnificent 7, AI-related equities (when presented as opportunity)
   - Lower-confidence: Stocks mentioned only in examples or case studies without explicit positive outlook

3. SEMANTIC PRECISION (Critical for Accuracy):
   - DISTINGUISH between:
     * "Questioning valuation/saturation" ≠ "Underperformed"
     * "Concerns about narrative fatigue" ≠ "Disappointed"
     * "Valuation concerns" ≠ "Performance decline"
   - When documents QUESTION or RAISE CONCERNS about valuation/saturation/narrative, state it as:
     "The document questions/raises concerns about [X]" NOT "X underperformed" or "X disappointed"
   - Only state "underperformed" or "disappointed" when documents explicitly state performance metrics or outcomes
   - For performance statements, require: specific metrics, comparisons, or explicit performance language

4. EXPLICIT vs IMPLIED RECOGNITION:
   - "Explicit" = directly named (e.g., "Microsoft", "Apple", "NVIDIA")
   - "Implied" = clearly referenced through groups/sectors (e.g., "Magnificent 7" = Apple, Microsoft, NVIDIA, etc.)
   - When documents mention groups like "Magnificent 7", "tech sector", "AI-related equities", treat as valid
   - DO NOT say "documents do not explicitly state" when they mention groups/sectors that clearly refer to stocks

5. CROSS-DOCUMENT SYNTHESIS:
   - When comparing forecast vs mid-year: actively map themes, stocks, and predictions
   - Create explicit mappings: "Forecast said X → Mid-year shows Y"
   - For "played out" questions: identify forecast themes and find corresponding mid-year commentary
   - Look for: theme names, stock mentions, sector references, valuation concerns, risk factors

6. CONFIDENCE THRESHOLDS & EVIDENCE WEIGHTING:
   - HIGH CONFIDENCE: Explicit mentions with investment conviction, specific metrics, direct comparisons
   - MEDIUM CONFIDENCE: Implied references through groups/sectors, thematic mentions
   - LOW CONFIDENCE: Illustrative mentions, passing references, ambiguous statements
   - When presenting answers, prioritize high-confidence items first
   - For lower-confidence items, qualify with: "mentioned illustratively" or "referenced in context of"

7. STRUCTURED OUTPUT - TABLE FORMATTING (CRITICAL - STRICT ENFORCEMENT):
   - For table requests: create proper markdown table with all requested columns
   - CRITICAL TABLE RULES (MUST FOLLOW):
     * Each cell MAX 60 characters (approximately 10-12 words MAX)
     * Use ULTRA-SHORT phrases, NOT sentences
     * Format: "Key1; Key2" (semicolon-separated, max 2-3 points)
     * Use abbreviations: U.S., EM, AI, EPS, etc.
     * NO articles (a, an, the) unless necessary
     * NO conjunctions (and, or, but) - use semicolons instead
   - GOOD examples:
     * "Rate cuts; U.S./Japan outperform EM"
     * "AI capex surge; power infra focus"
     * "15% EPS growth; beats market 8%"
   - BAD examples (TOO LONG):
     * "Expect normalization of policy rates; supports developed markets over emerging markets" (TOO LONG - split or shorten)
     * "Focus on AI, power, infrastructure, and security; significant capital inflows anticipated" (TOO LONG - remove words)
   - Extract from BOTH forecast and mid-year documents
   - Use "Yes"/"No"/"Partial" for supported column
   - Citations: [N] format only
   - Put ALL detailed explanations BELOW the table, never in cells

8. CITATION PRECISION:
   - Cite specific claims, not generic sections
   - For each claim, cite the document number [N] that contains the supporting evidence
   - For comparisons, cite both forecast [N] and mid-year [M] sources separately
   - Avoid generic citations - each citation should point to specific supporting evidence

9. ANALYST REASONING (not legal parsing):
   - Think like a senior research analyst, not a legal document parser
   - Connect related concepts across documents
   - Synthesize information rather than requiring exact word matches
   - But maintain semantic precision - don't over-interpret or mischaracterize statements

When citing, use the format: [N] where N corresponds to the document number."""),
        ("human", """Question: {question}

Retrieved Documents:
{formatted_documents}

{comparison_instructions}

Instructions:
- Answer as a senior equity research analyst: extract, synthesize, and compare with precision
- For "equity market themes": extract ONLY equity-specific investment themes, NOT macro/strategy themes
- For "highlighted stocks": prioritize stocks with investment conviction, distinguish from illustrative mentions
- Maintain semantic precision: distinguish "questions valuation" from "underperformed"
- Use confidence thresholds: prioritize high-confidence items, qualify lower-confidence ones
- For comparison questions, actively map forecast → mid-year outcomes with specific evidence
- Cite all claims with [N] format, citing specific evidence not generic sections
- If the question asks for a table: 
  * CRITICAL: Each cell MAX 60 characters (10-12 words MAX)
  * Use ULTRA-SHORT phrases: "Key1; Key2" format
  * Good: "Rate cuts; U.S./Japan outperform EM"
  * Bad: "Expect normalization of policy rates; supports developed markets over emerging markets" (TOO LONG)
  * Remove articles (a, an, the) and conjunctions (and, or) - use semicolons
  * Use abbreviations: U.S., EM, AI, EPS, etc.
  * Put ALL detailed explanations BELOW the table, never in cells
  * Count characters: if a cell exceeds 60 chars, shorten it immediately
- Be thorough but precise: extract relevant information while maintaining analytical accuracy""")
    ])
    
    # Add comparison-specific instructions if both sources are present
    comparison_instructions = ""
    if forecast_docs and midyear_docs:
        comparison_instructions = """
COMPARISON MODE ACTIVATED:
- You have documents from BOTH forecast and mid-year sources
- Actively compare: What was forecasted → What actually happened
- Map themes, stocks, risks, and valuations across time periods
- For "played out" questions: identify forecast themes and find corresponding mid-year commentary
- For structured tables: extract forecast views and mid-year reality, then determine if supported
"""
    
    # Initialize LLM (using cost-effective model)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Generate answer
    messages = prompt.format_messages(
        question=question,
        formatted_documents=formatted_docs,
        comparison_instructions=comparison_instructions
    )
    response = llm.invoke(messages)
    answer = response.content
    
    # Post-process to enforce table cell length limits (60 chars max)
    answer = truncate_table_cells(answer, max_length=60)
    
    # Append citation reference table for clarity with proper spacing
    citation_table = create_citation_reference_table(documents)
    if citation_table:
        # Ensure there's proper spacing before the citation table
        if not answer.endswith('\n'):
            answer = answer + "\n"
        answer = answer + citation_table
    
    return {
        **state,
        "answer": answer
    }


def route_decision(state: RAGState) -> str:
    """
    Conditional edge function: route based on router decision.
    
    Returns the route determined by router_node to direct flow to appropriate retriever.
    """
    return state["route"]


def build_rag_graph():
    """
    Build and compile the LangGraph RAG pipeline.
    
    Creates a state graph with the following workflow:
    1. router -> routes to forecast/midyear/both retriever
    2. retriever -> retrieves relevant documents
    3. synthesis -> generates answer with citations
    
    Returns:
        Compiled LangGraph workflow ready for execution
    """
    # Create the graph
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("forecast_retriever", forecast_retriever_node)
    workflow.add_node("midyear_retriever", midyear_retriever_node)
    workflow.add_node("both_retriever", both_retriever_node)
    workflow.add_node("synthesis", synthesis_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "forecast": "forecast_retriever",
            "midyear": "midyear_retriever",
            "both": "both_retriever"
        }
    )
    
    # All retrieval nodes go to synthesis
    workflow.add_edge("forecast_retriever", "synthesis")
    workflow.add_edge("midyear_retriever", "synthesis")
    workflow.add_edge("both_retriever", "synthesis")
    
    # Synthesis goes to END
    workflow.add_edge("synthesis", END)
    
    # Compile and return
    return workflow.compile()

