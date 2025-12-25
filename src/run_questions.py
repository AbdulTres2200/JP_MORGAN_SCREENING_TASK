"""
Run all screening questions through the RAG system.

This module executes the complete set of screening questions and saves results
to both JSON and Markdown formats in the outputs/ directory.
"""
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph import build_rag_graph


def ask_question(graph, question: str, question_num: str):
    """
    Ask a question to the RAG system and display the result.
    
    Args:
        graph: The compiled LangGraph workflow
        question: The question to ask
        question_num: Question number/identifier for tracking
        
    Returns:
        dict: Result data including question, answer, route, and metadata
    """
    print("\n" + "=" * 80)
    print(f"{question_num}")
    print("=" * 80)
    print(f"Question: {question}\n")
    print("Processing...\n")
    
    # Run the graph with the question
    initial_state = {
        "question": question,
        "route": "",
        "documents": [],
        "answer": ""
    }
    
    # Invoke the graph
    result = graph.invoke(initial_state)
    
    # Display results
    print("=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(result["answer"])
    print("=" * 80)
    print(f"\nRoute: {result['route']}")
    print(f"Documents retrieved: {len(result['documents'])}")
    if result['documents']:
        sources = [doc.metadata.get('source') for doc in result['documents']]
        print(f"Sources: {set(sources)}")
    print("\n")
    
    # Prepare result data for saving
    result_data = {
        "question_num": question_num,
        "question": question,
        "answer": result["answer"],
        "route": result["route"],
        "documents_retrieved": len(result["documents"]),
        "sources": list(set([doc.metadata.get('source') for doc in result['documents']])) if result['documents'] else []
    }
    
    return result_data


def save_results(results):
    """
    Save questions and answers to files (JSON and Markdown).
    
    Creates timestamped output files in the outputs/ directory:
    - JSON format for programmatic access
    - Markdown format for human readability
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    json_file = output_dir / f"rag_results_{timestamp}.json"
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(results),
        "results": results
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to JSON: {json_file}")
    
    # Save as Markdown (readable format)
    md_file = output_dir / f"rag_results_{timestamp}.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# RAG Screening Task - Questions and Answers\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Questions:** {len(results)}\n\n")
        f.write("---\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"## {result['question_num']}\n\n")
            f.write(f"**Question:** {result['question']}\n\n")
            f.write(f"**Route:** {result['route']}\n\n")
            f.write(f"**Documents Retrieved:** {result['documents_retrieved']}\n\n")
            if result['sources']:
                f.write(f"**Sources:** {', '.join(result['sources'])}\n\n")
            f.write("**Answer:**\n\n")
            f.write(f"{result['answer']}\n\n")
            f.write("---\n\n")
    
    print(f"Results saved to Markdown: {md_file}")
    print(f"\nOutput directory: {output_dir}")


def main():
    """
    Run all screening questions through the RAG system.
    
    Executes the complete set of screening questions organized by category:
    - Q1: Forecasted Equity Themes
    - Q2: Mid-Year Reality Check
    - Q3: Stock-Level Comparison
    - Q4: Valuation and Risk
    - Q5: Structured Output
    
    Results are saved to timestamped files in outputs/ directory.
    """
    print("=" * 80)
    print("RAG Screening Task - Running All Questions")
    print("=" * 80)
    
    # Build the graph once
    print("Building RAG graph...")
    graph = build_rag_graph()
    print("Graph built successfully!\n")
    
    # Define all questions
    questions = [
        {
            "num": "Q1 – Forecasted Equity Themes",
            "questions": [
                "According to Outlook 2025: Which equity market themes were expected to perform well in 2025?",
                "According to Outlook 2025: Which specific stocks or groups of stocks (e.g. Apple, Microsoft, Magnificent 7, AI-related equities) were highlighted?"
            ]
        },
        {
            "num": "Q2 – Mid-Year Reality Check",
            "questions": [
                "According to Mid-Year Outlook 2025: Which forecasted themes played out as expected?",
                "According to Mid-Year Outlook 2025: Which underperformed or disappointed?"
            ]
        },
        {
            "num": "Q3 – Stock-Level Comparison",
            "questions": [
                "Identify at least two named stocks (e.g. Apple, Microsoft, NVIDIA): What was implied or stated about them in the 2025 forecast? How their performance or outlook is described at mid-year 2025"
            ]
        },
        {
            "num": "Q4 – Valuation and Risk",
            "questions": [
                "What valuation or risk concerns were highlighted at the start of 2025? Which of those risks materialized by mid-year, according to J.P. Morgan?"
            ]
        },
        {
            "num": "Q5 – Structured Output",
            "questions": [
                "Produce the following table: | Stock / Theme | 2025 Forecast View | Mid-Year 2025 Reality | Supported? (Yes/No) | Citation |"
            ]
        }
    ]
    
    # Store all results
    all_results = []
    
    # Run all questions
    for q_group in questions:
        print("\n" + "#" * 80)
        print(f"# {q_group['num']}")
        print("#" * 80)
        
        for i, question in enumerate(q_group['questions'], 1):
            question_id = f"{q_group['num']} - Part {i}"
            result = ask_question(graph, question, question_id)
            all_results.append(result)
    
    print("\n" + "=" * 80)
    print("All questions completed!")
    print("=" * 80)
    
    # Save results to files
    save_results(all_results)


if __name__ == "__main__":
    main()

