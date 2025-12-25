# RAG Screening Task - J.P. Morgan Outlook 2025

A Retrieval-Augmented Generation (RAG) system for answering questions about J.P. Morgan's 2025 Outlook and Mid-Year Outlook documents. The system uses LangGraph to route questions to appropriate document sources and synthesizes answers with analyst-level reasoning.

## Features

- **Intelligent Routing**: Automatically routes questions to forecast, mid-year, or both document sources
- **Analyst-Level Reasoning**: Generates answers with equity-focused analysis, distinguishing between equity themes and macro themes
- **Cross-Document Synthesis**: Compares forecast predictions with mid-year reality
- **Structured Output**: Produces well-formatted tables with concise, readable cells
- **Citation Support**: Includes citation reference tables for all answers

## Project Structure

```
Rag_screening_task/
├── src/
│   ├── ingest.py          # Document ingestion and vectorstore management
│   ├── graph.py            # RAG graph pipeline and synthesis logic
│   ├── main.py             # Simple CLI for single questions
│   └── run_questions.py    # Run all screening questions
├── tests/
│   └── test_retrieving.py  # Test retrieval functionality
├── pdfs/
│   ├── forecast/           # Outlook 2025 PDF files
│   └── midyear/            # Mid-Year Outlook 2025 PDF files
├── chroma_db/              # ChromaDB vector database (created after ingestion)
├── outputs/                # Generated results (JSON and Markdown)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Prerequisites

- Python 3.11 or higher
- OpenAI API key
- PDF documents in `pdfs/forecast/` and `pdfs/midyear/` directories

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Rag_screening_task
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env.local` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Configuration

### Environment Variables

The system uses a `.env.local` file for configuration:

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Technical Parameters

#### Embedding Model
- **Model**: `text-embedding-3-small`
- **Purpose**: Generates vector embeddings for document chunks
- **Dimensions**: 1536 (default for text-embedding-3-small)

#### Language Model
- **Model**: `gpt-4o-mini`
- **Purpose**: Generates answers from retrieved documents
- **Temperature**: 0 (deterministic outputs)
- **Reason**: Cost-effective model suitable for screening tasks

#### Chunking Strategy
- **Chunk Size**: 2500 tokens
- **Chunk Overlap**: 400 tokens
- **Tokenizer**: `cl100k_base` (OpenAI's tokenizer via tiktoken)
- **Separators**: `["\n\n\n", "\n\n", "\n", ". ", " ", ""]` (respects paragraph boundaries)
- **Rationale**: Larger chunks preserve complete arguments and sections, critical for screening tasks

#### Retrieval Parameters
- **k (Number of Documents)**: 12
- **Retrieval Method**: Maximum Marginal Relevance (MMR)
- **MMR Diversity**: 0.7 (higher = more diverse results)
- **MMR Fetch k**: min(k × 3, 50) = 36 candidates
- **Rationale**: MMR balances relevance and diversity, preventing redundant chunks

#### Vector Database
- **Database**: ChromaDB (persistent vector store)
- **Location**: `chroma_db/` directory
- **Collection Name**: `rag_documents`
- **Storage**: SQLite backend with vector embeddings

## Usage

### 1. Ingest Documents

First, place your PDF files in the appropriate directories:
- `pdfs/forecast/` - Outlook 2025 documents
- `pdfs/midyear/` - Mid-Year Outlook 2025 documents

Then run the ingestion script:

```bash
python src/ingest.py
```

This will:
- Load all PDFs from both directories
- Chunk documents (2500 tokens, 400 overlap)
- Generate embeddings using `text-embedding-3-small`
- Store in ChromaDB at `chroma_db/`

**Note**: Re-run ingestion if you change chunking parameters or add new documents.

### 2. Run All Screening Questions

To run the complete set of screening questions:

```bash
python src/run_questions.py
```

This executes all questions and saves results to:
- `outputs/rag_results_YYYYMMDD_HHMMSS.json` (structured data)
- `outputs/rag_results_YYYYMMDD_HHMMSS.md` (human-readable)

### 3. Ask Individual Questions

For testing single questions:

```bash
python src/main.py
```

Or use the function programmatically:

```python
from src.main import ask_question

result = ask_question("According to Outlook 2025: Which equity market themes were expected to perform well in 2025?")
print(result["answer"])
```

### 4. Test Retrieval

To verify retrieval functionality:

```bash
python tests/test_retrieving.py
```

## How It Works

### 1. Document Ingestion (`ingest.py`)

1. **Load PDFs**: Extracts text from PDF files in `pdfs/forecast/` and `pdfs/midyear/`
2. **Chunk Documents**: Splits into 2500-token chunks with 400-token overlap
3. **Generate Embeddings**: Creates vector embeddings using OpenAI's `text-embedding-3-small`
4. **Store in ChromaDB**: Persists embeddings with metadata (source, document name, page number)

### 2. Question Routing (`graph.py`)

The router analyzes the question and routes to:
- **"forecast"**: Questions about Outlook 2025 / forecast documents
- **"midyear"**: Questions about Mid-Year Outlook 2025
- **"both"**: Questions requiring comparison between forecast and mid-year

### 3. Document Retrieval

- Retrieves k=12 documents using MMR for diversity
- Filters by source (forecast/midyear/both) based on routing
- Returns documents with metadata for citation

### 4. Answer Synthesis

1. **Format Documents**: Numbers documents with citations [1], [2], etc.
2. **Apply Analyst Prompts**: Uses equity-focused reasoning rules:
   - Distinguishes equity themes from macro themes
   - Identifies highlighted stocks vs illustrative mentions
   - Maintains semantic precision (valuation concerns ≠ underperformance)
   - Enables cross-document synthesis
3. **Generate Answer**: Uses `gpt-4o-mini` to synthesize answer
4. **Post-Process**: Truncates table cells to max 60 characters for readability
5. **Add Citations**: Appends citation reference table

## Example Questions

The system handles various question types:

- **Forecast Questions**: "According to Outlook 2025: Which equity market themes were expected to perform well in 2025?"
- **Mid-Year Questions**: "According to Mid-Year Outlook 2025: Which forecasted themes played out as expected?"
- **Comparison Questions**: "What was implied or stated about Microsoft in the 2025 forecast? How is its performance described at mid-year 2025?"
- **Structured Output**: "Produce a table with Stock/Theme, 2025 Forecast View, Mid-Year 2025 Reality, Supported? (Yes/No), Citation"

## Output Format

Answers include:
- **Main Answer**: Synthesized response with citations
- **Tables**: Formatted markdown tables (cells max 60 characters)
- **Citation Reference**: Table mapping [N] citations to source, document, and page

Example citation format:
```
[1] Source: forecast, Document: outlook-2025-building-on-strength, Page: 6
```

## Key Design Decisions

### Why 2500-token chunks?
- Preserves complete arguments and sections
- Better for screening tasks requiring context
- Reduces fragmentation of related information

### Why MMR instead of simple similarity?
- Prevents redundant chunks from same document
- Increases diversity of retrieved information
- Better coverage of different themes/sections

### Why k=12 documents?
- Balances recall (finding relevant info) with precision
- Provides sufficient context for synthesis
- MMR ensures diversity even with larger k

### Why gpt-4o-mini?
- Cost-effective for screening tasks
- Sufficient capability for document synthesis
- Deterministic outputs (temperature=0) for consistency

## Troubleshooting

### "No documents loaded" error
- Ensure PDF files exist in `pdfs/forecast/` and `pdfs/midyear/`
- Check file permissions
- Verify PDFs are not corrupted

### "Persistence directory does not exist"
- Run `python src/ingest.py` first to create the vectorstore

### "OpenAI API key not found"
- Create `.env.local` file in project root
- Add `OPENAI_API_KEY=your_key_here`

### Poor retrieval quality
- Re-ingest documents with different chunking parameters
- Adjust k value in `ingest.py` (default: 12)
- Modify MMR diversity parameter (default: 0.7)

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- `langchain` & `langgraph`: RAG framework and graph orchestration
- `langchain-openai`: OpenAI integration
- `chromadb`: Vector database
- `tiktoken`: Token counting for accurate chunking
- `pypdf`: PDF text extraction

## Out of Scope / Future Improvements

This project serves as a demonstration of RAG system understanding and implementation. The following areas are identified for future enhancement given more time and deeper context understanding:

### Table Formatting
- **Current State**: Tables are post-processed to enforce 60-character cell limits for readability
- **Future Work**: Refine table formatting logic to better handle complex financial data, improve column alignment, and enhance visual presentation

### Citation System
- **Current State**: Citations include page number and chunk index from the source documents
- **Future Work**: Improve citation granularity and accuracy:
  - More precise section-level citations
  - Better handling of cross-references
  - Enhanced citation formatting for different document types

### Analyst-Level Prompt Quality
- **Current State**: Prompt includes equity-focused reasoning rules and semantic precision guidelines
- **Future Work**: With deeper understanding of the financial context and screening requirements:
  - Refine prompt engineering for more nuanced analyst-level reasoning
  - Better distinction between investment conviction levels
  - Enhanced cross-document synthesis capabilities
  - More sophisticated handling of implied vs explicit information

### Additional Enhancements
- Fine-tune chunking strategy based on document structure analysis
- Implement confidence scoring for retrieved documents
- Add support for multi-turn conversations
- Enhance error handling and edge case management

**Note**: This project prioritizes demonstrating core RAG capabilities, technical implementation, and system architecture. The improvements listed above would be addressed in a production environment with dedicated time for domain-specific refinement.

 
