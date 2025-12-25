import os
import tiktoken
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load .env.local from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env.local")


def load_pdfs_from_folder(base_path):
    """Load all PDF files from forecast and midyear folders."""
    all_docs = []
    project_root = Path(__file__).parent.parent
    base_path_full = project_root / base_path

    for doc_type in ["forecast", "midyear"]:
        folder = base_path_full / doc_type

        if not folder.exists():
            print(f"Warning: Folder {folder} does not exist")
            continue

        files = os.listdir(str(folder))
        pdf_files = [f for f in files if f.lower().endswith(".pdf") and not f.startswith(".")]
        
        if not pdf_files:
            print(f"Warning: No PDF files found in {folder}")
            continue

        for file in pdf_files:
            file_path = folder / file
            try:
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()

                # Add metadata to each document page
                for d in docs:
                    d.metadata["source"] = doc_type
                    d.metadata["doc_name"] = file.replace(".pdf", "").replace("_", " ")

                all_docs.extend(docs)
                print(f"Loaded {len(docs)} pages from {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

    return all_docs


def chunk_documents(documents, chunk_size_tokens=2500, chunk_overlap_tokens=400):
    """
    Split documents into chunks based on token count.
    
    Uses larger chunks (2500 tokens) for argument-level content that preserves
    complete thoughts and sections, which is critical for screening tasks.
    """
    # Use OpenAI's tokenizer to count tokens accurately
    encoding = tiktoken.get_encoding("cl100k_base")
    
    def length_function(text):
        return len(encoding.encode(text))
    
    # Use separators that respect document structure
    # Prioritize paragraph breaks, then sentences, then words
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
        length_function=length_function,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]  # Respect paragraph boundaries
    )
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk index metadata for better tracking
    for i, chunk in enumerate(chunks):
        if "chunk_index" not in chunk.metadata:
            chunk.metadata["chunk_index"] = i
        # Preserve original page info
        if "page" in chunk.metadata:
            chunk.metadata["page_num"] = chunk.metadata["page"]
    
    return chunks


def create_embeddings_and_store(chunks, persist_directory=None, model="text-embedding-3-small", collection_name="rag_documents"):
    """Generate embeddings and store in ChromaDB vector database."""
    if not chunks:
        raise ValueError("Cannot create embeddings: chunks list is empty")
    
    if persist_directory is None:
        project_root = Path(__file__).parent.parent
        persist_directory = str(project_root / "chroma_db")
    
    # Ensure directory exists
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    
    embeddings = OpenAIEmbeddings(model=model)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    print(f"Vectorstore saved to {persist_directory} with collection '{collection_name}'")
    return vectorstore


def ingest_documents(base_path="pdfs", chunk_size_tokens=2500, chunk_overlap_tokens=400, persist_directory=None, embedding_model="text-embedding-3-small"):
    """Complete ingestion pipeline: load PDFs, chunk, embed, and store."""
    documents = load_pdfs_from_folder(base_path)
    
    if not documents:
        raise ValueError(f"No documents loaded from {base_path}. Please ensure PDF files exist in forecast/ and midyear/ folders.")
    
    print(f"Loaded {len(documents)} document pages")
    
    chunks = chunk_documents(documents, chunk_size_tokens=chunk_size_tokens, chunk_overlap_tokens=chunk_overlap_tokens)
    print(f"Created {len(chunks)} chunks")
    
    vectorstore = create_embeddings_and_store(chunks, persist_directory=persist_directory, model=embedding_model)
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")
    return vectorstore


def load_vectorstore(persist_directory=None, embedding_model="text-embedding-3-small", collection_name="rag_documents"):
    """Load existing ChromaDB vectorstore from disk."""
    if persist_directory is None:
        project_root = Path(__file__).parent.parent
        persist_directory = str(project_root / "chroma_db")
    
    if not Path(persist_directory).exists():
        raise ValueError(f"Persistence directory {persist_directory} does not exist. Please run ingestion first.")
    
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    return vectorstore


def create_retriever(vectorstore, source_filter=None, k=12, use_mmr=True, mmr_diversity=0.7):
    """
    Create a retriever with optional source filtering and MMR for diversity.
    
    Args:
        vectorstore: Chroma vectorstore instance
        source_filter: None (both), "forecast", or "midyear"
        k: Number of documents to retrieve (increased for better recall)
        use_mmr: Use Maximum Marginal Relevance for diverse results
        mmr_diversity: MMR diversity parameter (0-1, higher = more diverse)
    
    Returns:
        Retriever instance
    """
    search_kwargs = {"k": k}
    
    if use_mmr:
        # Use MMR to balance relevance and diversity
        search_kwargs["fetch_k"] = min(k * 3, 50)  # Fetch more candidates for MMR
        search_kwargs["lambda_mult"] = mmr_diversity
    
    if source_filter is not None:
        search_kwargs["filter"] = {"source": source_filter}
    
    retriever = vectorstore.as_retriever(search_type="mmr" if use_mmr else "similarity", search_kwargs=search_kwargs)
    return retriever


def get_forecast_retriever(persist_directory=None, k=12, embedding_model="text-embedding-3-small", use_mmr=True):
    """Get retriever for forecast documents only."""
    vectorstore = load_vectorstore(persist_directory, embedding_model)
    return create_retriever(vectorstore, source_filter="forecast", k=k, use_mmr=use_mmr)


def get_midyear_retriever(persist_directory=None, k=12, embedding_model="text-embedding-3-small", use_mmr=True):
    """Get retriever for mid-year documents only."""
    vectorstore = load_vectorstore(persist_directory, embedding_model)
    return create_retriever(vectorstore, source_filter="midyear", k=k, use_mmr=use_mmr)


def get_combined_retriever(persist_directory=None, k=12, embedding_model="text-embedding-3-small", use_mmr=True):
    """Get retriever for both forecast and mid-year documents."""
    vectorstore = load_vectorstore(persist_directory, embedding_model)
    return create_retriever(vectorstore, source_filter=None, k=k, use_mmr=use_mmr)


if __name__ == "__main__":
    vectorstore = ingest_documents()
    print("Ingestion complete!")