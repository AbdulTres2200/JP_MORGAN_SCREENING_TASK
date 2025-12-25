"""
Test script for verifying document retrieval functionality.

This script tests the retrieval system to ensure:
- Vectorstore loads correctly
- Retrievers work with proper source filtering
- Documents are retrieved with correct metadata
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingest import load_vectorstore, get_forecast_retriever, get_midyear_retriever, get_combined_retriever
import chromadb

# Check ChromaDB collections
print("Checking ChromaDB collections...")
persist_dir = str(project_root / "chroma_db")
client = chromadb.PersistentClient(path=persist_dir)
collections = client.list_collections()
print(f"Found {len(collections)} collections:")
for col in collections:
    print(f"  - {col.name}: {col.count()} documents")

# Load vectorstore
print("\nLoading vectorstore...")
try:
    vectorstore = load_vectorstore()
    count = vectorstore._collection.count()
    print(f"Collection count: {count}")
except Exception as e:
    print(f"Error loading: {e}")
    # Try to load the first available collection
    if collections:
        collection_name = collections[0].name
        print(f"Trying to load collection: {collection_name}")
        vectorstore = load_vectorstore(collection_name=collection_name)
        count = vectorstore._collection.count()
        print(f"Collection count: {count}")

# Test combined retriever (no filter)
print("\n=== Testing Combined Retriever (no filter) ===")
combined_retriever = get_combined_retriever(k=4)
docs = combined_retriever.invoke("economic outlook")
print(f"Found {len(docs)} documents")
for i, doc in enumerate(docs[:2], 1):
    print(f"  {i}. Source: {doc.metadata.get('source')}, Doc: {doc.metadata.get('doc_name')}")
    print(f"     Preview: {doc.page_content[:100]}...")

# Test forecast retriever
print("\n=== Testing Forecast Retriever ===")
forecast_retriever = get_forecast_retriever(k=4)
docs = forecast_retriever.invoke("economic outlook")
print(f"Found {len(docs)} documents")
for i, doc in enumerate(docs[:2], 1):
    print(f"  {i}. Source: {doc.metadata.get('source')}, Doc: {doc.metadata.get('doc_name')}")

# Test mid-year retriever
print("\n=== Testing Mid-Year Retriever ===")
midyear_retriever = get_midyear_retriever(k=4)
docs = midyear_retriever.invoke("economic outlook")
print(f"Found {len(docs)} documents")
for i, doc in enumerate(docs[:2], 1):
    print(f"  {i}. Source: {doc.metadata.get('source')}, Doc: {doc.metadata.get('doc_name')}")