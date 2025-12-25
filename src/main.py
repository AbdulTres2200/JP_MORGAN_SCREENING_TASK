"""
Simple CLI interface for asking individual questions to the RAG system.

This module provides a lightweight interface for testing single questions.
For running all screening questions, use run_questions.py instead.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph import build_rag_graph


def ask_question(question: str):
    """
    Ask a question to the RAG system.
    
    Args:
        question: The question to ask
        
    Returns:
        dict: Result containing answer, route, and retrieved documents
    """
    # Build the graph once
    graph = build_rag_graph()
    
    # Initialize state and run the graph
    initial_state = {
        "question": question,
        "route": "",
        "documents": [],
        "answer": ""
    }
    
    # Invoke the graph workflow
    result = graph.invoke(initial_state)
    
    return result


if __name__ == "__main__":
    # Simple demo: ask a sample question
    print("=" * 80)
    print("RAG Screening Task - J.P. Morgan Outlook 2025")
    print("=" * 80)
    print("\nThis system can answer questions about:")
    print("  - 2025 Forecast (Outlook 2025)")
    print("  - Mid-Year 2025 Reality (Mid-Year Outlook 2025)")
    print("  - Comparisons between forecast and mid-year")
    print("\n" + "=" * 80)
    
    # Example question
    test_question = "According to Outlook 2025: Which equity market themes were expected to perform well in 2025?"
    
    print(f"\nQuestion: {test_question}\n")
    print("Processing...\n")
    
    result = ask_question(test_question)
    
    print("=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(result["answer"])
    print("=" * 80)
    print(f"\nRoute: {result['route']}")
    print(f"Documents retrieved: {len(result['documents'])}")

