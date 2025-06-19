"""Tests for LLM functionality."""

import logging
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMProcessor


def test_llm():
    """Test LLM functionality."""
    logging.basicConfig(level=logging.INFO)

    llm = LLMProcessor()

    print("LLM available:", llm.is_available())
    print("Model name:", llm.model_name)

    if llm.is_available():
        # Test basic generation
        response = llm.generate_response(
            "What is the capital of France?", max_tokens=50
        )
        print("Test response:", response)

        # Test RAG response
        context = ["Military personnel must follow chain of command protocols."]
        rag_response = llm.generate_rag_response("What are command protocols?", context)
        print("RAG response:", rag_response)
    else:
        print("LLM not available - install Ollama and run: ollama pull", llm.model_name)


if __name__ == "__main__":
    test_llm()
