"""Tests for RAG functionality."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import SimpleRAG


def test_rag_init():
    """Test RAG initialization."""
    rag = SimpleRAG()
    assert rag is not None
    assert rag.collection_name == "iris_all_MiniLM_L6_v2"


def test_rag_init_with_custom_path():
    """Test RAG initialization with custom collection name."""
    # Test that we can create RAG with a custom embedding model
    rag_custom = SimpleRAG(embedding_model="all-mpnet-base-v2")
    assert rag_custom.collection_name == "iris_all_mpnet_base_v2"


def test_query_basic():
    """Test basic query functionality."""
    rag = SimpleRAG()
    result = rag.query("test question")
    assert isinstance(result, str)
    # Should return error message since no documents are loaded
    expected_msg = "Please load documents first using load_documents()"
    assert result == expected_msg


def test_query_empty():
    """Test query with empty string."""
    rag = SimpleRAG()
    result = rag.query("")
    assert result == "Please provide a valid question."


def test_query_whitespace():
    """Test query with only whitespace."""
    rag = SimpleRAG()
    result = rag.query("   ")
    assert result == "Please provide a valid question."


def test_load_documents_method_exists():
    """Test that load_documents method exists."""
    rag = SimpleRAG()
    # Method should exist and be callable
    assert hasattr(rag, "load_documents")
    assert callable(getattr(rag, "load_documents"))


def test_is_ready():
    """Test system readiness check."""
    rag = SimpleRAG()
    # Should be False until documents are loaded
    assert not rag.is_ready()


def test_get_stats():
    """Test get_stats method."""
    rag = SimpleRAG()
    stats = rag.get_stats()
    assert isinstance(stats, dict)
    # Stats should contain basic database information
    assert "documents" in stats
    assert "chunks" in stats
