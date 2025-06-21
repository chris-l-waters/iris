"""Tests for RAG functionality."""

import sys
import os
import pytest
import tempfile
import shutil
import chromadb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import SimpleRAG


@pytest.fixture
def temp_database_dir(monkeypatch):
    """Create a temporary directory for test database, ensure cleanup."""
    temp_dir = tempfile.mkdtemp(prefix="iris_test_rag_db_")

    # Patch ALL possible database paths to use our temp dir
    monkeypatch.setattr("src.vectorstore.DEFAULT_CHROMA_PERSIST_DIR", temp_dir)
    monkeypatch.setattr("src.config.DEFAULT_CHROMA_PERSIST_DIR", temp_dir)

    # Also patch the ChromaDB client creation to use temp dir
    original_init = chromadb.PersistentClient.__init__

    def patched_init(self, path=None, **kwargs):
        # Force all ChromaDB clients to use our temp directory
        return original_init(self, path=temp_dir, **kwargs)

    monkeypatch.setattr(chromadb.PersistentClient, "__init__", patched_init)

    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_rag_init(temp_database_dir):
    """Test RAG initialization."""
    rag = SimpleRAG()
    assert rag is not None
    assert rag.collection_name == "iris_all_MiniLM_L6_v2"


def test_rag_init_with_custom_path(temp_database_dir):
    """Test RAG initialization with custom collection name."""
    # Test that we can create RAG with a custom embedding model
    rag_custom = SimpleRAG(embedding_model="all-mpnet-base-v2")
    assert rag_custom.collection_name == "iris_all_mpnet_base_v2"


def test_query_basic(temp_database_dir):
    """Test basic query functionality."""
    rag = SimpleRAG()
    result = rag.query("test question")
    assert isinstance(result, str)
    # Should return error message since no documents are loaded
    expected_msg = "Please load documents first using load_documents()"
    assert result == expected_msg


def test_query_empty(temp_database_dir):
    """Test query with empty string."""
    rag = SimpleRAG()
    result = rag.query("")
    assert result == "Please provide a valid question."


def test_query_whitespace(temp_database_dir):
    """Test query with only whitespace."""
    rag = SimpleRAG()
    result = rag.query("   ")
    assert result == "Please provide a valid question."


def test_load_documents_method_exists(temp_database_dir):
    """Test that load_documents method exists."""
    rag = SimpleRAG()
    # Method should exist and be callable
    assert hasattr(rag, "load_documents")
    assert callable(getattr(rag, "load_documents"))


def test_is_ready(temp_database_dir):
    """Test system readiness check."""
    rag = SimpleRAG()
    # Should be False until documents are loaded
    assert not rag.is_ready()


def test_get_stats(temp_database_dir):
    """Test get_stats method."""
    rag = SimpleRAG()
    stats = rag.get_stats()
    assert isinstance(stats, dict)
    # Stats should contain basic database information
    assert "documents" in stats
    assert "chunks" in stats
