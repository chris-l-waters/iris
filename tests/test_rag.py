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

    # Cleanup after test - Windows-safe approach
    if os.path.exists(temp_dir):
        # Try to close any ChromaDB clients that might still be open
        try:
            # Force garbage collection to close any lingering connections
            import gc

            gc.collect()

            # Additional Windows-specific cleanup
            import platform

            if platform.system() == "Windows":
                # Add a small delay to allow file handles to be released
                import time

                time.sleep(0.1)
        except Exception:
            pass

        # Retry deletion with error handling
        max_retries = 3
        for attempt in range(max_retries):
            try:
                shutil.rmtree(temp_dir)
                break
            except (OSError, PermissionError) as e:
                if attempt == max_retries - 1:
                    # Final attempt failed, log but don't fail test
                    print(f"Warning: Could not clean up temp directory {temp_dir}: {e}")
                else:
                    # Wait before retry
                    import time

                    time.sleep(0.1)


def test_rag_init(temp_database_dir):
    """Test RAG initialization."""
    rag = SimpleRAG()
    try:
        assert rag is not None
        assert rag.collection_name == "iris_all_MiniLM_L6_v2"
    finally:
        # Ensure proper cleanup
        if hasattr(rag, "vectorstore") and hasattr(rag.vectorstore, "close"):
            rag.vectorstore.close()


def test_rag_init_with_custom_path(temp_database_dir):
    """Test RAG initialization with custom collection name."""
    # Test that we can create RAG with a custom embedding model
    rag_custom = SimpleRAG(embedding_model="all-mpnet-base-v2")
    try:
        assert rag_custom.collection_name == "iris_all_mpnet_base_v2"
    finally:
        # Ensure proper cleanup
        if hasattr(rag_custom, "vectorstore") and hasattr(
            rag_custom.vectorstore, "close"
        ):
            rag_custom.vectorstore.close()


def test_query_basic(temp_database_dir):
    """Test basic query functionality."""
    rag = SimpleRAG()
    try:
        result = rag.query("test question")
        assert isinstance(result, str)
        # Should return error message since no documents are loaded
        expected_msg = "Please load documents first using load_documents()"
        assert result == expected_msg
    finally:
        # Ensure proper cleanup
        if hasattr(rag, "vectorstore") and hasattr(rag.vectorstore, "close"):
            rag.vectorstore.close()


def test_query_empty(temp_database_dir):
    """Test query with empty string."""
    rag = SimpleRAG()
    try:
        result = rag.query("")
        assert result == "Please provide a valid question."
    finally:
        # Ensure proper cleanup
        if hasattr(rag, "vectorstore") and hasattr(rag.vectorstore, "close"):
            rag.vectorstore.close()


def test_query_whitespace(temp_database_dir):
    """Test query with only whitespace."""
    rag = SimpleRAG()
    try:
        result = rag.query("   ")
        assert result == "Please provide a valid question."
    finally:
        # Ensure proper cleanup
        if hasattr(rag, "vectorstore") and hasattr(rag.vectorstore, "close"):
            rag.vectorstore.close()


def test_load_documents_method_exists(temp_database_dir):
    """Test that load_documents method exists."""
    rag = SimpleRAG()
    try:
        # Method should exist and be callable
        assert hasattr(rag, "load_documents")
        assert callable(getattr(rag, "load_documents"))
    finally:
        # Ensure proper cleanup
        if hasattr(rag, "vectorstore") and hasattr(rag.vectorstore, "close"):
            rag.vectorstore.close()


def test_is_ready(temp_database_dir):
    """Test system readiness check."""
    rag = SimpleRAG()
    try:
        # Should be False until documents are loaded
        assert not rag.is_ready()
    finally:
        # Ensure proper cleanup
        if hasattr(rag, "vectorstore") and hasattr(rag.vectorstore, "close"):
            rag.vectorstore.close()


def test_get_stats(temp_database_dir):
    """Test get_stats method."""
    rag = SimpleRAG()
    try:
        stats = rag.get_stats()
        assert isinstance(stats, dict)
        # Stats should contain basic database information
        assert "documents" in stats
        assert "chunks" in stats
    finally:
        # Ensure proper cleanup
        if hasattr(rag, "vectorstore") and hasattr(rag.vectorstore, "close"):
            rag.vectorstore.close()
