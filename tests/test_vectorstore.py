"""Tests for vector store functionality."""

import logging
import os
import shutil
import sys
import tempfile

import chromadb
import numpy as np
import pytest

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.documents import DocumentProcessor
from src.vectorstore import SimpleVectorStore


@pytest.fixture
def mock_embedding_model(monkeypatch):
    """Mock SentenceTransformer to avoid model downloads and provide test isolation."""

    def mock_init(self, model_name):
        """Mock SentenceTransformer.__init__ to avoid model downloads."""
        self.model_name = model_name

    def mock_encode(_self, texts, **_kwargs):
        """Mock SentenceTransformer.encode to return dummy embeddings."""
        # Return realistic embedding dimensions (384 for all-MiniLM-L6-v2)
        return np.random.rand(len(texts), 384).astype(np.float32)

    # Mock the SentenceTransformer class
    monkeypatch.setattr("sentence_transformers.SentenceTransformer.__init__", mock_init)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer.encode", mock_encode)


@pytest.fixture
def temp_database_dir(monkeypatch):
    """Create a temporary directory for test database, ensure cleanup."""
    temp_dir = tempfile.mkdtemp(prefix="iris_test_db_")

    # Patch ALL possible database paths to use our temp dir
    monkeypatch.setattr("src.vectorstore.DEFAULT_CHROMA_PERSIST_DIR", temp_dir)
    monkeypatch.setattr("src.config.DEFAULT_CHROMA_PERSIST_DIR", temp_dir)

    # Import and patch the config module constants directly
    import src.config as config_module

    monkeypatch.setattr(config_module, "DEFAULT_CHROMA_PERSIST_DIR", temp_dir)

    # Patch the config instance
    from src.config import config

    monkeypatch.setattr(
        config,
        "_config",
        {
            **config._config,
            "chromadb": {**config._config.get("chromadb", {}), "persist_dir": temp_dir},
        },
    )

    # Also patch the ChromaDB client creation to use temp dir
    original_init = chromadb.PersistentClient.__init__

    def patched_init(self, **kwargs):
        # Force all ChromaDB clients to use our temp directory
        kwargs.pop("path", None)  # Remove any existing path
        return original_init(self, path=temp_dir, **kwargs)

    monkeypatch.setattr(chromadb.PersistentClient, "__init__", patched_init)

    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.mark.skipif(
    not os.path.exists(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "policies",
            "test",
        )
    ),
    reason="Test PDFs not available in CI",
)
def test_vector_store(mock_embedding_model, temp_database_dir):
    """Test vector store functionality with complete isolation."""
    # Fixtures are used implicitly through pytest's dependency injection
    _ = mock_embedding_model, temp_database_dir  # Acknowledge fixture usage

    # Create test documents with mocked embeddings
    processor = DocumentProcessor()
    test_dir = "policies/test"
    test_files = [f for f in os.listdir(test_dir) if f.endswith(".pdf")][:2]

    # Process documents
    docs = []
    for filename in test_files:
        file_path = os.path.join(test_dir, filename)
        try:
            doc = processor.process_document(file_path, quiet=True)
            if doc:
                docs.append(doc)
        except (OSError, IOError, ValueError) as e:
            logging.warning("Failed to process %s: %s", filename, e)

    if not docs:
        pytest.skip("No documents could be processed")

    # Create embeddings and initialize store
    chunks, embeddings, metadata = processor.create_embeddings_for_docs(docs)
    store = SimpleVectorStore(collection_name="test_vectorstore_isolated")

    try:
        # Add documents and test search
        store.add_documents(docs, embeddings, metadata)
        query_embedding = np.random.rand(384).astype(np.float32)
        results = store.similarity_search(query_embedding, top_k=3)

        # Verify results
        assert isinstance(results, list), "Results should be a list"
        assert len(results) <= 3, "Should return at most 3 results"
        assert len(results) <= len(chunks), "Cannot return more results than chunks"

        # Verify stats
        stats = store.get_stats()
        assert stats["documents"] > 0, "Should have processed documents"
        assert stats["chunks"] == len(chunks), "Should have correct chunk count"

        # Verify result structure
        for result in results:
            assert "filename" in result, "Result should have filename"
            assert "chunk_index" in result, "Result should have chunk_index"
            assert "chunk_text" in result, "Result should have chunk_text"
            assert "similarity" in result, "Result should have similarity score"
            assert isinstance(result["similarity"], (int, float)), (
                "Similarity should be numeric"
            )

    finally:
        store.close()


if __name__ == "__main__":
    # Cannot run directly due to fixture dependencies
    pytest.main([__file__])
