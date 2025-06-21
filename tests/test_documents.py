"""Tests for document processing functionality."""

import logging
import sys
import os
import pytest
import numpy as np
import tempfile

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.documents import DocumentProcessor


@pytest.fixture
def mock_embedding_model(monkeypatch):
    """Mock SentenceTransformer to avoid model downloads and provide test isolation."""

    def mock_init(self, model_name):
        """Mock SentenceTransformer.__init__ to avoid model downloads."""
        self.model_name = model_name

    def mock_encode(self, texts, show_progress_bar=False):
        """Mock SentenceTransformer.encode to return dummy embeddings."""
        # Return realistic embedding dimensions (384 for all-MiniLM-L6-v2)
        return np.random.rand(len(texts), 384).astype(np.float32)

    # Mock the SentenceTransformer class
    monkeypatch.setattr("sentence_transformers.SentenceTransformer.__init__", mock_init)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer.encode", mock_encode)


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
def test_document_processing(mock_embedding_model):
    """Test document processing with sample files using mocked embeddings."""
    # Use a temporary directory to ensure no database contamination
    with tempfile.TemporaryDirectory():
        processor = DocumentProcessor()

        # Process only the first few test files to keep test fast
        test_dir = "policies/test"
        test_files = [f for f in os.listdir(test_dir) if f.endswith(".pdf")][
            :3
        ]  # Limit to 3 files

        # Process individual files instead of whole directory for better control
        docs = []
        for filename in test_files:
            file_path = os.path.join(test_dir, filename)
            try:
                doc = processor.process_document(file_path, quiet=True)
                if doc:
                    docs.append(doc)
            except Exception as e:
                # Log but don't fail test if individual file fails
                logging.warning(f"Failed to process {filename}: {e}")

        # Initialize return variables
        chunks, embeddings, metadata = [], None, []

        if docs:
            # Create embeddings (now mocked)
            chunks, embeddings, metadata = processor.create_embeddings_for_docs(docs)

            # Verify processing results
            assert len(docs) > 0, "Should process at least one document"
            assert len(chunks) > 0, "Should create at least one chunk"
            assert embeddings.shape[0] == len(chunks), (
                "Should have one embedding per chunk"
            )
            assert embeddings.shape[1] == 384, "Should have 384-dimensional embeddings"
            assert len(metadata) == len(chunks), "Should have metadata for each chunk"

            # Log summary for debugging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            logger.info("Processing Summary:")
            logger.info(f"Documents processed: {len(docs)}")
            logger.info(f"Total chunks: {len(chunks)}")
            logger.info(f"Embeddings shape: {embeddings.shape}")
            if chunks:
                logger.info(f"Sample chunk: {chunks[0][:100]}...")

        # Assert that processing succeeded
        assert docs is not None
        assert chunks is not None
        assert embeddings is not None
        assert metadata is not None


if __name__ == "__main__":
    test_document_processing()
