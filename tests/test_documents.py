"""Tests for document processing functionality."""

import logging
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.documents import DocumentProcessor


def test_document_processing():
    """Test document processing with sample files."""
    processor = DocumentProcessor()

    # Process test directory
    docs = processor.process_directory("policies/test")

    # Initialize return variables
    chunks, embeddings, metadata = [], None, []

    if docs:
        # Create embeddings
        chunks, embeddings, metadata = processor.create_embeddings_for_docs(docs)

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("Processing Summary:")
        logger.info(f"Documents processed: {len(docs)}")
        logger.info(f"Total chunks: {len(chunks)}")
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"Sample chunk: {chunks[0][:100]}...")

    # Assert that processing succeeded
    assert docs is not None
    assert chunks is not None
    assert embeddings is not None
    assert metadata is not None


if __name__ == "__main__":
    test_document_processing()
