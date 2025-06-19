"""Tests for vector store functionality."""

import logging
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vectorstore import SimpleVectorStore
from src.documents import DocumentProcessor


def test_vector_store():
    """Test vector store functionality."""
    # Process documents
    processor = DocumentProcessor()
    docs = processor.process_directory("policies/test")

    if not docs:
        logging.getLogger(__name__).warning("No documents to process!")
        return

    # Create embeddings
    _, embeddings, metadata = processor.create_embeddings_for_docs(docs)

    # Initialize vector store
    store = SimpleVectorStore(collection_name="test_iris")

    # Add documents to store
    store.add_documents(docs, embeddings, metadata)

    # Test search
    query = "military personnel policy"
    query_embedding = processor.generate_embeddings([query])[0]

    results = store.similarity_search(query_embedding, top_k=3)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Search Results for '%s':", query)
    for i, result in enumerate(results, 1):
        logger.info("%s. %s (chunk %s)", i, result["filename"], result["chunk_index"])
        logger.info("   Similarity: %.3f", result["similarity"])
        logger.info("   Text: %s...", result["chunk_text"][:150])

    # Show stats
    stats = store.get_stats()
    logger.info("Vector Store Stats:")
    for key, value in stats.items():
        logger.info("  %s: %s", key, value)

    store.close()
    return store


if __name__ == "__main__":
    test_vector_store()
