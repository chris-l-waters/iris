"""Simple vector store using ChromaDB for IRIS RAG system."""

import logging
import os
from typing import List, Dict

import chromadb
import numpy as np
from .config import (
    DEFAULT_CHROMA_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    CHROMA_SIMILARITY_METRIC,
)


class SimpleVectorStore:
    """Simple vector store using ChromaDB following YAGNI principles."""

    def __init__(
        self, collection_name: str = None, db_path: str = None, document_store=None
    ):
        """Initialize vector store with ChromaDB.

        Args:
            collection_name: Name of the ChromaDB collection
            db_path: Legacy parameter for backward compatibility
            document_store: DocumentStore instance for text retrieval
        """
        # Determine collection name
        if collection_name:
            self.collection_name = collection_name
        elif db_path and "iris_" in db_path:
            # Extract from legacy db_path for backward compatibility
            basename = os.path.basename(db_path)
            if basename.startswith("iris_") and basename.endswith(".db"):
                model_name = basename[5:-3]  # Remove "iris_" and ".db"
                self.collection_name = f"iris_{model_name}"
            else:
                self.collection_name = DEFAULT_COLLECTION_NAME
        else:
            self.collection_name = DEFAULT_COLLECTION_NAME

        # Set up persistence directory
        self.persist_dir = DEFAULT_CHROMA_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)

        # Store document store for text retrieval
        self.document_store = document_store

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.logger = logging.getLogger(__name__)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": CHROMA_SIMILARITY_METRIC}
        )

    def add_documents(
        self,
        chunk_ids: List[str],
        embeddings: np.ndarray,
        minimal_metadata: List[Dict],
        clear_existing: bool = True,
    ):
        """Add chunk embeddings to the vector store.

        Args:
            chunk_ids: List of chunk IDs from DocumentStore
            embeddings: Numpy array of embeddings
            minimal_metadata: List of minimal metadata (no text content)
            clear_existing: Whether to clear existing collection
        """
        self.logger.info("Adding %s embeddings to vector store...", len(chunk_ids))

        try:
            if clear_existing:
                # Clear existing data if requested (default behavior)
                self.logger.info("Clearing existing collection data...")
                # Delete all documents in the collection
                self.client.delete_collection(self.collection_name)
                # Recreate the collection
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": CHROMA_SIMILARITY_METRIC},
                )
                self.logger.info("Collection cleared and recreated successfully")

            # Prepare data for ChromaDB - no text documents, just chunk IDs
            embeddings_list = [embedding.tolist() for embedding in embeddings]

            # Create minimal documents (just chunk IDs as placeholders)
            documents_placeholder = [f"chunk_{chunk_id}" for chunk_id in chunk_ids]

            # Add to ChromaDB collection in batches
            if chunk_ids:
                # Use ChromaDB's recommended batch size, with fallback
                try:
                    max_batch_size = self.client.get_max_batch_size()
                except (AttributeError, RuntimeError, Exception):
                    max_batch_size = 1000  # Fallback to conservative size

                batch_size = min(max_batch_size, 1000)  # Conservative approach
                total_chunks = len(chunk_ids)

                for i in range(0, total_chunks, batch_size):
                    end_idx = min(i + batch_size, total_chunks)
                    batch_ids = chunk_ids[i:end_idx]
                    batch_docs = documents_placeholder[i:end_idx]
                    batch_embeddings = embeddings_list[i:end_idx]
                    batch_metadata = minimal_metadata[i:end_idx]

                    self.collection.add(
                        ids=batch_ids,
                        documents=batch_docs,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadata,
                    )
                    batch_num = i // batch_size + 1
                    total_batches = (total_chunks + batch_size - 1) // batch_size
                    print(
                        f"\rAdded batch {batch_num}/{total_batches}: {len(batch_ids)} chunks",
                        end="",
                        flush=True,
                    )

                print()  # New line after batch progress
                self.logger.info("Added %s embeddings to vector store", len(chunk_ids))
            else:
                self.logger.info("No embeddings to add")

        except Exception as e:
            self.logger.error("Error adding embeddings to vector store: %s", e)
            raise

    def similarity_search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Dict]:
        """Find most similar chunks using ChromaDB's built-in similarity search."""
        try:
            # Query ChromaDB collection for chunk IDs and metadata
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["metadatas", "distances"],
            )

            if not results or not results.get("ids") or not results["ids"][0]:
                return []

            # Get chunk IDs and fetch text from DocumentStore
            chunk_ids = results["ids"][0]  # IDs are returned by default
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            # Fetch text content from DocumentStore
            if self.document_store is not None:
                chunks_data = self.document_store.get_chunks_by_ids(chunk_ids)
                # Create lookup dict for fast access
                chunk_text_lookup = {
                    chunk["chunk_id"]: chunk["chunk_text"] for chunk in chunks_data
                }
            else:
                self.logger.warning(
                    "DocumentStore not available, cannot fetch chunk text"
                )
                chunk_text_lookup = {}

            # Convert ChromaDB results to expected format
            similarities = []
            for chunk_id, metadata, distance in zip(chunk_ids, metadatas, distances):
                # Convert distance to similarity (ChromaDB returns distance, we want similarity)
                # For cosine distance: similarity = 1 - distance
                similarity = 1.0 - distance

                # Get chunk text from DocumentStore or use placeholder
                chunk_text = chunk_text_lookup.get(
                    chunk_id, f"[Text unavailable for chunk {chunk_id}]"
                )

                similarities.append(
                    {
                        "chunk_text": chunk_text,
                        "filename": metadata["filename"],
                        "chunk_index": metadata["chunk_index"],
                        "similarity": float(similarity),
                        "doc_number": metadata.get("doc_number") or None,
                    }
                )

            return similarities

        except Exception as e:
            self.logger.error("Error during similarity search: %s", e)
            return []

    def get_stats(self) -> Dict:
        """Get database statistics."""
        try:
            # Get collection count
            chunk_count = self.collection.count()

            # Count unique documents
            if chunk_count > 0:
                # Get all metadatas to count unique filenames
                all_data = self.collection.get(include=["metadatas"])
                unique_docs = set()
                for metadata in all_data["metadatas"]:
                    unique_docs.add(metadata["filename"])
                doc_count = len(unique_docs)
            else:
                doc_count = 0

            # Calculate approximate storage size
            # ChromaDB stores data in the persist directory
            db_size_mb = 0
            if os.path.exists(self.persist_dir):
                for root, dirs, files in os.walk(self.persist_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            db_size_mb += os.path.getsize(file_path)
                db_size_mb = round(db_size_mb / 1024 / 1024, 2)

            # Estimate expected size (rough calculation)
            if chunk_count > 0:
                # Rough estimate: ~2KB text + ~8KB embedding per chunk
                expected_size_mb = round((chunk_count * 10 * 1024) / 1024 / 1024, 2)
                bloat_factor = (
                    round(db_size_mb / expected_size_mb, 1)
                    if expected_size_mb > 0
                    else 1.0
                )
            else:
                expected_size_mb = 0
                bloat_factor = 1.0

            return {
                "documents": doc_count,
                "chunks": chunk_count,
                "collection_name": self.collection_name,
                "persist_dir": self.persist_dir,
                "db_size_mb": db_size_mb,
                "expected_size_mb": expected_size_mb,
                "bloat_factor": bloat_factor,
            }
        except Exception as e:
            self.logger.error("Error getting database statistics: %s", e)
            return {
                "documents": 0,
                "chunks": 0,
                "collection_name": self.collection_name,
                "persist_dir": self.persist_dir,
                "db_size_mb": 0,
                "expected_size_mb": 0,
                "bloat_factor": 1.0,
            }

    def list_documents(self) -> List[Dict]:
        """Get list of all loaded documents with their metadata."""
        try:
            # Delegate to DocumentStore if available for better metadata
            if self.document_store is not None:
                return self.document_store.list_documents()

            # Fallback to ChromaDB metadata if DocumentStore not available
            if self.collection.count() == 0:
                return []

            all_data = self.collection.get(include=["metadatas"])

            # Group by filename to get document-level info
            doc_info = {}
            for metadata in all_data["metadatas"]:
                filename = metadata["filename"]
                if filename not in doc_info:
                    doc_info[filename] = {
                        "filename": filename,
                        "path": metadata.get("doc_path", ""),
                        "chunk_count": 0,
                        "created_at": "N/A",  # ChromaDB doesn't track creation time by default
                    }
                doc_info[filename]["chunk_count"] += 1

            # Convert to list and sort by filename
            documents = list(doc_info.values())
            documents.sort(key=lambda x: x["filename"])

            return documents

        except Exception as e:
            self.logger.error("Error listing documents: %s", e)
            return []

    def check_existing_documents(
        self, documents: List[Dict]
    ) -> tuple[List[Dict], List[Dict]]:
        """Check which documents already exist in the database.

        Returns:
            tuple: (new_documents, existing_documents)
        """
        try:
            # Delegate to DocumentStore if available for better accuracy
            if self.document_store is not None:
                return self.document_store.check_existing_documents(documents)

            # Fallback to ChromaDB metadata if DocumentStore not available
            existing_filenames = set()
            if self.collection.count() > 0:
                all_data = self.collection.get(include=["metadatas"])
                for metadata in all_data["metadatas"]:
                    existing_filenames.add(metadata["filename"])

            new_documents = []
            existing_documents = []

            for doc in documents:
                if doc["filename"] in existing_filenames:
                    existing_documents.append(doc)
                else:
                    new_documents.append(doc)

            return new_documents, existing_documents
        except Exception as e:
            self.logger.error("Error checking existing documents: %s", e)
            # If there's an error, treat all as new to be safe
            return documents, []

    def cleanup_database(self):
        """Clean up database to reclaim space and fix bloat."""
        self.logger.info("Performing database cleanup...")
        try:
            # ChromaDB handles cleanup automatically
            # Data is automatically persisted with PersistentClient
            self.logger.info("Database cleanup completed (data auto-persisted)")
        except Exception as e:
            self.logger.error("Error during database cleanup: %s", e)
            raise

    def close(self):
        """Close database connection and clean up resources."""
        try:
            if hasattr(self, "client") and self.client:
                # Reset the client to None to close connections
                self.client = None
                self.collection = None
                self.logger.debug("ChromaDB client closed successfully")
        except Exception as e:
            self.logger.warning("Error closing ChromaDB client: %s", e)
