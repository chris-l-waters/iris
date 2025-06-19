"""Core RAG implementation for DOD directive querying."""

import os
from typing import List, Optional
from .documents import DocumentProcessor
from .vectorstore import SimpleVectorStore
from .llm import LLMProcessor


def get_major_group(doc_number):
    if not doc_number:
        return None
    return int(doc_number.split(".")[0][:1]) * 1000


class SimpleRAG:
    """Simple RAG implementation with document retrieval."""

    def __init__(
        self,
        db_path: str = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize RAG system with vector store and LLM."""
        # Create model-specific collection name
        if db_path is None:
            # Ensure database directory exists
            os.makedirs("database", exist_ok=True)
            db_model_name = embedding_model.replace("/", "_").replace("-", "_")
            self.collection_name = f"iris_{db_model_name}"
            # Keep db_path for backward compatibility (used in CLI output)
            self.db_path = f"database/iris_{db_model_name}.db"
        else:
            # Extract collection name from legacy db_path
            if "iris_" in db_path and db_path.endswith(".db"):
                basename = os.path.basename(db_path)
                model_name_part = basename[5:-3]  # Remove "iris_" and ".db"
                self.collection_name = f"iris_{model_name_part}"
            else:
                self.collection_name = "iris_default"
            self.db_path = db_path
        self.embedding_model = embedding_model
        self.processor = DocumentProcessor(embedding_model)
        self.vector_store = None
        self.llm = LLMProcessor(model_name)
        self._documents_loaded = False

    def _ensure_vector_store(self):
        """Lazy load vector store."""
        if self.vector_store is None:
            self.vector_store = SimpleVectorStore(collection_name=self.collection_name)
        return self.vector_store

    def load_documents(
        self,
        document_dir: str = "policies/test",
        force_reload: bool = False,
        verbose: bool = False,
    ):
        """Load and process documents into vector store."""
        print(f"Loading documents from {document_dir}")

        # Process documents
        print("Processing PDF documents (extracting text and creating chunks)...")
        docs = self.processor.process_directory(document_dir, verbose=verbose)
        if not docs:
            print("No documents found to process!")
            return

        # Check for existing documents unless force_reload is True
        store = self._ensure_vector_store()
        clear_existing = force_reload

        if not force_reload:
            # Check what documents already exist
            new_docs, existing_docs = store.check_existing_documents(docs)

            if existing_docs:
                print(f"Found {len(existing_docs)} documents already in database:")
                for doc in existing_docs:
                    print(f"  - {doc['filename']}")

            if not new_docs:
                print(
                    "All documents already exist in database. Use force_reload=True to reload."
                )
                self._documents_loaded = True
                return

            print(f"Processing {len(new_docs)} new documents")
            docs_to_process = new_docs
        else:
            print("Force reload requested - will replace all existing documents")
            docs_to_process = docs

        # Create embeddings for documents to process
        total_chunks = sum(doc["chunk_count"] for doc in docs_to_process)
        print(f"Generating embeddings for {total_chunks} chunks...")
        chunks, embeddings, metadata = self.processor.create_embeddings_for_docs(
            docs_to_process
        )

        # Store in vector database
        store.add_documents(docs, embeddings, metadata, clear_existing=clear_existing)

        self._documents_loaded = True
        print(f"Loaded {len(docs_to_process)} documents with {len(chunks)} chunks")

    def retrieve_context(
        self, question: str, top_k: int = 3, return_metadata: bool = False
    ):
        """Retrieve relevant context for a question."""
        store = self._ensure_vector_store()

        # Generate query embedding
        query_embedding = self.processor.generate_embeddings([question])[0]

        # Search for similar chunks
        results = store.similarity_search(query_embedding, top_k=top_k)

        # Rerank results based on DOD document number similarity
        if results and len(results) > 1:
            top_doc_number = results[0].get("doc_number")
            if top_doc_number and len(top_doc_number) >= 4:
                top_prefix = top_doc_number[:4]  # First 4 digits (subgroup)
                top_major_group = get_major_group(top_doc_number)

                # Apply boost to results with matching numbers
                for result in results[1:]:  # Skip the top result
                    result_doc_number = result.get("doc_number")
                    if result_doc_number:
                        if result_doc_number.startswith(top_prefix):
                            # Same subgroup (first 4 digits)
                            result["similarity"] *= 1.3
                        elif get_major_group(result_doc_number) == top_major_group:
                            # Same major group (first digit * 1000)
                            result["similarity"] *= 1.1

                # Re-sort by updated similarity scores
                results.sort(key=lambda x: x["similarity"], reverse=True)

        if return_metadata:
            return results
        else:
            # Extract text from results
            context_chunks = [result["chunk_text"] for result in results]
            return context_chunks

    def query(
        self,
        question: str,
        use_context: bool = True,
        use_llm: bool = True,
        max_k: int = None,
        return_context: bool = False,
        keep_model_loaded: bool = False,
    ):
        """Process a query using RAG pipeline."""
        if not question.strip():
            return "Please provide a valid question."

        if not self._documents_loaded:
            return "Please load documents first using load_documents()"

        if not use_context:
            return f"Simple response for: '{question}' (no context retrieval)"

        # Retrieve relevant context with smart limits
        # Large default for auto-fitting
        retrieval_limit = max_k if max_k else 50

        if return_context:
            # Get full metadata for context display
            context_results = self.retrieve_context(
                question, top_k=retrieval_limit, return_metadata=True
            )
            context_chunks = [result["chunk_text"] for result in context_results]
        else:
            context_chunks = self.retrieve_context(question, top_k=retrieval_limit)
            context_results = None

        if not context_chunks:
            return f"No relevant policy information found for: '{question}'"

        if use_llm and self.llm.is_available():
            # Use LLM to generate comprehensive response
            keep_alive = -1 if keep_model_loaded else None
            response = self.llm.generate_rag_response(
                question,
                context_chunks,
                max_k,
                False,
                keep_alive,  # Always False for LLM, we handle context separately
            )
            if return_context:
                return response, context_results
            return response

        # Fallback to simple context display
        context_preview = context_chunks[0][:200] + "..." if context_chunks else ""
        response = f"""Based on DOD policies, here's what I found regarding '{question}':

Relevant policy excerpt:
{context_preview}

{'Note: LLM not available, showing simplified response.' if use_llm else ''}

Found {len(context_chunks)} relevant policy sections."""
        if return_context:
            return response, context_results
        return response

    def get_stats(self) -> dict:
        """Get system statistics."""
        # Always try to get stats, even if vector_store isn't initialized yet
        store = self._ensure_vector_store()
        return store.get_stats()

    def is_ready(self) -> bool:
        """Check if RAG system is ready to process queries."""
        return self._documents_loaded

    def is_llm_ready(self) -> bool:
        """Check if LLM is available for enhanced responses."""
        return self.llm.is_available()

    def list_loaded_documents(self) -> List[dict]:
        """Get list of all documents loaded in the database."""
        store = self._ensure_vector_store()
        return store.list_documents()
