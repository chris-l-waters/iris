"""Core RAG implementation for DOD directive querying."""

import os
from typing import List, Optional, Dict
from .documents import DocumentProcessor
from .vectorstore import SimpleVectorStore
from .llm import LLMProcessor
from .config import config


def get_major_group(doc_number):
    """Extract major group from document number (first digit * 1000)."""
    if not doc_number:
        return None
    return int(doc_number.split(".")[0][:1]) * 1000


def get_subgroup(doc_number):
    """Extract subgroup from document number (second digit * 100)."""
    if not doc_number or len(doc_number.split(".")[0]) < 2:
        return None
    return int(doc_number.split(".")[0][:2]) * 100


def get_specific_range(doc_number):
    """Extract specific range from document number (first 3 digits * 10)."""
    if not doc_number or len(doc_number.split(".")[0]) < 3:
        return None
    return int(doc_number.split(".")[0][:3]) * 10


class DocumentAwareRanker:
    """Enhanced ranking system that boosts related chunks based on document similarity."""

    def __init__(self):
        """Initialize ranker with configuration from config.yml."""
        self.enabled = config.ranking_enabled
        self.same_document_boost = config.ranking_same_document_boost
        self.same_doc_number_boost = config.ranking_same_doc_number_boost
        self.same_specific_range_boost = config.ranking_same_specific_range_boost
        self.same_subgroup_boost = config.ranking_same_subgroup_boost
        self.same_major_group_boost = config.ranking_same_major_group_boost
        self.adjacent_chunk_boost = config.ranking_adjacent_chunk_boost
        self.near_chunk_boost = config.ranking_near_chunk_boost
        self.nearby_chunk_boost = config.ranking_nearby_chunk_boost

    def get_adjacency_boost(self, top_result: Dict, current_result: Dict) -> float:
        """Calculate adjacency boost factor based on chunk distance within same document."""
        # Adjacency only applies within same document file
        top_filename = top_result.get("filename")
        current_filename = current_result.get("filename")

        if not top_filename or not current_filename or top_filename != current_filename:
            return 1.0  # No adjacency boost across different files

        top_chunk_idx = top_result.get("chunk_index")
        current_chunk_idx = current_result.get("chunk_index")

        if top_chunk_idx is None or current_chunk_idx is None:
            return 1.0  # No adjacency boost if chunk indices are missing

        distance = abs(current_chunk_idx - top_chunk_idx)

        if distance == 1:
            return self.adjacent_chunk_boost  # Adjacent chunks (±1)
        elif distance == 2:
            return self.near_chunk_boost  # Near chunks (±2)
        elif distance <= 5:
            return self.nearby_chunk_boost  # Nearby chunks (±3-5)
        else:
            return 1.0  # Same document but not adjacent

    def rank_results(self, results: List[Dict]) -> List[Dict]:
        """Apply document-aware ranking to search results."""
        if not self.enabled or not results or len(results) <= 1:
            return results

        # Create a copy to avoid modifying original results
        ranked_results = [result.copy() for result in results]

        # Get top result as reference for boosting decisions
        top_result = ranked_results[0]
        top_filename = top_result.get("filename")
        top_doc_number = top_result.get("doc_number")

        # Extract reference patterns for comparison
        top_specific_range = None
        top_subgroup = None
        top_major_group = None
        if top_doc_number:
            top_specific_range = get_specific_range(top_doc_number)
            top_subgroup = get_subgroup(top_doc_number)
            top_major_group = get_major_group(top_doc_number)

        # Apply de-boosting to unrelated chunks (preserves top result's position)
        # Process all results except the top one (which stays unchanged)
        for i in range(1, len(ranked_results)):
            result = ranked_results[i]
            result_filename = result.get("filename")
            result_doc_number = result.get("doc_number")

            # Track what relationship this chunk has to top result
            relationship = None
            original_similarity = result["similarity"]

            # Check relationship hierarchy (highest priority first)
            if result_filename and top_filename and result_filename == top_filename:
                relationship = "same_document"
            elif (
                result_doc_number
                and top_doc_number
                and result_doc_number == top_doc_number
            ):
                relationship = "same_doc_number"
            elif (
                result_doc_number
                and top_specific_range
                and get_specific_range(result_doc_number) == top_specific_range
            ):
                relationship = "same_specific_range"
            elif (
                result_doc_number
                and top_subgroup
                and get_subgroup(result_doc_number) == top_subgroup
            ):
                relationship = "same_subgroup"
            elif (
                result_doc_number
                and top_major_group
                and get_major_group(result_doc_number) == top_major_group
            ):
                relationship = "same_major_group"
            else:
                relationship = "unrelated"

            # Apply document relationship boost
            document_boost = 1.0
            if relationship == "same_document":
                document_boost = self.same_document_boost
            elif relationship == "same_doc_number":
                document_boost = self.same_doc_number_boost
            elif relationship == "same_specific_range":
                document_boost = self.same_specific_range_boost
            elif relationship == "same_subgroup":
                document_boost = self.same_subgroup_boost
            elif relationship == "same_major_group":
                document_boost = self.same_major_group_boost
            else:  # unrelated
                # Apply de-boost factor as inverse of the weakest related boost
                document_boost = 1.0 / self.same_major_group_boost

            # Apply adjacency boost (multiplicative with document boost)
            adjacency_boost = self.get_adjacency_boost(top_result, result)

            # Calculate total boost (multiplicative)
            total_boost = document_boost * adjacency_boost
            result["similarity"] *= total_boost

            # Store debugging information
            result["_original_similarity"] = original_similarity
            result["_relationship"] = relationship
            result["_document_boost"] = document_boost
            result["_adjacency_boost"] = adjacency_boost
            result["_total_boost"] = total_boost

        # Re-sort by updated similarity scores
        ranked_results.sort(key=lambda x: x["similarity"], reverse=True)

        return ranked_results


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

        # Initialize DocumentStore
        from .documents import DocumentStore

        self.document_store = DocumentStore("database/documents.sqlite")

        # Initialize DocumentProcessor with DocumentStore
        self.processor = DocumentProcessor(
            embedding_model, document_store=self.document_store
        )
        self.vector_store = None
        self.llm = LLMProcessor(model_name)
        self.ranker = DocumentAwareRanker()
        self._documents_loaded = False

    def _ensure_vector_store(self):
        """Lazy load vector store."""
        if self.vector_store is None:
            self.vector_store = SimpleVectorStore(
                collection_name=self.collection_name, document_store=self.document_store
            )
        return self.vector_store

    def load_documents(
        self,
        document_dir: str = "policies/test",
        force_reload: bool = False,
        verbose: bool = False,
    ):
        """Load and process documents into vector store."""
        print(f"Loading documents from {document_dir}")

        # Check if DocumentStore has existing chunks (additional embeddings flow)
        if not force_reload and self.document_store:
            existing_chunks = self.document_store.get_all_chunks()
            if existing_chunks:
                print(f"Found {len(existing_chunks)} existing chunks in DocumentStore")
                print(
                    "Using existing chunks for additional embeddings (skipping document processing)"
                )

                # Use existing chunks from DocumentStore - no document processing needed
                chunk_ids, embeddings, minimal_metadata = (
                    self.processor.create_embeddings_for_docs([])
                )

                # Store in vector database
                store = self._ensure_vector_store()
                store.add_documents(
                    chunk_ids, embeddings, minimal_metadata, clear_existing=False
                )

                self._documents_loaded = True
                print(f"Generated embeddings for {len(chunk_ids)} existing chunks")
                return

        # New database flow - process documents
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
        chunk_ids, embeddings, minimal_metadata = (
            self.processor.create_embeddings_for_docs(docs_to_process)
        )

        # Store in vector database
        store.add_documents(
            chunk_ids, embeddings, minimal_metadata, clear_existing=clear_existing
        )

        self._documents_loaded = True
        print(f"Loaded {len(docs_to_process)} documents with {len(chunk_ids)} chunks")

    def retrieve_context(
        self,
        question: str,
        top_k: int = 3,
        return_metadata: bool = False,
        use_cross_encoder: bool = False,
    ):
        """Retrieve relevant context for a question."""
        store = self._ensure_vector_store()

        # Generate query embedding
        query_embedding = self.processor.generate_embeddings([question])[0]

        # Determine initial search count based on cross-encoder usage
        if use_cross_encoder:
            # Retrieve more candidates for reranking
            from .cross_encoder import get_cross_encoder_manager

            cross_encoder = get_cross_encoder_manager()
            if cross_encoder.is_available():
                initial_top_k = max(top_k, config.cross_encoder_rerank_top_k)
            else:
                initial_top_k = top_k
                use_cross_encoder = False  # Fallback to vector-only
        else:
            initial_top_k = top_k

        # Search for similar chunks
        results = store.similarity_search(query_embedding, top_k=initial_top_k)

        # Apply enhanced document-aware ranking
        results = self.ranker.rank_results(results)

        # Apply cross-encoder reranking if requested
        if use_cross_encoder:
            # Prepare passages for reranking - convert results to expected format
            passages = []
            for result in results:
                passage = {
                    "text": result.get("chunk_text", ""),
                    "metadata": {
                        "filename": result.get("filename"),
                        "chunk_index": result.get("chunk_index"),
                        "doc_number": result.get("doc_number"),
                        "vector_score": result.get("similarity_score", 0.0),
                    },
                }
                # Copy any additional metadata
                for key, value in result.items():
                    if key not in [
                        "chunk_text",
                        "filename",
                        "chunk_index",
                        "doc_number",
                        "similarity_score",
                    ]:
                        passage["metadata"][key] = value
                passages.append(passage)

            # Rerank with cross-encoder
            reranked_passages = cross_encoder.rerank_passages(question, passages)

            # Convert back to original format
            results = []
            for passage in reranked_passages[:top_k]:  # Limit to requested top_k
                result = {
                    "chunk_text": passage["text"],
                    "cross_encoder_score": passage.get("cross_encoder_score"),
                }
                # Copy metadata back
                result.update(passage["metadata"])
                results.append(result)
        else:
            # Limit to requested top_k for standard vector search
            results = results[:top_k]

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
        use_cross_encoder: bool = False,
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
                question,
                top_k=retrieval_limit,
                return_metadata=True,
                use_cross_encoder=use_cross_encoder,
            )
            context_chunks = [result["chunk_text"] for result in context_results]
        else:
            # We need metadata for citations even when not returning context
            context_results = self.retrieve_context(
                question,
                top_k=retrieval_limit,
                return_metadata=True,
                use_cross_encoder=use_cross_encoder,
            )
            context_chunks = [result["chunk_text"] for result in context_results]

        if not context_chunks:
            return f"No relevant policy information found for: '{question}'"

        if use_llm and self.llm.is_available():
            # Use LLM to generate comprehensive response
            keep_alive = -1 if keep_model_loaded else None
            if return_context:
                # Get both response and fitted chunks from LLM
                response, fitted_chunks = self.llm.generate_rag_response(
                    question,
                    context_chunks,
                    max_k,
                    return_context=True,
                    keep_alive=keep_alive,
                    context_results=context_results,
                )
                # Convert fitted chunks back to metadata format for display
                fitted_results = []
                for chunk_text in fitted_chunks:
                    # Find the corresponding metadata for each fitted chunk
                    for result in context_results:
                        if result["chunk_text"] == chunk_text:
                            fitted_results.append(result)
                            break
                return response, fitted_results
            else:
                response = self.llm.generate_rag_response(
                    question,
                    context_chunks,
                    max_k,
                    return_context=False,
                    keep_alive=keep_alive,
                    context_results=context_results,
                )
                return response

        # Fallback to simple context display
        context_preview = context_chunks[0][:200] + "..." if context_chunks else ""
        response = f"""Based on DOD policies, here's what I found regarding '{question}':

Relevant policy excerpt:
{context_preview}

{"Note: LLM not available, showing simplified response." if use_llm else ""}

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
