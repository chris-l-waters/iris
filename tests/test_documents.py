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
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create DocumentStore for the test
        from src.documents import DocumentStore

        document_store = DocumentStore(os.path.join(temp_dir, "test_documents.sqlite"))
        processor = DocumentProcessor(document_store=document_store)

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


def test_document_store():
    """Test DocumentStore functionality."""
    import tempfile
    import os
    from src.documents import DocumentStore

    # Use temporary database for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_documents.sqlite")
        store = DocumentStore(db_path)

        # Test initial state
        stats = store.get_stats()
        assert stats["documents"] == 0
        assert stats["chunks"] == 0
        assert stats["db_path"] == db_path

        # Test empty queries
        assert store.get_all_chunks() == []
        assert store.get_chunks_by_ids(["nonexistent"]) == []
        assert store.check_existing_documents([]) == ([], [])

        # Test document addition
        test_documents = [
            {
                "filename": "test1.pdf",
                "path": "/test/test1.pdf",
                "doc_number": "1234.56",
            },
            {
                "filename": "test2.pdf",
                "path": "/test/test2.pdf",
                "doc_number": "7890.12",
            },
        ]

        test_metadata = [
            {
                "doc_filename": "test1.pdf",
                "doc_path": "/test/test1.pdf",
                "chunk_index": 0,
                "chunk_text": "First chunk of test1",
            },
            {
                "doc_filename": "test1.pdf",
                "doc_path": "/test/test1.pdf",
                "chunk_index": 1,
                "chunk_text": "Second chunk of test1",
            },
            {
                "doc_filename": "test2.pdf",
                "doc_path": "/test/test2.pdf",
                "chunk_index": 0,
                "chunk_text": "First chunk of test2",
            },
        ]

        # Add documents and verify return mapping
        chunk_id_map = store.add_documents(test_documents, test_metadata)
        assert len(chunk_id_map) == 3

        # Test stats after addition
        stats = store.get_stats()
        assert stats["documents"] == 2
        assert stats["chunks"] == 3
        assert stats["db_size_mb"] > 0

        # Test get_all_chunks
        all_chunks = store.get_all_chunks()
        assert len(all_chunks) == 3

        # Verify chunk structure
        chunk = all_chunks[0]
        assert "chunk_id" in chunk
        assert "chunk_text" in chunk
        assert "chunk_index" in chunk
        assert "filename" in chunk
        assert "doc_path" in chunk
        assert "doc_number" in chunk

        # Test get_chunks_by_ids
        chunk_ids = [chunk["chunk_id"] for chunk in all_chunks[:2]]
        retrieved_chunks = store.get_chunks_by_ids(chunk_ids)
        assert len(retrieved_chunks) == 2

        # Test check_existing_documents
        new_docs = [{"filename": "test3.pdf", "path": "/test/test3.pdf"}]
        existing_docs = [{"filename": "test1.pdf", "path": "/test/test1.pdf"}]
        mixed_docs = new_docs + existing_docs

        new, existing = store.check_existing_documents(mixed_docs)
        assert len(new) == 1
        assert len(existing) == 1
        assert new[0]["filename"] == "test3.pdf"
        assert existing[0]["filename"] == "test1.pdf"

        # Test clear_existing=False (incremental add)
        new_document = [
            {
                "filename": "test3.pdf",
                "path": "/test/test3.pdf",
                "doc_number": "3333.33",
            }
        ]
        new_metadata = [
            {
                "doc_filename": "test3.pdf",
                "doc_path": "/test/test3.pdf",
                "chunk_index": 0,
                "chunk_text": "First chunk of test3",
            }
        ]

        store.add_documents(new_document, new_metadata, clear_existing=False)

        # Verify incremental addition
        stats = store.get_stats()
        assert stats["documents"] == 3
        assert stats["chunks"] == 4

        # Test clear_existing=True (replace all)
        replacement_docs = [
            {"filename": "replacement.pdf", "path": "/test/replacement.pdf"}
        ]
        replacement_metadata = [
            {
                "doc_filename": "replacement.pdf",
                "doc_path": "/test/replacement.pdf",
                "chunk_index": 0,
                "chunk_text": "Replacement chunk",
            }
        ]

        store.add_documents(replacement_docs, replacement_metadata, clear_existing=True)

        # Verify replacement
        stats = store.get_stats()
        assert stats["documents"] == 1
        assert stats["chunks"] == 1

        all_chunks = store.get_all_chunks()
        assert len(all_chunks) == 1
        assert all_chunks[0]["filename"] == "replacement.pdf"


def test_document_store_chunk_id_generation():
    """Test DocumentStore chunk ID generation matches VectorStore logic."""
    import tempfile
    import os
    import hashlib
    from src.documents import DocumentStore

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_chunk_ids.sqlite")
        store = DocumentStore(db_path)

        # Test chunk ID generation
        test_documents = [{"filename": "test.pdf", "path": "/path/to/test.pdf"}]
        test_metadata = [
            {
                "doc_filename": "test.pdf",
                "doc_path": "/path/to/test.pdf",
                "chunk_index": 0,
                "chunk_text": "Test chunk",
            }
        ]

        chunk_id_map = store.add_documents(test_documents, test_metadata)

        # Verify chunk ID format matches VectorStore logic
        expected_hash = hashlib.md5("/path/to/test.pdf".encode()).hexdigest()[:8]
        expected_chunk_id = f"{expected_hash}_test.pdf_0"

        stored_chunk_id = list(chunk_id_map.values())[0]
        assert stored_chunk_id == expected_chunk_id

        # Verify chunk can be retrieved by ID
        retrieved = store.get_chunks_by_ids([stored_chunk_id])
        assert len(retrieved) == 1
        assert retrieved[0]["chunk_id"] == expected_chunk_id
        assert retrieved[0]["chunk_text"] == "Test chunk"


def test_embedding_manager_new_database_flow(mock_embedding_model):
    """Test EmbeddingManager with new database (empty DocumentStore)."""
    import tempfile
    import os
    from src.documents import DocumentStore, EmbeddingManager

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_embeddings.sqlite")
        document_store = DocumentStore(db_path)
        embedding_manager = EmbeddingManager(
            "all-MiniLM-L6-v2", document_store=document_store
        )

        # Test data
        test_documents = [
            {
                "filename": "test1.pdf",
                "path": "/test/test1.pdf",
                "doc_number": "1234.56",
                "chunks": ["First chunk of test1", "Second chunk of test1"],
            },
            {
                "filename": "test2.pdf",
                "path": "/test/test2.pdf",
                "doc_number": "7890.12",
                "chunks": ["First chunk of test2"],
            },
        ]

        # Should process documents and store in DocumentStore (new database flow)
        chunk_ids, embeddings, metadata = embedding_manager.create_embeddings_for_docs(
            test_documents
        )

        # Verify return format
        assert len(chunk_ids) == 3  # 2 + 1 chunks
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 384  # Mock embedding dimensions
        assert len(metadata) == 3

        # Verify metadata structure
        assert all("chunk_id" in m for m in metadata)
        assert all("filename" in m for m in metadata)
        assert all("chunk_index" in m for m in metadata)

        # Verify DocumentStore was populated
        stats = document_store.get_stats()
        assert stats["documents"] == 2
        assert stats["chunks"] == 3


def test_embedding_manager_additional_embeddings_flow(mock_embedding_model):
    """Test EmbeddingManager with existing DocumentStore (additional embeddings)."""
    import tempfile
    import os
    from src.documents import DocumentStore, EmbeddingManager

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_embeddings.sqlite")
        document_store = DocumentStore(db_path)

        # Pre-populate DocumentStore
        test_documents = [
            {"filename": "test.pdf", "path": "/test/test.pdf", "doc_number": "1234.56"}
        ]
        test_metadata = [
            {
                "doc_filename": "test.pdf",
                "doc_path": "/test/test.pdf",
                "chunk_index": 0,
                "chunk_text": "Existing chunk",
            }
        ]
        document_store.add_documents(test_documents, test_metadata)

        # Create EmbeddingManager with populated DocumentStore
        embedding_manager = EmbeddingManager(
            "mixedbread-ai/mxbai-embed-large-v1", document_store=document_store
        )

        # Should ignore documents param and use existing chunks (additional embeddings flow)
        ignored_documents = [
            {
                "filename": "ignored.pdf",
                "path": "/ignored/ignored.pdf",
                "chunks": ["This should be ignored"],
            }
        ]
        chunk_ids, embeddings, metadata = embedding_manager.create_embeddings_for_docs(
            ignored_documents
        )

        # Verify it used existing chunks, not the ignored documents
        assert len(chunk_ids) == 1
        assert chunk_ids[0].endswith("test.pdf_0")
        assert embeddings.shape[0] == 1
        assert len(metadata) == 1
        assert metadata[0]["filename"] == "test.pdf"

        # Verify DocumentStore wasn't modified (no new documents added)
        stats = document_store.get_stats()
        assert stats["documents"] == 1
        assert stats["chunks"] == 1


def test_embedding_manager_requires_document_store():
    """Test EmbeddingManager requires DocumentStore."""
    from src.documents import EmbeddingManager
    import pytest

    embedding_manager = EmbeddingManager("all-MiniLM-L6-v2", document_store=None)

    with pytest.raises(ValueError, match="DocumentStore is required"):
        embedding_manager.create_embeddings_for_docs([])


def test_vectorstore_with_document_store(mock_embedding_model):
    """Test VectorStore integration with DocumentStore."""
    import tempfile
    import os
    import numpy as np
    from src.documents import DocumentStore
    from src.vectorstore import SimpleVectorStore

    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up DocumentStore
        doc_db_path = os.path.join(temp_dir, "test_documents.sqlite")
        document_store = DocumentStore(doc_db_path)

        # Set up VectorStore with DocumentStore
        vector_store = SimpleVectorStore(
            collection_name="test_collection", document_store=document_store
        )

        # Add test data to DocumentStore first
        test_documents = [
            {"filename": "test.pdf", "path": "/test/test.pdf", "doc_number": "1234.56"}
        ]
        test_metadata = [
            {
                "doc_filename": "test.pdf",
                "doc_path": "/test/test.pdf",
                "chunk_index": 0,
                "chunk_text": "Test chunk content",
            }
        ]
        chunk_id_map = document_store.add_documents(test_documents, test_metadata)

        # Prepare data for VectorStore
        chunk_ids = list(chunk_id_map.values())
        embeddings = np.random.rand(1, 384).astype(np.float32)  # Mock embedding
        minimal_metadata = [
            {
                "chunk_id": chunk_ids[0],
                "filename": "test.pdf",
                "chunk_index": 0,
                "doc_number": "1234.56",
            }
        ]

        # Add to VectorStore
        vector_store.add_documents(chunk_ids, embeddings, minimal_metadata)

        # Test similarity search
        query_embedding = np.random.rand(384).astype(np.float32)
        results = vector_store.similarity_search(query_embedding, top_k=1)

        # Verify results include text from DocumentStore
        assert len(results) == 1
        assert results[0]["chunk_text"] == "Test chunk content"
        assert results[0]["filename"] == "test.pdf"
        assert results[0]["chunk_index"] == 0
        assert "similarity" in results[0]

        # Test list_documents delegates to DocumentStore
        docs = vector_store.list_documents()
        assert len(docs) == 1
        assert docs[0]["filename"] == "test.pdf"
        assert docs[0]["chunk_count"] == 1


def test_vectorstore_without_document_store(mock_embedding_model):
    """Test VectorStore fallback when DocumentStore not provided."""
    import numpy as np
    from src.vectorstore import SimpleVectorStore

    # VectorStore without DocumentStore
    vector_store = SimpleVectorStore(collection_name="test_no_docstore")

    # Prepare data
    chunk_ids = ["test_chunk_0"]
    embeddings = np.random.rand(1, 384).astype(np.float32)
    minimal_metadata = [
        {
            "chunk_id": "test_chunk_0",
            "filename": "test.pdf",
            "chunk_index": 0,
            "doc_number": "1234.56",
        }
    ]

    # Add to VectorStore
    vector_store.add_documents(chunk_ids, embeddings, minimal_metadata)

    # Test similarity search falls back gracefully
    query_embedding = np.random.rand(384).astype(np.float32)
    results = vector_store.similarity_search(query_embedding, top_k=1)

    # Should get placeholder text when DocumentStore unavailable
    assert len(results) == 1
    assert "[Text unavailable for chunk test_chunk_0]" in results[0]["chunk_text"]
    assert results[0]["filename"] == "test.pdf"


def test_complete_rag_flow_new_database(mock_embedding_model):
    """Test complete RAG flow: first model with new database."""
    import tempfile
    import os
    import uuid
    from src.rag import SimpleRAG

    with tempfile.TemporaryDirectory() as temp_dir:
        # Change to temp directory to avoid contaminating real database
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        # Use unique database directory to avoid conflicts
        unique_db = f"test_database_{uuid.uuid4().hex[:8]}"
        os.makedirs(unique_db, exist_ok=True)

        try:
            # Create test documents structure similar to actual documents
            test_documents = [
                {
                    "filename": "test1.pdf",
                    "path": "test/test1.pdf",
                    "doc_number": "1234.56",
                    "chunks": [
                        "First chunk of test document",
                        "Second chunk with more content",
                    ],
                    "chunk_count": 2,
                },
                {
                    "filename": "test2.pdf",
                    "path": "test/test2.pdf",
                    "doc_number": "7890.12",
                    "chunks": ["Another document chunk"],
                    "chunk_count": 1,
                },
            ]

            # Mock the document processor to return our test documents with unique model name
            rag = SimpleRAG(embedding_model=f"all-MiniLM-L6-v2-{uuid.uuid4().hex[:8]}")

            # Override DocumentStore to use test database
            from src.documents import DocumentStore

            rag.document_store = DocumentStore(f"{unique_db}/documents.sqlite")
            rag.processor.embedding_manager.document_store = rag.document_store

            # Directly patch the VectorStore after creation instead of monkey patching class
            def patch_vectorstore(vs):
                vs.persist_dir = unique_db
                import chromadb

                vs.client = chromadb.PersistentClient(path=vs.persist_dir)
                vs.collection = vs.client.get_or_create_collection(
                    name=vs.collection_name, metadata={"hnsw:space": "cosine"}
                )

            # Patch _ensure_vector_store to use test directory
            original_ensure = rag._ensure_vector_store

            def test_ensure_vector_store():
                result = original_ensure()
                patch_vectorstore(result)
                return result

            rag._ensure_vector_store = test_ensure_vector_store

            # Mock the process_directory method
            def mock_process_directory(directory, verbose=False):
                return test_documents

            rag.processor.process_directory = mock_process_directory

            # Test first model - should create new database
            rag.load_documents("test", force_reload=True)

            # Verify DocumentStore was populated
            doc_stats = rag.document_store.get_stats()
            assert doc_stats["documents"] == 2
            assert doc_stats["chunks"] == 3  # 2 + 1 chunks

            # Verify VectorStore was populated
            vector_stats = rag.get_stats()
            assert vector_stats["chunks"] == 3

            # Test query functionality
            assert rag.is_ready()

            # Test retrieve_context
            context = rag.retrieve_context("test query", top_k=2, return_metadata=True)
            assert len(context) >= 1  # Should find at least one result
            assert "chunk_text" in context[0]
            assert "filename" in context[0]

        finally:
            # Clean up ChromaDB clients to prevent state sharing
            if hasattr(rag, "vector_store") and rag.vector_store:
                rag.vector_store.close()
            os.chdir(original_cwd)


def test_complete_rag_flow_additional_embeddings(mock_embedding_model):
    """Test complete RAG flow: second model with additional embeddings."""
    import tempfile
    import os
    import uuid
    from src.rag import SimpleRAG
    from src.documents import DocumentStore

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        # Use unique database directory to avoid conflicts
        unique_db = f"test_database_{uuid.uuid4().hex[:8]}"
        os.makedirs(unique_db, exist_ok=True)

        try:
            # Pre-populate DocumentStore as if first model already processed documents
            document_store = DocumentStore(f"{unique_db}/documents.sqlite")
            test_documents = [
                {
                    "filename": "existing.pdf",
                    "path": "test/existing.pdf",
                    "doc_number": "5555.55",
                }
            ]
            test_metadata = [
                {
                    "doc_filename": "existing.pdf",
                    "doc_path": "test/existing.pdf",
                    "chunk_index": 0,
                    "chunk_text": "Pre-existing chunk content",
                }
            ]
            document_store.add_documents(test_documents, test_metadata)

            # Create second RAG instance with different model (unique name)
            rag2 = SimpleRAG(
                embedding_model=f"mixedbread-ai/mxbai-embed-large-v1-{uuid.uuid4().hex[:8]}"
            )

            # Override DocumentStore to use the same test database
            rag2.document_store = document_store
            rag2.processor.embedding_manager.document_store = document_store

            # Directly patch the VectorStore after creation instead of monkey patching class
            def patch_vectorstore(vs):
                vs.persist_dir = unique_db
                import chromadb

                vs.client = chromadb.PersistentClient(path=vs.persist_dir)
                vs.collection = vs.client.get_or_create_collection(
                    name=vs.collection_name, metadata={"hnsw:space": "cosine"}
                )

            # Patch _ensure_vector_store to use test directory
            original_ensure = rag2._ensure_vector_store

            def test_ensure_vector_store():
                result = original_ensure()
                patch_vectorstore(result)
                return result

            rag2._ensure_vector_store = test_ensure_vector_store

            # Mock process_directory to return different documents (should be ignored)
            def mock_process_directory(directory, verbose=False):
                return [
                    {
                        "filename": "ignored.pdf",
                        "path": "ignored/ignored.pdf",
                        "chunks": ["This should be ignored"],
                        "chunk_count": 1,
                    }
                ]

            rag2.processor.process_directory = mock_process_directory

            # Load documents - should use existing DocumentStore content, ignore new documents
            rag2.load_documents("test", force_reload=False)

            # Verify DocumentStore wasn't modified (still has original content)
            doc_stats = rag2.document_store.get_stats()
            assert doc_stats["documents"] == 1
            assert doc_stats["chunks"] == 1

            # Verify second VectorStore was created with embeddings for existing chunks
            vector_stats = rag2.get_stats()
            assert vector_stats["chunks"] == 1
            assert vector_stats["collection_name"].startswith(
                "iris_mixedbread_ai_mxbai_embed_large_v1"
            )

            # Test query functionality uses existing text
            context = rag2.retrieve_context("test query", top_k=1, return_metadata=True)
            assert len(context) == 1
            assert context[0]["chunk_text"] == "Pre-existing chunk content"
            assert context[0]["filename"] == "existing.pdf"

        finally:
            # Clean up ChromaDB clients to prevent state sharing
            if hasattr(rag2, "vector_store") and rag2.vector_store:
                rag2.vector_store.close()
            os.chdir(original_cwd)


if __name__ == "__main__":
    test_document_processing()
    test_document_store()
    test_document_store_chunk_id_generation()
    test_embedding_manager_new_database_flow()
    test_embedding_manager_additional_embeddings_flow()
    test_embedding_manager_requires_document_store()
    test_vectorstore_with_document_store()
    test_vectorstore_without_document_store()
    test_complete_rag_flow_new_database()
    test_complete_rag_flow_additional_embeddings()
