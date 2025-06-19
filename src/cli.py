"""Simple CLI interface for testing the RAG system."""

import argparse
import logging
import sys
from typing import List
from pathlib import Path

from .hardware import get_hardware_info
from .rag import SimpleRAG
from .embedding_models import print_model_comparison
from .documents import save_processed_documents, load_processed_documents


def _configure_logging(verbose: bool) -> None:
    """Configure logging levels for document processing."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    else:
        logging.basicConfig(level=logging.ERROR, format="%(message)s")
        # Suppress verbose PDF parsing output
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        logging.getLogger("pdfplumber").setLevel(logging.ERROR)


def _print_stats_and_status(rag: "SimpleRAG") -> None:
    """Print database stats and LLM status."""
    stats = rag.get_stats()
    print(f"\nChromaDB collection: {rag.collection_name}")
    print("Database Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Show LLM status
    llm_status = (
        "Available"
        if rag.is_llm_ready()
        else "Not available (install llama-cpp-python and download model)"
    )
    print(f"LLM Status: {llm_status}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="IRIS - DOD Directive RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli --info           Show hardware information
  python -m src.cli --query "What is the policy on leave?"
  python -m src.cli --test           Run basic system test
        """,
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Show hardware information and recommended model",
    )

    parser.add_argument(
        "--embedding-info",
        action="store_true",
        help="Show comparison of available embedding models",
    )

    parser.add_argument(
        "--query", type=str, help="Query the RAG system with a question"
    )

    parser.add_argument("--test", action="store_true", help="Run basic system test")

    parser.add_argument(
        "--load-docs",
        action="store_true",
        help="Load documents into RAG system (clears existing data)",
    )

    parser.add_argument(
        "--doc-dirs",
        nargs="+",
        default=["policies/test"],
        help="One or more directories containing policy documents "
        "(e.g., policies/test or policies/dodd policies/dodi policies/dodm)",
    )

    parser.add_argument(
        "--save-intermediate",
        type=str,
        help="Save processed documents to JSON file (without embeddings). "
        "Useful for faster iteration when testing different embedding models - "
        "PDF processing only happens once, then use --load-intermediate to quickly generate embeddings.",
    )

    parser.add_argument(
        "--load-intermediate",
        type=str,
        help="Load processed documents from JSON file and generate embeddings. "
        "Use this with files created by --save-intermediate to skip PDF processing step.",
    )

    parser.add_argument(
        "--pull-model",
        type=str,
        help="Pull a model using Ollama (examples: llama3.2:1b-instruct-q4_K_M, llama3.2:3b-instruct-q4_K_M, "
        "mistral:7b-instruct-q4_K_M)",
    )

    parser.add_argument(
        "--model-name", type=str, help="Ollama model name (any Ollama-supported model)"
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show retrieved context chunks with query response",
    )

    parser.add_argument(
        "--max-k",
        type=int,
        help="Maximum chunks to retrieve from database (default: fit as many as possible in context)",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        choices=[
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "BAAI/bge-base-en-v1.5",
        ],
        help="Embedding model for semantic search (default: all-MiniLM-L6-v2)",
    )

    parser.add_argument(
        "--keep-model-loaded",
        action="store_true",
        help="Keep model loaded in memory indefinitely (faster subsequent queries)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed processing information during document loading",
    )

    args = parser.parse_args()

    # Show help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        return

    try:
        if args.info:
            show_hardware_info()
        elif args.embedding_info:
            print_model_comparison()
        elif args.pull_model:
            pull_model_cli(args.pull_model)
        elif args.load_docs:
            load_documents(
                args.doc_dirs,
                args.embedding_model,
                args.save_intermediate,
                args.verbose,
            )
        elif args.load_intermediate:
            load_from_intermediate(
                args.load_intermediate, args.embedding_model, args.verbose
            )
        elif args.query:
            # Process the query with all specified parameters (model, context
            # display, search params)
            process_query(
                args.query,
                args.model_name,
                args.show_context,
                args.max_k,
                args.embedding_model,
                args.keep_model_loaded,
            )
        elif args.test:
            run_system_test()
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except (ImportError, FileNotFoundError, OSError) as e:
        print(f"Error: {e}")
        sys.exit(1)


def show_hardware_info() -> None:
    """Display hardware information."""
    print("Hardware Information:")
    print("=" * 40)

    info = get_hardware_info()
    print(f"RAM: {info['ram_gb']} GB")
    print(f"CPU Cores: {info['cpu_cores']}")
    print(f"GPU Available: {'Yes' if info['has_gpu'] else 'No'}")
    print(f"Recommended Model: {info['recommended_model']}")


def load_documents(
    doc_dirs: List[str],
    embedding_model: str = "all-MiniLM-L6-v2",
    save_intermediate: str = None,
    verbose: bool = False,
) -> None:
    """Load documents from one or more directories into RAG system."""
    _configure_logging(verbose)

    if verbose:
        print(
            f"Loading documents from {len(doc_dirs)} {'directory' if len(doc_dirs) == 1 else 'directories'}:"
        )
        for doc_dir in doc_dirs:
            print(f"  {doc_dir}")
        print(f"Using embedding model: {embedding_model}")
        print("=" * 50)

    rag = SimpleRAG(embedding_model=embedding_model)
    all_docs = []

    # Set quiet mode for document processor if not verbose
    if rag.processor:
        rag.processor.quiet_mode = not verbose

    # For single directory, use the old SimpleRAG.load_documents method (database rebuild)
    if len(doc_dirs) == 1 and not save_intermediate:
        doc_dir = doc_dirs[0]

        if not verbose:
            print(f"Loading documents from: {doc_dir}")
            print(f"Using embedding model: {embedding_model}")
            print("=" * 40)

        # Show existing database stats if any
        existing_stats = rag.get_stats()
        if existing_stats.get("documents", 0) > 0:
            print(
                f"Existing collection: {existing_stats['documents']} docs, "
                f"{existing_stats['db_size_mb']}MB"
            )
            print("Will clear and rebuild collection...")
            # Reset vector store so it gets recreated with clear_existing=True
            rag.vector_store = None

        rag.load_documents(doc_dir, force_reload=True, verbose=verbose)
        _print_stats_and_status(rag)
        return

    # For multiple directories or save_intermediate, use document collection approach
    for i, doc_dir in enumerate(doc_dirs, 1):
        if verbose:
            print(f"\nProcessing directory {i}/{len(doc_dirs)}: {doc_dir}")

            # Get file count for this directory
            pdf_files = list(Path(doc_dir).glob("*.pdf"))
            print(f"Found {len(pdf_files)} PDF files in {doc_dir}")
        else:
            # Show progress through files in directory
            pdf_files = list(Path(doc_dir).glob("*.pdf"))

            # Process files with progress indicator
            docs = []
            for file_idx, pdf_path in enumerate(pdf_files, 1):
                print(
                    f"\rProcessing {doc_dir} ({i}/{len(doc_dirs)}, {file_idx}/{len(pdf_files)} files)...",
                    end="",
                    flush=True,
                )
                doc_data = rag.processor.process_document(str(pdf_path), quiet=True)
                if doc_data:
                    docs.append(doc_data)

            if docs:
                all_docs.extend(docs)

            print(" done")  # Complete the progress line

        if verbose:
            docs = rag.processor.process_directory(
                doc_dir, show_dir_info=False, verbose=verbose
            )
            if docs:
                all_docs.extend(docs)

    if all_docs:
        # Save intermediate file if requested
        if save_intermediate:
            save_processed_documents(all_docs, save_intermediate)
            print(
                f"Intermediate file saved. Use --load-intermediate {save_intermediate} to generate embeddings later."
            )
            return

        total_chunks = sum(doc["chunk_count"] for doc in all_docs)
        print(f"\nGenerating embeddings for all {total_chunks} chunks...")
        _, embeddings, metadata = rag.processor.create_embeddings_for_docs(all_docs)

        # Add all documents to vector store at once
        # pylint: disable=protected-access
        store = rag._ensure_vector_store()
        store.add_documents(all_docs, embeddings, metadata)
        print("Added all documents to vector database")

        # pylint: disable=protected-access
        rag._documents_loaded = True

    _print_stats_and_status(rag)


def load_from_intermediate(
    intermediate_file: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    verbose: bool = False,
) -> None:
    """Load processed documents from intermediate JSON file and generate embeddings."""
    _configure_logging(verbose)

    print(f"Loading intermediate file: {intermediate_file}")
    print(f"Using embedding model: {embedding_model}")
    print("=" * 50)

    # Load processed documents
    all_docs = load_processed_documents(intermediate_file)

    if all_docs:
        # Initialize RAG system
        rag = SimpleRAG(embedding_model=embedding_model)

        total_chunks = sum(doc["chunk_count"] for doc in all_docs)
        print(f"\nGenerating embeddings for all {total_chunks} chunks...")
        _, embeddings, metadata = rag.processor.create_embeddings_for_docs(all_docs)

        # Add all documents to vector store at once
        # pylint: disable=protected-access
        store = rag._ensure_vector_store()
        store.add_documents(all_docs, embeddings, metadata)
        print("Added all documents to vector database")

        _print_stats_and_status(rag)
    else:
        print("No documents found in intermediate file")


def pull_model_cli(model_name: str) -> None:
    """Pull a model using Ollama."""
    print(f"Pulling model: {model_name}")
    print("=" * 40)

    try:
        import ollama

        print(f"Running: ollama pull {model_name}")
        ollama.pull(model_name)
        print(f"\nModel {model_name} pulled successfully")

        # Show model info
        models = ollama.list()
        for model in models.get("models", []):
            if model_name in model["name"]:
                size_gb = model["size"] / (1024**3)
                print(f"Size: {size_gb:.1f} GB")
                break

    except ImportError:
        print("ERROR: Ollama not installed. Run: pip install ollama")
    except Exception as e:
        print(f"ERROR: Failed to pull model: {e}")
        print("Make sure Ollama is running: ollama serve")
        print(f"Or try manually: ollama pull {model_name}")


def process_query(
    question: str,
    model_name: str = None,
    show_context: bool = False,
    max_k: int = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    keep_model_loaded: bool = False,
) -> None:
    """Process a user query."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"Processing query: {question}")
    print(f"Using embedding model: {embedding_model}")
    print("-" * 40)

    rag = SimpleRAG(model_name=model_name, embedding_model=embedding_model)

    # Check if documents are already in the database
    stats = rag.get_stats()
    if stats.get("documents", 0) > 0:
        print(f"Using existing collection: {rag.collection_name}")
        print(f"Contains {stats['documents']} documents and {stats['chunks']} chunks")
        # pylint: disable=protected-access
        rag._documents_loaded = True
    else:
        print(f"No existing collection found for {embedding_model}")
        print("Load documents first with:")
        print(
            f"   python -m src.cli --load-docs "
            f"--doc-dirs policies/dodd policies/dodi policies/dodm "
            f"--embedding-model {embedding_model}"
        )
        return

    # Show LLM status
    if rag.is_llm_ready():
        print("Using LLM for enhanced responses")
    else:
        print("WARNING: LLM not available - using basic responses")

    # Execute query (get fitted context if requested)
    if show_context:
        result = rag.query(
            question,
            max_k=max_k,
            return_context=True,
            keep_model_loaded=keep_model_loaded,
        )
        if isinstance(result, tuple):
            response, context_results = result
            print(f"\nRetrieved Context Chunks ({len(context_results)} chunks found):")
            print("-" * 60)
            for i, chunk_result in enumerate(context_results, 1):
                filename = chunk_result.get("filename", "Unknown")
                doc_number = chunk_result.get("doc_number", "")
                similarity = chunk_result.get("similarity", 0)
                chunk_text = chunk_result.get("chunk_text", "")

                print(f"\n[Chunk {i}] Source: {filename}")
                if doc_number:
                    print(f"Document Number: {doc_number}")
                print(f"Similarity: {similarity:.3f}")
                print("-" * 40)
                print(chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text)
            print("-" * 60)
        else:
            response = result
    else:
        response = rag.query(question, max_k=max_k, keep_model_loaded=keep_model_loaded)

    print("\nResponse:")
    print(response)


def run_system_test() -> None:
    """Run basic system tests."""
    print("Running system tests...")
    print("=" * 40)

    # Test hardware detection
    print("Testing hardware detection...")
    info = get_hardware_info()
    print(f"  RAM: {info['ram_gb']} GB")

    # Test RAG initialization
    print("Testing RAG initialization...")
    rag = SimpleRAG()

    # Test document loading
    print("Testing document loading...")
    rag.load_documents("policies/test")

    # Test query with context
    print("Testing query with context...")
    response = rag.query("military personnel policy")
    print(f"  Response preview: {response[:100]}...")

    # Show stats
    stats = rag.get_stats()
    print(f"Database contains {stats.get('documents', 0)} documents")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
