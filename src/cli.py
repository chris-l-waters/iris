"""Simple CLI interface for testing the RAG system."""

import argparse
import logging
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

from .documents import load_processed_documents
from .embedding_models import print_model_comparison
from .hardware import get_hardware_info
from .rag import SimpleRAG


def _process_document_worker(pdf_path_str: str) -> dict:
    """Worker function for parallel document processing."""
    from .rag import SimpleRAG

    # Create a fresh RAG instance for this worker process
    rag = SimpleRAG()

    try:
        doc_data = rag.processor.process_document(pdf_path_str, quiet=True)
        return doc_data
    except Exception as e:
        # Return error info instead of failing silently
        return {"error": True, "filename": Path(pdf_path_str).name, "message": str(e)}


def _process_documents_streaming_to_json(
    doc_dirs: List[str], output_file: str, verbose: bool = False
) -> None:
    """Process documents in streaming fashion, writing to JSON immediately to minimize RAM usage."""
    import json

    total_processed = 0

    # Open JSON file for streaming write
    with open(output_file, "w") as f:
        f.write("[\n")  # Start JSON array
        first_doc = True

        for i, doc_dir in enumerate(doc_dirs, 1):
            pdf_files = list(Path(doc_dir).glob("*.pdf"))

            if verbose:
                print(f"\nProcessing directory {i}/{len(doc_dirs)}: {doc_dir}")
                print(f"Found {len(pdf_files)} PDF files in {doc_dir}")

            if pdf_files:
                # Process in smaller batches to control memory
                batch_size = max(
                    1, min(8, len(pdf_files))
                )  # Process 8 files at a time max
                dir_processed = 0

                for batch_start in range(0, len(pdf_files), batch_size):
                    batch_files = pdf_files[batch_start : batch_start + batch_size]
                    batch_num = batch_start // batch_size + 1
                    total_batches = (len(pdf_files) - 1) // batch_size + 1

                    if verbose:
                        print(f"  Processing batch {batch_num}/{total_batches}")
                    else:
                        # Update progress line in place
                        print(
                            f"\rProcessing {doc_dir} ({i}/{len(doc_dirs)}, batch {batch_num}/{total_batches})...",
                            end="",
                            flush=True,
                        )

                    # Process this batch
                    docs = _process_documents_parallel(
                        batch_files,
                        doc_dir,
                        i,
                        len(doc_dirs),
                        verbose,
                        batch_num,
                        total_batches,
                    )

                    # Write each document to JSON immediately (remove raw_text to save space)
                    for doc in docs:
                        if not first_doc:
                            f.write(",\n")
                        # Remove raw_text and clean_text to save file size
                        doc_copy = doc.copy()
                        doc_copy.pop("raw_text", None)
                        doc_copy.pop("clean_text", None)
                        json.dump(doc_copy, f, indent=2)
                        first_doc = False
                        total_processed += 1
                        dir_processed += 1

                    # Flush to disk after each batch
                    f.flush()

                    # Clear memory
                    del docs

                # Print final status for this directory
                if not verbose:
                    print(
                        f"\rProcessing {doc_dir} ({i}/{len(doc_dirs)}) done ({dir_processed} processed)"
                    )
                else:
                    print(
                        f"  Completed directory {i}/{len(doc_dirs)}: {dir_processed} documents processed"
                    )

        f.write("\n]")  # End JSON array

    print(f"Processed {total_processed} documents to {output_file}")


def _process_documents_streaming_to_vectorstore(
    doc_dirs: List[str], rag, verbose: bool = False
) -> None:
    """Process documents in batches, adding to vector store immediately to minimize RAM usage."""
    total_processed = 0
    total_chunks = 0

    # pylint: disable=protected-access
    store = rag._ensure_vector_store()

    for i, doc_dir in enumerate(doc_dirs, 1):
        pdf_files = list(Path(doc_dir).glob("*.pdf"))

        if verbose:
            print(f"\nProcessing directory {i}/{len(doc_dirs)}: {doc_dir}")
            print(f"Found {len(pdf_files)} PDF files in {doc_dir}")

        if pdf_files:
            # Process in smaller batches to control memory
            batch_size = max(1, min(8, len(pdf_files)))  # Process 8 files at a time max

            for batch_start in range(0, len(pdf_files), batch_size):
                batch_files = pdf_files[batch_start : batch_start + batch_size]

                if verbose:
                    print(
                        f"  Processing batch {batch_start // batch_size + 1}/{(len(pdf_files) - 1) // batch_size + 1}"
                    )

                # Process this batch
                batch_num = batch_start // batch_size + 1
                total_batches = (len(pdf_files) - 1) // batch_size + 1
                docs = _process_documents_parallel(
                    batch_files,
                    doc_dir,
                    i,
                    len(doc_dirs),
                    verbose,
                    batch_num,
                    total_batches,
                )

                if docs:
                    # Generate embeddings for this batch only
                    batch_chunks = sum(doc["chunk_count"] for doc in docs)
                    if verbose:
                        print(f"    Generating embeddings for {batch_chunks} chunks...")

                    _, embeddings, metadata = rag.processor.create_embeddings_for_docs(
                        docs
                    )

                    # Add this batch to vector store
                    store.add_documents(docs, embeddings, metadata)

                    total_processed += len(docs)
                    total_chunks += batch_chunks

                    if verbose:
                        print(f"    Added {len(docs)} documents to vector store")

                    # Clear memory
                    del docs, embeddings, metadata

    print(f"Processed {total_processed} documents with {total_chunks} total chunks")

    # pylint: disable=protected-access
    rag._documents_loaded = True


def _process_documents_parallel(
    pdf_files: List[Path],
    doc_dir: str,
    dir_idx: int,
    total_dirs: int,
    verbose: bool = False,
    batch_num: int = 1,
    total_batches: int = 1,
) -> List[dict]:
    """Process multiple documents in parallel with progress tracking."""
    if not pdf_files:
        return []

    # Determine number of worker processes
    # Use conservative worker count to avoid overwhelming the system
    # Each PDF processing is CPU/memory intensive, so limit workers
    max_workers = max(1, min(len(pdf_files), max(2, multiprocessing.cpu_count() // 4)))

    if verbose:
        print(
            f"Processing {len(pdf_files)} files with {max_workers} parallel workers..."
        )

    docs = []
    completed = 0
    errors = 0

    # Convert Path objects to strings for the worker function
    pdf_path_strs = [str(pdf_path) for pdf_path in pdf_files]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(_process_document_worker, pdf_path_str): pdf_path_str
            for pdf_path_str in pdf_path_strs
        }

        # Process completed tasks as they finish
        for future in as_completed(future_to_path):
            pdf_path_str = future_to_path[future]
            filename = Path(pdf_path_str).name

            try:
                result = future.result()
                completed += 1

                if result and not result.get("error", False):
                    docs.append(result)
                    if verbose:
                        print(f"✓ Completed {filename} ({completed}/{len(pdf_files)})")
                    else:
                        # Show per-file progress within the batch
                        print(
                            f"\rProcessing {doc_dir} ({dir_idx}/{total_dirs}, batch {batch_num}/{total_batches}, {completed}/{len(pdf_files)} files)...",
                            end="",
                            flush=True,
                        )
                else:
                    errors += 1
                    if verbose:
                        error_msg = (
                            result.get("message", "Unknown error")
                            if result
                            else "Failed to process"
                        )
                        print(f"✗ Failed {filename}: {error_msg}")

            except Exception as e:
                errors += 1
                completed += 1
                if verbose:
                    print(f"✗ Error processing {filename}: {e}")

    if not verbose:
        # Don't print completion message here - let the caller handle it
        pass
    elif verbose and errors > 0:
        print(f"Completed with {errors} errors out of {len(pdf_files)} files")

    return docs


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
            "mixedbread-ai/mxbai-embed-large-v1",
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

    # For multiple directories or save_intermediate, use streaming approach to minimize RAM usage
    if save_intermediate:
        # For intermediate save, use streaming JSON writing
        _process_documents_streaming_to_json(doc_dirs, save_intermediate, verbose)
        print(
            f"Intermediate file saved. Use --load-intermediate {save_intermediate} to generate embeddings later."
        )
        return
    else:
        # For direct embedding generation, use batched processing to control memory usage
        _process_documents_streaming_to_vectorstore(doc_dirs, rag, verbose)

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
            print(
                f"\nContext Chunks Used in Prompt ({len(context_results)} chunks fitted):"
            )
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
