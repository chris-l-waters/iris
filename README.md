# 👁️ IRIS - Issuances Retrieval and Intelligence Search

A portfolio project demonstrating edge AI deployment for querying DOD Directives using RAG (Retrieval Augmented Generation). Features a modern web interface with folder selection, real-time processing output, and adaptive hardware detection.

## Features

- **DOD Document Intelligence**: Context-aware retrieval that understands DOD numbering conventions (DODD/DODI/DODM) and clusters related policies
- **Semantic Search**: Vector similarity search using SentenceTransformers embeddings with contextual responses
- **Adaptive Hardware Detection**: Automatically detects system capabilities and recommends optimal configurations
- **Web Interface**: Intuitive GUI with folder selection, live terminal output, and real-time status monitoring
- **Offline Operation**: Complete offline functionality with lightweight, file-based vector database

## Quick Start

### Download DOD Issuances
To generate the vector database, you will need to manually download the policies from https://esd.whs.mil/. This README assumes you will save them to policies/dodi, policies/dodm, and policies/dodd. As of 19 June 2025 there were 1005 documents retrievable from this site.

### Setup
```bash
# Install Ollama (prerequisite for LLM responses)
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh
# Windows: Download from https://ollama.ai/download

# Create virtual environment and install dependencies
make setup

# Or manually:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Start the Web Interface
```bash
# Launch the web interface
python3 gui/app.py

# Opens automatically at http://localhost:8080
```

## Requirements

### System Requirements
- **Python**: 3.10+
- **RAM**: 4GB minimum, 8GB+ recommended
- **GPU**: Not required, highly recommended for speed
- **Storage**: 2GB+ for full document collection
- **OS**: Windows, macOS, or Linux

### Dependencies
- **Document Processing**: pdfplumber, SentenceTransformers
- **Vector Database**: ChromaDB, NumPy
- **Hardware Detection**: psutil, GPUtil
- **LLM Integration**: Ollama

### Hardware Recommendations

#### LLM Models (Text Generation with Ollama - Quantized Q4)
- **2-4GB RAM**: Llama 3.2 1B Instruct Q4 (basic LLM functionality)
- **4-6GB RAM**: Llama 3.2 3B Instruct Q4 models (standard LLM performance)
- **8-12GB RAM**: Mistral 7B Q4 models (high-quality LLM performance)
- **GPU**: Automatic GPU acceleration when available (Ollama)

#### Embedding Models (Semantic Search)
- **4-8GB RAM**: all-MiniLM-L6-v2 (fast processing - default)
- **8-16GB RAM**: all-mpnet-base-v2 (policy documents)
- **16GB+ RAM**: BAAI/bge-base-en-v1.5 (high accuracy)

## Using the GUI

#### Initial Setup
1. **Start Ollama Server**: Use the "Ollama Server Control" panel to start the Ollama server
2. **Select Document Folders**: Click "Select Document Folders" to choose directories containing PDF files
3. **Choose Embedding Model**: Select your preferred embedding model for processing (all-MiniLM-L6-v2, all-mpnet-base-v2, or BAAI/bge-base-en-v1.5)
4. **Process Documents**: Click "Process and Generate Embeddings" to start document processing

#### Document Processing
- **Live Terminal Output**: Watch real-time command output in the terminal-style display
- **Progress Monitoring**: See exactly what files are being processed and current status
- **Auto-completion**: Process completes automatically and updates system status

#### Querying
1. **Model Selection**: Choose your preferred LLM model - it loads automatically when selected
2. **Real-time Status**: Monitor system readiness with live status indicators for docs, models, and server
3. **Ask Questions**: Query your processed documents and get intelligent, contextual responses

### Example Queries
- "Who is the decision authority for an ACAT ID program?"
- "What is the policy on personnel security?"
- "How are performance evaluations conducted?"

## LLM Model Considerations

GPU acceleration will be used if available; querying delays may be frustrating on CPU-only systems, even at the lowest end configuration.

| Model | Parameters | RAM Usage | Speed | Quality | Best For |
|-------|------------|-----------|-------|---------|----------|
| Llama 3.2 1B Instruct Q4 | 1B | 2-4GB | Very Fast | OK | Low-end systems |
| Llama 3.2 3B Instruct Q4 | 3.2B | 4-6GB | Fast | Good | Standard systems |
| Mistral 7B Q4 | 7B | 8-12GB | Medium | Very Good | High-end systems |

#### Embedding Model Performance

| Model | Quality Score | Speed | RAM Usage | Best For |
|-------|---------------|-------|-----------|----------|
| all-MiniLM-L6-v2 | 81.3 | Fastest | 200MB | Fast processing |
| all-mpnet-base-v2 | 84.8 | Slower | 800MB | Policy documents |
| BAAI/bge-base-en-v1.5 | 85.2 | Slowest | 850MB | High accuracy |

## Advanced CLI Usage Examples

For advanced users or automation, CLI commands are available:

```bash
# Check hardware compatibility
python3 -m src.cli --info

# Load single directory
python3 -m src.cli --load-docs --doc-dirs policies/test

# Load multiple directories (full pipeline)
python3 -m src.cli --load-docs --doc-dirs policies/dodd policies/dodi policies/dodm

# Save intermediate processing results (faster iteration)
python3 -m src.cli --load-docs --doc-dirs policies/dodd policies/dodi policies/dodm --save-intermediate processed_docs.json

# Load from intermediate JSON and generate embeddings
python3 -m src.cli --load-intermediate processed_docs.json --embedding-model all-MiniLM-L6-v2

# Show detailed processing information (verbose mode)
python3 -m src.cli --load-docs --doc-dirs policies/dodd policies/dodi policies/dodm --verbose

# Query from command line
python3 -m src.cli --query "What are the requirements for security clearances?"

# Use any Ollama-supported model
python3 -m src.cli --query "your question" --model-name any-ollama-model
```

**Note**: The CLI supports any model available in Ollama. The GUI focuses on the recommended models for optimal user experience, but CLI users can specify any model name that Ollama supports.

### Intermediate File Workflow (Advanced)

For development and testing, you can split document processing into two phases:

**Phase 1: Process PDFs (slow, ~10+ minutes for full dataset)**
```bash
# Process all documents and save to intermediate JSON
python3 -m src.cli --load-docs --doc-dirs policies/dodd policies/dodi policies/dodm --save-intermediate processed_docs.json
```

**Phase 2: Generate Embeddings (fast, ~2-10 minutes per model)**
```bash
# Test different embedding models quickly
python3 -m src.cli --load-intermediate processed_docs.json --embedding-model all-MiniLM-L6-v2
python3 -m src.cli --load-intermediate processed_docs.json --embedding-model all-mpnet-base-v2
python3 -m src.cli --load-intermediate processed_docs.json --embedding-model BAAI/bge-base-en-v1.5
```

**Benefits:**
- **Time Savings**: PDF processing happens only once, then test multiple embedding models quickly
- **Development Efficiency**: Iterate on embedding models without re-processing PDFs
- **Debugging**: Inspect intermediate JSON to understand document processing results

## Architecture

```
src/
├── documents.py          # PDF processing, text extraction, and embedding generation
├── vectorstore.py        # ChromaDB vector database with similarity search
├── rag.py                # Complete RAG pipeline with document retrieval
├── llm.py                # LLM integration using Ollama
├── hardware.py           # Hardware detection and model recommendation
├── embedding_models.py   # Embedding model utilities
├── config.py             # Configuration management (YAML-based)
├── cli.py                # Command-line interface
├── error_utils.py        # Error handling utilities
├── logging_utils.py      # Logging configuration
└── __init__.py           # Package initialization

gui/
├── app.py                # Flask web interface with SSE streaming and task management
├── templates/
│   └── index.html        # Main web interface with terminal display
└── static/
    └── style.css         # CSS with terminal styling and responsive design

database/                 # ChromaDB vector database (auto-created)
├── chroma.sqlite3       # ChromaDB metadata and collections
└── [uuid-dirs]/         # Collection data (binary files)

policies/                 # DOD document collection (downloaded separately)
├── dodd/                 # DOD Directives (259 files)
├── dodi/                 # DOD Instructions (638 files)
├── dodm/                 # DOD Manuals (112 files)
└── test/                 # Test subset (25 files)

tests/                    # Test suite
├── test_documents.py     # Document processing tests
├── test_hardware.py      # Hardware detection tests
├── test_llm.py           # LLM integration tests
├── test_rag.py           # RAG functionality tests
└── test_vectorstore.py   # Vector database tests

debug/                    # Development debugging tools
├── rag_debug.py          # RAG pipeline debugging
├── pipeline_trace.py     # Processing pipeline analysis
└── ...                   # Various debugging utilities

config.yml                # YAML configuration file
start_gui.py              # GUI launcher script
Makefile                  # Build and setup automation
```

### Data Flow
1. **Folder Selection** → Browser dialog → Selected paths → Backend validation
2. **Document Processing** → PDF extraction → DOD number extraction → Text chunks → Live terminal output → Embeddings → Model-specific Vector DB
3. **Real-time Streaming** → Server-Sent Events → Terminal display → Progress feedback
4. **Query Processing** → Embedding → Similarity search → DOD context-aware ranking → Context retrieval → LLM response
5. **GUI Management** → Status monitoring → Model hot-loading → Server control

#### Document Processing Pipeline
- **Step 1**: PDF processing extracts and cleans text, creates chunks, identifies DOD numbers
- **Step 2**: Optional intermediate JSON save enables faster iteration and debugging
- **Step 3**: Embedding generation converts text chunks to vectors for semantic search
- **Step 4**: Vector database storage in model-specific ChromaDB files with DOD metadata

#### DOD Document Intelligence Features
- **Number Extraction**: Robust extraction of DOD directive numbers (DODD/DODI/DODM) from filenames and content
- **Context-Aware Ranking**: Related policies automatically cluster together in search results
- **Hierarchical Understanding**: System recognizes DOD's logical document organization (major groups, subgroups)
- **Smart Boosting**: Documents in the same series receive similarity boosts (1.3x for subgroups, 1.1x for major groups)

## Performance Benchmarks
### M4 Mac, 16gb RAM: 
- Document extraction : 4 min 13 s, 1005 files, 13171 chunks (Default configuration)
- Chunk embeddings:
  - all-MiniLM-L6-v2: 74 seconds
  - all-mpnet-base-v2: 8 min 51 seconds
  - BAAI/bge-base-en-v1.5: 10 min 30 seconds
  - 965.3mb vector databse for all three models together

#### Query Response Performance Benchmarks

Test Query: *"What are the requirements for security clearances?"*

| Embedding Model | llama3.2:1b-instruct-q4_K_M | llama3.2:3b-instruct-q4_K_M | mistral:7b-instruct-q4_K_M |
|----------------|------------|------------------------------|----------------------------|
| **all-MiniLM-L6-v2** | 13s | 29.5s | 54.3s |
| **all-mpnet-base-v2** | 13s | 21.6s\* | 32.75s |
| **BAAI/bge-base-en-v1.5** | 13s | 30.8s | 32.4s† |

\* most detailed; † best answer

## Development Status

### Recently Completed
- ✅ **Modern Web Interface**: Complete GUI with intuitive folder selection and responsive design
- ✅ **Live Terminal Output**: Real-time streaming of document processing with terminal-style display
- ✅ **Flexible Document Processing**: Support for any PDF folders with live progress feedback
- ✅ **Server-Sent Events**: Streaming command output via SSE for real-time user feedback
- ✅ **Automatic Model Management**: Models load automatically when selected with status indicators
- ✅ **Ollama Server Control**: Start/stop server directly from GUI with real-time status
- ✅ **Multiple Embedding Models**: Process documents with different models and instant database switching
- ✅ **Hardware Detection**: Automatic recommendations with manual override options

### Not yet implemented
- **Testing on other systems**: will be done following initial commit.

### Planned Features
1. **Web Backend Performance Refactor**: Hybrid architecture combining CLI power with web performance
2. Windows installer with model downloading
3. MacOS installer and distribution optimization
4. Performance optimization and final polish

## License

Apache License 2.0
