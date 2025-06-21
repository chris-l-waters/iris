# 👁️ IRIS - Issuances Retrieval and Intelligent Search

A portfolio project demonstrating edge AI deployment for querying DOD Directives using RAG (Retrieval Augmented Generation). Features a modern web interface with folder selection, real-time processing output, and adaptive hardware detection.

## Features

- **DOD Document Intelligence**: Context-aware retrieval with hierarchical boosting for document relationships AND adjacent chunk proximity within same documents
- **Semantic Search**: Vector similarity search using SentenceTransformers embeddings with contextual responses
- **Adaptive Hardware Detection**: Automatically detects system capabilities and recommends optimal configurations
- **Web Interface**: Intuitive GUI with folder selection, live terminal output, and real-time status monitoring
- **Offline Operation**: Complete offline functionality with lightweight, file-based vector database

## Quick Start

### Download DOD Issuances
To generate the vector database, you will need to manually download the policies from https://esd.whs.mil/. This README assumes you will save them to policies/dodi, policies/dodm, and policies/dodd. As of 19 June 2025 there were 1005 documents retrievable from this site.

Alternatively, a .torrent with prebuilt vector databases and policy PDFs can be used: 
```magnet:?xt=urn:btih:566FC09B4D96054B309BCD5EBA69B3CE971A77A0&dn=iris_database&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=udp%3A%2F%2Ftracker.openbittorrent.com%3A80%2Fannounce```

### Setup (Windows)

1. **Install Prerequisites**
  - Python 3.10+: Download from https://python.org/downloads/
    - Be sure to check "Add Python to PATH" during installation
    - Test: Open Command Prompt → python --version
  - Ollama: Download from https://ollama.ai/download
    - Install the Windows executable
    - Test: ollama --version
2. **Setup Virtual Environment**
  - Open Command Prompt in your IRIS folder:
    ```powershell
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```

  3. **Start Ollama Server**
  - In one Command Prompt window (keep running):
      ```powershell
      ollama serve
      ```
  4. **Launch IRIS GUI**
   - In another Command Prompt window:
      ```powershell   
      cd path\to\iris
      venv\Scripts\activate
      python gui\app.py
      ```

### Setup (Mac/Linux)
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
- **2-4GB RAM**: TinyLlama 1.1B Chat Q4 (minimum spec functionality)
- **4-6GB RAM**: Llama 3.2 1B Instruct Q4 (basic LLM functionality)  
- **6-8GB RAM**: Llama 3.2 3B Instruct Q4 (standard LLM performance)
- **8-12GB RAM**: Gemma2 9B Instruct Q4 (high-quality performance)
- **12GB+ RAM**: Phi4 Mini (premium performance)
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
| TinyLlama 1.1B Chat Q4 | 1.1B | 2-4GB | Fastest | Basic | Minimum systems |
| Llama 3.2 1B Instruct Q4 | 1B | 4-6GB | Very Fast | OK | Low-end systems |
| Llama 3.2 3B Instruct Q4 | 3.2B | 6-8GB | Fast | Good | Standard systems |
| Gemma2 9B Instruct Q4 | 9B | 8-12GB | Medium | Very Good | High-end systems |
| Phi4 Mini | 14B | 12GB+ | Slower | Excellent | Premium systems |

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

## Configuration and Customization

### Prompt Templates

IRIS uses configurable prompt templates stored in `config.yml` for maximum flexibility. You can customize how the system instructs LLMs to respond to queries by editing the prompt templates:

```yaml
# config.yml
prompts:
  simple:      # For TinyLlama models
    template: |
      Context from DOD policies:
      {context}
      
      Question: {question}
      
      Instructions: Answer based ONLY on the context above...
  
  instruct:    # For Llama 3.2 models  
    template: |
      ### System:
      You are an expert assistant for Department of Defense policy questions...
  
  mistral:     # For Mistral models
    template: |
      [INST] You are a helpful assistant that answers questions...
```

**Key benefits:**
- **Easy experimentation**: Test different prompt strategies without code changes
- **Model-specific optimization**: Different templates for different model families
- **Version control**: Prompt changes are tracked in git
- **Quick iteration**: Modify prompts and test immediately

### System Configuration

The `config.yml` file controls all major system parameters:

```yaml
# Model behavior
llm:
  default_context_window: 4096    # Context length for all models
  fallback_temperature: 0.7       # Response randomness

# Document processing
document_processing:
  default_chunk_size: 500         # Words per document chunk
  default_chunk_overlap: 50       # Overlap between chunks

# Document-aware ranking with adjacency boosting
ranking:
  same_document_boost: 1.5          # Same filename
  adjacent_chunk_boost: 1.3         # Adjacent chunks (±1)
  near_chunk_boost: 1.15            # Near chunks (±2)
  nearby_chunk_boost: 1.05          # Nearby chunks (±3-5)

# Hardware detection
# (automatically configured based on system capabilities)
```

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
├── llm.py                # LLM integration using Ollama with configurable prompts
├── hardware.py           # Hardware detection and model recommendation
├── embedding_models.py   # Embedding model utilities
├── config.py             # Configuration management (YAML-based) with prompt templates
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

config.yml                # YAML configuration file with model settings and prompt templates
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
- Most of the reason this runs slowly is due to the pdfplumber table detection. Luckily, it only needs to be done once.

#### DOD Document Intelligence Features
- **Number Extraction**: Robust extraction of DOD directive numbers (DODD/DODI/DODM) from filenames and content
- **Context-Aware Ranking**: Related policies automatically cluster together in search results
- **Hierarchical Understanding**: System recognizes DOD's logical document organization (major groups, subgroups)
- **Smart Boosting**: Documents in the same series receive similarity boosts, plus adjacent chunks within the same document receive multiplicative proximity boosts
- **Adjacency Boosting**: Sequential chunks (before/after relevant content) are boosted to provide better context and narrative flow

## Performance Benchmarks
#### Document Extraction + Embeddings

| System Configuration | Document Extraction | all-MiniLM-L6-v2 | all-mpnet-base-v2 | BAAI/bge-base-en-v1.5 | Vector DB Size |
|----------------------|--------------------|--------------------|-------------------|----------------------|----------------|
| **M4 MacBook Pro**<br>10 core CPU/GPU, 16GB RAM | 11m 54s | 52s | 6m 2s | 6m 58s | 742mb |
| **AMD 9800X3D**<br>64GB RAM, Pop!_OS 22.04 | XXX | XXX | XXX | XXX | XXX |
| **AMD 5600X + GTX 1080**<br>32GB RAM, 8GB VRAM, Pop!_OS 22.04 | XXX | XXX | XXX | XXX | XXX |

#### Query Response Performance Benchmarks
##### Test Query: *"What are the requirements for security clearances?"*

| M4, 10 core CPU/GPU | tinyllama:1.1b-chat-v1-q4_K_M | llama3.2:1b-instruct-q4_K_M | llama3.2:3b-instruct-q4_K_M | gemma2:9b-instruct-q4_K_M | phi4-mini:latest |
|----------------|------------|------------|------------------------------|----------------------------|------------------|
| **all-MiniLM-L6-v2** | 6.4s | 9.4s | 6.5s | 53s | 35s |
| **all-mpnet-base-v2** | 9.4s† | 9.4s | 10.5s | 58s†* | 38.4s |
| **BAAI/bge-base-en-v1.5** | 9.5s | 7.9s† | 17.1s† | 62s | 33.8s† |

| 9800X3D | tinyllama:1.1b-chat-v1-q4_K_M | llama3.2:1b-instruct-q4_K_M | llama3.2:3b-instruct-q4_K_M | gemma2:9b-instruct-q4_K_M | phi4-mini:latest |
|----------------|------------|------------|------------------------------|----------------------------|------------------|
| **all-MiniLM-L6-v2** | XXX | XXX | XXX | XXX | XXX |
| **all-mpnet-base-v2** | XXX | XXX | XXX*† | XXX | XXX |
| **BAAI/bge-base-en-v1.5** | XXX | XXX | XXX | XXX | XXX |

| 5600X/GTX 1080  | tinyllama:1.1b-chat-v1-q4_K_M | llama3.2:1b-instruct-q4_K_M | llama3.2:3b-instruct-q4_K_M | gemma2:9b-instruct-q4_K_M | phi4-mini:latest |
|----------------|------------|------------|------------------------------|----------------------------|------------------|
| **all-MiniLM-L6-v2** | XXX | XXX | XXX* | XXX | XXX |
| **all-mpnet-base-v2** | XXX | XXX | XXX | XXX | XXX |
| **BAAI/bge-base-en-v1.5** | XXX | XXX | XXX† | XXX | XXX |

† - best per model * best overall

Windows 10 was tested for installation and verification, not benchmarking; I only have Windows VM's available and don't run it on bare metal.

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

### Future, possible features
1. **Web Backend Performance Refactor**: Hybrid architecture combining CLI power with web performance
2. Windows installer with model downloading
3. MacOS installer and distribution optimization

## License

Apache License 2.0
