# 👁️ IRIS - Issuances Retrieval and Intelligent Search

A portfolio project demonstrating edge AI deployment for querying DOD Directives using RAG (Retrieval Augmented Generation). Features a modern web interface with folder selection, real-time processing output, and adaptive hardware detection.

## Features

- **Advanced Retrieval System**: 
  - **Cross-encoder Reranking**: Optional two-stage retrieval with cross-encoder models for improved result quality
  - **Document-aware Ranking**: Hierarchical boosting for document relationships and adjacent chunk proximity
- **Semantic Search**: Vector similarity search using SentenceTransformers embeddings with contextual responses
- **Adaptive Hardware Detection**: Automatically detects system capabilities and recommends optimal configurations
- **Centralized Configuration**: All models and settings managed through `config.yml` for easy customization
- **Web Interface**: Intuitive GUI with folder selection, live terminal output, and real-time status monitoring
- **Offline Operation**: Complete offline functionality with lightweight, file-based vector database

## Quick Start

### Download DOD Issuances
To generate the vector database, you will need to manually download the policies from https://esd.whs.mil/. This README assumes you will save them to policies/dodi, policies/dodm, and policies/dodd. As of 19 June 2025 there were 1005 documents retrievable from this site.

Alternatively, a .torrent with prebuilt vector databases and policy PDFs can be used. It will likely take several hours to generate embeddings with the highest performer if you lack a CUDA-capable GPU or Apple silicon. This will get you querying much, much faster: 
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
- **Cross-encoder Reranking**: sentence-transformers (cross-encoder models)
- **Hardware Detection**: psutil, GPUtil
- **LLM Integration**: Ollama
- **Configuration**: PyYAML
- **Web Interface**: Flask

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
- **16GB+ RAM**: mixedbread-ai/mxbai-embed-large-v1 (best retrieval)

## Using the GUI

#### Initial Setup
1. **Choose Embedding Model**: Select your preferred embedding model for processing (all-MiniLM-L6-v2, all-mpnet-base-v2, or mixedbread-ai/mxbai-embed-large-v1)
2. **Select Document Folders**: Click "Select Document Folders" to choose directories containing PDF files
3. **Process Documents**: Click "Process and Generate Embeddings" to start document processing

#### Document Processing
- **Live Terminal Output**: Watch real-time command output in the terminal-style display
- **Progress Monitoring**: See exactly what files are being processed and current status
- **Auto-completion**: Process completes automatically and updates system status

#### Querying
1. **Start Ollama Server**: Use the "Ollama Server Control" panel to start the Ollama server
2. **Model Selection**: Choose your preferred LLM and embedding models - they load automatically when selected
3. **Advanced Options**: Enable cross-encoder reranking for improved result quality (slower but more accurate)
4. **Real-time Status**: Monitor system readiness with live status indicators for docs, models, and server
5. **Ask Questions**: Query your processed documents and get intelligent, contextual responses with automatic citations

### Example Queries
- "Who is the decision authority for an ACAT ID program?"
- "When is a CAPE AOA requried for an MDAP?"
- "How are performance evaluations conducted?"

**Sample Response with Citations:**
```
Response:
Based on DOD policies: For an ACAT ID program, the decision authority is typically either the Defense Acquisition Executive (DAE) or a designee. This can be inferred from Section 2430(d)(3)(A) of Title 10 U.S.C., which states that for programs where "the USD(A&S) has designated [an alternate MDA],... the Secretary of the Military Department concerned, or designee," may request reversion back to the SAE. Additionally, it is mentioned in Section 2430(d)(2), stating: "The service acquisition executive (SAE)...will review ACAT IB programs unless otherwise specified." Therefore, for an ACAT ID program specifically designated as such by the USD(A&S) and not delegated elsewhere within DoD Policy, either the DAE or a designee would typically have decision authority.

Sources: 5000.01, 5000.02, 5000.82, 5000.85
```

## LLM Model Considerations

GPU acceleration will be used if available; querying delays may be frustrating on CPU-only systems, even at the lowest end configuration.

| Model | Parameters | RAM Usage | Speed | Quality | Best For |
|-------|------------|-----------|-------|---------|----------|
| TinyLlama 1.1B Chat Q4 | 1.1B | 2-4GB | Fastest | Basic | Minimum systems |
| Llama 3.2 1B Instruct Q4 | 1B | 4-6GB | Very Fast | Good | Low-end systems |
| Llama 3.2 3B Instruct Q4 | 3.2B | 6-8GB | Fast | Very Good | Standard systems |
| Gemma2 9B Instruct Q4 | 9B | 8-12GB | Medium | Excellent | High-end systems |
| Phi4 Mini | 14B | 12GB+ | Slower | Best | Premium systems |

#### Embedding Model Performance

| Model | Quality Score | Speed | RAM Usage | Best For |
|-------|---------------|-------|-----------|----------|
| all-MiniLM-L6-v2 | 81.3 | Fastest | 200MB | Fast processing |
| all-mpnet-base-v2 | 84.8 | Fast | 800MB | Policy documents |
| mixedbread-ai/mxbai-embed-large-v1 | 87.2 | Medium | 1200MB | Best retrieval |

## Advanced CLI Usage Examples

For advanced users or automation, CLI commands are available:

```bash
# Check hardware compatibility
python3 -m src.cli --info

# Load single directory
python3 -m src.cli --load-docs --doc-dirs policies/test

# Load multiple directories (full pipeline)
python3 -m src.cli --load-docs --doc-dirs policies/dodd policies/dodi policies/dodm

# Show detailed processing information (verbose mode)
python3 -m src.cli --load-docs --doc-dirs policies/dodd policies/dodi policies/dodm --verbose

# Query from command line
python3 -m src.cli --query "What are the requirements for security clearances?"

# Query with cross-encoder reranking for improved accuracy
python3 -m src.cli --query "Who has ACAT delegation authority?" --xencode

# Use specific embedding model
python3 -m src.cli --query "your question" --embedding-model mixedbread-ai/mxbai-embed-large-v1

# Use any Ollama-supported model
python3 -m src.cli --query "your question" --model-name any-ollama-model

# Show retrieved context chunks with query response
python3 -m src.cli --query "your question" --show-context
```

**Note**: The CLI supports any model available in Ollama. The GUI focuses on the recommended models for optimal user experience, but CLI users can specify any model name that Ollama supports.

## Configuration and Customization

### Centralized Configuration

IRIS uses a centralized `config.yml` file for all system settings, making customization easy and consistent across the entire application.

#### Model Configurations

Both LLM and embedding models are configured in `config.yml`:

```yaml
# LLM Models (automatically detected by GUI)
llm:
  models:
    "llama3.2:1b-instruct-q4_K_M":
      tier: 1
      max_tokens: 400
      temperature: 0.7
      context_window: 4096
      prompt_style: "instruct"

# Embedding Models (automatically detected by GUI)  
embedding_models:
  "all-MiniLM-L6-v2":
    tier: 0
    ram_usage_mb: 200
    quality_score: 81.3
    best_for: "Fast processing"
```

#### Cross-encoder Reranking

Configure cross-encoder settings for improved retrieval quality:

```yaml
retrieval:
  cross_encoder:
    model_name: "cross-encoder/ms-marco-MiniLM-L6-v2"
    rerank_top_k: 20      # Candidates to rerank
    final_top_k: 10       # Final results returned
    batch_size: 8         # Inference batch size
    max_length: 512       # Max tokens per query+passage pair
```

#### Prompt Templates

Customize how the system instructs LLMs to respond:

```yaml
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
python3 -m src.cli --load-intermediate processed_docs.json --embedding-model mixedbread-ai/mxbai-embed-large-v1
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
├── chroma.sqlite3        # ChromaDB metadata and collections
├── documents.sqlite3     # Document chunks (shared between embeddings)
└── [uuid-dirs]/          # Collection data (binary files)

(policies/)               # DOD document collection (downloaded separately)
├── (dodd/)               # DOD Directives (259 files)
├── (dodi/)               # DOD Instructions (638 files)
├── (dodm/)               # DOD Manuals (112 files)
└── (test/)               # Test subset (25 files)

tests/                    # Test suite
├── test_documents.py     # Document processing tests
├── test_hardware.py      # Hardware detection tests
├── test_llm.py           # LLM integration tests
├── test_rag.py           # RAG functionality tests
└── test_vectorstore.py   # Vector database tests

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
- Pdfplumber table detection and vector generation both take significant time, especially on the advanced embedding models. Luckily, it only needs to be done once.

#### DOD Document Intelligence Features
- **Number Extraction**: Robust extraction †of DOD directive numbers (DODD/DODI/DODM) from filenames and content
- **Context-Aware Ranking**: Related policies automatically cluster together in search results
- **Hierarchical Understanding**: System recognizes DOD's logical document organization (major groups, subgroups)
- **Smart Boosting**: Documents in the same series receive similarity boosts, plus adjacent chunks within the same document receive multiplicative proximity boosts
- **Adjacency Boosting**: Sequential chunks (before/after relevant content) are boosted to provide better context and narrative flow

## Performance Benchmarks
#### Document Extraction + Embeddings (310MB total)

| System Configuration | Document Extraction | all-MiniLM-L6-v2 | all-mpnet-base-v2 | mixedbread-ai/mxbai-embed-large-v1 |
|----------------------|--------------------|--------------------|-------------------|----------------------|
| **M4 Mac Mini**<br>10 core CPU/GPU, 16GB RAM | X | 1m 42s | 13m 42s | 49m 16s |
| **AMD 9800X3D**<br>64GB RAM, Pop!_OS 22.04 | 4m 59s | 3m 17s | 42m 49s | (chose not to run) |

#### Query Response Performance Benchmarks
##### Test Query: *"What ACAT levels delegate decision authority to the service components?"*

| M4, 10 core CPU/GPU | tinyllama:1.1b-chat-v1-q4_K_M | llama3.2:1b-instruct-q4_K_M | llama3.2:3b-instruct-q4_K_M | gemma2:9b-instruct-q4_K_M | phi4-mini:latest |
|----------------|------------|------------|------------------------------|----------------------------|------------------|
| **all-MiniLM-L6-v2** | 6.3s | 11.4s | 11.2s | 51.1s | 33.8s |
| **all-mpnet-base-v2** | 5.6s | 6.5s | 11.3s | 62.7s | 32.9s |
| **mixedbread-ai/mxbai-embed-large-v1** | 8.1s† | 7.2s† | 21.0s† | 58.0s†* | 44.1s† |

| 9800X3D/6950XT | tinyllama:1.1b-chat-v1-q4_K_M | llama3.2:1b-instruct-q4_K_M | llama3.2:3b-instruct-q4_K_M | gemma2:9b-instruct-q4_K_M | phi4-mini:latest |
|----------------|------------|------------|------------------------------|----------------------------|------------------|
| **all-MiniLM-L6-v2** | X | X | X | X | X |
| **all-mpnet-base-v2** | X | X | X | X | X |
| **mixedbread-ai/mxbai-embed-large-v1** | X | X | X | X | X |

| 9800X3D (no GPU) | tinyllama:1.1b-chat-v1-q4_K_M | llama3.2:1b-instruct-q4_K_M | llama3.2:3b-instruct-q4_K_M | gemma2:9b-instruct-q4_K_M | phi4-mini:latest |
|----------------|------------|------------|------------------------------|----------------------------|------------------|
| **all-MiniLM-L6-v2** | X | X | X | X | X |
| **all-mpnet-base-v2** | X | X | X | X | X |
| **mixedbread-ai/mxbai-embed-large-v1** | X | X | X | X | X |

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

# Contact

**Dr. Christopher Waters** - chris@dr-w.co  - [Github account](https://github.com/chris-l-waters)