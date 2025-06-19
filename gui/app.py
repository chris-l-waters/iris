"""Flask web interface for IRIS RAG system."""

import os
import re
import subprocess
import sys
import time
import webbrowser
from threading import Thread
import psutil
from flask import Flask, render_template, request, jsonify

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.hardware import get_hardware_info  # noqa: E402
from src.config import (  # noqa: E402
    DEFAULT_WEB_PORT,
    DEFAULT_WEB_HOST,
    DEFAULT_QUERY_TIMEOUT,
)

app = Flask(__name__)

# Track model hot-loading state
HOT_LOADED_MODEL = None

# Track ollama server process
OLLAMA_SERVER_PROCESS = None

# Track running background tasks
running_tasks = {}

# Security: Define allowed values for user inputs
ALLOWED_EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
}

ALLOWED_LLM_MODELS = {
    "llama3.2:1b-instruct-q4_K_M",
    "llama3.2:3b-instruct-q4_K_M",
    "mistral:7b-instruct-q4_K_M",
}


def validate_input(value, max_length=1000):
    """Validate user input for security."""
    if not value:
        return True  # Empty values are ok

    # Check length
    if len(value) > max_length:
        return False

    # Check for dangerous characters that could be used for injection
    if re.search(r"[;&|`$(){}\[\]<>\\]", value):
        return False

    return True


def validate_model_name(model_name, allowed_models):
    """Validate model name against allowed list."""
    return model_name in allowed_models or not model_name  # Empty is ok


def check_model_availability(model_name):
    """Check if a model is available in Ollama."""
    try:
        import ollama

        models_response = ollama.list()
        # Type annotation to help pylint understand the structure
        model_list = [model.model for model in getattr(models_response, "models", [])]
        return model_name in model_list
    except (ImportError, ConnectionError, AttributeError):
        return False


def safe_ollama_call(model_name, operation="load"):
    """Safely call ollama operations without subprocess injection."""
    try:
        import ollama

        if operation == "load":
            ollama.generate(
                model=model_name, prompt="", options={"num_predict": 1}, keep_alive=-1
            )
        elif operation == "unload":
            ollama.generate(
                model=model_name, prompt="", options={"num_predict": 1}, keep_alive=0
            )
        return True, None
    except ImportError as e:
        return False, f"Ollama library not installed: {str(e)}"
    except ConnectionError as e:
        return False, f"Cannot connect to Ollama server: {str(e)}"
    except (RuntimeError, ValueError, OSError) as e:
        # Handle ollama.ResponseError and other ollama-specific errors
        error_msg = str(e)
        if "not found" in error_msg.lower():
            return (
                False,
                f"Model '{model_name}' not found. "
                f"Please pull it first with: ollama pull {model_name}",
            )
        if "status code: 404" in error_msg:
            return (
                False,
                f"Model '{model_name}' not available. "
                f"Try pulling it with: ollama pull {model_name}",
            )
        else:
            return False, f"Ollama error: {error_msg}"


def is_ollama_server_running():
    """Check if ollama server is running by checking for process."""
    try:
        # Check if we can connect to ollama
        import ollama

        ollama.list()
        return True
    except (ImportError, ConnectionError, RuntimeError):
        return False


def start_ollama_server():
    """Start ollama server process."""
    global OLLAMA_SERVER_PROCESS
    try:
        # Start ollama serve in background
        OLLAMA_SERVER_PROCESS = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # Give it a moment to start
        time.sleep(2)
        return True, "Server started successfully"
    except FileNotFoundError:
        return False, "Ollama not found. Please install ollama first."
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"Error starting server: {str(e)}"


def stop_ollama_server():
    """Stop ollama server process gracefully."""
    global OLLAMA_SERVER_PROCESS
    try:
        if OLLAMA_SERVER_PROCESS and OLLAMA_SERVER_PROCESS.poll() is None:
            # Try graceful shutdown first
            OLLAMA_SERVER_PROCESS.terminate()
            try:
                OLLAMA_SERVER_PROCESS.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                OLLAMA_SERVER_PROCESS.kill()
                OLLAMA_SERVER_PROCESS.wait()
            OLLAMA_SERVER_PROCESS = None
            return True, "Server stopped successfully"
        else:
            # Try to find and kill any existing ollama serve processes
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    if (
                        proc.info["name"] == "ollama"
                        and "serve" in proc.info["cmdline"]
                    ):
                        proc.terminate()
                        proc.wait(timeout=5)
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.TimeoutExpired,
                ):
                    pass
            return True, "Any existing servers stopped"
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"Error stopping server: {str(e)}"


@app.route("/")
def index():
    """Main interface page."""
    # Get hardware info for display
    hardware_info = get_hardware_info()

    # Check if documents are loaded
    docs_loaded = check_docs_loaded()

    # Check if Ollama is available
    llm_available = is_ollama_server_running()

    # System is ready when both docs and LLM are available
    system_ready = docs_loaded and llm_available

    return render_template(
        "index.html",
        hardware_info=hardware_info,
        docs_loaded=docs_loaded,
        system_ready=system_ready,
    )


@app.route("/query", methods=["POST"])
def query():
    """Process a query using the RAG system."""
    global HOT_LOADED_MODEL
    question = request.form.get("question", "").strip()
    embedding_model = request.form.get("embedding_model", "all-MiniLM-L6-v2")
    llm_model = request.form.get("llm_model", "")

    # Security: Validate all inputs
    if not question:
        return jsonify({"error": "Please provide a question"})

    if not validate_input(question, max_length=2000):
        return jsonify({"error": "Invalid question format or too long"})

    if not validate_model_name(embedding_model, ALLOWED_EMBEDDING_MODELS):
        return jsonify({"error": "Invalid embedding model"})

    if not validate_model_name(llm_model, ALLOWED_LLM_MODELS):
        return jsonify({"error": "Invalid LLM model"})

    try:
        # Build command with model parameters
        cmd = [sys.executable, "-m", "src.cli", "--query", question]

        # Add embedding model if not default
        if embedding_model and embedding_model != "all-MiniLM-L6-v2":
            cmd.extend(["--embedding-model", embedding_model])

        # Add LLM model if specified
        if llm_model:
            cmd.extend(["--model-name", llm_model])

        # Add keep_model_loaded flag if model is hot-loaded
        if HOT_LOADED_MODEL == llm_model:
            cmd.append("--keep-model-loaded")

        # Run the query through the CLI (increased timeout for model loading scenarios)
        timeout = (
            120 if HOT_LOADED_MODEL == llm_model else DEFAULT_QUERY_TIMEOUT
        )  # Longer timeout if model might need loading
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        if result.returncode == 0:
            # Update HOT_LOADED_MODEL if auto was used and query succeeded
            if not llm_model:  # Auto was used
                hardware_info = get_hardware_info()
                HOT_LOADED_MODEL = hardware_info["recommended_model"]
            elif llm_model:  # Specific model was used
                HOT_LOADED_MODEL = llm_model

            return jsonify({"response": result.stdout})

        return jsonify({"error": f"Query failed: {result.stderr}"})

    except subprocess.TimeoutExpired:
        return jsonify({"error": f"Query timed out ({timeout}s limit)"})
    except (FileNotFoundError, OSError) as e:
        return jsonify({"error": f"Error processing query: {str(e)}"})


@app.route("/load_docs", methods=["POST"])
def load_docs():
    """Load documents into the RAG system."""
    embedding_model = request.form.get("embedding_model", "all-MiniLM-L6-v2")
    selected_folders_json = request.form.get("selected_folders", "[]")

    # Security: Validate embedding model
    if not validate_model_name(embedding_model, ALLOWED_EMBEDDING_MODELS):
        return jsonify({"error": "Invalid embedding model"})

    try:
        # Parse selected folders
        import json

        selected_folders = json.loads(selected_folders_json)

        if not selected_folders:
            return jsonify({"error": "No folders selected"})

        # For now, map folder names to our existing policies structure
        # In the future, this could accept full paths from the user's file system
        doc_dirs = []
        available_folders = ["dodd", "dodi", "dodm", "test"]

        for folder in selected_folders:
            if folder in available_folders:
                doc_dirs.append(f"policies/{folder}")
            else:
                # If folder doesn't match our structure, treat it as a relative path
                # This allows for future expansion to handle arbitrary paths
                doc_dirs.append(folder)

        # Build command with embedding model and selected directories
        cmd = [
            sys.executable,
            "-m",
            "src.cli",
            "--load-docs",
            "--doc-dirs",
        ]
        cmd.extend(doc_dirs)

        # Add embedding model if not default
        if embedding_model and embedding_model != "all-MiniLM-L6-v2":
            cmd.extend(["--embedding-model", embedding_model])

        # Start the task and return task ID for streaming
        import uuid

        task_id = str(uuid.uuid4())

        # Store task info globally (in production, use Redis or similar)
        global running_tasks
        if "running_tasks" not in globals():
            running_tasks = {}

        running_tasks[task_id] = {
            "cmd": cmd,
            "status": "starting",
            "output": [],
            "process": None,
        }

        return jsonify({"task_id": task_id, "message": "Document processing started"})

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid folder selection data"})
    except (ValueError, TypeError, OSError) as e:
        return jsonify({"error": f"Error starting document processing: {str(e)}"})


@app.route("/stream_task/<task_id>")
def stream_task(task_id):
    """Stream task output using Server-Sent Events."""
    from flask import Response

    def execute_task(task_id):
        """Execute the task in a separate thread."""
        global running_tasks
        if task_id not in running_tasks:
            return

        task = running_tasks[task_id]
        cmd = task["cmd"]

        try:
            task["status"] = "running"
            # Start the process
            process = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )
            task["process"] = process

            # Read output line by line
            for line in iter(process.stdout.readline, ""):
                if line:
                    task["output"].append(line.rstrip())

            # Wait for process to complete
            process.wait()

            if process.returncode == 0:
                task["status"] = "completed"
                task["output"].append("✅ Document processing completed successfully!")
            else:
                task["status"] = "failed"
                task["output"].append(
                    f"❌ Process failed with exit code {process.returncode}"
                )

        except (OSError, subprocess.SubprocessError) as e:
            task["status"] = "failed"
            task["output"].append(f"❌ Error: {str(e)}")

    def generate():
        import json

        global running_tasks
        if task_id not in running_tasks:
            yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
            return

        task = running_tasks[task_id]

        # Start the task if it hasn't been started
        if task["status"] == "starting":
            thread = Thread(target=execute_task, args=(task_id,))
            thread.daemon = True
            thread.start()

        # Stream output
        output_index = 0
        while True:
            # Send new output lines
            while output_index < len(task["output"]):
                line = task["output"][output_index]
                yield f"data: {json.dumps({'type': 'output', 'line': line})}\n\n"
                output_index += 1

            # Check if task is complete
            if task["status"] in ["completed", "failed"]:
                yield f"data: {json.dumps({'type': 'status', 'status': task['status']})}\n\n"
                break

            # Small delay to prevent busy loop
            time.sleep(0.1)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/hardware_info")
def get_hardware_info_endpoint():
    """Get current hardware information."""
    try:
        info = get_hardware_info()
        return jsonify(info)
    except (ImportError, OSError) as e:
        return jsonify({"error": f"Error getting hardware info: {str(e)}"})


@app.route("/available_models")
def available_models():
    """Get list of available models in Ollama."""
    try:
        import ollama

        models_response = ollama.list()
        # Type annotation to help pylint understand the structure
        model_list = [model.model for model in getattr(models_response, "models", [])]
        return jsonify({"models": model_list})
    except (ImportError, ConnectionError, AttributeError) as e:
        return jsonify(
            {"error": f"Error getting available models: {str(e)}", "models": []}
        )


@app.route("/load_model", methods=["POST"])
def load_model():
    """Load a model into Ollama for hot-loading."""
    llm_model = request.form.get("llm_model", "")

    if not llm_model:
        return jsonify({"error": "No model specified"})

    # Security: Validate model name
    if not validate_model_name(llm_model, ALLOWED_LLM_MODELS):
        return jsonify({"error": "Invalid model name"})

    # Check if ollama server is running first
    if not is_ollama_server_running():
        return jsonify(
            {"error": "Ollama server is not running. Please start the server first."}
        )

    # Check if model is available
    if not check_model_availability(llm_model):
        return jsonify(
            {
                "error": f"Model '{llm_model}' not found in Ollama. Please pull it first with: ollama pull {llm_model}"
            }
        )

    try:
        global HOT_LOADED_MODEL

        # Unload previous model if one is loaded
        if HOT_LOADED_MODEL and HOT_LOADED_MODEL != llm_model:
            safe_ollama_call(HOT_LOADED_MODEL, operation="unload")

        # Use safe ollama call to load new model
        success, error = safe_ollama_call(llm_model, operation="load")

        if success:
            HOT_LOADED_MODEL = llm_model
            return jsonify(
                {"success": True, "message": f"Model {llm_model} loaded successfully"}
            )

        return jsonify({"error": error})

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Error loading model: {str(e)}"})


@app.route("/unload_model", methods=["POST"])
def unload_model():
    """Unload model from Ollama using keep_alive=0."""
    global HOT_LOADED_MODEL

    if HOT_LOADED_MODEL:
        try:
            # Use safe ollama call instead of subprocess injection
            success, error = safe_ollama_call(HOT_LOADED_MODEL, operation="unload")

            if success:
                HOT_LOADED_MODEL = None
                return jsonify(
                    {"success": True, "message": "Model unloaded successfully"}
                )

            return jsonify({"error": f"Failed to unload model: {error}"})

        except (ImportError, ConnectionError) as e:
            return jsonify({"error": f"Error unloading model: {str(e)}"})

    HOT_LOADED_MODEL = None
    return jsonify({"success": True, "message": "No model was loaded"})


@app.route("/model_status")
def model_status():
    """Get current model loading status."""
    llm_model = request.args.get("model", "")

    if not llm_model:
        return jsonify({"loaded": False, "error": "No model specified"})

    # Check our internal hot-loaded state
    global HOT_LOADED_MODEL
    loaded = HOT_LOADED_MODEL == llm_model

    return jsonify({"loaded": loaded})


def check_docs_loaded(embedding_model="all-MiniLM-L6-v2"):
    """Check if documents are loaded for the specified embedding model."""
    try:
        database_dir = os.path.join(project_root, "database")
        if not os.path.exists(database_dir):
            return False

        # Check if ChromaDB collection exists for this embedding model
        chroma_db = os.path.join(database_dir, "chroma.sqlite3")
        if os.path.exists(chroma_db) and os.path.getsize(chroma_db) > 1024:
            try:
                import sqlite3

                conn = sqlite3.connect(chroma_db)
                cursor = conn.cursor()

                # Create collection name from embedding model
                db_model_name = embedding_model.replace("/", "_").replace("-", "_")
                collection_name = f"iris_{db_model_name}"

                # Check if collection exists for this embedding model
                cursor.execute(
                    "SELECT id FROM collections WHERE name = ?", (collection_name,)
                )
                collection_row = cursor.fetchone()

                if collection_row:
                    collection_id = collection_row[0]
                    # Check if collection has documents (use correct ChromaDB schema)
                    cursor.execute(
                        """
                        SELECT COUNT(*) FROM embeddings e 
                        JOIN segments s ON e.segment_id = s.id 
                        WHERE s.collection = ?
                    """,
                        (collection_id,),
                    )
                    doc_count = cursor.fetchone()[0]
                    conn.close()
                    return doc_count > 0

                conn.close()
            except (sqlite3.Error, OSError):
                # If we can't query ChromaDB, assume no docs loaded
                pass

        return False
    except (OSError, ImportError):
        return False


def get_available_embeddings():
    """Get list of embedding models that have databases."""
    try:
        database_dir = os.path.join(project_root, "database")
        if not os.path.exists(database_dir):
            return []

        available_embeddings = []

        # Check ChromaDB collections
        chroma_db = os.path.join(database_dir, "chroma.sqlite3")
        if os.path.exists(chroma_db) and os.path.getsize(chroma_db) > 1024:
            try:
                import sqlite3

                conn = sqlite3.connect(chroma_db)
                cursor = conn.cursor()

                # Get all collection names that start with "iris_"
                cursor.execute("SELECT name FROM collections WHERE name LIKE 'iris_%'")
                collections = cursor.fetchall()

                for (collection_name,) in collections:
                    # Check if collection has documents
                    cursor.execute(
                        """
                        SELECT COUNT(*) FROM embeddings e 
                        JOIN segments s ON e.segment_id = s.id 
                        JOIN collections c ON s.collection = c.id
                        WHERE c.name = ?
                    """,
                        (collection_name,),
                    )
                    doc_count = cursor.fetchone()[0]

                    if doc_count > 0:
                        # Extract model name from collection name
                        model_name = collection_name.replace("iris_", "")
                        # Convert back to original format
                        model_name = model_name.replace("_", "/").replace(
                            "BAAI/bge/base/en/v1.5", "BAAI/bge-base-en-v1.5"
                        )
                        model_name = model_name.replace(
                            "all/MiniLM/L6/v2", "all-MiniLM-L6-v2"
                        )
                        model_name = model_name.replace(
                            "all/mpnet/base/v2", "all-mpnet-base-v2"
                        )
                        available_embeddings.append(model_name)

                conn.close()
            except (sqlite3.Error, OSError):
                pass

        return sorted(list(set(available_embeddings)))
    except (OSError, ImportError):
        return []


@app.route("/status")
def status():
    """Get system status."""
    try:
        # Get embedding model from query parameter, default to all-MiniLM-L6-v2
        embedding_model = request.args.get("embedding_model", "all-MiniLM-L6-v2")

        # Check if documents are loaded for the specific embedding model
        docs_loaded = check_docs_loaded(embedding_model)

        # Check if Ollama is available
        try:
            import ollama

            ollama.list()  # Quick check if Ollama is responsive
            llm_available = True
        except (ImportError, ConnectionError, RuntimeError):
            llm_available = False

        # Get available embeddings
        available_embeddings = get_available_embeddings()

        return jsonify(
            {
                "docs_loaded": docs_loaded,
                "llm_available": llm_available,
                "system_ready": docs_loaded and llm_available,
                "embedding_model": embedding_model,
                "available_embeddings": available_embeddings,
            }
        )
    except (ValueError, TypeError) as e:
        return jsonify(
            {
                "docs_loaded": False,
                "llm_available": False,
                "system_ready": False,
                "error": str(e),
            }
        )


@app.route("/ollama_server_status")
def ollama_server_status():
    """Get ollama server status."""
    try:
        is_running = is_ollama_server_running()
        return jsonify({"running": is_running})
    except (OSError, RuntimeError) as e:
        return jsonify({"running": False, "error": str(e)})


@app.route("/start_ollama_server", methods=["POST"])
def start_ollama_server_endpoint():
    """Start ollama server."""
    try:
        success, message = start_ollama_server()
        if success:
            return jsonify({"success": True, "message": message})
        else:
            return jsonify({"error": message})
    except (OSError, RuntimeError) as e:
        return jsonify({"error": f"Error starting server: {str(e)}"})


@app.route("/stop_ollama_server", methods=["POST"])
def stop_ollama_server_endpoint():
    """Stop ollama server."""
    try:
        success, message = stop_ollama_server()
        if success:
            return jsonify({"success": True, "message": message})
        else:
            return jsonify({"error": message})
    except (OSError, RuntimeError) as e:
        return jsonify({"error": f"Error stopping server: {str(e)}"})


@app.route("/shutdown_everything", methods=["POST"])
def shutdown_everything():
    """Shutdown everything - unload models, stop ollama server, and terminate GUI."""
    try:
        global HOT_LOADED_MODEL, OLLAMA_SERVER_PROCESS

        # Unload any loaded model
        if HOT_LOADED_MODEL:
            safe_ollama_call(HOT_LOADED_MODEL, operation="unload")
            HOT_LOADED_MODEL = None

        # Stop ollama server
        stop_ollama_server()

        # Schedule Flask app shutdown after response is sent
        def shutdown_server():
            import signal

            # Give the response time to be sent
            time.sleep(3)
            # Terminate the process
            os.kill(os.getpid(), signal.SIGTERM)

        # Run shutdown in a separate thread
        shutdown_thread = Thread(target=shutdown_server)
        shutdown_thread.daemon = True
        shutdown_thread.start()

        return jsonify(
            {
                "success": True,
                "message": "All models unloaded, server stopped, and GUI shutting down...",
            }
        )
    except (OSError, RuntimeError) as e:
        return jsonify({"error": f"Error during shutdown: {str(e)}"})


def open_browser():
    """Open browser after a short delay."""
    time.sleep(1)
    webbrowser.open("http://localhost:8080")


if __name__ == "__main__":
    print("Starting IRIS Web Interface...")
    print("Opening browser at http://localhost:8080")

    # Open browser in a separate thread
    Thread(target=open_browser, daemon=True).start()

    # Run Flask app
    app.run(debug=False, port=DEFAULT_WEB_PORT, host=DEFAULT_WEB_HOST)
