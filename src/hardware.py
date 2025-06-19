"""Hardware detection utilities for adaptive model selection."""

from typing import Dict, Optional, Union
import psutil


def get_ram_gb() -> float:
    """Get total RAM in GB."""
    return psutil.virtual_memory().total / (1024**3)


def detect_gpu() -> bool:
    """Detect if GPU is available."""
    try:
        import GPUtil

        return len(GPUtil.getGPUs()) > 0
    except ImportError:
        return False


def recommend_model(ram_gb: Optional[float] = None) -> str:
    """Recommend Ollama model based on hardware capabilities."""
    if ram_gb is None:
        ram_gb = get_ram_gb()

    # Standard recommendations (based on quantized model requirements)
    if ram_gb >= 8:  # Systems with 8GB+ can handle Mistral 7B Q4
        return "mistral:7b-instruct-q4_K_M"
    if ram_gb >= 4:  # Systems with 4GB+ can handle Llama 3.2 3B Q4
        return "llama3.2:3b-instruct-q4_K_M"
    return "llama3.2:1b-instruct-q4_K_M"  # Minimum spec systems use Llama 3.2 1B Q4


def recommend_embedding_model(ram_gb: Optional[float] = None) -> str:
    """Recommend embedding model based on hardware capabilities and quality."""
    if ram_gb is None:
        ram_gb = get_ram_gb()

    # Choose best quality model that fits in available RAM
    # Simplified progression: fast -> policy-optimized -> high accuracy
    if ram_gb >= 16:  # High-end systems get high accuracy
        return "BAAI/bge-base-en-v1.5"  # 85.2 quality, 850MB RAM
    elif ram_gb >= 8:  # Standard systems get policy-optimized model
        return "all-mpnet-base-v2"  # 84.8 quality, 800MB RAM
    else:  # Lower-end systems get fast processing
        return "all-MiniLM-L6-v2"  # 81.3 quality, 200MB RAM


def get_hardware_info() -> Dict[str, Union[float, bool, int, str]]:
    """Get comprehensive hardware information."""
    ram_gb = get_ram_gb()
    has_gpu = detect_gpu()
    recommended_model = recommend_model(ram_gb)
    recommended_embedding = recommend_embedding_model(ram_gb)

    return {
        "ram_gb": round(ram_gb, 1),
        "has_gpu": has_gpu,
        "cpu_cores": psutil.cpu_count(),
        "recommended_model": recommended_model,
        "recommended_embedding": recommended_embedding,
    }
