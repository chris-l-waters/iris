"""Hardware detection utilities for adaptive model selection."""

from typing import Dict, Optional, Union
import psutil
from .config import config


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
    """Recommend Ollama model based on hardware capabilities and config tiers."""
    if ram_gb is None:
        ram_gb = get_ram_gb()

    # RAM-based tier selection (matching config.yml model requirements)
    if ram_gb >= 12:  # Tier 4: Premium spec (12GB+ RAM)
        tier = 4
    elif ram_gb >= 8:  # Tier 3: High spec (8-12GB RAM)
        tier = 3
    elif ram_gb >= 6:  # Tier 2: Standard spec (6-8GB RAM)
        tier = 2
    elif ram_gb >= 4:  # Tier 1: Low spec (4-6GB RAM)
        tier = 1
    else:  # Tier 0: Minimum spec (2-4GB RAM)
        tier = 0

    # Get models for the determined tier, fallback to lower tiers if needed
    for fallback_tier in range(tier, -1, -1):
        tier_models = config.get_models_by_tier(fallback_tier)
        if tier_models:
            # Return the first (and typically only) model for this tier
            return list(tier_models.keys())[0]

    # Final fallback if no models found (shouldn't happen with proper config)
    return "llama3.2:1b-instruct-q4_K_M"


def recommend_embedding_model(ram_gb: Optional[float] = None) -> str:
    """Recommend embedding model based on hardware capabilities and quality."""
    if ram_gb is None:
        ram_gb = get_ram_gb()

    # Choose best quality model that fits in available RAM
    # Simplified progression: fast -> policy-optimized -> best retrieval
    if ram_gb >= 16:  # High-end systems get best retrieval
        return "mixedbread-ai/mxbai-embed-large-v1"  # 87.2 quality, 1200MB RAM
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
