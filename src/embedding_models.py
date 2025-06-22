"""Embedding model configurations and performance specifications."""

from typing import Dict, Any
from .config import config


# Embedding model specifications loaded from config.yml
def get_embedding_model_configs() -> Dict[str, Any]:
    """Get embedding model configurations from config.yml."""
    return config.embedding_models


# For backwards compatibility
EMBEDDING_MODEL_CONFIGS = get_embedding_model_configs()


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific embedding model."""
    return config.embedding_models.get(model_name, {})


def print_model_comparison():
    """Print a comparison table of available embedding models."""
    print("\nEmbedding Model Comparison:")
    print("=" * 80)
    print(f"{'Model':<25} {'Quality':<8} {'Speed':<12} {'RAM':<8} {'Best For':<20}")
    print("-" * 80)

    for model_name, model_config in config.embedding_models.items():
        short_name = (
            model_name.rsplit("/", maxsplit=1)[-1] if "/" in model_name else model_name
        )
        print(
            f"{short_name:<25} "
            f"{model_config['quality_score']:<8} "
            f"{model_config['speed_sentences_per_sec']:<12} "
            f"{model_config['ram_usage_mb']}MB{'':<4} "
            f"{model_config['best_for']:<20}"
        )

    print("\nRecommendations:")
    print("  • 4-8GB RAM:   all-MiniLM-L6-v2 (fast processing)")
    print("  • 8-16GB RAM:  all-mpnet-base-v2 (policy documents)")
    print("  • 16GB+ RAM:   mixedbread-ai/mxbai-embed-large-v1 (best retrieval)")
