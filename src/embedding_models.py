"""Embedding model configurations and performance specifications."""

from typing import Dict, Any

# Embedding model specifications with performance characteristics
EMBEDDING_MODEL_CONFIGS = {
    "all-MiniLM-L6-v2": {
        "description": "Fast, lightweight model (default)",
        "ram_usage_mb": 200,
        "speed_sentences_per_sec": 1000,
        "quality_score": 81.3,
        "best_for": "Fast processing",
    },
    "all-mpnet-base-v2": {
        "description": "High quality semantic understanding",
        "ram_usage_mb": 800,
        "speed_sentences_per_sec": 200,
        "quality_score": 84.8,
        "best_for": "Policy documents",
    },
    "BAAI/bge-base-en-v1.5": {
        "description": "State-of-the-art retrieval model",
        "ram_usage_mb": 850,
        "speed_sentences_per_sec": 180,
        "quality_score": 85.2,
        "best_for": "High accuracy",
    },
}


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific embedding model."""
    return EMBEDDING_MODEL_CONFIGS.get(model_name, {})


def print_model_comparison():
    """Print a comparison table of available embedding models."""
    print("\nEmbedding Model Comparison:")
    print("=" * 80)
    print(f"{'Model':<25} {'Quality':<8} {'Speed':<12} {'RAM':<8} {'Best For':<20}")
    print("-" * 80)

    for model_name, config in EMBEDDING_MODEL_CONFIGS.items():
        short_name = (
            model_name.rsplit("/", maxsplit=1)[-1] if "/" in model_name else model_name
        )
        print(
            f"{short_name:<25} "
            f"{config['quality_score']:<8} "
            f"{config['speed_sentences_per_sec']:<12} "
            f"{config['ram_usage_mb']}MB{'':<4} "
            f"{config['best_for']:<20}"
        )

    print("\nRecommendations:")
    print("  • 4-8GB RAM:  all-MiniLM-L6-v2 (fast processing)")
    print("  • 8-16GB RAM: all-mpnet-base-v2 (policy documents)")
    print("  • 16GB+ RAM:  BAAI/bge-base-en-v1.5 (high accuracy)")
