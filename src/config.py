"""Configuration loader for IRIS RAG system.

This module loads configuration from config.yml and provides backwards-compatible
constants for the rest of the application.
"""

import os
from typing import Dict, Any

import yaml

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yml")


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback to default values if config file doesn't exist
        return {
            "document_processing": {
                "default_chunk_size": 500,
                "default_chunk_overlap": 50,
                "enable_sectional_chunking": True,
                "max_section_chunk_size": 800,
                "section_split_threshold": 1200,
                "preserve_subsection_integrity": True,
                "sectional_chunking": {
                    "target_chunk_size": 700,
                    "min_chunk_size": 200,
                    "max_chunk_size": 1200,
                    "enable_hierarchical_grouping": True,
                },
                "header_footer_zone_ratio": 0.1,
                "line_tolerance_pixels": 5,
                "min_header_footer_chars": 50,
                "max_header_footer_words": 3,
                "min_line_length": 15,
                "first_page_sample_chars": 2000,
            },
            "web_interface": {"default_port": 8080, "default_host": "localhost"},
            "timeouts": {"default_query_timeout": 60, "default_load_docs_timeout": 300},
            "chromadb": {
                "default_persist_dir": "database",
                "default_collection_name": "iris_documents",
                "similarity_metric": "cosine",
            },
            "ranking": {
                "same_document_boost": 1.5,
                "same_doc_number_boost": 1.4,
                "same_specific_range_boost": 1.35,
                "same_subgroup_boost": 1.3,
                "same_major_group_boost": 1.2,
                "enable_document_aware_ranking": True,
                "apply_boosts_to_top_result": True,
            },
            "llm": {
                "default_context_window": 4096,
                "fallback_max_tokens": 256,
                "fallback_temperature": 0.7,
                "fallback_top_p": 0.95,
                "models": {
                    "tinyllama:1.1b-chat-v1.0-q4_k_m": {
                        "tier": 0,
                        "max_tokens": 256,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "prompt_style": "simple",
                        "context_window": 2048,
                    },
                    "llama3.2:1b-instruct-q4_K_M": {
                        "tier": 1,
                        "max_tokens": 400,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "prompt_style": "instruct",
                        "context_window": 4096,
                    },
                    "llama3.2:3b-instruct-q4_K_M": {
                        "tier": 2,
                        "max_tokens": 512,
                        "temperature": 0.5,
                        "top_p": 0.95,
                        "prompt_style": "instruct",
                        "context_window": 4096,
                    },
                    "gemma2:9b-instruct-q4_K_M": {
                        "tier": 3,
                        "max_tokens": 1024,
                        "temperature": 0.6,
                        "top_p": 0.9,
                        "prompt_style": "instruct",
                        "context_window": 8192,
                    },
                    "phi4-mini:latest": {
                        "tier": 4,
                        "max_tokens": 1024,
                        "temperature": 0.6,
                        "top_p": 0.9,
                        "prompt_style": "instruct",
                        "context_window": 8192,
                    },
                },
            },
            "prompts": {},
        }


# Load configuration and create Config instance
_config = load_config()


class Config:
    """Configuration class providing access to YAML config values."""

    def __init__(self):
        self._config = _config

    def get(self, key_path: str, default=None):
        """Get config value using dot notation (e.g., 'llm.default_context_window')."""
        keys = key_path.split(".")
        value = self._config
        for key in keys:
            value = value.get(key, {})
            if not isinstance(value, dict) and key != keys[-1]:
                return default
        return value if value != {} else default

    # Backwards compatible properties for existing code
    @property
    def default_chunk_size(self):
        """Get default chunk size for text processing."""
        return self._config["document_processing"]["default_chunk_size"]

    @property
    def default_chunk_overlap(self):
        """Get default chunk overlap for text processing."""
        return self._config["document_processing"]["default_chunk_overlap"]

    # LLM configuration properties
    @property
    def llm_default_context_window(self):
        """Get default LLM context window size."""
        return self.get("llm.default_context_window", 4096)

    @property
    def llm_fallback_max_tokens(self):
        """Get fallback max tokens for LLM responses."""
        return self.get("llm.fallback_max_tokens", 256)

    @property
    def llm_fallback_temperature(self):
        """Get fallback temperature for LLM responses."""
        return self.get("llm.fallback_temperature", 0.7)

    @property
    def llm_fallback_top_p(self):
        """Get fallback top_p for LLM responses."""
        return self.get("llm.fallback_top_p", 0.95)

    def get_prompt_template(self, style: str) -> str:
        """Get prompt template for given style."""
        return self.get(f"prompts.{style}.template", "")

    @property
    def available_prompt_styles(self):
        """Get list of available prompt styles."""
        return list(self.get("prompts", {}).keys())

    def get_model_config(self, model_name: str) -> dict:
        """Get configuration for a specific model."""
        # Direct access to models dict since dot notation fails with colons in model names
        models = self.get("llm.models", {})
        model_config = models.get(model_name, {}).copy()

        if not model_config:
            # Return fallback configuration
            return {
                "max_tokens": self.llm_fallback_max_tokens,
                "temperature": self.llm_fallback_temperature,
                "top_p": self.llm_fallback_top_p,
                "prompt_style": "simple",
                "context_window": self.llm_default_context_window,
            }

        # Set context_window from model-specific config or use default
        if "context_window" not in model_config:
            model_config["context_window"] = self.llm_default_context_window

        return model_config

    @property
    def available_models(self):
        """Get list of available model names."""
        return list(self.get("llm.models", {}).keys())

    def get_models_by_tier(self, tier: int = None):
        """Get models by tier level (0-4) or all models with tier info."""
        models = self.get("llm.models", {})
        if tier is None:
            # Return all models with their tier info
            return {name: config for name, config in models.items()}
        else:
            # Return models matching specific tier
            return {
                name: config
                for name, config in models.items()
                if config.get("tier") == tier
            }

    def get_model_tier(self, model_name: str) -> int:
        """Get tier level for a specific model."""
        models = self._config.get("llm", {}).get("models", {})
        model_config = models.get(model_name, {})
        return model_config.get("tier", -1)

    # Ranking configuration properties
    @property
    def ranking_enabled(self):
        """Check if document-aware ranking is enabled."""
        return self.get("ranking.enable_document_aware_ranking", True)

    @property
    def ranking_same_document_boost(self):
        """Get boost factor for same document chunks."""
        return self.get("ranking.same_document_boost", 1.5)

    @property
    def ranking_same_doc_number_boost(self):
        """Get boost factor for same document number chunks."""
        return self.get("ranking.same_doc_number_boost", 1.4)

    @property
    def ranking_same_specific_range_boost(self):
        """Get boost factor for same specific range chunks."""
        return self.get("ranking.same_specific_range_boost", 1.35)

    @property
    def ranking_same_subgroup_boost(self):
        """Get boost factor for same subgroup chunks."""
        return self.get("ranking.same_subgroup_boost", 1.3)

    @property
    def ranking_same_major_group_boost(self):
        """Get boost factor for same major group chunks."""
        return self.get("ranking.same_major_group_boost", 1.2)

    @property
    def ranking_apply_to_top_result(self):
        """Check if boosts should be applied to top result."""
        return self.get("ranking.apply_boosts_to_top_result", True)

    @property
    def ranking_adjacent_chunk_boost(self):
        """Get boost factor for adjacent chunks (±1 distance)."""
        return self.get("ranking.adjacent_chunk_boost", 1.3)

    @property
    def ranking_near_chunk_boost(self):
        """Get boost factor for near chunks (±2 distance)."""
        return self.get("ranking.near_chunk_boost", 1.15)

    @property
    def ranking_nearby_chunk_boost(self):
        """Get boost factor for nearby chunks (±3-5 distance)."""
        return self.get("ranking.nearby_chunk_boost", 1.05)


# Create global config instance
config = Config()

# Backwards compatible constants (for existing code)
DEFAULT_CHUNK_SIZE = config.default_chunk_size
DEFAULT_CHUNK_OVERLAP = config.default_chunk_overlap

# Sectional chunking constants
ENABLE_SECTIONAL_CHUNKING = _config["document_processing"]["enable_sectional_chunking"]
MAX_SECTION_CHUNK_SIZE = _config["document_processing"]["max_section_chunk_size"]
SECTION_SPLIT_THRESHOLD = _config["document_processing"]["section_split_threshold"]
PRESERVE_SUBSECTION_INTEGRITY = _config["document_processing"][
    "preserve_subsection_integrity"
]

# Hierarchical sectional chunking constants
SECTIONAL_CHUNKING_CONFIG = _config["document_processing"]["sectional_chunking"]
TARGET_CHUNK_SIZE = SECTIONAL_CHUNKING_CONFIG["target_chunk_size"]
MIN_CHUNK_SIZE = SECTIONAL_CHUNKING_CONFIG["min_chunk_size"]
MAX_CHUNK_SIZE = SECTIONAL_CHUNKING_CONFIG["max_chunk_size"]
ENABLE_HIERARCHICAL_GROUPING = SECTIONAL_CHUNKING_CONFIG["enable_hierarchical_grouping"]

# Performance optimization constants
ENABLE_SECTION_CACHING = _config["document_processing"]["enable_section_caching"]
MAX_CACHE_ENTRIES = _config["document_processing"]["max_cache_entries"]

HEADER_FOOTER_ZONE_RATIO = _config["document_processing"]["header_footer_zone_ratio"]
LINE_TOLERANCE_PIXELS = _config["document_processing"]["line_tolerance_pixels"]
MIN_HEADER_FOOTER_CHARS = _config["document_processing"]["min_header_footer_chars"]
MAX_HEADER_FOOTER_WORDS = _config["document_processing"]["max_header_footer_words"]
MIN_LINE_LENGTH = _config["document_processing"]["min_line_length"]
FIRST_PAGE_SAMPLE_CHARS = _config["document_processing"]["first_page_sample_chars"]
DEFAULT_WEB_PORT = _config["web_interface"]["default_port"]
DEFAULT_WEB_HOST = _config["web_interface"]["default_host"]
DEFAULT_QUERY_TIMEOUT = _config["timeouts"]["default_query_timeout"]
DEFAULT_LOAD_DOCS_TIMEOUT = _config["timeouts"]["default_load_docs_timeout"]
DEFAULT_CHROMA_PERSIST_DIR = _config["chromadb"]["default_persist_dir"]
DEFAULT_COLLECTION_NAME = _config["chromadb"]["default_collection_name"]
CHROMA_SIMILARITY_METRIC = _config["chromadb"]["similarity_metric"]


def get_config() -> Dict[str, Any]:
    """Get the full configuration dictionary."""
    return _config


def reload_config() -> None:
    """Reload configuration from file (useful for tests)."""
    global _config
    _config = load_config()

    # Update module-level constants
    globals().update(
        {
            "DEFAULT_CHUNK_SIZE": _config["document_processing"]["default_chunk_size"],
            "DEFAULT_CHUNK_OVERLAP": _config["document_processing"][
                "default_chunk_overlap"
            ],
            "HEADER_FOOTER_ZONE_RATIO": _config["document_processing"][
                "header_footer_zone_ratio"
            ],
            "LINE_TOLERANCE_PIXELS": _config["document_processing"][
                "line_tolerance_pixels"
            ],
            "MIN_HEADER_FOOTER_CHARS": _config["document_processing"][
                "min_header_footer_chars"
            ],
            "MAX_HEADER_FOOTER_WORDS": _config["document_processing"][
                "max_header_footer_words"
            ],
            "MIN_LINE_LENGTH": _config["document_processing"]["min_line_length"],
            "FIRST_PAGE_SAMPLE_CHARS": _config["document_processing"][
                "first_page_sample_chars"
            ],
            "DEFAULT_WEB_PORT": _config["web_interface"]["default_port"],
            "DEFAULT_WEB_HOST": _config["web_interface"]["default_host"],
            "DEFAULT_QUERY_TIMEOUT": _config["timeouts"]["default_query_timeout"],
            "DEFAULT_LOAD_DOCS_TIMEOUT": _config["timeouts"][
                "default_load_docs_timeout"
            ],
            "DEFAULT_CHROMA_PERSIST_DIR": _config["chromadb"]["default_persist_dir"],
            "DEFAULT_COLLECTION_NAME": _config["chromadb"]["default_collection_name"],
            "CHROMA_SIMILARITY_METRIC": _config["chromadb"]["similarity_metric"],
        }
    )
