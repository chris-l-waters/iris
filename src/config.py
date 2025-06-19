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


# Create global config instance
config = Config()

# Backwards compatible constants (for existing code)
DEFAULT_CHUNK_SIZE = config.default_chunk_size
DEFAULT_CHUNK_OVERLAP = config.default_chunk_overlap
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
