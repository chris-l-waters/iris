"""Cross-encoder reranking module for IRIS RAG system.

This module provides cross-encoder functionality to rerank initial vector search
results for improved relevance scoring in the retrieval-augmented generation pipeline.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

from .config import config

logger = logging.getLogger(__name__)


class CrossEncoderManager:
    """Manages cross-encoder model for passage reranking.

    Uses sentence-transformers CrossEncoder to rerank query-passage pairs
    for more accurate relevance scoring than vector similarity alone.
    """

    def __init__(self):
        """Initialize CrossEncoderManager with lazy loading."""
        self._model = None
        self._model_loaded = False
        self._load_error = None

    def _load_model(self) -> bool:
        """Load cross-encoder model on first use.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        if self._model_loaded:
            return self._model is not None

        try:
            logger.info(
                "Loading cross-encoder model: %s", config.cross_encoder_model_name
            )
            start_time = time.time()

            # Import here to avoid dependency issues if sentence-transformers not available
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(config.cross_encoder_model_name)
            load_time = time.time() - start_time

            logger.info("Cross-encoder model loaded successfully in %.2fs", load_time)
            self._model_loaded = True
            return True

        except Exception as e:
            self._load_error = str(e)
            logger.error("Failed to load cross-encoder model: %s", e)
            self._model_loaded = True  # Mark as attempted
            return False

    def _truncate_text_pair(self, query: str, passage: str) -> Tuple[str, str]:
        """Truncate query-passage pair to fit within token limits.

        Strategy: Keep full query, truncate passage from end if needed.

        Args:
            query: The search query text
            passage: The passage text to potentially truncate

        Returns:
            Tuple of (query, truncated_passage)
        """
        # Simple heuristic: 512 tokens ≈ 400 words ≈ 2000 characters
        max_chars = config.cross_encoder_max_length * 4
        query_chars = len(query)

        if query_chars + len(passage) <= max_chars:
            return query, passage

        # Reserve space for query and separator, truncate passage
        available_for_passage = max_chars - query_chars - 20  # Buffer for separators

        if available_for_passage < 100:  # Minimum passage length
            logger.warning(
                "Query too long (%d chars), may affect cross-encoder performance",
                query_chars,
            )
            return query, passage[:100]

        truncated_passage = passage[:available_for_passage]

        # Try to break at sentence end for better coherence
        last_period = truncated_passage.rfind(".")
        last_space = truncated_passage.rfind(" ")

        if last_period > available_for_passage * 0.8:  # If period is in last 20%
            truncated_passage = truncated_passage[: last_period + 1]
        elif last_space > available_for_passage * 0.9:  # If space is in last 10%
            truncated_passage = truncated_passage[:last_space]

        if len(truncated_passage) < len(passage):
            logger.debug(
                "Truncated passage from %d to %d characters",
                len(passage),
                len(truncated_passage),
            )

        return query, truncated_passage

    def rerank_passages(self, query: str, passages: List[Dict]) -> List[Dict]:
        """Rerank passages using cross-encoder model.

        Args:
            query: The search query
            passages: List of passage dictionaries with 'text' field and metadata

        Returns:
            List of reranked passage dictionaries with cross_encoder_score added
        """
        if not passages:
            logger.debug("No passages to rerank")
            return []

        # Load model if not already loaded
        if not self._load_model():
            logger.warning(
                "Cross-encoder model not available, returning original ranking"
            )
            return passages

        try:
            start_time = time.time()

            # Determine how many passages to rerank
            rerank_count = min(len(passages), config.cross_encoder_rerank_top_k)
            passages_to_rerank = passages[:rerank_count]

            logger.debug("Reranking %d passages with cross-encoder", rerank_count)

            # Prepare query-passage pairs with truncation
            query_passage_pairs = []
            valid_indices = []

            for i, passage in enumerate(passages_to_rerank):
                passage_text = passage.get("text", "")
                if not passage_text.strip():
                    logger.warning("Empty passage text at index %d, skipping", i)
                    continue

                truncated_query, truncated_passage = self._truncate_text_pair(
                    query, passage_text
                )
                query_passage_pairs.append((truncated_query, truncated_passage))
                valid_indices.append(i)

            if not query_passage_pairs:
                logger.warning("No valid query-passage pairs for reranking")
                return passages

            # Perform batch prediction
            batch_size = config.cross_encoder_batch_size
            all_scores = []

            for i in range(0, len(query_passage_pairs), batch_size):
                batch = query_passage_pairs[i : i + batch_size]
                batch_scores = self._model.predict(batch)
                all_scores.extend(batch_scores)

            # Add cross-encoder scores to passages
            reranked_passages = []
            score_index = 0

            for i, passage in enumerate(passages_to_rerank):
                passage_copy = passage.copy()

                if i in valid_indices:
                    cross_encoder_score = float(all_scores[score_index])
                    passage_copy["cross_encoder_score"] = cross_encoder_score
                    score_index += 1
                else:
                    # Assign low score to invalid passages
                    passage_copy["cross_encoder_score"] = -10.0

                reranked_passages.append(passage_copy)

            # Sort by cross-encoder scores (descending)
            reranked_passages.sort(
                key=lambda x: x.get("cross_encoder_score", -10.0), reverse=True
            )

            # Return final top-k after reranking
            final_count = min(len(reranked_passages), config.cross_encoder_final_top_k)
            final_passages = reranked_passages[:final_count]

            # Add any remaining original passages that weren't reranked
            if len(passages) > rerank_count:
                remaining_passages = passages[rerank_count:]
                final_passages.extend(remaining_passages)

            rerank_time = time.time() - start_time
            logger.info("Cross-encoder reranking completed in %.3fs", rerank_time)

            # Log score distribution for debugging
            scores = [
                p.get("cross_encoder_score")
                for p in final_passages
                if "cross_encoder_score" in p
            ]
            if scores:
                logger.debug(
                    "Cross-encoder scores: min=%.3f, max=%.3f, mean=%.3f",
                    min(scores),
                    max(scores),
                    sum(scores) / len(scores),
                )

            return final_passages

        except Exception as e:
            logger.error("Error during cross-encoder reranking: %s", e)
            logger.warning("Falling back to original vector search ranking")
            return passages

    def is_available(self) -> bool:
        """Check if cross-encoder model is available.

        Returns:
            bool: True if model can be loaded/is loaded, False otherwise.
        """
        return self._load_model()

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the cross-encoder model.

        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": config.cross_encoder_model_name,
            "rerank_top_k": str(config.cross_encoder_rerank_top_k),
            "final_top_k": str(config.cross_encoder_final_top_k),
            "batch_size": str(config.cross_encoder_batch_size),
            "max_length": str(config.cross_encoder_max_length),
            "loaded": str(self._model_loaded),
            "available": str(self.is_available()),
            "load_error": self._load_error or "None",
        }


# Global instance for lazy loading
_cross_encoder_manager: Optional[CrossEncoderManager] = None


def get_cross_encoder_manager() -> CrossEncoderManager:
    """Get global CrossEncoderManager instance.

    Returns:
        CrossEncoderManager: Global instance with lazy initialization.
    """
    global _cross_encoder_manager
    if _cross_encoder_manager is None:
        _cross_encoder_manager = CrossEncoderManager()
    return _cross_encoder_manager
