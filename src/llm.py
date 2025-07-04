"""LLM integration for IRIS RAG system using Ollama."""

import logging
from typing import Optional, List

from .hardware import get_hardware_info
from .config import config

# Model configurations are now loaded from config.yml


class LLMProcessor:
    """Simple LLM processor using Ollama following YAGNI principles."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize LLM processor."""
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name or self._get_default_model_name()
        self.model_config = self._get_model_config()

    def _get_default_model_name(self) -> str:
        """Get default model name based on hardware."""
        hardware_info = get_hardware_info()
        return hardware_info["recommended_model"]

    def _get_model_config(self) -> dict:
        """Get configuration for the current model."""
        # Get model config from config.yml
        model_config = config.get_model_config(self.model_name)

        # Ensure n_ctx is set (use model-specific context_window or default)
        model_config["n_ctx"] = model_config.get(
            "context_window", config.llm_default_context_window
        )

        return model_config

    def _ensure_ollama(self) -> bool:
        """Ensure Ollama is available and model is pulled."""
        try:
            import ollama
        except ImportError:
            self.logger.error("Ollama not installed. Run: pip install ollama")
            return False

        try:
            # Try to connect to Ollama - this will auto-start if installed
            models_response = ollama.list()
            model_names = [model["model"] for model in models_response["models"]]

            # Check if our model is available (handle different naming formats)
            model_available = any(
                self.model_name in name or name.startswith(self.model_name)
                for name in model_names
            )

            if not model_available:
                self.logger.warning("Model %s not found in Ollama", self.model_name)
                self.logger.info("Available models: %s", model_names)
                self.logger.info("Pull the model with: ollama pull %s", self.model_name)
                return False

            self.logger.info("Using Ollama model: %s", self.model_name)
            return True

        except (ConnectionError, TimeoutError, AttributeError) as e:
            self.logger.error("Ollama error: %s", e)
            return False

    def is_available(self) -> bool:
        """Check if LLM is available and ready."""
        try:
            import importlib.util

            if importlib.util.find_spec("ollama") is None:
                return False
        except ImportError:
            return False
        return self._ensure_ollama()

    def generate_response(
        self, prompt: str, max_tokens: int = None, keep_alive=None
    ) -> str:
        """Generate response using Ollama."""
        if not self.is_available():
            return self._fallback_response()

        try:
            import ollama
        except ImportError:
            return self._fallback_response()

        try:
            # Use model-specific parameters
            max_tokens = max_tokens or self.model_config["max_tokens"]

            # Token counting for context window management
            prompt_tokens = self.count_tokens(prompt)
            self.logger.info(
                "Token usage - Prompt: %s chars (%s tokens), Response: %s tokens, Context: %s tokens",
                len(prompt),
                prompt_tokens,
                max_tokens,
                self.model_config["n_ctx"],
            )

            # Generate response using Ollama
            generate_params = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "num_ctx": self.model_config["n_ctx"],
                    "num_predict": max_tokens,
                    "temperature": self.model_config["temperature"],
                    "top_p": self.model_config["top_p"],
                    "stop": ["Human:", "Assistant:", "[/INST]", "###"],
                },
            }

            # Add keep_alive if specified
            if keep_alive is not None:
                generate_params["keep_alive"] = keep_alive

            # Final safety check before using ollama
            if ollama is None:
                self.logger.error("Ollama became unavailable during request")
                return self._fallback_response()

            response = ollama.generate(**generate_params)

            return response["response"].strip()

        except (ConnectionError, TimeoutError, KeyError, AttributeError) as e:
            self.logger.error("Error generating Ollama response: %s", e)
            return self._fallback_response()

    def _fallback_response(self) -> str:
        """Fallback response when LLM is not available."""
        return (
            "LLM not available. Please install Ollama and pull a model. "
            f"Run: ollama pull {self.model_name}"
        )

    def create_rag_prompt(self, question: str, context_chunks: List[str]) -> str:
        """Create a structured prompt for RAG responses."""
        # Context chunks are already filtered by fit_chunks_to_context()
        context = "\n\n".join(context_chunks)
        prompt_style = self.model_config["prompt_style"]

        # Get template from config
        template = config.get_prompt_template(prompt_style)
        if not template:
            # Fallback to default template if style not found
            self.logger.warning(
                "Prompt style '%s' not found in config, using fallback", prompt_style
            )
            template = config.get_prompt_template("fallback")

        if not template:
            # Ultimate fallback if config is missing
            self.logger.error(
                "No prompt templates found in config, using hardcoded fallback"
            )
            template = """Context from DOD policies:
{context}

Question: {question}

Instructions: Answer based ONLY on the context above. If the context doesn't contain enough information, say "The provided policies do not contain sufficient information to answer this question."

Answer:"""

        return template.format(context=context, question=question)

    def count_tokens(self, text: str) -> int:
        """Count tokens using estimation (Ollama doesn't expose tokenizer)."""
        # Use improved estimation based on text characteristics
        words = len(text.split())
        chars = len(text)

        # Better estimation: consider both word count and character count
        # Most models use ~4 chars per token on average
        word_estimate = int(words * 1.3)
        char_estimate = int(chars / 4)

        # Use the higher estimate to be safe
        return max(word_estimate, char_estimate)

    def fit_chunks_to_context(
        self, question: str, context_chunks: List[str], max_chunks: int = None
    ) -> tuple[List[str], int]:
        """Fit chunks by testing estimated prompt size."""
        context_window = self.model_config["n_ctx"]
        max_tokens = self.model_config["max_tokens"]

        # Reserve space for response generation + safety buffer
        available_for_prompt = context_window - max_tokens - 64

        # Start with all chunks (or max_chunks limit)
        chunks_to_test = context_chunks[:max_chunks] if max_chunks else context_chunks

        # Iteratively test prompt sizes
        selected_chunks = []
        final_prompt_tokens = 0

        for i in range(len(chunks_to_test)):
            test_chunks = chunks_to_test[: i + 1]

            # Create actual formatted prompt with current chunks
            test_prompt = self.create_rag_prompt(question, test_chunks)
            prompt_tokens = self.count_tokens(test_prompt)

            if prompt_tokens <= available_for_prompt:
                selected_chunks = test_chunks
                final_prompt_tokens = prompt_tokens
                self.logger.debug(
                    "Chunk %s: prompt now %s tokens", i + 1, prompt_tokens
                )
            else:
                self.logger.debug(
                    "Stopping at chunk %s: prompt would be %s tokens (exceeds %s)",
                    i + 1,
                    prompt_tokens,
                    available_for_prompt,
                )
                break

        # Log final results
        if max_chunks and len(selected_chunks) == max_chunks:
            self.logger.debug("Reached max_chunks limit: %s", max_chunks)
        elif len(selected_chunks) < len(chunks_to_test):
            self.logger.debug("Stopped due to token budget limit")

        utilization = (
            (final_prompt_tokens / available_for_prompt) * 100
            if available_for_prompt > 0
            else 0
        )
        self.logger.info(
            "Final prompt: %s tokens (%.1f%% utilization)",
            final_prompt_tokens,
            utilization,
        )

        return selected_chunks, final_prompt_tokens

    def extract_citations(
        self, fitted_chunks: List[str], context_results: List[dict]
    ) -> List[str]:
        """Extract unique document numbers from fitted chunks."""
        if not context_results:
            return []

        doc_numbers = set()

        for chunk_text in fitted_chunks:
            for result in context_results:
                if result["chunk_text"] == chunk_text:
                    doc_num = result.get("doc_number")
                    if doc_num and doc_num.strip():
                        doc_numbers.add(doc_num.strip())
                    break

        return sorted(list(doc_numbers))

    def format_response_with_citations(
        self, response: str, citations: List[str]
    ) -> str:
        """Append citations to response."""
        if not citations:
            return response

        citation_text = ", ".join(citations)
        return f"{response}\n\nSources: {citation_text}"

    def generate_rag_response(
        self,
        question: str,
        context_chunks: List[str],
        max_chunks: int = None,
        return_context: bool = False,
        keep_alive=None,
        context_results: List[dict] = None,
    ):
        """Generate a response using RAG context."""
        if not context_chunks:
            return "No relevant policy information found for: '" + question + "'"

        # Dynamically fit chunks to available context space
        fitted_chunks, prompt_tokens = self.fit_chunks_to_context(
            question, context_chunks, max_chunks
        )

        if not fitted_chunks:
            return "Context chunks too large for available token space"

        self.logger.info(
            "RAG: Using %s of %s chunks (%s tokens) - Model: %s, Context: %s",
            len(fitted_chunks),
            len(context_chunks),
            prompt_tokens,
            self.model_name,
            self.model_config["n_ctx"],
        )

        # Create final prompt (already tested to fit in context window)
        prompt = self.create_rag_prompt(question, fitted_chunks)
        response = self.generate_response(prompt, keep_alive=keep_alive)

        # Clean up the response based on prompt style
        response = response.strip()

        # Clean up common prefixes
        prefixes_to_remove = [
            "Based on the DOD policies above,",
            "Based on the context provided,",
            "According to the DOD policies,",
        ]

        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix) :].strip()
                break

        # Extract citations from fitted chunks
        citations = self.extract_citations(fitted_chunks, context_results or [])

        # Format response with citations
        base_response = "Based on DOD policies: " + response
        final_response = self.format_response_with_citations(base_response, citations)

        if return_context:
            return final_response, fitted_chunks
        return final_response
