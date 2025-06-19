"""Error handling utilities for IRIS RAG system."""

import logging
import functools
from typing import Callable, Any, Optional, Type


class IrisError(Exception):
    """Base exception class for IRIS-specific errors."""


class DocumentProcessingError(IrisError):
    """Raised when document processing fails."""


class EmbeddingError(IrisError):
    """Raised when embedding generation fails."""


class VectorStoreError(IrisError):
    """Raised when vector store operations fail."""


class LLMError(IrisError):
    """Raised when LLM operations fail."""


def handle_errors(
    logger: Optional[logging.Logger] = None,
    reraise: bool = True,
    default_return: Any = None,
    error_types: tuple = (Exception,),
) -> Callable:
    """
    Decorator for standardized error handling.

    Args:
        logger: Logger instance to use for error logging
        reraise: Whether to reraise the exception after logging
        default_return: Value to return if exception is caught and not reraised
        error_types: Tuple of exception types to catch

    Returns:
        Decorated function with error handling
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if logger is None:
                log = logging.getLogger(func.__module__)
            else:
                log = logger

            try:
                return func(*args, **kwargs)
            except error_types as e:
                log.error("Error in %s: %s", func.__name__, str(e))
                if reraise:
                    raise
                return default_return

        return wrapper

    return decorator


def log_and_return_error(
    error_msg: str,
    logger: Optional[logging.Logger] = None,
    exception_type: Type[Exception] = IrisError,
) -> None:
    """
    Log an error message and raise an exception.

    Args:
        error_msg: Error message to log and include in exception
        logger: Logger instance to use
        exception_type: Type of exception to raise
    """
    if logger:
        logger.error(error_msg)
    else:
        logging.error(error_msg)
    raise exception_type(error_msg)


def safe_execute(
    func: Callable,
    *args,
    logger: Optional[logging.Logger] = None,
    error_msg: str = "Operation failed",
    default_return: Any = None,
    **kwargs,
) -> Any:
    """
    Safely execute a function with standardized error handling.

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        logger: Logger instance to use
        error_msg: Error message prefix
        default_return: Value to return on error
        **kwargs: Keyword arguments for the function

    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        full_msg = f"{error_msg}: {str(e)}"
        if logger:
            logger.error(full_msg)
        else:
            logging.error(full_msg)
        return default_return


class ErrorContext:
    """Context manager for standardized error handling."""

    def __init__(
        self,
        operation_name: str,
        logger: Optional[logging.Logger] = None,
        reraise: bool = True,
    ):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.reraise = reraise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, _exc_tb):
        if exc_type is not None:
            self.logger.error("Error in %s: %s", self.operation_name, str(exc_val))
            if not self.reraise:
                return True  # Suppress the exception
        return False  # Let the exception propagate
