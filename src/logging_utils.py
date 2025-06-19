"""Logging utilities for IRIS RAG system."""

import logging
from typing import Optional


class QuietLogger:
    """A logger wrapper that respects quiet mode settings."""

    def __init__(
        self, logger: Optional[logging.Logger] = None, quiet_mode: bool = False
    ):
        """Initialize with optional logger and quiet mode setting."""
        self.logger = logger or logging.getLogger(__name__)
        self.quiet_mode = quiet_mode

    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message if not in quiet mode."""
        if not self.quiet_mode:
            self.logger.info(message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message if not in quiet mode."""
        if not self.quiet_mode:
            self.logger.debug(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Always log warning messages regardless of quiet mode."""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """Always log error messages regardless of quiet mode."""
        self.logger.error(message, *args, **kwargs)

    def set_quiet_mode(self, quiet_mode: bool) -> None:
        """Update quiet mode setting."""
        self.quiet_mode = quiet_mode


def create_quiet_logger(name: str, quiet_mode: bool = False) -> QuietLogger:
    """Create a QuietLogger instance for the given module name."""
    logger = logging.getLogger(name)
    return QuietLogger(logger, quiet_mode)
