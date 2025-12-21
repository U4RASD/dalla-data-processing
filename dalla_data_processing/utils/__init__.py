"""Utility functions for text processing."""

from dalla_data_processing.utils.logger import get_logger, logger, setup_logging

__all__ = ["simple_word_tokenize", "logger", "get_logger", "setup_logging"]


def __getattr__(name):
    """Lazy load modules with optional dependencies."""
    if name == "simple_word_tokenize":
        from dalla_data_processing.utils.tokenize import simple_word_tokenize

        return simple_word_tokenize
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
