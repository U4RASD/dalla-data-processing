"""
Dalla Data Processing

A comprehensive toolkit for processing Arabic text data with support for:
- Deduplication
- Stemming and morphological analysis
- Quality checking
- Readability scoring
"""

try:
    from dalla_data_processing._version import version as __version__
except ImportError:
    # Fallback for development without installation
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("dalla-data-processing")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"


# Lazy imports - only import when actually used, not at package load time
def __getattr__(name):
    """Lazy load heavy modules only when accessed."""
    if name == "DatasetManager":
        from dalla_data_processing.core.dataset import DatasetManager

        return DatasetManager
    elif name == "simple_word_tokenize":
        from dalla_data_processing.utils.tokenize import simple_word_tokenize

        return simple_word_tokenize
    elif name == "stem":
        from dalla_data_processing.stemming import stem

        return stem
    elif name == "stem_dataset":
        from dalla_data_processing.stemming import stem_dataset

        return stem_dataset
    elif name == "DatasetPacker":
        from dalla_data_processing.packing import DatasetPacker

        return DatasetPacker
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "DatasetManager",
    "simple_word_tokenize",
    "stem",
    "stem_dataset",
    "DatasetPacker",
    "__version__",
]
