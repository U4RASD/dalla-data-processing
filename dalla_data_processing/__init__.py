"""
Dalla Data Processing

A comprehensive toolkit for processing Arabic text data with support for:
- Deduplication
- Stemming and morphological analysis
- Quality checking
- Readability scoring
"""

__version__ = "0.0.1"

try:
    from dalla_data_processing.core.dataset import DatasetManager

    _has_dataset = True
except ImportError:
    _has_dataset = False
    DatasetManager = None

try:
    from dalla_data_processing.utils.tokenize import simple_word_tokenize

    _has_tokenize = True
except ImportError:
    _has_tokenize = False
    simple_word_tokenize = None

try:
    from dalla_data_processing.stemming import stem, stem_dataset

    _has_stemming = True
except ImportError:
    _has_stemming = False
    stem = None
    stem_dataset = None

__all__ = ["DatasetManager", "simple_word_tokenize", "stem", "stem_dataset", "__version__"]
