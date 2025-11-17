"""
Dalla Data Processing

A comprehensive toolkit for processing Arabic text data with support for:
- Deduplication
- Stemming and morphological analysis
- Quality checking
- Readability scoring
"""

try:
    from dalla.core.dataset import DatasetManager

    _has_dataset = True
except ImportError:
    _has_dataset = False
    DatasetManager = None

try:
    from dalla.utils.tokenize import simple_word_tokenize

    _has_tokenize = True
except ImportError:
    _has_tokenize = False
    simple_word_tokenize = None

__all__ = ["DatasetManager", "simple_word_tokenize"]
