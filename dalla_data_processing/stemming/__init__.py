"""Stemming and morphological analysis module.

This module provides Arabic stemming and morphological tokenization
functionality using CAMeL Tools disambiguators.
"""

from dalla_data_processing.stemming.stemmer import stem, stem_dataset

__all__ = ["stem_dataset", "stem"]
