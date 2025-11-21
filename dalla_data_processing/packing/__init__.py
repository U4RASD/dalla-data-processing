"""
Dataset packing module for efficient LLM training.

This module provides functionality to pack datasets by combining multiple
examples into fixed-length sequences, optimizing for efficient training.
"""

from dalla_data_processing.packing.dataset_packer import DatasetPacker

__all__ = ["DatasetPacker"]
