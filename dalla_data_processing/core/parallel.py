"""
Parallel processing utilities for efficient dataset operations.

This module provides utilities for parallel processing of datasets,
including batch processing, multiprocessing, and progress tracking.
"""

import multiprocessing
from collections.abc import Callable
from typing import Any

from datasets import Dataset
from tqdm import tqdm

from dalla_data_processing.utils.logger import get_logger

logger = get_logger(__name__)


class ParallelProcessor:
    """Utility class for parallel dataset processing."""

    @staticmethod
    def get_optimal_num_workers(num_workers: int | None = None) -> int:
        """
        Get optimal number of workers for parallel processing.

        Args:
            num_workers: Requested number of workers (None for auto)

        Returns:
            Optimal number of workers
        """
        cpu_count = multiprocessing.cpu_count()
        if num_workers is None:
            return max(1, cpu_count - 1)
        return min(num_workers, cpu_count)

    @staticmethod
    def process_dataset_parallel(
        dataset: Dataset,
        process_fn: Callable,
        num_proc: int | None = None,
        batched: bool = False,
        batch_size: int = 1000,
        desc: str | None = None,
        remove_columns: list[str] | None = None,
        **map_kwargs,
    ) -> Dataset:
        """
        Process a dataset in parallel using the map function.

        Args:
            dataset: Dataset to process
            process_fn: Function to apply to each example/batch
            num_proc: Number of processes (None for auto)
            batched: Whether to process in batches
            batch_size: Batch size when batched=True
            desc: Description for progress bar
            remove_columns: Columns to remove after processing
            **map_kwargs: Additional arguments for dataset.map()

        Returns:
            Processed dataset

        Example:
            >>> def process_text(example):
            ...     example['processed'] = example['text'].lower()
            ...     return example
            >>> processed = ParallelProcessor.process_dataset_parallel(
            ...     dataset, process_text, num_proc=4
            ... )
        """
        num_workers = ParallelProcessor.get_optimal_num_workers(num_proc)

        logger.info(f"Processing dataset with {num_workers} workers")
        logger.info(f"Batched: {batched}, Batch size: {batch_size if batched else 'N/A'}")

        return dataset.map(
            process_fn,
            num_proc=num_workers,
            batched=batched,
            batch_size=batch_size,
            desc=desc or "Processing dataset",
            remove_columns=remove_columns,
            **map_kwargs,
        )

    @staticmethod
    def process_in_batches(
        dataset: Dataset,
        process_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        batch_size: int = 1000,
        desc: str | None = None,
    ) -> Dataset:
        """
        Process dataset in batches with custom function.

        Args:
            dataset: Dataset to process
            process_fn: Function that takes a list of examples and returns processed list
            batch_size: Size of batches
            desc: Description for progress bar

        Returns:
            Processed dataset

        Example:
            >>> def batch_process(batch):
            ...     # Process batch of examples
            ...     return [{'text': ex['text'].upper()} for ex in batch]
            >>> result = ParallelProcessor.process_in_batches(
            ...     dataset, batch_process, batch_size=100
            ... )
        """
        logger.info(f"Processing dataset in batches of {batch_size}")

        processed_examples = []
        total_batches = (len(dataset) + batch_size - 1) // batch_size

        with tqdm(total=total_batches, desc=desc or "Processing batches") as pbar:
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i : i + batch_size]

                batch_list = [
                    {key: batch[key][j] for key in batch}
                    for j in range(len(batch[next(iter(batch))]))
                ]

                processed_batch = process_fn(batch_list)
                processed_examples.extend(processed_batch)
                pbar.update(1)

        return Dataset.from_list(processed_examples)

    @staticmethod
    def create_shards(
        dataset: Dataset,
        num_shards: int,
    ) -> list[Dataset]:
        """
        Split dataset into shards for parallel processing.

        Args:
            dataset: Dataset to shard
            num_shards: Number of shards to create

        Returns:
            List of dataset shards

        Example:
            >>> shards = ParallelProcessor.create_shards(dataset, 4)
            >>> # Process each shard independently
        """
        if num_shards <= 0:
            raise ValueError("num_shards must be positive")

        total_size = len(dataset)
        shard_size = (total_size + num_shards - 1) // num_shards

        shards = []
        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = min(start_idx + shard_size, total_size)
            if start_idx < total_size:
                shard_indices = list(range(start_idx, end_idx))
                shards.append(dataset.select(shard_indices))

        logger.info(f"Created {len(shards)} shards from dataset of {total_size} examples")
        return shards

    @staticmethod
    def process_with_multiprocessing(
        items: list[Any],
        process_fn: Callable,
        num_workers: int | None = None,
        desc: str | None = None,
    ) -> list[Any]:
        """
        Process a list of items using multiprocessing.

        Args:
            items: List of items to process
            process_fn: Function to apply to each item
            num_workers: Number of worker processes
            desc: Description for progress bar

        Returns:
            List of processed items

        Example:
            >>> def process_item(x):
            ...     return x * 2
            >>> results = ParallelProcessor.process_with_multiprocessing(
            ...     [1, 2, 3, 4], process_item, num_workers=2
            ... )
        """
        num_workers = ParallelProcessor.get_optimal_num_workers(num_workers)

        logger.info(f"Processing {len(items)} items with {num_workers} workers")

        if num_workers == 1:
            return [process_fn(item) for item in tqdm(items, desc=desc or "Processing items")]

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_fn, items),
                    total=len(items),
                    desc=desc or "Processing items",
                )
            )

        return results


class ProgressTracker:
    """Utility for tracking progress across multiple operations."""

    def __init__(self, total: int, desc: str | None = None):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items to track
            desc: Description for progress bar
        """
        self.pbar = tqdm(total=total, desc=desc or "Processing")
        self.current = 0

    def update(self, n: int = 1):
        """Update progress by n items."""
        self.pbar.update(n)
        self.current += n

    def set_description(self, desc: str):
        """Update progress bar description."""
        self.pbar.set_description(desc)

    def close(self):
        """Close the progress bar."""
        self.pbar.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()


def batch_iterator(iterable, batch_size: int):
    """
    Yield batches from an iterable.

    Args:
        iterable: Any iterable
        batch_size: Size of each batch

    Yields:
        Batches of items

    Example:
        >>> for batch in batch_iterator(range(10), batch_size=3):
        ...     print(batch)
        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]
        [9]
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
