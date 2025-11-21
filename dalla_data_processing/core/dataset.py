"""
Dataset I/O utilities for unified HuggingFace dataset handling.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from dalla_data_processing.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetManager:
    """Unified manager for HuggingFace dataset operations."""

    @staticmethod
    def load(
        path: str | Path,
        split: str | None = None,
        streaming: bool = False,
    ) -> Dataset | DatasetDict:
        """
        Load a HuggingFace dataset from disk.

        Args:
            path: Path to the dataset directory
            split: Optional split name to load (e.g., 'train', 'test')
            streaming: Whether to use streaming mode for large datasets

        Returns:
            Dataset or DatasetDict depending on the structure

        Example:
            >>> dm = DatasetManager()
            >>> dataset = dm.load("./data/my_dataset")
            >>> train_data = dm.load("./data/my_dataset", split="train")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        logger.info(f"Loading dataset from {path}")
        dataset = load_from_disk(str(path))

        if split is not None:
            if isinstance(dataset, DatasetDict):
                if split not in dataset:
                    raise ValueError(
                        f"Split '{split}' not found. Available splits: {list(dataset.keys())}"
                    )
                dataset = dataset[split]
            else:
                logger.warning(f"Split '{split}' specified but dataset has no splits")

        logger.info(f"Loaded dataset with {DatasetManager.get_size(dataset)} examples")
        return dataset

    @staticmethod
    def save(
        dataset: Dataset | DatasetDict,
        path: str | Path,
        overwrite: bool = False,
    ) -> None:
        """
        Save a HuggingFace dataset to disk.

        Args:
            dataset: Dataset or DatasetDict to save
            path: Path where the dataset will be saved
            overwrite: Whether to overwrite existing dataset

        Example:
            >>> dm = DatasetManager()
            >>> dm.save(processed_dataset, "./data/processed")
        """
        path = Path(path)

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Dataset path already exists: {path}. Use overwrite=True to replace."
            )

        logger.info(f"Saving dataset to {path}")
        dataset.save_to_disk(str(path))
        logger.info("Dataset saved successfully")

    @staticmethod
    def get_size(dataset: Dataset | DatasetDict) -> int:
        """
        Get the total number of examples in a dataset.

        Args:
            dataset: Dataset or DatasetDict

        Returns:
            Total number of examples
        """
        if isinstance(dataset, DatasetDict):
            return sum(len(ds) for ds in dataset.values())
        return len(dataset)

    @staticmethod
    def get_column_names(dataset: Dataset | DatasetDict) -> list[str]:
        """
        Get column names from a dataset.

        Args:
            dataset: Dataset or DatasetDict

        Returns:
            List of column names
        """
        if isinstance(dataset, DatasetDict):
            # Get columns from first split
            first_split = next(iter(dataset.values()))
            return first_split.column_names
        return dataset.column_names

    @staticmethod
    def add_column(
        dataset: Dataset,
        column_name: str,
        data: list[Any],
    ) -> Dataset:
        """
        Add a new column to a dataset.

        Args:
            dataset: Dataset to modify
            column_name: Name of the new column
            data: List of values for the new column

        Returns:
            Dataset with the new column added

        Example:
            >>> scores = [0.95, 0.87, 0.92, ...]
            >>> dataset = dm.add_column(dataset, "quality_score", scores)
        """
        if len(data) != len(dataset):
            raise ValueError(
                f"Data length ({len(data)}) must match dataset length ({len(dataset)})"
            )

        logger.info(f"Adding column '{column_name}' to dataset")
        return dataset.add_column(column_name, data)

    @staticmethod
    def map_column(
        dataset: Dataset,
        fn: Callable,
        input_column: str,
        output_column: str | None = None,
        batched: bool = False,
        batch_size: int = 1000,
        num_proc: int | None = None,
        desc: str | None = None,
    ) -> Dataset:
        """
        Apply a function to a column in the dataset.

        Args:
            dataset: Dataset to process
            fn: Function to apply to each example
            input_column: Name of the input column
            output_column: Name of the output column (if None, replaces input_column)
            batched: Whether to process in batches
            batch_size: Size of batches when batched=True
            num_proc: Number of processes for parallel processing
            desc: Description for progress bar

        Returns:
            Processed dataset

        Example:
            >>> def deduplicate_text(text):
            ...     return text.strip().lower()
            >>> dataset = dm.map_column(
            ...     dataset,
            ...     deduplicate_text,
            ...     "text",
            ...     "cleaned_text",
            ...     num_proc=4
            ... )
        """
        if input_column not in dataset.column_names:
            raise ValueError(f"Column '{input_column}' not found in dataset")

        output_col = output_column or input_column

        def process_fn(examples):
            if batched:
                results = [fn(item) for item in examples[input_column]]
            else:
                results = fn(examples[input_column])
            return {output_col: results}

        logger.info(f"Mapping function to column '{input_column}'")
        return dataset.map(
            process_fn,
            batched=batched,
            batch_size=batch_size,
            num_proc=num_proc,
            desc=desc or f"Processing {input_column}",
        )

    @staticmethod
    def filter_dataset(
        dataset: Dataset,
        fn: Callable,
        num_proc: int | None = None,
        desc: str | None = None,
    ) -> Dataset:
        """
        Filter dataset based on a condition.

        Args:
            dataset: Dataset to filter
            fn: Function that returns True for examples to keep
            num_proc: Number of processes for parallel processing
            desc: Description for progress bar

        Returns:
            Filtered dataset

        Example:
            >>> def is_high_quality(example):
            ...     return example['quality_score'] > 0.8
            >>> filtered = dm.filter_dataset(dataset, is_high_quality)
        """
        logger.info(f"Filtering dataset with {len(dataset)} examples")
        filtered = dataset.filter(fn, num_proc=num_proc, desc=desc or "Filtering dataset")
        logger.info(
            f"Filtered to {len(filtered)} examples ({len(filtered) / len(dataset) * 100:.1f}%)"
        )
        return filtered

    @staticmethod
    def select_columns(
        dataset: Dataset | DatasetDict,
        columns: list[str],
    ) -> Dataset | DatasetDict:
        """
        Select specific columns from a dataset.

        Args:
            dataset: Dataset or DatasetDict
            columns: List of column names to keep

        Returns:
            Dataset with only the specified columns
        """
        available_columns = DatasetManager.get_column_names(dataset)
        invalid_columns = set(columns) - set(available_columns)
        if invalid_columns:
            raise ValueError(f"Columns not found: {invalid_columns}")

        logger.info(f"Selecting columns: {columns}")
        if isinstance(dataset, DatasetDict):
            return DatasetDict({split: ds.select_columns(columns) for split, ds in dataset.items()})
        return dataset.select_columns(columns)

    @staticmethod
    def remove_columns(
        dataset: Dataset | DatasetDict,
        columns: list[str],
    ) -> Dataset | DatasetDict:
        """
        Remove specific columns from a dataset.

        Args:
            dataset: Dataset or DatasetDict
            columns: List of column names to remove

        Returns:
            Dataset without the specified columns
        """
        logger.info(f"Removing columns: {columns}")
        if isinstance(dataset, DatasetDict):
            return DatasetDict({split: ds.remove_columns(columns) for split, ds in dataset.items()})
        return dataset.remove_columns(columns)

    @staticmethod
    def concatenate(datasets: list[Dataset]) -> Dataset:
        """
        Concatenate multiple datasets.

        Args:
            datasets: List of datasets to concatenate

        Returns:
            Concatenated dataset
        """
        if not datasets:
            raise ValueError("Cannot concatenate empty list of datasets")

        logger.info(f"Concatenating {len(datasets)} datasets")
        return concatenate_datasets(datasets)

    @staticmethod
    def train_test_split(
        dataset: Dataset,
        test_size: float = 0.1,
        seed: int = 42,
    ) -> DatasetDict:
        """
        Split dataset into train and test sets.

        Args:
            dataset: Dataset to split
            test_size: Fraction of data to use for testing
            seed: Random seed for reproducibility

        Returns:
            DatasetDict with 'train' and 'test' splits
        """
        logger.info(f"Splitting dataset into train/test (test_size={test_size})")
        return dataset.train_test_split(test_size=test_size, seed=seed)

    @staticmethod
    def get_info(dataset: Dataset | DatasetDict) -> dict[str, Any]:
        """
        Get information about a dataset.

        Args:
            dataset: Dataset or DatasetDict

        Returns:
            Dictionary with dataset information
        """
        if isinstance(dataset, DatasetDict):
            return {
                "type": "DatasetDict",
                "splits": list(dataset.keys()),
                "total_examples": DatasetManager.get_size(dataset),
                "split_info": {
                    split: {
                        "num_examples": len(ds),
                        "columns": ds.column_names,
                        "features": str(ds.features),
                    }
                    for split, ds in dataset.items()
                },
            }
        else:
            return {
                "type": "Dataset",
                "num_examples": len(dataset),
                "columns": dataset.column_names,
                "features": str(dataset.features),
            }

    @staticmethod
    def print_info(dataset: Dataset | DatasetDict) -> None:
        """
        Print dataset information in a readable format.

        Args:
            dataset: Dataset or DatasetDict
        """
        info = DatasetManager.get_info(dataset)

        if info["type"] == "DatasetDict":
            print(f"\n{'=' * 60}")
            print("Dataset Dictionary")
            print(f"{'=' * 60}")
            print(f"Total examples: {info['total_examples']:,}")
            print(f"Splits: {', '.join(info['splits'])}")
            print()
            for split, split_info in info["split_info"].items():
                print(f"  {split}:")
                print(f"    Examples: {split_info['num_examples']:,}")
                print(f"    Columns: {', '.join(split_info['columns'])}")
            print(f"{'=' * 60}\n")
        else:
            print(f"\n{'=' * 60}")
            print("Dataset")
            print(f"{'=' * 60}")
            print(f"Examples: {info['num_examples']:,}")
            print(f"Columns: {', '.join(info['columns'])}")
            print(f"{'=' * 60}\n")
