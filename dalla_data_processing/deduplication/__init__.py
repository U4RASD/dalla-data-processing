"""
Deduplication module using onion algorithm.

This module provides a streamlined interface to the onion deduplication tool,
wrapping the complex multi-step process into a single API call.
"""

import gc
import shutil
import tempfile
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm

from dalla_data_processing.deduplication.onion_wrapper import find_onion_binary, run_onion
from dalla_data_processing.deduplication.postprocessing import extract_duplicates_from_csvs
from dalla_data_processing.deduplication.preprocessing import create_file_list, create_vert_files
from dalla_data_processing.utils.logger import get_logger

logger = get_logger(__name__)


def deduplicate_dataset(
    dataset: Dataset,
    column: str = "text",
    threshold: float = 0.8,
    return_pairs: bool = True,
    keep_vert_files: bool = False,
    vert_dir: Path | None = None,
    onion_binary: Path | None = None,
    work_dir: Path | None = None,
    calculate_scores: bool = False,
) -> Dataset:
    """
    Remove duplicate entries from dataset using onion algorithm.

    This function:
    1. Converts texts to vertical format (one word per line)
    2. Runs onion phase 1 (find duplicate groups)
    3. Optionally runs onion phase 2 (calculate similarity scores)
    4. Adds duplicate information to dataset

    Args:
        dataset: HuggingFace dataset
        column: Column to check for duplicates (default: "text")
        threshold: Similarity threshold 0.0-1.0 (default: 0.8)
                  Note: Onion uses this internally, output includes all pairs
        return_pairs: If True, return dataset with duplicate pairs info
                     If False, return deduplicated dataset (keeps first occurrence)
        keep_vert_files: Keep vertical format files for inspection (default: False)
        vert_dir: Directory to store vertical files (default: work_dir/vert_files)
                 Useful if you want to store vert files on a different disk
        onion_binary: Path to onion binary (auto-detected if None)
        work_dir: Working directory for temp files (auto-created if None)
        calculate_scores: Run phase 2 to calculate similarity scores (default: False)
                         Set to True if you need precise similarity measurements
                         Phase 1 is usually sufficient for deduplication

    Returns:
        Dataset with added columns:
        - If return_pairs=True:
          - duplicate_cluster: Cluster ID for duplicate groups
          - is_duplicate: Boolean indicating if doc has duplicates
          - duplicate_count: Number of duplicates found
        - If return_pairs=False:
          - Returns filtered dataset with duplicates removed

    Example:
        >>> # Get duplicate information
        >>> result = deduplicate_dataset(dataset, return_pairs=True)
        >>> duplicates = result.filter(lambda x: x['is_duplicate'])
        >>>
        >>> # Get deduplicated dataset
        >>> deduped = deduplicate_dataset(dataset, return_pairs=False)
        >>>
        >>> # Store vert files on different disk
        >>> deduped = deduplicate_dataset(
        ...     dataset,
        ...     vert_dir=Path("/mnt/large_disk/vert_files"),
        ...     keep_vert_files=True
        ... )

    Raises:
        FileNotFoundError: If onion binary not found
        RuntimeError: If onion execution fails
    """
    logger.info(f"Starting deduplication of {len(dataset)} examples")
    logger.info(f"Column: {column}, Threshold: {threshold}, Return pairs: {return_pairs}")

    # Check for onion binary
    if onion_binary is None:
        onion_binary = find_onion_binary()
    if onion_binary is None:
        raise FileNotFoundError(
            "Onion binary not found. Please install onion or set ONION_BINARY environment variable.\n"
            "Installation: https://corpus.tools/wiki/Onion"
        )

    logger.info(f"Using onion binary: {onion_binary}")

    # Create working directory
    if work_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="dalla_dedup_")
        work_dir = Path(temp_dir)
        cleanup_work_dir = True
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup_work_dir = False

    logger.info(f"Working directory: {work_dir}")

    try:
        # Step 1: Extract texts and IDs
        logger.info("Step 1: Extracting texts from dataset...")
        texts = []
        ids = []

        for idx, example in enumerate(tqdm(dataset, desc="Extracting texts")):
            text = example.get(column, "")
            texts.append(text)
            ids.append(f"doc_{idx}")

        logger.info(f"Extracted {len(texts)} texts")

        # Check text lengths (onion needs sufficient content)
        avg_tokens = sum(len(t.split()) for t in texts if t) / len(texts) if texts else 0
        min_tokens = min((len(t.split()) for t in texts if t), default=0)
        max_tokens = max((len(t.split()) for t in texts if t), default=0)

        logger.info(
            f"Text statistics: min={min_tokens}, avg={avg_tokens:.1f}, max={max_tokens} tokens"
        )

        if avg_tokens < 10:
            logger.warning(
                f"Texts are very short (avg {avg_tokens:.1f} tokens). "
                "Onion requires at least ~10-15 tokens per text for reliable duplicate detection. "
                "Short texts may be marked as 'bad' and not processed."
            )

        # Step 2: Create vertical format files
        logger.info("Step 2: Creating vertical format files...")
        # Use custom vert_dir if provided, otherwise use work_dir/vert_files
        vert_dir = work_dir / "vert_files" if vert_dir is None else Path(vert_dir)

        logger.info(f"Vertical files directory: {vert_dir}")
        file_paths, id_mapping = create_vert_files(texts, ids, vert_dir)

        # Free memory: texts list is no longer needed after creating vert files
        del texts
        gc.collect()

        # Step 3: Create file list for onion
        logger.info("Step 3: Creating file list...")
        file_list_path = work_dir / "file_list.txt"
        create_file_list(file_paths, file_list_path)

        # Step 4: Run onion phase 1 (find duplicates)
        logger.info("Step 4: Running onion phase 1 (finding duplicates)...")
        phase1_output = work_dir / "phase1_output"
        success, csv_dir = run_onion(
            file_list_path,
            phase1_output,
            dataset_name="phase1",
            onion_binary=onion_binary,
        )

        if not success or csv_dir is None:
            raise RuntimeError("Onion phase 1 failed")

        # Step 5: Extract duplicate file paths
        logger.info("Step 5: Extracting duplicate paths...")
        file_to_duplicates = extract_duplicates_from_csvs(csv_dir, file_list_path)

        # Free memory: id_mapping no longer needed
        del id_mapping
        gc.collect()

        if not file_to_duplicates:
            logger.info("No duplicates found!")
            if avg_tokens < 10:
                logger.warning(
                    "No duplicates detected. This might be because texts are too short for onion. "
                    "Onion requires texts with at least 10-15 tokens for reliable duplicate detection. "
                    "Consider using a different deduplication method for very short texts."
                )
            # Add empty duplicate columns
            return _add_empty_duplicate_columns(dataset)

        # Step 6: Create file list of only duplicates for phase 2
        logger.info(f"Step 6: Found {len(file_to_duplicates)} files with duplicates")
        duplicate_files = list(
            set(
                list(file_to_duplicates) + [d for dups in file_to_duplicates.values() for d in dups]
            )
        )

        duplicate_file_list = work_dir / "duplicate_files.txt"
        create_file_list([Path(f) for f in duplicate_files], duplicate_file_list)

        # Step 7: Optionally run onion phase 2 (calculate scores)
        csv_dir2 = None
        if calculate_scores:
            logger.info("Step 7: Running onion phase 2 (calculating similarity scores)...")
            phase2_output = work_dir / "phase2_output"
            success, csv_dir2 = run_onion(
                duplicate_file_list,
                phase2_output,
                dataset_name="phase2",
                onion_binary=onion_binary,
            )

            if not success:
                logger.warning("Onion phase 2 failed, using phase 1 results only")
                csv_dir2 = None
        else:
            logger.info("Step 7: Skipping phase 2 (calculate_scores=False)")

        # Free memory: duplicate_files list no longer needed
        del duplicate_files
        gc.collect()

        # Step 8: Parse results and add to dataset
        logger.info("Step 8: Processing results...")

        if return_pairs:
            # Add duplicate information columns
            result = _add_duplicate_columns(
                dataset,
                file_paths,
                ids,
                file_to_duplicates,
                csv_dir2,
                duplicate_file_list,
            )
        else:
            # Filter out duplicates (keep first occurrence)
            result = _filter_duplicates(
                dataset,
                file_paths,
                ids,
                file_to_duplicates,
            )

        logger.info("Deduplication complete!")

        # Free memory: cleanup large intermediate objects
        del file_paths, ids, file_to_duplicates
        gc.collect()

        return result

    finally:
        # Cleanup
        if cleanup_work_dir and not keep_vert_files:
            logger.info(f"Cleaning up working directory: {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)
        elif keep_vert_files:
            logger.info(f"Vertical files kept in: {work_dir}")


def _add_empty_duplicate_columns(dataset: Dataset) -> Dataset:
    """Add empty duplicate columns when no duplicates found."""

    def add_columns(example):
        example["duplicate_cluster"] = -1
        example["is_duplicate"] = False
        example["duplicate_count"] = 0
        return example

    return dataset.map(add_columns, desc="Adding duplicate columns")


def _add_duplicate_columns(
    dataset: Dataset,
    file_paths: list[Path],
    ids: list[str],
    file_to_duplicates: dict[str, set],
    csv_dir: Path | None,
    duplicate_file_list: Path,
) -> Dataset:
    """Add duplicate information columns to dataset."""

    # Build mapping from doc ID to cluster ID
    clusters = {}
    cluster_id = 0

    # Group duplicates into clusters
    processed = set()
    for file_path, duplicates in file_to_duplicates.items():
        if file_path in processed:
            continue

        # Find all connected duplicates (transitive closure)
        cluster = {file_path}
        cluster.update(duplicates)

        # Mark all as same cluster
        for f in cluster:
            # Get doc ID from file path
            for idx, fp in enumerate(file_paths):
                if str(fp) == str(f):
                    clusters[ids[idx]] = cluster_id
                    processed.add(f)

        cluster_id += 1

    logger.info(f"Created {cluster_id} duplicate clusters")

    # Add columns to dataset
    def add_duplicate_info(example, idx):
        doc_id = ids[idx]

        if doc_id in clusters:  # noqa: F821
            example["duplicate_cluster"] = clusters[doc_id]  # noqa: F821
            example["is_duplicate"] = True
            # Count how many others in same cluster
            cluster_size = sum(1 for cid in clusters.values() if cid == clusters[doc_id])  # noqa: F821
            example["duplicate_count"] = cluster_size - 1  # Exclude self
        else:
            example["duplicate_cluster"] = -1
            example["is_duplicate"] = False
            example["duplicate_count"] = 0

        return example

    result = dataset.map(add_duplicate_info, with_indices=True, desc="Adding duplicate info")

    # Log statistics
    num_duplicates = sum(1 for cid in clusters.values())
    logger.info(
        f"Found {num_duplicates} documents with duplicates ({num_duplicates / len(dataset) * 100:.1f}%)"
    )

    # Free memory
    del clusters, processed
    gc.collect()

    return result


def _filter_duplicates(
    dataset: Dataset,
    file_paths: list[Path],
    ids: list[str],
    file_to_duplicates: dict[str, set],
) -> Dataset:
    """Filter out duplicates, keeping only first occurrence."""

    # Build set of indices to keep
    to_remove = set()

    for _file_path, duplicates in file_to_duplicates.items():
        # Find indices of duplicates
        for dup in duplicates:
            for idx, fp in enumerate(file_paths):
                if str(fp) == str(dup):
                    to_remove.add(idx)

    indices_to_keep = [idx for idx in range(len(dataset)) if idx not in to_remove]

    logger.info(f"Keeping {len(indices_to_keep)} / {len(dataset)} documents")
    logger.info(f"Removed {len(to_remove)} duplicates ({len(to_remove) / len(dataset) * 100:.1f}%)")

    return dataset.select(indices_to_keep)


__all__ = ["deduplicate_dataset"]
