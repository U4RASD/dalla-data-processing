"""
Postprocessing utilities for onion output.

Parses CSV files from onion and extracts duplicate information.
"""

import contextlib
import csv
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from dalla_data_processing.utils.logger import get_logger

logger = get_logger(__name__)


def parse_onion_csv(csv_file: Path) -> list[tuple[str, int]]:
    """
    Parse a single onion CSV file.

    Onion CSV format:
    - First column: file path
    - Second column: line number (index in file list)

    Args:
        csv_file: Path to CSV file

    Returns:
        List of (file_path, line_number) tuples
    """
    entries = []

    try:
        with open(csv_file, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    file_path = row[0].strip().strip('"')
                    line_number = int(row[1])
                    entries.append((file_path, line_number))
    except Exception as e:
        logger.warning(f"Error parsing CSV {csv_file}: {e}")

    return entries


def extract_duplicates_from_csvs(
    csv_dir: Path,
    file_list_path: Path,
) -> dict[str, set[str]]:
    """
    Extract duplicate groups from onion CSV outputs.

    Each CSV file represents a group of duplicates.

    Args:
        csv_dir: Directory containing onion output CSVs
        file_list_path: Path to original file list (to resolve line numbers)

    Returns:
        Dictionary mapping each file to its set of duplicates
    """
    csv_dir = Path(csv_dir)

    file_list = []
    with open(file_list_path, encoding="utf-8") as f:
        file_list = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(file_list)} files from file list")

    csv_files = list(csv_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {csv_dir}")

    duplicate_groups = []

    for csv_file in tqdm(csv_files, desc="Parsing duplicate CSVs", unit="file"):
        entries = parse_onion_csv(csv_file)

        if not entries:
            logger.debug(f"CSV {csv_file.name} has no entries")
            continue

        logger.debug(f"CSV {csv_file.name}: {len(entries)} entries")

        group = set()
        for _, line_num in entries:
            if 1 <= line_num <= len(file_list):
                resolved_path = file_list[line_num - 1]
                group.add(resolved_path)
                logger.debug(f"  Line {line_num} -> {Path(resolved_path).name}")
            else:
                logger.warning(f"Invalid line number {line_num} in {csv_file.name}")

        logger.debug(f"  Group has {len(group)} unique files")
        if len(group) > 1:
            duplicate_groups.append(group)
            logger.info(f"Found duplicate group with {len(group)} files from {csv_file.name}")
        else:
            logger.debug(f"  Skipping group with only {len(group)} file(s)")

    logger.info(f"Found {len(duplicate_groups)} duplicate groups")

    file_to_duplicates = defaultdict(set)

    for group in duplicate_groups:
        for file_path in group:
            duplicates = group - {file_path}
            file_to_duplicates[file_path].update(duplicates)

    logger.info(f"Total files with duplicates: {len(file_to_duplicates)}")

    return dict(file_to_duplicates)


def create_duplicate_pairs_with_scores(
    csv_dir: Path,
    file_list_path: Path,
) -> list[dict]:
    """
    Create list of duplicate pairs with similarity scores.

    Parses onion phase 2 output which includes similarity scores.

    Args:
        csv_dir: Directory containing onion phase 2 CSVs
        file_list_path: Path to file list

    Returns:
        List of duplicate pair dictionaries with format:
        {
            'doc1': file_path_1,
            'doc2': file_path_2,
            'similarity': score (0.0-1.0)
        }
    """
    csv_dir = Path(csv_dir)

    file_list = []
    with open(file_list_path, encoding="utf-8") as f:
        file_list = [line.strip() for line in f if line.strip()]

    logger.info(f"Processing phase 2 outputs from {csv_dir}")

    pairs = []
    csv_files = list(csv_dir.glob("*.csv"))

    for csv_file in tqdm(csv_files, desc="Processing similarity scores", unit="file"):
        try:
            with open(csv_file, encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)

                if not rows:
                    continue

                for row in rows:
                    if len(row) < 2:
                        continue

                    file_path = row[0].strip().strip('"')
                    line_num = int(row[1]) if row[1].isdigit() else None

                    score = 1.0
                    if len(row) > 2:
                        with contextlib.suppress(ValueError, IndexError):
                            score = float(row[2])

                    if not file_path and line_num and 1 <= line_num <= len(file_list):
                        file_path = file_list[line_num - 1]

                    if file_path:
                        pairs.append(
                            {
                                "source_csv": csv_file.stem,
                                "file_path": file_path,
                                "similarity": score,
                            }
                        )

        except Exception as e:
            logger.warning(f"Error processing {csv_file}: {e}")

    logger.info(f"Extracted {len(pairs)} duplicate entries from CSVs")

    csv_groups = defaultdict(list)
    for entry in pairs:
        csv_groups[entry["source_csv"]].append(entry)

    duplicate_pairs = []
    for _, entries in csv_groups.items():
        if len(entries) < 2:
            continue

        source = entries[0]["file_path"]

        for entry in entries[1:]:
            duplicate_pairs.append(
                {
                    "doc1": source,
                    "doc2": entry["file_path"],
                    "similarity": entry["similarity"],
                }
            )

    logger.info(f"Created {len(duplicate_pairs)} duplicate pairs")

    return duplicate_pairs


__all__ = [
    "parse_onion_csv",
    "extract_duplicates_from_csvs",
    "create_duplicate_pairs_with_scores",
]
