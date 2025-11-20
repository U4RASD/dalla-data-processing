"""
Preprocessing utilities for deduplication.

Handles text normalization and conversion to vertical format for onion.
"""

import re
from pathlib import Path

from camel_tools.utils.dediac import dediac_ar
from tqdm import tqdm

from dalla_data_processing.utils.logger import get_logger

logger = get_logger(__name__)


def text_to_vertical(text: str, doc_id: str = "") -> str:
    """
    Convert text to vertical format (one word per line).

    This format is required by the onion deduplication algorithm.
    Uses CAMEL Tools tokenizer to properly split words and preserve spaces.

    Args:
        text: Input text (can be None, will be converted to empty string)
        doc_id: Document identifier (optional)

    Returns:
        Text in vertical format with optional doc tags
    """
    if text is None or not isinstance(text, str):
        text = ""

    text = dediac_ar(text)

    words = text.split()

    vertical = "\n".join(words)

    if doc_id:
        vertical = f"<doc name='{doc_id}'>\n{vertical}\n</doc>"

    return vertical


def create_vert_files(
    texts: list[str],
    ids: list[str],
    output_dir: Path,
) -> tuple[list[Path], dict[str, str]]:
    """
    Create vertical format files from texts.

    Args:
        texts: List of text strings
        ids: List of document IDs
        output_dir: Directory to save vert files

    Returns:
        Tuple of (list of file paths, mapping of original IDs to vert file paths)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating {len(texts)} vert files in {output_dir}")

    file_paths = []
    id_mapping = {}

    for idx, (text, doc_id) in enumerate(
        tqdm(zip(texts, ids, strict=False), total=len(texts), desc="Creating vert files")
    ):
        safe_filename = re.sub(r"[^\w\-_.]", "_", str(doc_id))
        if len(safe_filename) > 200:
            safe_filename = safe_filename[-200:]

        filename = f"{idx:08d}_{safe_filename}.txt"
        filepath = output_dir / filename
        vert_text = text_to_vertical(text, doc_id)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(vert_text)

        file_paths.append(filepath)
        id_mapping[str(doc_id)] = str(filepath)

    logger.info(f"Created {len(file_paths)} vert files")

    return file_paths, id_mapping


def create_file_list(file_paths: list[Path], output_file: Path) -> Path:
    """
    Create a text file listing all file paths (input for onion).

    Args:
        file_paths: List of file paths
        output_file: Path to output file list

    Returns:
        Path to created file list
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for filepath in file_paths:
            f.write(f"{filepath}\n")

    logger.info(f"Created file list with {len(file_paths)} entries: {output_file}")

    return output_file


__all__ = [
    "text_to_vertical",
    "create_vert_files",
    "create_file_list",
]
