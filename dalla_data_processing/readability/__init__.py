"""Readability scoring and ranking module using textstat."""

from datasets import Dataset

from dalla_data_processing.readability.ranking import compute_ranks_and_levels
from dalla_data_processing.readability.scorer import ReadabilityScorer
from dalla_data_processing.utils.logger import get_logger

logger = get_logger(__name__)


def score_readability(
    dataset: Dataset,
    column: str = "text",
    add_ranks: bool = True,
    num_proc: int | None = None,
) -> Dataset:
    """
    Score readability using Flesch and Osman methods, with optional ranking.

    Adds columns to dataset:
    - flesch_score: Flesch Reading Ease score
    - osman_score: Osman readability score

    If add_ranks=True, also adds (computed across entire dataset):
    - flesch_rank: Flesch rank (1 = lowest score)
    - osman_rank: Osman rank (1 = lowest score)
    - readability_level: Final readability level (0-4)

    Args:
        dataset: HuggingFace dataset
        column: Column to score
        add_ranks: Whether to add ranking columns (default: True)
        num_proc: Number of parallel processes

    Returns:
        Dataset with readability scores and optional rankings

    Example:
        >>> from dalla_data_processing.readability import score_readability
        >>> scored = score_readability(dataset)
        >>> # Columns: flesch_score, osman_score, readability_level, etc.
    """
    logger.info(f"Scoring readability of {len(dataset)} examples")
    logger.info(f"Column: {column}, Add ranks: {add_ranks}, Workers: {num_proc or 'auto'}")

    # Initialize scorer
    logger.info("Initializing readability scorer...")
    ReadabilityScorer()  # Initialize to verify dependencies are available
    logger.info("Scorer ready")

    # Step 1: Score all texts
    logger.info("Calculating Flesch and Osman scores...")

    def score_example(example):
        # Create scorer inside worker (for multiprocessing compatibility)
        from dalla_data_processing.readability.scorer import ReadabilityScorer

        worker_scorer = ReadabilityScorer()

        text = example.get(column, "")
        if not text:
            example["osman_score"] = None
            example["flesch_score"] = None
            return example

        osman_score, flesch_score = worker_scorer.score_text(text)
        example["osman_score"] = osman_score
        example["flesch_score"] = flesch_score
        return example

    scored_dataset = dataset.map(score_example, num_proc=num_proc, desc="Scoring readability")

    # Count how many valid scores we got
    valid_count = sum(
        1
        for ex in scored_dataset
        if ex.get("osman_score") is not None and ex.get("flesch_score") is not None
    )
    logger.info(f"Scoring complete for {len(scored_dataset)} examples")
    if valid_count == len(scored_dataset):
        logger.info(f"Successfully scored all {valid_count} examples")
    else:
        logger.info(
            f"Valid scores: {valid_count}/{len(scored_dataset)} ({valid_count / len(scored_dataset) * 100:.1f}%)"
        )
        if valid_count == 0:
            logger.error(
                "Failed to calculate scores for any examples. "
                "This indicates a problem with the text or textstat library."
            )
    logger.info(f"Scoring complete for {len(scored_dataset)} examples")
    logger.info(
        f"Valid scores: {valid_count}/{len(scored_dataset)} ({valid_count / len(scored_dataset) * 100:.1f}%)"
    )

    if valid_count == 0:
        logger.warning(
            "No valid readability scores calculated. "
            "Common causes: text too short (< 2 sentences), "
            "no complete sentences, or special characters only."
        )

    # Step 2: Add ranks if requested
    if add_ranks:
        logger.info("Computing ranks and readability levels...")
        scored_dataset = _add_ranks_to_dataset(scored_dataset)
        logger.info("Ranks and levels added")

    logger.info("Readability scoring complete!")
    return scored_dataset


def _add_ranks_to_dataset(dataset: Dataset) -> Dataset:
    """
    Add ranking columns to dataset based on scores.

    This computes ranks across the entire dataset and adds:
    - osman_rank, flesch_rank
    - readability_level (final 0-4 level)

    Args:
        dataset: Dataset with osman_score and flesch_score columns

    Returns:
        Dataset with ranking columns added
    """
    # Extract scores
    osman_scores = []
    flesch_scores = []
    valid_indices = []

    for i, example in enumerate(dataset):
        o_score = example.get("osman_score")
        f_score = example.get("flesch_score")

        # Only include examples with valid scores
        if o_score is not None and f_score is not None:
            osman_scores.append(float(o_score))
            flesch_scores.append(float(f_score))
            valid_indices.append(i)

    logger.info(f"Computing ranks for {len(valid_indices)} valid examples")

    if len(osman_scores) == 0:
        logger.error("No valid scores found - cannot compute ranks")
        logger.error(
            f"All {len(dataset)} examples have None scores. "
            "This should not happen with the fallback scoring system. "
            "Please report this as a bug."
        )
        # Still return the dataset with None rank columns
        return dataset

    # Compute ranks and levels
    o_ranks, f_ranks, final_levels = compute_ranks_and_levels(osman_scores, flesch_scores)

    # Create mapping from index to rank data
    rank_data = {}
    for idx, o_r, f_r, final_lvl in zip(
        valid_indices,
        o_ranks,
        f_ranks,
        final_levels,
        strict=False,
    ):
        rank_data[idx] = {
            "osman_rank": o_r,
            "flesch_rank": f_r,
            "readability_level": final_lvl,
        }

    # Add columns to dataset
    def add_rank_columns(example, idx):
        if idx in rank_data:
            example.update(rank_data[idx])
        else:
            # No valid scores - set to None
            example["osman_rank"] = None
            example["flesch_rank"] = None
            example["readability_level"] = None
        return example

    dataset = dataset.map(add_rank_columns, with_indices=True, desc="Adding ranks")

    # Log statistics
    if final_levels:
        level_counts = [final_levels.count(i) for i in range(5)]
        logger.info("Readability level distribution:")
        for i, count in enumerate(level_counts):
            pct = (count / len(final_levels)) * 100
            logger.info(f"  Level {i}: {count:,} ({pct:.1f}%)")

    return dataset


__all__ = ["score_readability"]
