"""
Ranking and binning logic for readability scores.

Converts raw Flesch and Osman scores into 5-level difficulty rankings.
"""

from dalla_data_processing.utils.logger import get_logger

logger = get_logger(__name__)

# Strategies for combining the Osman and Flesch bins into a final level.
WEIGHTED = "weighted"          # Osman-dominant weighted average (default)
CONSERVATIVE = "conservative"  # legacy regime-split (Option B3)
LEVEL_METHODS = (WEIGHTED, CONSERVATIVE)

# Default weight on the Osman bin for the "weighted" method. Osman is the more
# reliable signal for Arabic, so it dominates; Flesch only nudges the result.
OSMAN_WEIGHT = 0.8


def compute_ranks_and_levels(
    osman_scores: list[float],
    flesch_scores: list[float],
    method: str = WEIGHTED,
    osman_weight: float = OSMAN_WEIGHT,
) -> tuple[list[int], list[int], list[int]]:
    """
    Compute ranks and final readability levels.

    Methodology:
    1. Rank documents by Osman & Flesch (highest score = rank 1, easiest)
    2. Bin ranks into 5 levels (0-4) using quantiles (guarantees balanced bins)
    3. Decide the final level from the two bins (see decide_final_level)

    Args:
        osman_scores: List of Osman scores
        flesch_scores: List of Flesch scores
        method: How to combine the bins ("weighted" or "conservative")
        osman_weight: Weight on the Osman bin when method="weighted"

    Returns:
        Tuple of:
        - o_ranks: Osman ranks (list of ints)
        - f_ranks: Flesch ranks (list of ints)
        - final_levels: Final readability levels 0-4 (list of ints)
    """
    n = len(osman_scores)

    if n == 0:
        return ([], [], [])

    # Determine ranks (highest score => rank=1, easiest)
    sorted_osman_idx = sorted(range(n), key=lambda i: osman_scores[i], reverse=True)
    o_ranks = [0] * n
    for rank_i, doc_idx in enumerate(sorted_osman_idx):
        o_ranks[doc_idx] = rank_i + 1

    sorted_flesch_idx = sorted(range(n), key=lambda i: flesch_scores[i], reverse=True)
    f_ranks = [0] * n
    for rank_i, doc_idx in enumerate(sorted_flesch_idx):
        f_ranks[doc_idx] = rank_i + 1

    # Bin ranks into [0..4]
    o_bins = bin_ranks(o_ranks)
    f_bins = bin_ranks(f_ranks)

    # Decide final level
    final_levels = [
        decide_final_level(ob, fb, method=method, osman_weight=osman_weight)
        for ob, fb in zip(o_bins, f_bins, strict=True)
    ]

    return (o_ranks, f_ranks, final_levels)


def bin_ranks(ranks: list[int]) -> list[int]:
    """
    Map ranks into 5 bins (0..4) using quantile-based binning.

    This uses TRUE quantile binning (position-based) which guarantees approximately
    20% of documents in each bin, unlike percentile-threshold binning which can
    create unbalanced or empty bins when data is clustered.

    After ranking (where highest score = rank 1), lower rank numbers indicate easier text.
    This function bins rank 1 (easiest) to bin 0, and highest rank (hardest) to bin 4.

    Args:
        ranks: List of rank values (integers starting from 1)

    Returns:
        List of bin assignments (0-4, where 0=easiest, 4=hardest)

    Algorithm:
        1. Sort ranks in ascending order (rank 1 first = easiest)
        2. Assign bins based on position in sorted list
        3. First 20% (lowest ranks) → bin 0, last 20% (highest ranks) → bin 4

    Example:
        >>> bin_ranks([5, 4, 3, 2, 1, 1, 2, 3, 4, 5])
        [4, 3, 2, 1, 0, 0, 1, 2, 3, 4]
        # Rank 1 (easiest) → bin 0, Rank 5 (hardest) → bin 4
    """
    n = len(ranks)

    if n == 0:
        return []
    if n == 1:
        return [0]

    # Create (rank, original_index) pairs to track positions
    indexed_ranks = [(rank, i) for i, rank in enumerate(ranks)]

    # Sort by rank ASCENDING (rank 1 = easiest, should go to bin 0)
    indexed_ranks.sort(key=lambda x: x[0])

    # Assign bins based on position in sorted list
    bins = [0] * n

    for sorted_position, (_rank, orig_idx) in enumerate(indexed_ranks):
        # Calculate which quintile (0-4) this position falls into
        # Position 0 to n/5-1 → bin 0 (easiest 20%)
        # Position n/5 to 2n/5-1 → bin 1
        # ...
        # Position 4n/5 to n-1 → bin 4 (hardest 20%)
        bin_number = min(4, int((sorted_position * 5) / n))
        bins[orig_idx] = bin_number

    return bins


def decide_final_level(
    o_bin: int, f_bin: int, method: str = WEIGHTED, osman_weight: float = OSMAN_WEIGHT
) -> int:
    """
    Decide final readability level from the Osman and Flesch bins.

    Two strategies are available:

    "weighted" (default): an Osman-dominant weighted average,
        round(osman_weight * o_bin + (1 - osman_weight) * f_bin).
        For Arabic, Osman is the reliable signal (it carries Arabic-specific terms
        such as faseeh and complex/long-word ratios that hold up on undiacritised
        text), whereas Flesch depends on syllable counts that degrade without
        diacritics. Flesch therefore only nudges the level rather than overriding it.

    "conservative": the legacy regime-split (Option B3) — trust Osman when it
        indicates hardness (bins 3-4), trust Flesch when it indicates easiness
        (bins 0-1), take the harder bin on large disagreement, else average.

    Args:
        o_bin: Osman bin (0-4, 0=easiest, 4=hardest)
        f_bin: Flesch bin (0-4, 0=easiest, 4=hardest)
        method: "weighted" or "conservative"
        osman_weight: Weight on the Osman bin when method="weighted"

    Returns:
        Final level (0-4)

    Examples:
        >>> decide_final_level(4, 0)                          # weighted, Osman dominates
        3
        >>> decide_final_level(0, 4, method="conservative")   # easy: trust Flesch -> hard
        4
    """
    if method == WEIGHTED:
        return round(osman_weight * o_bin + (1 - osman_weight) * f_bin)
    if method == CONSERVATIVE:
        if o_bin >= 3:
            return o_bin
        if f_bin <= 1:
            return f_bin
        if abs(o_bin - f_bin) >= 2:
            return max(o_bin, f_bin)
        return (o_bin + f_bin + 1) // 2
    raise ValueError(f"Unknown level method {method!r}; expected one of {LEVEL_METHODS}")
