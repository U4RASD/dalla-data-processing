"""
Ranking and binning logic for readability scores.

Converts raw Flesch and Osman scores into 5-level difficulty rankings.
"""

from dalla_data_processing.utils.logger import get_logger

logger = get_logger(__name__)


def compute_ranks_and_levels(
    osman_scores: list[float], flesch_scores: list[float]
) -> tuple[list[int], list[int], list[int]]:
    """
    Compute ranks and final readability levels.

    Methodology:
    1. Rank documents by Osman & Flesch (highest score = rank 1, easiest)
    2. Bin ranks into 5 levels (0-4) using quantiles (guarantees balanced bins)
    3. Decide final level using smart conservative logic

    Args:
        osman_scores: List of Osman scores
        flesch_scores: List of Flesch scores

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
    final_levels = [decide_final_level(ob, fb) for ob, fb in zip(o_bins, f_bins, strict=True)]

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


def decide_final_level(o_bin: int, f_bin: int) -> int:
    """
    Decide final readability level from Osman and Flesch bins.

    Strategy (Option B3 - Smart Conservative):
    - Trust Osman when it indicates hardness (bins 3-4)
    - Trust Flesch when it indicates easiness (bins 0-1)
    - On complete disagreement (diff >= 2), be conservative (take harder)
    - On small disagreement (diff = 1), average them

    Philosophy:
    - Osman is the expert at identifying hard texts
    - Flesch is the expert at identifying easy texts
    - When metrics completely disagree, the text is unusual → mark as harder
    - When metrics slightly disagree, compromise with average

    Args:
        o_bin: Osman bin (0-4, 0=easiest, 4=hardest)
        f_bin: Flesch bin (0-4, 0=easiest, 4=hardest)

    Returns:
        Final level (0-4)

    Examples:
        >>> decide_final_level(4, 0)  # Osman=hard, Flesch=easy → trust Osman
        4
        >>> decide_final_level(0, 4)  # Osman=easy, Flesch=hard → trust Flesch (unusual, conservative)
        4
        >>> decide_final_level(1, 0)  # Both easy, Flesch=easier → trust Flesch
        0
        >>> decide_final_level(3, 4)  # Both hard, Osman=easier → trust Osman
        3
        >>> decide_final_level(2, 3)  # Small disagreement → average (2+3+1)//2 = 3
        3
    """
    # Strong Osman signal: text is hard (bins 3-4)
    if o_bin >= 3:
        return o_bin

    # Strong Flesch signal: text is easy (bins 0-1)
    if f_bin <= 1:
        return f_bin

    # Calculate disagreement magnitude
    diff = abs(o_bin - f_bin)

    # Complete disagreement (diff >= 2)
    if diff >= 2:
        return max(o_bin, f_bin)

    # Small disagreement (diff = 1) or agreement
    return (o_bin + f_bin + 1) // 2
