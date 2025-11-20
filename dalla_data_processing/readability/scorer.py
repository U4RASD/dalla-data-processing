"""
Readability scoring using textstat library (Flesch Reading Ease).

For Arabic-specific Osman scoring, we use a simplified formula.
"""

import textstat

from dalla_data_processing.utils.logger import get_logger

logger = get_logger(__name__)


class ReadabilityScorer:
    """Calculate readability scores for Arabic text."""

    def __init__(self):
        """Initialize the readability scorer."""
        try:
            self.textstat = textstat
            try:
                textstat.set_lang("ar")
            except Exception:
                logger.warning("Arabic language not available in textstat, using default")
        except ImportError as err:
            raise ImportError(
                "textstat library required. Install with: pip install textstat"
            ) from err

    def score_text(self, text: str) -> tuple[float | None, float | None]:
        """
        Score text using both Flesch and Osman methods.

        For very short texts where Flesch returns None, we use the Osman score.
        If Osman also fails, we use a simple fallback based on word length.

        Args:
            text: Text to score

        Returns:
            Tuple of (osman_score, flesch_score)
        """
        if not text or not text.strip():
            return (None, None)

        flesch_score = self._calculate_flesch(text)
        osman_score = self._calculate_osman(text)

        # If Flesch fails but Osman succeeds, use Osman for both
        if flesch_score is None and osman_score is not None:
            logger.info(f"Flesch failed, using Osman score ({osman_score:.1f}) for both metrics")
            flesch_score = osman_score

        # If both fail, use fallback as last resort
        elif flesch_score is None and osman_score is None:
            flesch_fallback, osman_fallback = self._calculate_fallback_scores(text)
            flesch_score = flesch_fallback
            osman_score = osman_fallback
            logger.info(
                f"Both Flesch and Osman failed, using fallback scores: O={osman_score:.1f}, F={flesch_score:.1f}"
            )

        return (osman_score, flesch_score)

    def _calculate_flesch(self, text: str) -> float | None:
        """
        Calculate Flesch Reading Ease score.

        Score range: 0-100+

        Args:
            text: Text to score

        Returns:
            Flesch score or None if error
        """
        try:
            score = self.textstat.flesch_reading_ease(text)
            if score is None:
                logger.debug(f"Flesch score is None for text (length={len(text)})")
                return None
            return float(score)
        except Exception as e:
            logger.warning(f"Error calculating Flesch score: {type(e).__name__}: {e}")
            return None

    def _calculate_osman(self, text: str) -> float | None:
        """
        Calculate Osman readability score for Arabic.

        Args:
            text: Text to score

        Returns:
            Osman score or None if error
        """
        try:
            score = self.textstat.osman(text)
            if score is None:
                logger.debug(f"Osman score is None for text (length={len(text)})")
                return None
            return float(score)

        except Exception as e:
            logger.warning(f"Error calculating Osman score: {type(e).__name__}: {e}")
            return None

    def _calculate_fallback_scores(self, text: str) -> tuple[float, float]:
        """
        Calculate simple fallback scores for very short texts.

        This is used when textstat returns None (usually for texts with < 2 sentences).
        We calculate simple metrics based on word/character counts.

        Args:
            text: Text to score

        Returns:
            Tuple of (osman_fallback, flesch_fallback)
        """
        words = text.split()
        num_words = len(words)
        num_chars = len(text.strip())

        # Average word length
        avg_word_len = num_chars / num_words if num_words > 0 else 0

        if avg_word_len <= 3:
            flesch_fallback = 90.0  # Very easy
        elif avg_word_len <= 5:
            flesch_fallback = 70.0  # Easy
        elif avg_word_len <= 7:
            flesch_fallback = 50.0  # Medium
        elif avg_word_len <= 9:
            flesch_fallback = 30.0  # Difficult
        else:
            flesch_fallback = 10.0  # Very difficult

        # Osman fallback: similar logic
        # Osman typically ranges 0-100+ for Arabic
        osman_fallback = flesch_fallback  # Use same score for simplicity

        logger.debug(
            f"Using fallback scores (words={num_words}, avg_word_len={avg_word_len:.1f}): "
            f"Flesch={flesch_fallback}, Osman={osman_fallback}"
        )

        return (osman_fallback, flesch_fallback)
