"""
Quality checking implementation for Arabic text using CAMEL Tools.

This module provides quality assessment by analyzing morphological features
and detecting errors in Arabic text.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from types import MethodType
from typing import Any

from camel_tools.data.catalogue import Catalogue
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.disambig.mle import MLEDisambiguator
from datasets import Dataset

from dalla_data_processing.core.parallel import ParallelProcessor
from dalla_data_processing.utils.logger import get_logger

logger = get_logger(__name__)

WORD_DELIMITERS = re.compile(r'[0-9#%?:\-+=~()\s\'"/\\*]+|[\[\]{}<>﴿﴾,.٫٪؟«»،؛]+')
SENTENCE_DELIMITERS = re.compile(r"[?\n\r.;:,.٫٪؟«»،؛]+")


class QualityChecker:
    """Quality checker for Arabic text using CAMEL Tools."""

    def __init__(self, timeout: int = 3600, model: str = "mle", use_gpu: bool = False):
        """
        Initialize quality checker.

        Args:
            timeout: Maximum time in seconds for processing a single text (default: 3600)
            model: Disambiguator model to use - "mle" or "bert" (default: "mle")
            use_gpu: Whether to use GPU for BERT model (default: False)
        """
        self.timeout = timeout
        self.model = model.lower()
        self.use_gpu = use_gpu
        self.disambiguator = None
        self.erroneous_words: dict[str, int] = {}

        if self.model not in ["mle", "bert"]:
            raise ValueError(f"Invalid model '{model}'. Must be 'mle' or 'bert'")

        logger.info(f"Initializing CAMEL Tools {self.model.upper()} disambiguator...")
        if self.model == "bert" and self.use_gpu:
            logger.info("GPU mode enabled for BERT")

        self._init_disambiguator()

    def _init_disambiguator(self):
        """Initialize and configure the disambiguator with caching."""
        # Install required CAMeL Tools packages based on model type
        logger.info("Checking CAMeL Tools data packages...")
        catalogue = Catalogue.load_catalogue()

        try:
            catalogue.download_package("morphology-db-msa-r13")
            catalogue.download_package("disambig-mle-calima-msa-r13")
            logger.info("msa-r13 packages installed")
        except Exception as e:
            logger.warning(f"Package installation warning: {e}")

        # Install BERT package if using BERT model
        if self.model == "bert":
            try:
                catalogue.download_package("disambig-bert-unfactored-all")
                logger.info("BERT package installed")
            except Exception as e:
                logger.warning(f"BERT package installation warning: {e}")

        if self.model == "mle":
            self.disambiguator = MLEDisambiguator.pretrained()
            logger.info("MLE disambiguator loaded")
        else:
            self.disambiguator = BERTUnfactoredDisambiguator.pretrained(use_gpu=self.use_gpu)
            logger.info(f"BERT disambiguator loaded (GPU: {self.use_gpu})")

        def cached_scored_analysis(disambiguator, word_dd):
            if word_dd in disambiguator._cache:
                return disambiguator._cache[word_dd]
            result = disambiguator._scored_analyses(word_dd)
            disambiguator._cache[word_dd] = result
            return result

        self.disambiguator._scored_analyses_cached = MethodType(
            cached_scored_analysis, self.disambiguator
        )
        self.disambiguator._score_fn = self.disambiguator._scored_analyses_cached

        logger.info("Disambiguator initialized with caching enabled")

    @staticmethod
    def is_arabic(word: str) -> bool:
        """
        Check if a word is Arabic.

        Args:
            word: Word to check

        Returns:
            True if word contains only Arabic characters
        """
        arabic_ranges = [
            (0x0600, 0x06FF),  # Arabic
            (0x0750, 0x077F),  # Arabic Supplement
            (0x08A0, 0x08FF),  # Arabic Extended-A
            (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
            (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
        ]

        arabic_numbers = range(0x0660, 0x066A)
        return all(
            any(start <= ord(char) <= end for start, end in arabic_ranges) for char in word
        ) and not all(ord(char) in arabic_numbers for char in word)

    def process_content(
        self, content: str, erroneous_words: dict[str, int]
    ) -> tuple[int, int, int, int]:
        """
        Process content and count errors.

        Args:
            content: Text content to process
            erroneous_words: Dictionary to track erroneous words

        Returns:
            Tuple of (total_words, error_count, no_analysis_count, foreign_count)
        """
        arabic_sentence_list = WORD_DELIMITERS.split(content)
        arabic_sentence_list = [word for word in arabic_sentence_list if word]

        if not arabic_sentence_list:
            return 0, 0, 0, 0

        morph_features = self.disambiguator.disambiguate(arabic_sentence_list)

        total_words = len(morph_features)
        err_count = 0
        err_no_analysis = 0
        err_foreign = 0

        for i, word in enumerate(arabic_sentence_list):
            if morph_features[i] is None or len(morph_features[i].analyses) == 0:
                err_count += 1
                if self.is_arabic(word):
                    erroneous_words[word] = erroneous_words.get(word, 0) + 1
                continue

            analyses = morph_features[i].analyses
            analysis_i = analyses[0].analysis

            if analysis_i["gloss"] == "NO_ANALYSIS":
                err_count += 1
                err_no_analysis += 1
                if self.is_arabic(word):
                    erroneous_words[word] = erroneous_words.get(word, 0) + 1

            elif analysis_i["gloss"] == word:
                err_count += 1
                err_foreign += 1
                if self.is_arabic(word):
                    erroneous_words[word] = erroneous_words.get(word, 0) + 1

        return total_words, err_count, err_no_analysis, err_foreign

    def process_full_content(
        self, content: str, erroneous_words: dict[str, int]
    ) -> tuple[float, float, float]:
        """
        Process full content by splitting into sentences.

        Args:
            content: Full text content
            erroneous_words: Dictionary to track erroneous words

        Returns:
            Tuple of (quality_score, arabic_error_percent, foreign_error_percent)
        """
        full_content_list = SENTENCE_DELIMITERS.split(content)

        total_words = 0
        err_count = 0
        err_no_analysis = 0
        err_foreign = 0

        for sentence in full_content_list:
            if sentence.strip():
                t, ec, ena, ef = self.process_content(sentence, erroneous_words)
                total_words += t
                err_count += ec
                err_no_analysis += ena
                err_foreign += ef

        if total_words == 0:
            return 0.0, 0.0, 0.0

        quality_score = 100 * (1 - (err_count / total_words))
        score_ar = 100 * (err_no_analysis / total_words)
        score_foreign = 100 * (err_foreign / total_words)

        return quality_score, score_ar, score_foreign

    def check_text_quality(
        self, text: str, erroneous_words: dict[str, int] | None = None
    ) -> dict[str, Any]:
        """
        Check quality of a single text with timeout protection.

        Args:
            text: Text to check
            erroneous_words: Optional dictionary to track erroneous words

        Returns:
            Dictionary with quality scores and status
        """
        if erroneous_words is None:
            erroneous_words = {}

        result = {
            "quality_score": 0.0,
            "arabic_error_percent": 0.0,
            "foreign_error_percent": 0.0,
            "error_code": 0,
            "error_message": None,
        }

        if not text or not isinstance(text, str):
            result["error_code"] = -1
            result["error_message"] = "Empty or invalid text"
            return result

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.process_full_content, text, erroneous_words)
            try:
                quality_score, score_ar, score_foreign = future.result(timeout=self.timeout)
                result["quality_score"] = quality_score
                result["arabic_error_percent"] = score_ar
                result["foreign_error_percent"] = score_foreign
            except FutureTimeoutError:
                result["error_code"] = -3
                result["error_message"] = f"Processing timeout ({self.timeout}s)"
                logger.warning(f"Text processing timeout after {self.timeout}s")
            except Exception as e:
                result["error_code"] = -2
                result["error_message"] = f"Processing error: {str(e)}"
                logger.error(f"Error processing text: {e}")

        return result

    def process_example(self, example: dict[str, Any], column: str) -> dict[str, Any]:
        """
        Process a single example from dataset.

        Args:
            example: Dataset example
            column: Column name to process

        Returns:
            Example with added quality scores
        """
        text = example.get(column, "")

        result = self.check_text_quality(text, self.erroneous_words)

        example["quality_score"] = result["quality_score"]
        example["arabic_error_percent"] = result["arabic_error_percent"]
        example["foreign_error_percent"] = result["foreign_error_percent"]
        example["quality_error_code"] = result["error_code"]

        if result["error_message"]:
            example["quality_error_message"] = result["error_message"]

        return example

    def get_erroneous_words(self) -> dict[str, int]:
        """
        Get dictionary of erroneous words found during processing.

        Returns:
            Dictionary mapping erroneous words to their occurrence count
        """
        return self.erroneous_words.copy()


def check_quality(
    dataset: Dataset,
    column: str = "text",
    min_score: float = 0.0,
    save_errors: bool = False,
    num_workers: int | None = None,
    timeout: int = 3600,
    model: str = "mle",
    use_gpu: bool = False,
) -> Dataset:
    """
    Check quality of texts in dataset and add quality score columns.

    Args:
        dataset: HuggingFace dataset
        column: Column name to check
        min_score: Minimum quality score to keep (0-100)
        save_errors: Whether to save erroneous words (logged if True)
        num_workers: Number of parallel workers (None for auto)
        timeout: Timeout per text in seconds
        model: Disambiguator model - "mle" or "bert" (default: "mle")
        use_gpu: Use GPU for BERT model (default: False, only for model="bert")

    Returns:
        Dataset with quality score columns added (and optionally filtered)

    Example:
        >>> # Using MLE (default, faster)
        >>> scored = check_quality(dataset, min_score=50.0)

        >>> # Using BERT (more accurate, slower)
        >>> scored = check_quality(dataset, model="bert", use_gpu=True)

        >>> # Columns added: quality_score, arabic_error_percent,
        >>> #                foreign_error_percent, quality_error_code
    """

    logger.info(f"Checking quality of {len(dataset)} examples")
    logger.info(f"Model: {model.upper()}, Column: {column}, Min score: {min_score}")
    logger.info(f"Timeout: {timeout}s, GPU: {use_gpu if model == 'bert' else 'N/A'}")

    if column not in dataset.column_names:
        raise ValueError(f"Column '{column}' not found in dataset")

    checker = QualityChecker(timeout=timeout, model=model, use_gpu=use_gpu)

    num_workers = ParallelProcessor.get_optimal_num_workers(num_workers)
    logger.info(f"Processing with {num_workers} workers")

    processed_dataset = dataset.map(
        lambda example: checker.process_example(example, column),
        num_proc=num_workers,
        desc="Quality checking",
    )

    original_size = len(dataset)
    avg_score = sum(processed_dataset["quality_score"]) / len(processed_dataset)
    logger.info(f"Average quality score: {avg_score:.2f}")

    if min_score > 0:
        logger.info(f"Filtering examples with score < {min_score}")
        processed_dataset = processed_dataset.filter(
            lambda x: x["quality_score"] >= min_score,
            num_proc=num_workers,
            desc=f"Filtering (min_score={min_score})",
        )

        filtered_size = len(processed_dataset)
        removed = original_size - filtered_size
        logger.info(
            f"Removed {removed:,} low-quality examples ({removed / original_size * 100:.1f}%)"
        )
        logger.info(f"Final dataset size: {filtered_size:,}")

    if save_errors:
        erroneous_words = checker.get_erroneous_words()
        logger.info(f"Found {len(erroneous_words)} unique erroneous words")

        if erroneous_words:
            sorted_errors = sorted(erroneous_words.items(), key=lambda x: x[1], reverse=True)[:20]
            logger.info("Top 20 erroneous words:")
            for word, count in sorted_errors:
                logger.info(f"  {word}: {count}")

    return processed_dataset
