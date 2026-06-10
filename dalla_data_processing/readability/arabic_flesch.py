"""
Arabic Flesch Reading Ease scoring.

textstat's flesch_reading_ease() cannot score Arabic: it counts syllables with
pyphen, which ships no Arabic hyphenation dictionary and raises KeyError. This
module instead counts Arabic syllables directly from diacritics, with a
character-length fallback for undiacritised text, so Flesch is computable for
Arabic in all cases.

Ported from the original OSMAN readability implementation by Mahmoud El-Haj
(OsmanReadability.java, Syllables.java).
"""

import re

# Short vowels (harakat): fatha, damma, kasra.
HARAKAT = ("َ", "ُ", "ِ")
# Long-vowel letters that turn a preceding haraka into a long syllable: alef, waw, yaa.
LONG_LETTERS = ("ا", "و", "ي")
# Stress marks: tanween fath, tanween damm, tanween kasr, shadda.
STRESS_MARKS = ("ً", "ٌ", "ٍ", "ّ")

PUNCT_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
DIGIT_PATTERN = re.compile(r"\d")
SENTENCE_PATTERN = re.compile(r"\n|(?<!\d)\.(?!\d)")


def count_all_syllables(text: str) -> tuple[int, int, int]:
    """
    Count Arabic short, long, and stress syllables.

    Long syllables are harakat followed by a long-vowel letter; the remaining
    harakat are short. Stress syllables are tanween and shadda marks. For
    undiacritised text (no short syllables found), short syllables are
    approximated from the stripped character length.

    Args:
        text: Text to analyse

    Returns:
        Tuple of (short_syllables, long_syllables, stress_syllables)
    """
    long_count = 0
    short_count = 0

    for haraka in HARAKAT:
        for i, char in enumerate(text):
            if char == haraka:
                if i + 1 < len(text) and text[i + 1] in LONG_LETTERS:
                    long_count += 1
                else:
                    short_count += 1

    stress_count = sum(text.count(mark) for mark in STRESS_MARKS)

    # Fallback for undiacritised text: approximate short syllables from length.
    if short_count == 0:
        stripped = (
            text.replace("ا", "")
            .replace("ى", "")
            .replace("?", "")
            .replace(".", "")
            .replace("!", "")
            .replace(",", "")
            .replace(" ", "")
        )
        short_count = len(stripped) - 2

    return short_count, long_count, stress_count


def count_syllables(text: str) -> int:
    """
    Count total syllables, weighting long and stress syllables double.

    Args:
        text: Text to analyse

    Returns:
        Total syllable count
    """
    short_syl, long_syl, stress_syl = count_all_syllables(text)
    return (long_syl * 2) + short_syl + (stress_syl * 2)


def count_words(text: str) -> int:
    """
    Count words after removing digits and punctuation.

    Args:
        text: Text to analyse

    Returns:
        Number of whitespace-separated words
    """
    cleaned = DIGIT_PATTERN.sub("", text)
    cleaned = PUNCT_PATTERN.sub("", cleaned)
    cleaned = re.sub(r" +", " ", cleaned.strip())
    return len(cleaned.split()) if cleaned else 0


def count_sentences(text: str) -> int:
    """
    Count sentences by splitting on newlines and non-decimal periods.

    Args:
        text: Text to analyse

    Returns:
        Number of sentences
    """
    # Match Java's String.split(regex), which discards trailing empty strings,
    # so a trailing period does not count as an extra empty sentence.
    parts = SENTENCE_PATTERN.split(text)
    while parts and parts[-1] == "":
        parts.pop()
    return len(parts)


def words_per_sentence(text: str) -> float:
    """Return the average number of words per sentence."""
    words = count_words(text)
    sentences = count_sentences(text)
    return words / sentences if sentences else float(words)


def syllables_per_word(text: str) -> float:
    """Return the average number of syllables per word."""
    words = count_words(text)
    return count_syllables(text) / words if words else 0.0


def arabic_flesch_reading_ease(text: str) -> float | None:
    """
    Calculate Arabic Flesch Reading Ease.

    Score = 206.835 - 1.015 * (words / sentence) - 84.6 * (syllables / word)

    Args:
        text: Text to score

    Returns:
        Flesch Reading Ease score, or None for empty or word-less text
    """
    if not text or not text.strip() or count_words(text) == 0:
        return None
    return 206.835 - (1.015 * words_per_sentence(text)) - (84.6 * syllables_per_word(text))
