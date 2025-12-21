# MIT License
#
# Copyright 2018-2024 New York University Abu Dhabi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Word-boundary tokenization utilities."""

import re

__all__ = ["simple_word_tokenize"]

# Compact mode: Arabic + Latin + digits
_ARABIC = (
    r"\u0621-\u063A"
    r"\u0641-\u064A"
    r"\u064B-\u0652"
    r"\u0653-\u0655"
    r"\u0670"
    r"\u0671-\u06D3"
    r"\u06D5-\u06FF"
    r"\u0750-\u077F"
    r"\u08A0-\u08FF"
    r"\uFB50-\uFDFF"
    r"\uFE70-\uFEFF"
)
_LATIN = r"a-zA-Z"
_DIGITS = r"0-9\u0660-\u0669\u06F0-\u06F9"
_COMPACT_CHARSET = _ARABIC + _LATIN + _DIGITS

# Full mode: Unicode letters/marks/numbers (via \w which covers all Unicode word chars)
_FULL_CHARSET = r"\w"

# Pre-compiled regexes for compact mode
_COMPACT_RE = re.compile(f"[{_COMPACT_CHARSET}]+|[^{_COMPACT_CHARSET}\\s]|\\s+")
_COMPACT_SPLIT_RE = re.compile(f"[{_ARABIC}{_LATIN}]+|[{_DIGITS}]+|[^{_COMPACT_CHARSET}\\s]|\\s+")

# Pre-compiled regexes for full mode
_FULL_RE = re.compile(r"\w+|[^\w\s]|\s+")
_FULL_SPLIT_RE = re.compile(r"[^\W\d]+|\d+|[^\w\s]|\s+")


def simple_word_tokenize(sentence, split_digits=False, mode="compact"):
    """Tokenize a sentence by splitting on whitespace and separating punctuation.

    Args:
        sentence: Sentence to tokenize.
        split_digits: Split digits from letters. Defaults to False.
        mode: "compact" (Arabic + Latin + digits) or "full" (all Unicode).
            Defaults to "compact".

    Returns:
        List of tokens.
    """
    if mode == "compact":
        if split_digits:
            return _COMPACT_SPLIT_RE.findall(sentence)
        return _COMPACT_RE.findall(sentence)
    elif mode == "full":
        if split_digits:
            return _FULL_SPLIT_RE.findall(sentence)
        return _FULL_RE.findall(sentence)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'compact' or 'full'.")
