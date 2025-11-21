"""Stemming and morphological analysis implementation.

This module contains all the implementation details for Arabic stemming
and morphological tokenization using CAMeL Tools.
"""

import os
import re
from collections import deque
from types import MethodType

from camel_tools.data.catalogue import Catalogue
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.utils.dediac import dediac_ar
from datasets import Dataset

from dalla_data_processing.utils.logger import get_logger
from dalla_data_processing.utils.tokenize import simple_word_tokenize

logger = get_logger(__name__)


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text."""
    _DIAC_RE = re.compile(
        r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u08D3-\u08FF]"
    )
    _TATWEEL_RE = re.compile(r"\u0640")
    _ALIF_RE = re.compile(r"[آأإٱ]")
    _ALIF_MAK_RE = re.compile(r"ى")
    _TEH_MARB_RE = re.compile(r"ة")

    text = _DIAC_RE.sub("", text)
    text = _TATWEEL_RE.sub("", text)
    text = _ALIF_RE.sub("ا", text)
    text = _ALIF_MAK_RE.sub("ي", text)
    text = _TEH_MARB_RE.sub("ه", text)
    return text


def has_diacritics(word):
    """Check if word has diacritics."""
    diacritic_marks = {
        "\u064b",
        "\u064c",
        "\u064d",
        "\u064e",
        "\u064f",
        "\u0650",
        "\u0651",
        "\u0652",
        "\u0670",
    }
    return any(char in diacritic_marks for char in word)


def apply_diacritics_to_segments_keep_markers(segments, diacritized_word, sep_token="<+>"):
    """Apply diacritics from original word to segmented tokens."""
    result = []
    diacritic_marks = {
        "\u064b",
        "\u064c",
        "\u064d",
        "\u064e",
        "\u064f",
        "\u0650",
        "\u0651",
        "\u0652",
        "\u0670",
    }
    sep_len = len(sep_token)

    leading_diacritics = []
    i = 0
    while i < len(diacritized_word) and diacritized_word[i] in diacritic_marks:
        leading_diacritics.append(diacritized_word[i])
        i += 1

    diacritic_index = len(leading_diacritics)

    for segment_idx, segment in enumerate(segments):
        if segment == sep_token:
            result.append(segment)
        else:
            diacritized_segment = []

            if segment_idx == 0 and leading_diacritics:
                diacritized_segment.extend(leading_diacritics)

            i = 0
            while i < len(segment):
                char = segment[i]
                if segment[i : i + sep_len] == sep_token:
                    diacritized_segment.append(sep_token)
                    i += sep_len
                    continue

                if diacritic_index < len(diacritized_word):
                    while (
                        diacritic_index < len(diacritized_word)
                        and diacritized_word[diacritic_index] in diacritic_marks
                    ):
                        diacritic_index += 1

                    if (
                        diacritic_index < len(diacritized_word)
                        and diacritized_word[diacritic_index] == char
                    ):
                        diacritized_segment.append(char)
                        diacritic_index += 1

                        while (
                            diacritic_index < len(diacritized_word)
                            and diacritized_word[diacritic_index] in diacritic_marks
                        ):
                            diacritized_segment.append(diacritized_word[diacritic_index])
                            diacritic_index += 1
                    else:
                        diacritized_segment.append(char)
                else:
                    diacritized_segment.append(char)

                i += 1

            result.append("".join(diacritized_segment))

    return result


def read_and_dediacritize(file_name):
    """Read words from file and dediacritize them."""
    words = []
    with open(file_name, encoding="utf-8") as file:
        for line in file:
            word = line.strip()
            dediacritized_word = dediac_ar(word)
            words.append(dediacritized_word)
    return words


def par_is_utf8_encoded(paragraph):
    """Check if paragraph is UTF-8 encoded."""
    try:
        paragraph.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False


def tokenize(text):
    """Tokenize text into words."""
    if par_is_utf8_encoded(text):
        text_list = simple_word_tokenize(text)
        return text_list
    else:
        return None


def merge_alef_and_alef_lam(input_list, sep_token="<+>"):
    """Merge specific Arabic morpheme patterns."""
    pattern = [f"\u0644{sep_token}".encode(), f"\u0627\u0644{sep_token}".encode()]
    replacement = f"\u0644\u0644{sep_token}"

    modified_list = []
    i = 0

    while i < len(input_list):
        if i < len(input_list) - 1:
            current_element = input_list[i].encode("utf-8")
            next_element = input_list[i + 1].encode("utf-8")

            if current_element == pattern[0] and next_element == pattern[1]:
                modified_list.append(replacement)
                i += 2
                continue

        modified_list.append(input_list[i])
        i += 1

    return modified_list


def process_NOAN_word(list_al_t, list_al, list_t, word, sep_token="<+>"):
    """Process words marked as NOAN (no analysis)."""
    alef_lam = b"\xd8\xa7\xd9\x84"
    taa_marbouta_detached = b"\xef\xba\x93"
    taa_marbouta_attached = b"\xd8\xa9"
    word_bytes = word.encode("utf-8")

    if (
        word_bytes.startswith(alef_lam)
        and (
            word_bytes.endswith(taa_marbouta_detached) or word_bytes.endswith(taa_marbouta_attached)
        )
        and word in list_al_t
    ):
        stripped_word = word[2:-1]
        first_part = word[0:2] + sep_token
        last_part = sep_token + word[-1]
        return [first_part, stripped_word, last_part]

    if word_bytes.startswith(alef_lam) and word in list_al:
        stripped_word = word[2:]
        first_part = word[0:2] + sep_token
        return [first_part, stripped_word]

    if word_bytes.endswith(taa_marbouta_detached) or word_bytes.endswith(taa_marbouta_attached):
        if word in list_t:
            stripped_word = word[:-1]
            last_part = sep_token + word[-1]
            return [stripped_word, last_part]

    return [word]


def merge_tokens(tokens, original_word, sep_token="<+>"):
    """Merge tokenized segments back into a word."""
    parts = []
    sep_len = len(sep_token)
    for tok in tokens:
        if tok == sep_token:
            parts.append("_")
        elif tok.endswith(sep_token):
            tok = tok[:-sep_len]
            parts.append(tok)
        elif tok.startswith(sep_token):
            tok = tok[sep_len:]
            parts.append(tok)
        elif tok.endswith("+"):
            tok = tok[:-1]
            parts.append(tok)
        elif tok.startswith("+"):
            tok = tok[1:]
            parts.append(tok)
        else:
            parts.append(tok)

    merged_word = "".join(parts)
    return merged_word


def split_token_on_t(list_toks, sep_token="<+>"):
    """Split tokens on taa marbouta character."""
    new_list = []
    taa_marbouta_detached = b"\xef\xba\x93"
    taa_marbouta_attached = b"\xd8\xa9"
    haa_attached = b"\xd9\x87"

    for token in list_toks:
        token_bytes = token.encode("utf-8")
        if (
            token_bytes.endswith(taa_marbouta_detached)
            or token_bytes.endswith(taa_marbouta_attached)
            or token_bytes.endswith(haa_attached)
        ):
            if token_bytes == b"\xd9\x87":
                token = sep_token + taa_marbouta_attached.decode("utf-8")
                new_list.append(token)
            else:
                part1 = token[:-1]
                part2 = sep_token + token[-1]
                new_list.append(part1)
                new_list.append(part2)
        else:
            new_list.append(token)

    return new_list


def replace_separator(toks, sep_token="<+>"):
    """Replace + with sep_token in tokens."""
    result = list(toks)

    for i, tok in enumerate(result):
        if tok.startswith("+"):
            result[i] = sep_token + tok[1:]
        if tok.endswith("+"):
            result[i] = tok[:-1] + sep_token
    return result


def morph_tokenize(
    words, disambiguator, list_al_t, list_al, list_t, scheme="d3tok", split=True, sep_token="<+>"
):
    """Generate morphological tokens for a list of words."""
    disambig_words = disambiguator.disambiguate(words)
    result = deque()
    err_disambig = []
    err_camel = []
    has_diacritics_in_par = False

    for original, disambig_word in zip(words, disambig_words, strict=False):
        scored_analyses = disambig_word.analyses
        original_word = original
        dediac_word = dediac_ar(original_word)

        if has_diacritics(original_word):
            has_diacritics_in_par = True

        if not scored_analyses:
            result.append(original_word)
            continue

        analysis = scored_analyses[0].analysis
        tok = dediac_ar(analysis.get(scheme, None))
        tok_bw = dediac_ar(analysis.get("bwtok", None))
        seg_d3 = dediac_ar(analysis.get("d3seg", None))

        taa_marbouta_detached = b"\xef\xba\x93"
        taa_marbouta_attached = b"\xd8\xa9"
        original_word_bytes = dediac_word.encode("utf-8")

        if original_word_bytes.endswith(taa_marbouta_attached) or original_word_bytes.endswith(
            taa_marbouta_detached
        ):
            if "+ة_+" in tok_bw or "+ه" in tok_bw or "+ة" in tok_bw:
                toks = tok.split("_")
                toks = split_token_on_t(toks, sep_token)
                toks = replace_separator(toks, sep_token)
                toks = merge_alef_and_alef_lam(toks, sep_token)
                merged_toks = dediac_ar(merge_tokens(toks, dediac_word, sep_token))

                d3_seg_tok = seg_d3.split("_")
                d3_seg_tok = split_token_on_t(d3_seg_tok, sep_token)
                d3_seg_tok = replace_separator(d3_seg_tok, sep_token)
                d3_seg_tok = merge_alef_and_alef_lam(d3_seg_tok, sep_token)
                merged_toks_seg = dediac_ar(merge_tokens(d3_seg_tok, dediac_word, sep_token))

                bw_toks = tok_bw.split("_")
                bw_toks = split_token_on_t(bw_toks, sep_token)
                bw_toks = replace_separator(bw_toks, sep_token)
                bw_toks = merge_alef_and_alef_lam(bw_toks, sep_token)
                merged_toks_bw = dediac_ar(merge_tokens(bw_toks, dediac_word, sep_token))

                if merged_toks == dediac_word and len(toks) > 1:
                    if has_diacritics(original):
                        toks = apply_diacritics_to_segments_keep_markers(toks, original, sep_token)
                    result.extend(toks)
                    continue

                elif merged_toks_seg == dediac_word and len(d3_seg_tok) > 1:
                    if has_diacritics(original):
                        d3_seg_tok = apply_diacritics_to_segments_keep_markers(
                            d3_seg_tok, original, sep_token
                        )
                    result.extend(d3_seg_tok)
                    continue

                elif merged_toks_bw == dediac_word and len(bw_toks) > 1:
                    if has_diacritics(original):
                        bw_toks = apply_diacritics_to_segments_keep_markers(
                            bw_toks, original, sep_token
                        )
                    result.extend(bw_toks)
                    continue

                else:
                    result.append(original_word)
                    err_disambig.append(dediac_word)
                    err_camel.append(merged_toks)
                    continue

        if tok is None or "NOAN" in tok:
            tok = process_NOAN_word(list_al_t, list_al, list_t, dediac_word, sep_token)
            if has_diacritics(original):
                toks = apply_diacritics_to_segments_keep_markers(tok, original, sep_token)
            else:
                toks = tok
            result.extend(toks)

        elif split:
            tok = dediac_ar(tok)
            toks = tok.split("_")
            toks = replace_separator(toks, sep_token)
            toks = merge_alef_and_alef_lam(toks, sep_token)
            merged_toks = dediac_ar(merge_tokens(toks, dediac_word, sep_token))

            bw_toks = tok_bw.split("_")
            bw_toks = replace_separator(bw_toks, sep_token)
            bw_toks = merge_alef_and_alef_lam(bw_toks, sep_token)
            merged_toks_bw = dediac_ar(merge_tokens(bw_toks, dediac_word, sep_token))

            d3_seg_tok = seg_d3.split("_")
            d3_seg_tok = replace_separator(d3_seg_tok, sep_token)
            d3_seg_tok = merge_alef_and_alef_lam(d3_seg_tok, sep_token)
            merged_toks_seg = dediac_ar(merge_tokens(d3_seg_tok, dediac_word, sep_token))

            if merged_toks == dediac_word and len(toks) > 1:
                if has_diacritics(original):
                    toks = apply_diacritics_to_segments_keep_markers(toks, original, sep_token)
                result.extend(toks)
            elif merged_toks_seg == dediac_word and len(d3_seg_tok) > 1:
                if has_diacritics(original):
                    d3_seg_tok = apply_diacritics_to_segments_keep_markers(
                        d3_seg_tok, original, sep_token
                    )
                result.extend(d3_seg_tok)
            elif merged_toks_bw == dediac_word and len(bw_toks) > 1:
                if has_diacritics(original):
                    bw_toks = apply_diacritics_to_segments_keep_markers(
                        bw_toks, original, sep_token
                    )
                result.extend(bw_toks)
            else:
                result.append(original_word)
                err_disambig.append(dediac_word)
                err_camel.append(merged_toks)

        else:
            tok = dediac_ar(tok)
            if tok == dediac_word:
                result.append(original_word)
            else:
                result.append(original_word)
                err_disambig.append(dediac_word)
                err_camel.append(tok)

    return list(result), err_disambig, err_camel, has_diacritics_in_par


def stem_dataset(
    dataset: Dataset,
    column: str = "text",
    sep_token: str = "<+>",
    normalize: bool = False,
    keep_diacritics: bool = True,
    num_proc: int | None = None,
    model: str = "mle",
    use_gpu: bool = False,
) -> Dataset:
    """
    Apply stemming and morphological analysis to dataset.

    Args:
        dataset: HuggingFace dataset
        column: Column to process
        sep_token: Separator token for morphological splits (default: '<+>')
        normalize: Apply Arabic normalization (default: False)
        keep_diacritics: Keep dediacritized column (default: True)
        num_proc: Number of parallel processes
        model: Disambiguator model to use - "mle" or "bert" (default: "mle")
        use_gpu: Whether to use GPU for BERT model (default: False)

    Returns:
        Dataset with {column}_stemmed and optionally {column}_dediac columns

    Example:
        >>> # Stem with defaults (MLE, keeps diacritics)
        >>> stemmed = stem_dataset(dataset)
        >>> # Result has 'text_stemmed' and 'text_dediac' columns

        >>> # Stem using BERT with GPU
        >>> stemmed = stem_dataset(dataset, model="bert", use_gpu=True)

        >>> # Stem without keeping diacritics
        >>> stemmed = stem_dataset(dataset, keep_diacritics=False)
        >>> # Result has only 'text_stemmed' column
    """
    model = model.lower()
    if model not in ["mle", "bert"]:
        raise ValueError(f"Invalid model '{model}'. Must be 'mle' or 'bert'")

    logger.info(f"Starting stemming of {len(dataset)} examples")
    logger.info(
        f"Model: {model.upper()}, Column: {column}, Sep token: {sep_token}, Normalize: {normalize}"
    )
    logger.info(f"Keep diacritics: {keep_diacritics}, Workers: {num_proc or 'auto'}")
    if model == "bert":
        logger.info(f"GPU: {use_gpu}")

    logger.info("Checking CAMeL Tools data packages...")
    catalogue = Catalogue.load_catalogue()
    try:
        catalogue.download_package("morphology-db-msa-r13")
        catalogue.download_package("disambig-mle-calima-msa-r13")
        logger.info("msa-r13 packages installed")
    except Exception as e:
        logger.warning(f"Package installation warning: {e}")

    if model == "bert":
        try:
            catalogue.download_package("disambig-bert-unfactored-all")
            logger.info("BERT package installed")
        except Exception as e:
            logger.warning(f"BERT package installation warning: {e}")

    logger.info("CAMeL Tools data packages ready")

    logger.info("Loading additional words lists...")
    words_dir = os.path.join(os.path.dirname(__file__), "data")
    list_al_t = set(read_and_dediacritize(os.path.join(words_dir, "words_al_t.txt")))
    list_al = set(read_and_dediacritize(os.path.join(words_dir, "words_al.txt")))
    list_t = set(read_and_dediacritize(os.path.join(words_dir, "words_t.txt")))
    logger.info("Loaded word list entries")

    logger.info(f"Initializing {model.upper()} disambiguator...")
    if model == "mle":
        disambiguator = MLEDisambiguator.pretrained("calima-msa-r13", cache_size=1_000_000)
    else:  # bert
        disambiguator = BERTUnfactoredDisambiguator.pretrained(use_gpu=use_gpu)
    logger.info("Disambiguator ready")

    def new_scored_analysis(self, word_dd):
        if word_dd in self._cache:
            return self._cache[word_dd]
        result = self._scored_analyses(word_dd)
        self._cache[word_dd] = result
        return result

    disambiguator._scored_analyses_cached = MethodType(new_scored_analysis, disambiguator)
    disambiguator._score_fn = disambiguator._scored_analyses_cached

    def process_row(row):
        text = row.get(column, "")
        if not text:
            row[f"{column}_stemmed"] = ""
            if keep_diacritics:
                row[f"{column}_dediac"] = ""
            return row

        word_list = tokenize(text)
        if word_list is None:
            row[f"{column}_stemmed"] = text
            if keep_diacritics:
                row[f"{column}_dediac"] = dediac_ar(text)
            return row

        tokenized, _, _, has_diacs = morph_tokenize(
            word_list, disambiguator, list_al_t, list_al, list_t, sep_token=sep_token
        )

        if tokenized is not None:
            tokenized = merge_alef_and_alef_lam(tokenized, sep_token)
            stemmed = "".join(tokenized)

            if normalize:
                stemmed = normalize_arabic(stemmed)

            row[f"{column}_stemmed"] = stemmed

            if keep_diacritics:
                row[f"{column}_dediac"] = dediac_ar(stemmed)
        else:
            row[f"{column}_stemmed"] = text
            if keep_diacritics:
                row[f"{column}_dediac"] = dediac_ar(text)

        return row

    logger.info("Starting morphological tokenization...")
    result = dataset.map(process_row, num_proc=num_proc, desc="Stemming")

    logger.info(f"Stemming complete! Processed {len(result)} examples")
    return result


def stem(
    text: str | list[str],
    sep_token: str = "<+>",
    normalize: bool = False,
    keep_diacritics: bool = False,
    model: str = "mle",
    use_gpu: bool = False,
) -> str | list[str]:
    """
    Stem Arabic text or list of texts.

    Args:
        text: Single string or list of strings to stem
        sep_token: Separator token for morphological splits (default: '<+>')
        normalize: Apply Arabic normalization (default: False)
        keep_diacritics: Keep diacritics in output (default: False)
        model: Disambiguator model to use - "mle" or "bert" (default: "mle")
        use_gpu: Whether to use GPU for BERT model (default: False)

    Returns:
        Stemmed text in the same format as input (string or list of strings)

    Example:
        >>> # Stem a single string
        >>> stemmed = stem("النص العربي")
        >>> # Returns: "ال<+>نص ال<+>عربي"

        >>> # Stem a list of strings
        >>> stemmed = stem(["النص العربي", "مثال آخر"])
        >>> # Returns: ["ال<+>نص ال<+>عربي", "مثال آخر"]

        >>> # Stem with BERT model and GPU
        >>> stemmed = stem("النص", model="bert", use_gpu=True)
    """
    # Validate model parameter
    model = model.lower()
    if model not in ["mle", "bert"]:
        raise ValueError(f"Invalid model '{model}'. Must be 'mle' or 'bert'")

    # Track whether input was a single string
    is_single_string = isinstance(text, str)

    # Convert single string to list for uniform processing
    text_list = [text] if is_single_string else text

    # Validate all items are strings
    if not all(isinstance(t, str) for t in text_list):
        raise TypeError("All items in text list must be strings")

    logger.info(f"Initializing {model.upper()} disambiguator...")
    catalogue = Catalogue.load_catalogue()
    try:
        catalogue.download_package("morphology-db-msa-r13")
        catalogue.download_package("disambig-mle-calima-msa-r13")
        logger.info("msa-r13 packages installed")
    except Exception as e:
        logger.warning(f"Package installation warning: {e}")

    if model == "bert":
        try:
            catalogue.download_package("disambig-bert-unfactored-all")
            logger.info("BERT package installed")
        except Exception as e:
            logger.warning(f"BERT package installation warning: {e}")

    if model == "mle":
        disambiguator = MLEDisambiguator.pretrained("calima-msa-r13", cache_size=1_000_000)
    else:  # bert
        disambiguator = BERTUnfactoredDisambiguator.pretrained(use_gpu=use_gpu)

    # Add caching to disambiguator
    def new_scored_analysis(self, word_dd):
        if word_dd in self._cache:
            return self._cache[word_dd]
        result = self._scored_analyses(word_dd)
        self._cache[word_dd] = result
        return result

    disambiguator._scored_analyses_cached = MethodType(new_scored_analysis, disambiguator)
    disambiguator._score_fn = disambiguator._scored_analyses_cached

    # Load word lists
    words_dir = os.path.join(os.path.dirname(__file__), "data")
    list_al_t = set(read_and_dediacritize(os.path.join(words_dir, "words_al_t.txt")))
    list_al = set(read_and_dediacritize(os.path.join(words_dir, "words_al.txt")))
    list_t = set(read_and_dediacritize(os.path.join(words_dir, "words_t.txt")))

    # Process each text
    results = []
    for txt in text_list:
        if not txt:
            results.append("")
            continue

        word_list = tokenize(txt)
        if word_list is None:
            stemmed = dediac_ar(txt) if not keep_diacritics else txt
            results.append(stemmed)
            continue

        tokenized, _, _, has_diacs = morph_tokenize(
            word_list, disambiguator, list_al_t, list_al, list_t, sep_token=sep_token
        )

        if tokenized is not None:
            tokenized = merge_alef_and_alef_lam(tokenized, sep_token)
            stemmed = "".join(tokenized)

            if normalize:
                stemmed = normalize_arabic(stemmed)

            if not keep_diacritics:
                stemmed = dediac_ar(stemmed)

            results.append(stemmed)
        else:
            stemmed = dediac_ar(txt) if not keep_diacritics else txt
            results.append(stemmed)

    # Return in the same format as input
    return results[0] if is_single_string else results
