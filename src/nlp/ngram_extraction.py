"""N-gram extraction utilities for text analysis."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

# Preprocessing patterns
PUNCTUATION_PATTERN = re.compile(r"[^\w\s\-]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def preprocess_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_punctuation: bool = True,
) -> str:
    """Preprocess text for n-gram extraction.

    Args:
        text: Input text.
        lowercase: Convert to lowercase.
        remove_punctuation: Remove punctuation marks.

    Returns:
        Preprocessed text.
    """
    if not text:
        return ""

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = PUNCTUATION_PATTERN.sub(" ", text)

    # Normalize whitespace
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


def tokenize(text: str) -> List[str]:
    """Tokenize text into words.

    Args:
        text: Preprocessed text.

    Returns:
        List of word tokens.
    """
    if not text:
        return []
    return text.split()


def extract_ngrams(
    tokens: List[str],
    n: int,
) -> List[Tuple[str, ...]]:
    """Extract n-grams from a list of tokens.

    Args:
        tokens: List of word tokens.
        n: N-gram size (1=unigram, 2=bigram, etc).

    Returns:
        List of n-gram tuples.
    """
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def ngram_to_string(ngram: Tuple[str, ...]) -> str:
    """Convert n-gram tuple to space-separated string."""
    return " ".join(ngram)


def extract_all_ngrams(
    text: str,
    *,
    n_min: int = 1,
    n_max: int = 3,
    preprocess: bool = True,
    remove_punctuation: bool = True,
) -> Dict[str, Dict[str, int | str]]:
    """Extract all n-grams from text with frequency counts.

    Args:
        text: Input text.
        n_min: Minimum n-gram size.
        n_max: Maximum n-gram size.
        preprocess: Apply preprocessing.
        remove_punctuation: Remove punctuation during preprocessing.

    Returns:
        Dict mapping ngram string to {'frequency': count, 'ngram_length': n}.
    """
    if preprocess:
        text = preprocess_text(text, remove_punctuation=remove_punctuation)

    tokens = tokenize(text)

    result: Dict[str, Dict[str, int | str]] = {}

    for n in range(n_min, n_max + 1):
        ngrams = extract_ngrams(tokens, n)
        for ngram in ngrams:
            ngram_str = ngram_to_string(ngram)
            if ngram_str not in result:
                result[ngram_str] = {"frequency": 0, "ngram_length": n}
            result[ngram_str]["frequency"] += 1

    return result


def extract_ngrams_from_corpus(
    texts: List[str],
    *,
    n_min: int = 1,
    n_max: int = 3,
    preprocess: bool = True,
    remove_punctuation: bool = True,
    min_frequency: int = 1,
) -> List[Dict]:
    """Extract n-grams from a corpus of texts.

    Args:
        texts: List of text documents.
        n_min: Minimum n-gram size.
        n_max: Maximum n-gram size.
        preprocess: Apply preprocessing.
        remove_punctuation: Remove punctuation.
        min_frequency: Minimum frequency to include.

    Returns:
        List of dicts with 'ngram', 'frequency', 'ngram_length', 'document_frequency'.
    """
    # Count n-gram frequencies across corpus
    ngram_counts: Counter = Counter()
    ngram_doc_counts: Counter = Counter()  # How many docs contain each ngram
    ngram_lengths: Dict[str, int] = {}

    for text in texts:
        if preprocess:
            text = preprocess_text(text, remove_punctuation=remove_punctuation)

        tokens = tokenize(text)
        doc_ngrams: Set[str] = set()

        for n in range(n_min, n_max + 1):
            ngrams = extract_ngrams(tokens, n)
            for ngram in ngrams:
                ngram_str = ngram_to_string(ngram)
                ngram_counts[ngram_str] += 1
                ngram_lengths[ngram_str] = n
                doc_ngrams.add(ngram_str)

        # Count document frequency
        for ngram_str in doc_ngrams:
            ngram_doc_counts[ngram_str] += 1

    # Build result list
    result = []
    for ngram_str, freq in ngram_counts.items():
        if freq >= min_frequency:
            result.append({
                "ngram": ngram_str,
                "frequency": freq,
                "ngram_length": ngram_lengths[ngram_str],
                "document_frequency": ngram_doc_counts[ngram_str],
            })

    return result


def count_unigrams(texts: List[str], *, preprocess: bool = True) -> Counter:
    """Count unigram frequencies across a corpus.

    Args:
        texts: List of text documents.
        preprocess: Apply preprocessing.

    Returns:
        Counter of word -> frequency.
    """
    counts: Counter = Counter()
    for text in texts:
        if preprocess:
            text = preprocess_text(text)
        tokens = tokenize(text)
        counts.update(tokens)
    return counts


def filter_ngrams_by_words(
    ngrams: List[Dict],
    allowed_words: Optional[Set[str]] = None,
    forbidden_words: Optional[Set[str]] = None,
) -> List[Dict]:
    """Filter n-grams by allowed/forbidden word lists.

    Args:
        ngrams: List of n-gram dicts.
        allowed_words: If provided, only keep n-grams where all words are in this set.
        forbidden_words: If provided, remove n-grams containing any of these words.

    Returns:
        Filtered list of n-gram dicts.
    """
    result = []
    for ng in ngrams:
        words = ng["ngram"].split()

        if forbidden_words:
            if any(w in forbidden_words for w in words):
                continue

        if allowed_words:
            if not all(w in allowed_words for w in words):
                continue

        result.append(ng)

    return result
