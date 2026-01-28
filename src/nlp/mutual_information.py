"""Mutual Information (MI/PMI) calculation for n-gram analysis."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ngram_extraction import (
    count_unigrams,
    extract_ngrams_from_corpus,
    preprocess_text,
    tokenize,
)


def compute_mi(
    ngram_freq: int,
    total_ngrams: int,
    word_freqs: List[int],
    total_unigrams: int,
    *,
    smoothing: float = 0.0,
) -> float:
    """Compute Pointwise Mutual Information (PMI) for an n-gram.

    MI(w1, ..., wn) = log2(P(w1,...,wn) / (P(w1) * ... * P(wn)))

    Args:
        ngram_freq: Frequency of the n-gram.
        total_ngrams: Total n-gram count in corpus.
        word_freqs: Frequencies of each word in the n-gram.
        total_unigrams: Total unigram count in corpus.
        smoothing: Add-k smoothing constant.

    Returns:
        PMI score (log base 2).
    """
    if ngram_freq <= 0 or total_ngrams <= 0 or total_unigrams <= 0:
        return float("-inf")

    if any(f <= 0 for f in word_freqs):
        return float("-inf")

    # P(ngram)
    p_ngram = (ngram_freq + smoothing) / (total_ngrams + smoothing)

    # Product of P(word) for each word
    p_product = 1.0
    for wf in word_freqs:
        p_word = (wf + smoothing) / (total_unigrams + smoothing)
        p_product *= p_word

    if p_product <= 0:
        return float("-inf")

    return math.log2(p_ngram / p_product)


def compute_npmi(
    ngram_freq: int,
    total_ngrams: int,
    word_freqs: List[int],
    total_unigrams: int,
    *,
    smoothing: float = 0.0,
) -> float:
    """Compute Normalized Pointwise Mutual Information (NPMI).

    NPMI = PMI / -log2(P(ngram))

    Returns value in [-1, 1] range:
    - +1: words always co-occur
    - 0: independent
    - -1: never co-occur

    Args:
        ngram_freq: Frequency of the n-gram.
        total_ngrams: Total n-gram count in corpus.
        word_freqs: Frequencies of each word in the n-gram.
        total_unigrams: Total unigram count in corpus.
        smoothing: Add-k smoothing constant.

    Returns:
        NPMI score in [-1, 1].
    """
    pmi = compute_mi(ngram_freq, total_ngrams, word_freqs, total_unigrams, smoothing=smoothing)

    if not math.isfinite(pmi):
        return float("-inf")

    p_ngram = (ngram_freq + smoothing) / (total_ngrams + smoothing)
    if p_ngram <= 0:
        return float("-inf")

    denominator = -math.log2(p_ngram)
    if denominator <= 0:
        return float("-inf")

    return pmi / denominator


def analyze_corpus_mi(
    texts: List[str],
    *,
    n_min: int = 2,
    n_max: int = 3,
    min_frequency: int = 2,
    smoothing: float = 0.0,
) -> List[Dict]:
    """Compute MI scores for all n-grams in a corpus.

    Args:
        texts: List of text documents.
        n_min: Minimum n-gram size (default 2 for bigrams).
        n_max: Maximum n-gram size.
        min_frequency: Minimum n-gram frequency to include.
        smoothing: Add-k smoothing constant.

    Returns:
        List of dicts with 'ngram', 'frequency', 'ngram_length', 'mi', 'npmi'.
    """
    # Extract all n-grams including unigrams for word frequencies
    ngrams = extract_ngrams_from_corpus(
        texts,
        n_min=1,  # Need unigrams for MI calculation
        n_max=n_max,
        min_frequency=1,
    )

    # Build unigram frequency dict
    unigram_freqs: Dict[str, int] = {}
    for ng in ngrams:
        if ng["ngram_length"] == 1:
            unigram_freqs[ng["ngram"]] = ng["frequency"]

    total_unigrams = sum(unigram_freqs.values())

    # Compute MI for n-grams (n >= 2)
    result = []
    for ng in ngrams:
        n = ng["ngram_length"]
        if n < n_min or ng["frequency"] < min_frequency:
            continue

        words = ng["ngram"].split()
        word_freqs = [unigram_freqs.get(w, 0) for w in words]

        # Total n-grams of this length
        total_ngrams = sum(
            x["frequency"] for x in ngrams if x["ngram_length"] == n
        )

        mi = compute_mi(
            ng["frequency"],
            total_ngrams,
            word_freqs,
            total_unigrams,
            smoothing=smoothing,
        )

        npmi = compute_npmi(
            ng["frequency"],
            total_ngrams,
            word_freqs,
            total_unigrams,
            smoothing=smoothing,
        )

        result.append({
            "ngram": ng["ngram"],
            "frequency": ng["frequency"],
            "ngram_length": n,
            "document_frequency": ng.get("document_frequency", 0),
            "mi": mi if math.isfinite(mi) else None,
            "npmi": npmi if math.isfinite(npmi) else None,
        })

    return result


def zscore_normalize(
    values: List[Optional[float]],
) -> List[Optional[float]]:
    """Z-score normalize a list of values.

    Args:
        values: List of numeric values (None values are ignored).

    Returns:
        List of z-scored values (None preserved).
    """
    valid = [v for v in values if v is not None and math.isfinite(v)]
    if len(valid) < 2:
        return values

    mean = np.mean(valid)
    std = np.std(valid)

    if std == 0:
        return [0.0 if v is not None and math.isfinite(v) else None for v in values]

    return [
        ((v - mean) / std) if v is not None and math.isfinite(v) else None
        for v in values
    ]


def normalize_mi_by_length(ngrams: List[Dict]) -> List[Dict]:
    """Z-score normalize MI values within each n-gram length group.

    Args:
        ngrams: List of n-gram dicts with 'mi' and 'ngram_length' fields.

    Returns:
        Same list with 'mi_z' field added.
    """
    # Group by length
    by_length: Dict[int, List[int]] = {}  # length -> indices
    for i, ng in enumerate(ngrams):
        n = ng["ngram_length"]
        by_length.setdefault(n, []).append(i)

    # Normalize within each length group
    for indices in by_length.values():
        mi_values = [ngrams[i].get("mi") for i in indices]
        mi_z = zscore_normalize(mi_values)
        for idx, z in zip(indices, mi_z):
            ngrams[idx]["mi_z"] = z

    return ngrams


def compute_effect_size(
    freq_target: int,
    total_target: int,
    freq_reference: int,
    total_reference: int,
    *,
    smoothing: float = 0.5,
) -> float:
    """Compute effect size (log ratio) between target and reference corpus.

    effect_size = log2((freq_target/total_target) / (freq_reference/total_reference))

    Positive = overrepresented in target
    Negative = overrepresented in reference
    Zero = proportionally similar

    Args:
        freq_target: Frequency in target corpus.
        total_target: Total tokens in target corpus.
        freq_reference: Frequency in reference corpus.
        total_reference: Total tokens in reference corpus.
        smoothing: Add-k smoothing constant.

    Returns:
        Effect size (log2 ratio).
    """
    if total_target <= 0 or total_reference <= 0:
        return 0.0

    rate_target = (freq_target + smoothing) / (total_target + 2 * smoothing)
    rate_reference = (freq_reference + smoothing) / (total_reference + 2 * smoothing)

    if rate_reference <= 0:
        return float("inf") if rate_target > 0 else 0.0

    return math.log2(rate_target / rate_reference)


def g_test(
    freq_target: int,
    total_target: int,
    freq_reference: int,
    total_reference: int,
) -> Tuple[float, float]:
    """Compute G-test (log-likelihood ratio) for corpus comparison.

    Args:
        freq_target: Frequency in target corpus.
        total_target: Total tokens in target corpus.
        freq_reference: Frequency in reference corpus.
        total_reference: Total tokens in reference corpus.

    Returns:
        Tuple of (G statistic, p-value).
    """
    from scipy.stats import chi2

    eps = 1e-12

    k1, n1 = freq_target, total_target
    k2, n2 = freq_reference, total_reference

    p = (k1 + k2) / (n1 + n2)
    p1 = k1 / n1 if n1 > 0 else 0
    p2 = k2 / n2 if n2 > 0 else 0

    def safe_log(x):
        return math.log(max(x, eps))

    # Avoid log(0) issues
    if p <= 0 or p >= 1 or p1 <= 0 or p2 <= 0:
        return 0.0, 1.0

    try:
        g = 2 * (
            k1 * safe_log(p1 / p)
            + (n1 - k1) * safe_log((1 - p1) / (1 - p))
            + k2 * safe_log(p2 / p)
            + (n2 - k2) * safe_log((1 - p2) / (1 - p))
        )
    except (ValueError, ZeroDivisionError):
        return 0.0, 1.0

    if g < 0:
        g = 0.0

    p_value = 1 - chi2.cdf(g, df=1)

    return g, p_value


def compare_with_reference(
    target_ngrams: List[Dict],
    reference_ngrams: List[Dict],
    total_target: int,
    total_reference: int,
) -> List[Dict]:
    """Compare n-gram frequencies between target and reference corpus.

    Args:
        target_ngrams: N-grams from target corpus with 'ngram', 'frequency'.
        reference_ngrams: N-grams from reference corpus.
        total_target: Total tokens in target corpus.
        total_reference: Total tokens in reference corpus.

    Returns:
        Target n-grams with added 'effect_size', 'g_stat', 'p_value', 'significant'.
    """
    # Build reference frequency lookup
    ref_freq: Dict[str, int] = {
        ng["ngram"]: ng["frequency"] for ng in reference_ngrams
    }

    for ng in target_ngrams:
        ngram = ng["ngram"]
        freq_target = ng["frequency"]
        freq_reference = ref_freq.get(ngram, 0)

        ng["freq_reference"] = freq_reference
        ng["effect_size"] = compute_effect_size(
            freq_target, total_target,
            freq_reference, total_reference,
        )

        g_stat, p_value = g_test(
            freq_target, total_target,
            freq_reference, total_reference,
        )
        ng["g_stat"] = g_stat
        ng["p_value"] = p_value
        ng["significant"] = p_value < 0.05

    return target_ngrams


def filter_meaningful_phrases(
    ngrams: List[Dict],
    *,
    min_mi_z: float = 0.0,
    min_effect_size: float = 0.0,
    require_significant: bool = True,
    min_frequency: int = 3,
    min_doc_frequency: int = 2,
) -> List[Dict]:
    """Filter n-grams to meaningful phrases based on MI and domain specificity.

    Args:
        ngrams: List of n-gram dicts with MI and effect size scores.
        min_mi_z: Minimum z-scored MI to include.
        min_effect_size: Minimum effect size (log ratio) to include.
        require_significant: Only include statistically significant phrases.
        min_frequency: Minimum raw frequency.
        min_doc_frequency: Minimum document frequency.

    Returns:
        Filtered list of meaningful phrases.
    """
    result = []
    for ng in ngrams:
        # Skip unigrams
        if ng.get("ngram_length", 1) < 2:
            continue

        # Frequency filters
        if ng.get("frequency", 0) < min_frequency:
            continue
        if ng.get("document_frequency", 0) < min_doc_frequency:
            continue

        # MI filter
        mi_z = ng.get("mi_z")
        if mi_z is not None and mi_z < min_mi_z:
            continue

        # Effect size filter
        effect_size = ng.get("effect_size")
        if effect_size is not None and effect_size < min_effect_size:
            continue

        # Significance filter
        if require_significant and not ng.get("significant", True):
            continue

        result.append(ng)

    return result
