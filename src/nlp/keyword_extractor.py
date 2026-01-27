"""Keyword extraction using KeyBERT (semantic) and TF-IDF (corpus-level)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def extract_keywords_keybert(
    text: str,
    top_n: int = 15,
    *,
    keyphrase_ngram_range: Tuple[int, int] = (1, 2),
    diversity: float = 0.5,
) -> List[Tuple[str, float]]:
    """Extract keywords from a single document using KeyBERT.

    Args:
        text: Input text.
        top_n: Number of keywords to extract.
        keyphrase_ngram_range: N-gram range for keyphrases.
        diversity: MMR diversity parameter (0=no diversity, 1=max).

    Returns:
        List of (keyword, score) tuples sorted by relevance.
    """
    try:
        from keybert import KeyBERT
    except ImportError:
        # Fallback to simple TF-IDF if KeyBERT not installed
        return _fallback_keywords(text, top_n)

    kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words="english",
        top_n=top_n,
        use_mmr=True,
        diversity=diversity,
    )
    return keywords


def extract_keywords_tfidf(
    documents: List[str],
    top_n: int = 20,
    *,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> List[List[Tuple[str, float]]]:
    """Extract discriminative keywords per document using TF-IDF.

    Args:
        documents: List of document texts.
        top_n: Number of keywords per document.
        max_features: Maximum vocabulary size.
        ngram_range: N-gram range.

    Returns:
        List of keyword lists, one per document. Each is [(keyword, tfidf_score), ...].
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not documents:
        return []

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        lowercase=True,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    results: List[List[Tuple[str, float]]] = []
    for i in range(tfidf_matrix.shape[0]):
        row = np.asarray(tfidf_matrix[i].todense()).ravel()
        top_indices = np.argsort(row)[::-1][:top_n]
        keywords = [(str(feature_names[j]), float(row[j])) for j in top_indices if row[j] > 0]
        results.append(keywords)

    return results


def compute_keyword_aspect_mi(
    keywords_per_job: List[List[str]],
    aspects_per_job: List[Dict[str, List[str]]],
    aspect_name: str,
) -> Dict[str, float]:
    """Compute mutual information between keywords and a specific aspect.

    Args:
        keywords_per_job: Keywords extracted per job.
        aspects_per_job: Aspects dict per job.
        aspect_name: Which aspect to compute MI for.

    Returns:
        Dict mapping keyword -> MI score with the aspect.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_selection import mutual_info_classif

    n = len(keywords_per_job)
    if n == 0:
        return {}

    # Build keyword documents
    kw_docs = [" ".join(kws) for kws in keywords_per_job]

    # Build binary target: does job have this aspect?
    y = np.array([1 if aspects_per_job[i].get(aspect_name) else 0 for i in range(n)])

    if y.sum() == 0 or y.sum() == n:
        return {}

    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform(kw_docs)
    feature_names = vectorizer.get_feature_names_out()

    mi = mutual_info_classif(X, y, discrete_features=True, random_state=42)
    return {str(feature_names[i]): float(mi[i]) for i in range(len(feature_names)) if mi[i] > 0}


def _fallback_keywords(text: str, top_n: int) -> List[Tuple[str, float]]:
    """Simple word-frequency fallback when KeyBERT is not available."""
    import re
    from collections import Counter

    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "need", "must",
        "this", "that", "these", "those", "we", "you", "they", "it", "its",
        "our", "your", "their", "as", "if", "not", "no", "so", "up", "out",
        "about", "into", "over", "after", "such", "also", "more", "other",
    }

    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.-]{1,}\b", text.lower())
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    counter = Counter(words)
    total = sum(counter.values()) or 1
    return [(word, count / total) for word, count in counter.most_common(top_n)]
