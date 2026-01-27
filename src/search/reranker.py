"""Optional cross-encoder reranking for search results."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

_reranker = None


def _get_reranker():
    """Lazy-load cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        except ImportError:
            return None
    return _reranker


def rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """Rerank chunks using a cross-encoder model.

    Args:
        query: Original search query.
        chunks: Candidate chunks from hybrid search.
        top_k: Number of results to return after reranking.

    Returns:
        Reranked list of chunk dicts.
    """
    model = _get_reranker()
    if model is None or not chunks:
        return chunks[:top_k]

    pairs = [(query, ch.get("text", "")) for ch in chunks]
    scores = model.predict(pairs)

    scored: List[Tuple[float, int]] = [(float(s), i) for i, s in enumerate(scores)]
    scored.sort(key=lambda x: x[0], reverse=True)

    return [chunks[i] for _, i in scored[:top_k]]
