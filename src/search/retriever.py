"""Hybrid search retrieval combining FAISS vector search and BM25.

Extracted and enhanced from rag.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .embedder import embed_query
from .indexer import IndexArtifacts, load_artifacts, simple_tokenize


class HybridRetriever:
    """Hybrid vector + keyword retriever with filtering support.

    Usage:
        retriever = HybridRetriever.from_artifacts()
        results = retriever.search("machine learning", k=8)
        filtered = retriever.search("Python", k=8, where=lambda c: c.get("meta", {}).get("location") == "Amsterdam")
    """

    def __init__(self, artifacts: IndexArtifacts):
        self.chunks = artifacts.chunks
        self.faiss_index = artifacts.faiss_index
        self.bm25 = artifacts.bm25

    @classmethod
    def from_artifacts(cls, artifact_dir: Optional[Path] = None) -> "HybridRetriever":
        """Load from disk artifacts."""
        return cls(load_artifacts(artifact_dir))

    def search(
        self,
        query: str,
        k: int = 8,
        alpha: float = 0.55,
        *,
        where: Optional[Callable[[Dict[str, Any]], bool]] = None,
        oversample: int = 120,
    ) -> List[Dict[str, Any]]:
        """Hybrid search with optional metadata filtering.

        Args:
            query: Search query.
            k: Number of results.
            alpha: Weight for vector vs BM25 (1.0 = vectors only).
            where: Optional filter predicate on chunk dicts.
            oversample: Candidates to fetch before filtering.

        Returns:
            List of chunk dicts, ranked by hybrid score.
        """
        ranked_indices = self._hybrid_search(query, k=max(k, oversample) if where else k, alpha=alpha)

        if where:
            results = [self.chunks[i] for i in ranked_indices if where(self.chunks[i])]
            return results[:k]

        return [self.chunks[i] for i in ranked_indices]

    def search_semantic(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        """Vector-only search."""
        return self.search(query, k=k, alpha=1.0)

    def search_keyword(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        """BM25-only search."""
        return self.search(query, k=k, alpha=0.0)

    def get_chunks_for_job(self, job_key: str) -> List[Dict[str, Any]]:
        """Return all chunks belonging to a specific job."""
        return [c for c in self.chunks if c.get("job_key") == job_key]

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Return all chunks."""
        return self.chunks

    def get_all_job_keys(self) -> List[str]:
        """Return unique job keys in order of first appearance."""
        seen: set[str] = set()
        keys: List[str] = []
        for c in self.chunks:
            jk = c.get("job_key", "")
            if jk and jk not in seen:
                seen.add(jk)
                keys.append(jk)
        return keys

    def _hybrid_search(self, query: str, k: int = 12, alpha: float = 0.55) -> List[int]:
        """Return indices of top chunks using hybrid scoring."""
        # Vector search
        qv = embed_query(query)
        sims, idxs = self.faiss_index.search(qv, min(k, len(self.chunks)))
        vec_scores = sims[0]
        vec_idxs = idxs[0]

        # BM25 search
        bm_scores = self.bm25.get_scores(simple_tokenize(query))

        # Combine candidates
        candidates = set(vec_idxs.tolist()) | set(np.argsort(bm_scores)[-k:].tolist())
        candidates = [c for c in candidates if 0 <= c < len(self.chunks)]

        if not candidates:
            return []

        # Normalize BM25 scores
        bm_sub = np.array([bm_scores[i] for i in candidates], dtype="float32")
        bm_max = bm_sub.max()
        if bm_max > 0:
            bm_sub = bm_sub / (bm_max + 1e-6)

        # Build vector sub-scores
        vec_map = {int(i): float(s) for i, s in zip(vec_idxs, vec_scores)}
        vec_sub = np.array([vec_map.get(int(i), 0.0) for i in candidates], dtype="float32")

        # Hybrid score
        hybrid = alpha * vec_sub + (1 - alpha) * bm_sub
        order = np.argsort(hybrid)[::-1][:k]
        return [int(candidates[i]) for i in order]
