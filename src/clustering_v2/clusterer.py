"""HDBSCAN clustering (replaces KMeans)."""

from __future__ import annotations

from typing import Optional

import numpy as np


def cluster_hdbscan(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
) -> np.ndarray:
    """Cluster embeddings using HDBSCAN.

    HDBSCAN advantages over KMeans:
    - No need to specify k upfront
    - Detects arbitrary-shaped clusters
    - Identifies noise points (label=-1)

    Args:
        embeddings: Input array (n_samples, n_features). Typically 2D UMAP coords
                   or high-dimensional embeddings.
        min_cluster_size: Minimum cluster size.
        min_samples: Minimum samples for core points.
        metric: Distance metric.
        cluster_selection_method: "eom" (excess of mass) or "leaf".

    Returns:
        Cluster labels array (n_samples,). -1 indicates noise.
    """
    try:
        import hdbscan
    except ImportError:
        # Fallback to KMeans
        return _fallback_kmeans(embeddings, min_cluster_size)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )
    labels = clusterer.fit_predict(embeddings)
    return np.asarray(labels)


def _fallback_kmeans(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Fallback KMeans clustering when HDBSCAN is not available."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = embeddings.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)

    # Auto-determine k
    max_k = min(12, max(2, n // min_cluster_size))
    min_k = 2
    best_k = min_k
    best_score = -1.0

    for k in range(min_k, max_k + 1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(embeddings)
            if len(set(labels)) <= 1:
                continue
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    return km.fit_predict(embeddings)
