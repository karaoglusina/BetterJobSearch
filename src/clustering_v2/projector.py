"""UMAP 2D projection (replaces TruncatedSVD)."""

from __future__ import annotations

from typing import Optional

import numpy as np


def project_umap(
    embeddings: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Project high-dimensional embeddings to 2D using UMAP.

    Args:
        embeddings: Input array of shape (n_samples, n_features).
        n_neighbors: UMAP locality parameter.
        min_dist: Minimum distance between points in 2D.
        metric: Distance metric.
        n_components: Output dimensions (default 2 for scatter).
        random_state: Random seed for reproducibility.

    Returns:
        numpy array of shape (n_samples, n_components).
    """
    try:
        import umap
    except ImportError:
        # Fallback to TruncatedSVD if umap not installed
        from sklearn.decomposition import TruncatedSVD
        n_comp = min(n_components, embeddings.shape[1] - 1, embeddings.shape[0] - 1)
        if n_comp < 1:
            return np.zeros((embeddings.shape[0], n_components))
        svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
        result = svd.fit_transform(embeddings)
        if result.shape[1] < n_components:
            pad = np.zeros((result.shape[0], n_components - result.shape[1]))
            result = np.hstack([result, pad])
        return result

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=n_components,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def project_umap_transform(
    train_embeddings: np.ndarray,
    new_embeddings: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit UMAP on train data and transform new data into same space.

    Returns:
        Tuple of (train_2d, new_2d).
    """
    try:
        import umap
    except ImportError:
        train_2d = project_umap(train_embeddings, random_state=random_state)
        new_2d = project_umap(new_embeddings, random_state=random_state)
        return train_2d, new_2d

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=random_state,
    )
    train_2d = reducer.fit_transform(train_embeddings)
    new_2d = reducer.transform(new_embeddings)
    return train_2d, new_2d
