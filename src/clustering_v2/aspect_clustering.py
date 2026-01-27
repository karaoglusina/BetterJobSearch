"""Dynamic re-clustering by aspect or free-text concept."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..models.cluster import ClusterResult, ClusterInfo
from .projector import project_umap
from .clusterer import cluster_hdbscan
from .labeler import label_clusters


def cluster_by_aspect(
    job_ids: List[str],
    aspect_data: Dict[str, Dict[str, List[str]]],
    texts: List[str],
    titles: List[str],
    aspect_name: str,
    *,
    embeddings: Optional[np.ndarray] = None,
    min_cluster_size: int = 5,
) -> ClusterResult:
    """Re-cluster jobs by a specific aspect.

    For structured aspects (skills, tools, language): builds binary feature matrix
    and runs UMAP + HDBSCAN on those features.

    For semantic aspects (domain, culture) or when embeddings are provided:
    uses embeddings directly.

    Args:
        job_ids: List of job IDs.
        aspect_data: Dict of job_id -> {aspect_name: [values]}.
        texts: Job texts (for cluster labeling).
        titles: Job titles (for cluster labeling).
        aspect_name: Which aspect to cluster by.
        embeddings: Optional pre-computed embeddings (n_jobs, dim).
        min_cluster_size: Minimum HDBSCAN cluster size.

    Returns:
        ClusterResult with projection coordinates and cluster assignments.
    """
    n_jobs = len(job_ids)
    if n_jobs < 2:
        return _empty_result(aspect_name, job_ids)

    # Collect all unique values for this aspect across jobs
    all_values: set[str] = set()
    for jid in job_ids:
        vals = aspect_data.get(jid, {}).get(aspect_name, [])
        all_values.update(v.lower() for v in vals)

    if not all_values:
        # No data for this aspect — fall back to embeddings if available
        if embeddings is not None:
            return _cluster_from_embeddings(
                job_ids, embeddings, texts, titles, aspect_name, min_cluster_size
            )
        return _empty_result(aspect_name, job_ids)

    # Build binary feature matrix (job x aspect_value)
    value_list = sorted(all_values)
    value_idx = {v: i for i, v in enumerate(value_list)}
    features = np.zeros((n_jobs, len(value_list)), dtype="float32")

    for i, jid in enumerate(job_ids):
        vals = aspect_data.get(jid, {}).get(aspect_name, [])
        for v in vals:
            idx = value_idx.get(v.lower())
            if idx is not None:
                features[i, idx] = 1.0

    # If feature matrix is too sparse, fall back to embeddings
    nonzero_rows = np.sum(features.any(axis=1))
    if nonzero_rows < max(2, n_jobs * 0.1) and embeddings is not None:
        return _cluster_from_embeddings(
            job_ids, embeddings, texts, titles, aspect_name, min_cluster_size
        )

    return _cluster_from_features(
        job_ids, features, texts, titles, aspect_name, min_cluster_size
    )


def cluster_by_concept(
    job_ids: List[str],
    embeddings: np.ndarray,
    concept: str,
    texts: List[str],
    titles: List[str],
    *,
    min_cluster_size: int = 5,
) -> ClusterResult:
    """Re-cluster by a free-text concept.

    Computes concept embedding, weights job embeddings by cosine similarity
    to the concept, then re-projects with UMAP.

    Args:
        job_ids: List of job IDs.
        embeddings: Job embeddings (n_jobs, dim).
        concept: Free-text concept string.
        texts: Job texts for labeling.
        titles: Job titles for labeling.
        min_cluster_size: Min HDBSCAN cluster size.
    """
    from ..search.embedder import embed_texts

    n_jobs = len(job_ids)
    if n_jobs < 2:
        return _empty_result(concept, job_ids)

    # Compute concept embedding
    concept_emb = embed_texts([concept], show_progress=False)[0]

    # Weight job embeddings by cosine similarity to concept
    # (embeddings are already L2-normalized)
    similarities = embeddings @ concept_emb
    # Softmax-like weighting
    weights = np.exp(similarities * 3)  # temperature=3 for sharper weighting
    weights = weights / (weights.sum() + 1e-8)

    # Weight each dimension
    weighted_emb = embeddings * weights[:, np.newaxis]

    return _cluster_from_embeddings(
        job_ids, weighted_emb, texts, titles, concept, min_cluster_size
    )


def _cluster_from_features(
    job_ids: List[str],
    features: np.ndarray,
    texts: List[str],
    titles: List[str],
    aspect_name: str,
    min_cluster_size: int,
) -> ClusterResult:
    """UMAP + HDBSCAN on a feature matrix."""
    coords_2d = project_umap(features)
    labels = cluster_hdbscan(coords_2d, min_cluster_size=min_cluster_size, min_samples=3)

    cluster_infos = label_clusters(labels, texts, titles)
    noise_count = int(np.sum(labels == -1))

    return ClusterResult(
        aspect=aspect_name,
        n_clusters=len(cluster_infos),
        clusters=cluster_infos,
        job_ids=job_ids,
        x=coords_2d[:, 0].tolist(),
        y=coords_2d[:, 1].tolist(),
        cluster_ids=labels.tolist(),
        noise_count=noise_count,
    )


def _cluster_from_embeddings(
    job_ids: List[str],
    embeddings: np.ndarray,
    texts: List[str],
    titles: List[str],
    aspect_name: str,
    min_cluster_size: int,
) -> ClusterResult:
    """UMAP + HDBSCAN on embeddings."""
    coords_2d = project_umap(embeddings)
    labels = cluster_hdbscan(coords_2d, min_cluster_size=min_cluster_size, min_samples=3)

    cluster_infos = label_clusters(labels, texts, titles)
    noise_count = int(np.sum(labels == -1))

    return ClusterResult(
        aspect=aspect_name,
        n_clusters=len(cluster_infos),
        clusters=cluster_infos,
        job_ids=job_ids,
        x=coords_2d[:, 0].tolist(),
        y=coords_2d[:, 1].tolist(),
        cluster_ids=labels.tolist(),
        noise_count=noise_count,
    )


def _empty_result(aspect_name: str, job_ids: List[str]) -> ClusterResult:
    """Return empty cluster result."""
    return ClusterResult(
        aspect=aspect_name,
        n_clusters=0,
        clusters=[],
        job_ids=job_ids,
        x=[0.0] * len(job_ids),
        y=[0.0] * len(job_ids),
        cluster_ids=[0] * len(job_ids),
        noise_count=0,
    )
