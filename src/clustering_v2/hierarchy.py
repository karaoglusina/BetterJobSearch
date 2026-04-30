"""Sub-clustering within HDBSCAN top clusters, plus level-1 c-TF-IDF labels."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .labeler import compute_ctfidf


def _sub_partition(xy: np.ndarray, max_k: int = 4) -> np.ndarray:
    """Partition a cluster's 2D points into 2-max_k sub-clusters with KMeans.

    Chooses k by size so tiny clusters get k=2 and larger ones get more sub-topics.
    Returns sub-cluster ids in [0, k).
    """
    n = xy.shape[0]
    if n < 20:
        return np.zeros(n, dtype=int)

    from sklearn.cluster import KMeans

    if n < 40:
        k = 2
    elif n < 100:
        k = 3
    else:
        k = min(max_k, 4)

    try:
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        return km.fit_predict(xy).astype(int)
    except Exception:
        return np.zeros(n, dtype=int)


def build_hierarchical_labels(
    *,
    x: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray,
    docs_per_cluster: Dict[int, List[str]],
    level0_labels: Dict[int, str],
    meaningful_phrases: Optional[List[str]] = None,
    min_sub_points: int = 20,
) -> List[Dict[str, Any]]:
    """Compute topic labels at two levels for Atlas-style zoom-aware rendering.

    Level 0: one record per top-level cluster at its centroid. Uses the existing
    `level0_labels` map (produced by compute_ctfidf in the regular pipeline).

    Level 1: sub-clusters inside each top cluster, KMeans-partitioned in 2D,
    relabeled with a fresh c-TF-IDF pass scoped to that cluster's documents.
    Only top clusters with >= `min_sub_points` points get sub-labels.

    Args:
        x, y: Full 2D coordinate arrays.
        cluster_ids: Top-level HDBSCAN labels (-1 = noise).
        docs_per_cluster: Short doc string per job, grouped by top cluster_id.
        level0_labels: cluster_id -> label for level 0 (pre-computed).
        meaningful_phrases: Optional whitelist passed to c-TF-IDF.
        min_sub_points: Minimum points in a top cluster to bother splitting.

    Returns:
        List of {text, x, y, level, priority, cluster_id, sub_id} records.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cluster_ids = np.asarray(cluster_ids)

    results: List[Dict[str, Any]] = []

    unique = sorted(int(c) for c in set(cluster_ids.tolist()) if c >= 0)

    for cid in unique:
        mask = cluster_ids == cid
        n = int(mask.sum())
        if n == 0:
            continue

        cx = float(x[mask].mean())
        cy = float(y[mask].mean())
        label = level0_labels.get(cid) or f"Cluster {cid}"

        results.append({
            "text": label,
            "x": cx,
            "y": cy,
            "level": 0,
            "priority": n,
            "cluster_id": cid,
            "sub_id": -1,
        })

        if n < min_sub_points:
            continue

        docs = docs_per_cluster.get(cid, [])
        if len(docs) < min_sub_points:
            continue

        xy = np.column_stack([x[mask], y[mask]])
        sub_ids = _sub_partition(xy)
        unique_subs = sorted(set(int(s) for s in sub_ids.tolist()))
        if len(unique_subs) < 2:
            continue

        sub_docs: Dict[int, List[str]] = {}
        for i, sid in enumerate(sub_ids.tolist()):
            if i < len(docs):
                sub_docs.setdefault(int(sid), []).append(docs[i])

        try:
            sub_kw = compute_ctfidf(
                sub_docs,
                top_n=5,
                meaningful_phrases=meaningful_phrases,
            )
        except Exception:
            sub_kw = {}

        for sid in unique_subs:
            sub_mask = np.zeros(cluster_ids.shape[0], dtype=bool)
            sub_mask_indices = np.where(mask)[0][sub_ids == sid]
            sub_mask[sub_mask_indices] = True
            sub_n = int(sub_mask.sum())
            if sub_n == 0:
                continue

            sx = float(x[sub_mask].mean())
            sy = float(y[sub_mask].mean())

            kws = sub_kw.get(sid, [])
            if not kws:
                continue
            sub_label = ", ".join(kws[:2])

            # Skip level-1 labels that are identical to the parent level-0 label
            if sub_label.strip().lower() == label.strip().lower():
                continue

            results.append({
                "text": sub_label,
                "x": sx,
                "y": sy,
                "level": 1,
                "priority": sub_n,
                "cluster_id": cid,
                "sub_id": int(sid),
            })

    return results
