"""Cluster endpoints: get clusters by aspect, cluster by concept."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

router = APIRouter(tags=["clusters"])

# Aspects that use binary feature matrices (from aspects.jsonl)
STRUCTURED_ASPECTS = {"skills", "tools", "language", "remote_policy", "experience", "education", "benefits", "domain", "culture"}


class ConceptClusterRequest(BaseModel):
    concept: str


def _build_job_map(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group chunks into jobs with metadata."""
    job_map: Dict[str, Dict[str, Any]] = {}
    for idx, ch in enumerate(chunks):
        jk = ch.get("job_key", "")
        if not jk:
            continue
        if jk not in job_map:
            meta = ch.get("meta", {})
            job_map[jk] = {
                "job_id": jk,
                "title": meta.get("title", ""),
                "company": meta.get("company", ""),
                "chunk_indices": [],
            }
        job_map[jk]["chunk_indices"].append(idx)
    return job_map


def _get_job_embeddings_from_faiss(faiss_index, job_map: Dict[str, Dict[str, Any]], job_ids: List[str]) -> np.ndarray:
    """Reconstruct per-job embeddings by averaging chunk embeddings from the FAISS index."""
    dim = faiss_index.d
    embeddings = np.zeros((len(job_ids), dim), dtype="float32")

    for i, jk in enumerate(job_ids):
        chunk_indices = job_map[jk]["chunk_indices"]
        chunk_vecs = np.zeros((len(chunk_indices), dim), dtype="float32")
        for j, ci in enumerate(chunk_indices):
            chunk_vecs[j] = faiss_index.reconstruct(int(ci))
        embeddings[i] = chunk_vecs.mean(axis=0)

    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    return embeddings


def _build_aspect_features(
    job_ids: List[str],
    aspect_data: Dict[str, Dict[str, List[str]]],
    aspect_name: str,
) -> np.ndarray | None:
    """Build a binary feature matrix for a structured aspect.

    Returns None if insufficient data for meaningful clustering.
    """
    # Collect all unique values for this aspect across all jobs
    all_values: set[str] = set()
    for jid in job_ids:
        vals = aspect_data.get(jid, {}).get(aspect_name, [])
        all_values.update(v.lower() for v in vals)

    if len(all_values) < 2:
        return None

    value_list = sorted(all_values)
    value_idx = {v: i for i, v in enumerate(value_list)}
    n_jobs = len(job_ids)
    features = np.zeros((n_jobs, len(value_list)), dtype="float32")

    jobs_with_data = 0
    for i, jid in enumerate(job_ids):
        vals = aspect_data.get(jid, {}).get(aspect_name, [])
        if vals:
            jobs_with_data += 1
        for v in vals:
            idx = value_idx.get(v.lower())
            if idx is not None:
                features[i, idx] = 1.0

    # Need at least 10% of jobs to have data for this aspect
    if jobs_with_data < max(2, n_jobs * 0.05):
        return None

    return features


def _generate_cluster_labels(
    chunks: List[Dict[str, Any]],
    job_map: Dict[str, Dict[str, Any]],
    job_ids: List[str],
    labels: np.ndarray,
    titles: List[str],
) -> Dict[int, str]:
    """Generate descriptive cluster labels using c-TF-IDF keywords."""
    from ...clustering_v2.labeler import compute_ctfidf

    # Group chunk texts by cluster
    docs_per_cluster: Dict[int, List[str]] = {}
    titles_per_cluster: Dict[int, List[str]] = {}
    for i, jk in enumerate(job_ids):
        cid = int(labels[i])
        if cid < 0:
            continue
        # Collect first few chunk texts for this job
        chunk_texts = [chunks[ci].get("text", "") for ci in job_map[jk]["chunk_indices"][:3]]
        doc = " ".join(chunk_texts)[:500]
        docs_per_cluster.setdefault(cid, []).append(doc)
        titles_per_cluster.setdefault(cid, []).append(titles[i])

    if not docs_per_cluster:
        return {}

    try:
        keywords = compute_ctfidf(docs_per_cluster, top_n=5)
        return {
            cid: ", ".join(kws[:3]) if kws else f"Cluster {cid}"
            for cid, kws in keywords.items()
        }
    except Exception:
        return {int(l): f"Cluster {l}" for l in set(labels) if l >= 0}


def _compute_clusters(
    chunks: List[Dict[str, Any]],
    faiss_index,
    aspect: str,
    aspect_data: Dict[str, Dict[str, List[str]]],
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute clusters synchronously (runs in thread pool)."""
    from ...clustering_v2.cache import ClusterCache

    # Build a cache key that includes custom params
    has_custom_params = (n_neighbors != 15 or min_dist != 0.1 or min_cluster_size is not None)
    cache_key = aspect
    if has_custom_params:
        cache_key = f"{aspect}_nn{n_neighbors}_md{min_dist}"
        if min_cluster_size is not None:
            cache_key += f"_mc{min_cluster_size}"

    cache = ClusterCache()

    # Check cache first
    cached = cache.load(cache_key)
    if cached is not None:
        records = cached.to_dict(orient="records")
        for r in records:
            r.setdefault("title", "")
            r.setdefault("company", "")
        return {
            "aspect": aspect,
            "n_jobs": len(cached),
            "data": records,
        }

    # Group chunks into jobs
    job_map = _build_job_map(chunks)
    job_ids = list(job_map.keys())
    titles = [job_map[jk]["title"] for jk in job_ids]
    companies = [job_map[jk]["company"] for jk in job_ids]

    if len(job_ids) < 2:
        return {"aspect": aspect, "n_jobs": len(job_ids), "data": []}

    from ...clustering_v2.projector import project_umap
    from ...clustering_v2.clusterer import cluster_hdbscan

    umap_kwargs = {"n_neighbors": n_neighbors, "min_dist": min_dist}

    # For structured aspects, try to use binary feature matrix first
    features = None
    if aspect in STRUCTURED_ASPECTS and aspect_data:
        features = _build_aspect_features(job_ids, aspect_data, aspect)

    if features is not None:
        coords = project_umap(features, **umap_kwargs)
    else:
        embeddings = _get_job_embeddings_from_faiss(faiss_index, job_map, job_ids)
        coords = project_umap(embeddings, **umap_kwargs)

    actual_min_cluster = min_cluster_size if min_cluster_size is not None else max(3, len(job_ids) // 20)
    labels = cluster_hdbscan(coords, min_cluster_size=actual_min_cluster)

    # Generate descriptive labels using c-TF-IDF
    label_map = _generate_cluster_labels(chunks, job_map, job_ids, labels, titles)

    cache.save(
        cache_key, job_ids, coords[:, 0].tolist(), coords[:, 1].tolist(),
        labels.tolist(), label_map, titles=titles, companies=companies,
    )

    data = [
        {
            "job_id": job_ids[i],
            "title": titles[i],
            "company": companies[i],
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "cluster_id": int(labels[i]),
            "cluster_label": label_map.get(int(labels[i]), "Noise"),
        }
        for i in range(len(job_ids))
    ]

    return {"aspect": aspect, "n_jobs": len(job_ids), "data": data}


def _compute_concept_clusters(chunks: List[Dict[str, Any]], faiss_index, concept: str) -> Dict[str, Any]:
    """Compute concept-based clusters synchronously (runs in thread pool)."""
    job_map = _build_job_map(chunks)
    job_ids = list(job_map.keys())
    titles = [job_map[jk]["title"] for jk in job_ids]
    companies = [job_map[jk]["company"] for jk in job_ids]

    if len(job_ids) < 2:
        return {"concept": concept, "n_jobs": len(job_ids), "data": []}

    embeddings = _get_job_embeddings_from_faiss(faiss_index, job_map, job_ids)

    # Build short texts for cluster labeling
    texts = []
    for jk in job_ids:
        chunk_texts = [chunks[ci].get("text", "") for ci in job_map[jk]["chunk_indices"][:3]]
        texts.append(" ".join(chunk_texts)[:500])

    from ...clustering_v2.aspect_clustering import cluster_by_concept as do_cluster
    result = do_cluster(job_ids, embeddings, concept, texts, titles)

    # Generate descriptive labels for concept clusters
    cluster_ids_arr = np.array(result.cluster_ids)
    label_map = _generate_cluster_labels(chunks, job_map, job_ids, cluster_ids_arr, titles)

    data = [
        {
            "job_id": result.job_ids[i],
            "title": titles[i],
            "company": companies[i],
            "x": result.x[i],
            "y": result.y[i],
            "cluster_id": result.cluster_ids[i],
            "cluster_label": label_map.get(result.cluster_ids[i], "Noise"),
        }
        for i in range(len(result.job_ids))
    ]

    return {
        "concept": concept,
        "n_jobs": len(job_ids),
        "n_clusters": result.n_clusters,
        "data": data,
    }


@router.get("/clusters/{aspect}")
async def get_clusters(
    request: Request,
    aspect: str = "default",
    n_neighbors: int = Query(15, ge=2, le=100),
    min_dist: float = Query(0.1, ge=0.0, le=1.0),
    min_cluster_size: Optional[int] = Query(None, ge=2, le=200),
) -> Dict[str, Any]:
    """Get cluster data for a specific aspect."""
    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="Search artifacts not loaded")

    aspect_data = getattr(request.app.state, "aspect_data", {})

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, partial(
            _compute_clusters, retriever.chunks, retriever.faiss_index, aspect, aspect_data,
            n_neighbors=n_neighbors, min_dist=min_dist, min_cluster_size=min_cluster_size,
        )
    )


@router.post("/clusters/concept")
async def cluster_by_concept(request: Request, body: ConceptClusterRequest) -> Dict[str, Any]:
    """Cluster jobs by a free-text concept."""
    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="Search artifacts not loaded")

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, partial(_compute_concept_clusters, retriever.chunks, retriever.faiss_index, body.concept)
    )
