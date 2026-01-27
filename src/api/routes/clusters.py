"""Cluster endpoints: get clusters by aspect, cluster by concept."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["clusters"])


class ConceptClusterRequest(BaseModel):
    concept: str


@router.get("/clusters/{aspect}")
async def get_clusters(request: Request, aspect: str = "default") -> Dict[str, Any]:
    """Get cluster data for a specific aspect.

    For the scatter plot visualization. Returns x, y coordinates
    and cluster assignments for each job.
    """
    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="Search artifacts not loaded")

    from ...clustering_v2.cache import ClusterCache
    cache = ClusterCache()

    # Check cache
    cached = cache.load(aspect)
    if cached is not None:
        return {
            "aspect": aspect,
            "n_jobs": len(cached),
            "data": cached.to_dict(orient="records"),
        }

    # Compute fresh projection
    # Group chunks into jobs and get texts
    job_map: Dict[str, Dict[str, Any]] = {}
    for ch in retriever.chunks:
        jk = ch.get("job_key", "")
        if jk not in job_map:
            meta = ch.get("meta", {})
            job_map[jk] = {
                "job_id": jk,
                "title": meta.get("title", ""),
                "company": meta.get("company", ""),
                "texts": [],
            }
        job_map[jk]["texts"].append(ch.get("text", ""))

    job_ids = list(job_map.keys())
    texts = [" ".join(job_map[jk]["texts"])[:2000] for jk in job_ids]
    titles = [job_map[jk]["title"] for jk in job_ids]

    if len(job_ids) < 2:
        return {"aspect": aspect, "n_jobs": len(job_ids), "data": []}

    # Embed and project
    from ...search.embedder import embed_texts
    from ...clustering_v2.projector import project_umap
    from ...clustering_v2.clusterer import cluster_hdbscan

    embeddings = embed_texts(texts, show_progress=False)
    coords = project_umap(embeddings)
    labels = cluster_hdbscan(coords, min_cluster_size=max(3, len(job_ids) // 20))

    # Cache result
    label_map = {int(l): f"Cluster {l}" for l in set(labels) if l >= 0}
    cache.save(aspect, job_ids, coords[:, 0].tolist(), coords[:, 1].tolist(), labels.tolist(), label_map)

    data = [
        {
            "job_id": job_ids[i],
            "title": titles[i],
            "company": job_map[job_ids[i]]["company"],
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "cluster_id": int(labels[i]),
            "cluster_label": label_map.get(int(labels[i]), "Noise"),
        }
        for i in range(len(job_ids))
    ]

    return {"aspect": aspect, "n_jobs": len(job_ids), "data": data}


@router.post("/clusters/concept")
async def cluster_by_concept(request: Request, body: ConceptClusterRequest) -> Dict[str, Any]:
    """Cluster jobs by a free-text concept."""
    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="Search artifacts not loaded")

    # Group chunks into jobs
    job_map: Dict[str, Dict[str, Any]] = {}
    for ch in retriever.chunks:
        jk = ch.get("job_key", "")
        if jk not in job_map:
            meta = ch.get("meta", {})
            job_map[jk] = {
                "job_id": jk,
                "title": meta.get("title", ""),
                "company": meta.get("company", ""),
                "texts": [],
            }
        job_map[jk]["texts"].append(ch.get("text", ""))

    job_ids = list(job_map.keys())
    texts = [" ".join(job_map[jk]["texts"])[:2000] for jk in job_ids]
    titles = [job_map[jk]["title"] for jk in job_ids]

    if len(job_ids) < 2:
        return {"concept": body.concept, "n_jobs": len(job_ids), "data": []}

    from ...search.embedder import embed_texts
    from ...clustering_v2.aspect_clustering import cluster_by_concept as do_cluster

    embeddings = embed_texts(texts, show_progress=False)
    result = do_cluster(job_ids, embeddings, body.concept, texts, titles)

    data = [
        {
            "job_id": result.job_ids[i],
            "title": titles[i],
            "company": job_map[job_ids[i]]["company"],
            "x": result.x[i],
            "y": result.y[i],
            "cluster_id": result.cluster_ids[i],
            "cluster_label": "",
        }
        for i in range(len(result.job_ids))
    ]

    return {
        "concept": body.concept,
        "n_jobs": len(job_ids),
        "n_clusters": result.n_clusters,
        "data": data,
    }
