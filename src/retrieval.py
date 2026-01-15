from __future__ import annotations

from typing import Any, Dict, List
import random
import hashlib

from . import rag


def _ensure_loaded() -> None:
    if getattr(rag, "_CACHE", {}).get("chunks"):
        return
    rag.load_cache()


def retrieve(job_id: str, query: str, k: int = 8) -> List[Dict[str, Any]]:
    """Return chunks for a single job_id (mapped to rag.Chunk.job_key).

    Output schema: [{chunk_id, job_id, title, company, text, score?}]
    """
    _ensure_loaded()
    # Use filtered retrieval to constrain to a job_key
    chs = rag.retrieve_filtered(query, where=lambda c: getattr(c, "job_key", None) == job_id, k=k)
    out: List[Dict[str, Any]] = []
    for c in chs:
        out.append(
            {
                "chunk_id": getattr(c, "chunk_id_v2", getattr(c, "chunk_id", "")),
                "job_id": getattr(c, "job_key", ""),
                "title": c.meta.get("title"),
                "company": c.meta.get("company"),
                "text": c.text,
            }
        )
    return out


def retrieve_global(query: str, k: int = 8) -> List[Dict[str, Any]]:
    """Return top-k chunks across jobs.

    Output schema: [{chunk_id, job_id, title, company, text}]
    """
    _ensure_loaded()
    chs = rag.retrieve(query, k=k)
    out: List[Dict[str, Any]] = []
    for c in chs:
        out.append(
            {
                "chunk_id": getattr(c, "chunk_id_v2", getattr(c, "chunk_id", "")),
                "job_id": getattr(c, "job_key", ""),
                "title": c.meta.get("title"),
                "company": c.meta.get("company"),
                "text": c.text,
            }
        )
    return out


def sample_chunks(job_query: str, sample_size: int = 1000, per_job_cap: int = 3, seed: int = 42) -> List[Dict[str, Any]]:
    """Balanced sample combining full-text (chunk) retrieval and title match, then seeded per-job picks.

    Strategy:
      1) Run semantic retrieval over chunks using the query (broad top-K) to collect job_ids by rank
      2) Union with jobs whose titles contain the query
      3) For each job_id in that combined order, pick up to `per_job_cap` chunks at random (seeded)
      4) Backfill with additional retrieved chunks if still below `sample_size`
    """
    _ensure_loaded()
    all_chunks = rag._CACHE.get("chunks") or []
    q = (job_query or "").lower().strip()

    # 1) Semantic retrieval to get a wide set of jobs ranked by relevance
    topk = min(len(all_chunks), max(sample_size * 10, 500))
    retrieved = rag.retrieve(job_query, k=topk)
    ranked_jobs: List[str] = []
    seen_jobs: set[str] = set()
    for c in retrieved:
        jk = getattr(c, "job_key", None)
        if jk and jk not in seen_jobs:
            seen_jobs.add(jk)
            ranked_jobs.append(jk)

    # 2) Title matches (case-insensitive) → add to the pool preserving order of first appearance
    title_jobs: List[str] = []
    seen_title: set[str] = set()
    for c in all_chunks:
        title = (c.meta.get("title") or "").lower()
        jk = getattr(c, "job_key", None)
        if jk and (q and q in title) and jk not in seen_title:
            seen_title.add(jk)
            title_jobs.append(jk)

    # Combined job order: retrieved rank first, then title-matched add-ons
    combined_jobs: List[str] = []
    seen_combined: set[str] = set()
    for jk in ranked_jobs + title_jobs:
        if jk and jk not in seen_combined:
            seen_combined.add(jk)
            combined_jobs.append(jk)

    # Build job -> chunks map preserving chunk order as in artifacts
    by_job: Dict[str, List[Any]] = {}
    for c in all_chunks:
        jk = getattr(c, "job_key", None)
        if not jk:
            continue
        by_job.setdefault(jk, []).append(c)

    # 3) Seeded per-job random pick (deterministic given seed + job_id)
    rows: List[Dict[str, Any]] = []
    for jk in combined_jobs:
        arr = by_job.get(jk, [])
        if not arr:
            continue
        # Stable per-job seed from job_id
        h = hashlib.md5(str(jk).encode("utf-8")).hexdigest()
        job_seed = (int(h[:8], 16) ^ int(seed)) & 0xFFFFFFFF
        rng = random.Random(job_seed)
        if len(arr) <= per_job_cap:
            picks = list(range(len(arr)))
        else:
            picks = sorted(rng.sample(range(len(arr)), per_job_cap))
        for idx in picks:
            c = arr[idx]
            rows.append(
                {
                    "chunk_id": getattr(c, "chunk_id_v2", getattr(c, "chunk_id", "")),
                    "job_id": getattr(c, "job_key", ""),
                    "title": c.meta.get("title"),
                    "company": c.meta.get("company"),
                    "text": c.text,
                }
            )
            if len(rows) >= sample_size:
                return rows

    # 4) Backfill with additional retrieved chunks to hit sample_size
    if len(rows) < sample_size:
        backfill = rag.retrieve(job_query, k=sample_size * 2)
        seen = {r["chunk_id"] for r in rows}
        for c in backfill:
            cid = getattr(c, "chunk_id_v2", getattr(c, "chunk_id", ""))
            if cid in seen:
                continue
            rows.append(
                {
                    "chunk_id": cid,
                    "job_id": getattr(c, "job_key", ""),
                    "title": c.meta.get("title"),
                    "company": c.meta.get("company"),
                    "text": c.text,
                }
            )
            seen.add(cid)
            if len(rows) >= sample_size:
                break

    return rows


__all__ = ["retrieve", "retrieve_global", "sample_chunks"]


