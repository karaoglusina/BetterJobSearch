"""Job listing and detail endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(tags=["jobs"])


@router.get("/jobs")
async def list_jobs(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    location: Optional[str] = None,
    company: Optional[str] = None,
    title_contains: Optional[str] = None,
) -> Dict[str, Any]:
    """List jobs with filtering and pagination."""
    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="Search artifacts not loaded")

    # Group chunks by job
    jobs: Dict[str, Dict[str, Any]] = {}
    for ch in retriever.chunks:
        jk = ch.get("job_key", "")
        if jk not in jobs:
            meta = ch.get("meta", {})
            jobs[jk] = {
                "job_id": jk,
                "title": meta.get("title", ""),
                "company": meta.get("company", ""),
                "location": meta.get("location", ""),
                "url": meta.get("jobUrl", ""),
                "days_old": meta.get("days_old"),
                "contract_type": meta.get("contractType"),
                "work_type": meta.get("workType"),
                "salary": meta.get("salary"),
                "n_chunks": 0,
            }
        jobs[jk]["n_chunks"] = jobs[jk].get("n_chunks", 0) + 1

    result = list(jobs.values())

    # Filters
    if location:
        loc_lower = location.lower()
        result = [j for j in result if loc_lower in (j.get("location") or "").lower()]
    if company:
        comp_lower = company.lower()
        result = [j for j in result if comp_lower in (j.get("company") or "").lower()]
    if title_contains:
        tc_lower = title_contains.lower()
        result = [j for j in result if tc_lower in (j.get("title") or "").lower()]

    total = len(result)
    paginated = result[skip: skip + limit]

    return {"total": total, "skip": skip, "limit": limit, "jobs": paginated}


@router.get("/jobs/{job_id:path}")
async def get_job(request: Request, job_id: str) -> Dict[str, Any]:
    """Get full job detail including all chunks."""
    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="Search artifacts not loaded")

    chunks = retriever.get_chunks_for_job(job_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    sorted_chunks = sorted(chunks, key=lambda c: c.get("order", 0))
    meta = sorted_chunks[0].get("meta", {})

    return {
        "job_id": job_id,
        "title": meta.get("title", ""),
        "company": meta.get("company", ""),
        "location": meta.get("location", ""),
        "url": meta.get("jobUrl", ""),
        "days_old": meta.get("days_old"),
        "contract_type": meta.get("contractType"),
        "work_type": meta.get("workType"),
        "salary": meta.get("salary"),
        "n_chunks": len(sorted_chunks),
        "sections": list(set(c.get("section", "") for c in sorted_chunks if c.get("section"))),
        "full_text": "\n\n".join(c.get("text", "") for c in sorted_chunks),
        "chunks": [
            {
                "chunk_id": c.get("chunk_id_v2", c.get("chunk_id", "")),
                "text": c.get("text", ""),
                "section": c.get("section"),
                "order": c.get("order", 0),
            }
            for c in sorted_chunks
        ],
    }
