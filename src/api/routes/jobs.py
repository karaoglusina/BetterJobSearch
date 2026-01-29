"""Job listing and detail endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from ...nlp.section_detector import detect_sections_structural

router = APIRouter(tags=["jobs"])


def _remove_chunk_overlap(texts: List[str]) -> str:
    """Reconstruct full text from overlapping chunks by removing duplicated portions.

    The chunker prepends up to 150 chars from the end of chunk N to the start
    of chunk N+1. This function detects and strips that overlap.
    """
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]

    result_parts = [texts[0]]
    for i in range(1, len(texts)):
        prev = texts[i - 1]
        curr = texts[i]
        # Find the longest suffix of prev that matches a prefix of curr
        best_overlap = 0
        # Check overlap lengths from longest plausible down to shortest
        max_check = min(len(prev), len(curr), 250)
        for length in range(max_check, 0, -1):
            if curr.startswith(prev[-length:]):
                best_overlap = length
                break
        if best_overlap > 0:
            trimmed = curr[best_overlap:].lstrip()
            if trimmed:
                result_parts.append(trimmed)
        else:
            result_parts.append(curr)

    return "\n\n".join(part for part in result_parts if part.strip())


@router.get("/jobs")
async def list_jobs(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    location: Optional[str] = None,
    company: Optional[str] = None,
    title_contains: Optional[str] = None,
    language: Optional[str] = None,
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
                "language": meta.get("language", ""),
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
    if language:
        lang_lower = language.lower()
        result = [j for j in result if (j.get("language") or "").lower() == lang_lower]

    total = len(result)
    paginated = result[skip: skip + limit]

    return {"total": total, "skip": skip, "limit": limit, "jobs": paginated}


@router.get("/languages")
async def list_languages(request: Request) -> Dict[str, Any]:
    """List detected languages and their counts."""
    retriever = request.app.state.retriever
    if retriever is None:
        return {"languages": []}

    # Count languages from chunk metadata (dedupe by job_key)
    seen_jobs: set = set()
    counts: Dict[str, int] = {}
    for ch in retriever.chunks:
        jk = ch.get("job_key", "")
        if jk in seen_jobs:
            continue
        seen_jobs.add(jk)
        lang = ch.get("meta", {}).get("language", "")
        if lang:
            counts[lang] = counts.get(lang, 0) + 1

    sorted_langs = sorted(counts.items(), key=lambda x: -x[1])
    return {"languages": [{"code": code, "count": count} for code, count in sorted_langs]}


class BatchJobsRequest(BaseModel):
    job_ids: List[str]


@router.post("/jobs/batch")
async def get_jobs_batch(request: Request, body: BatchJobsRequest) -> Dict[str, Any]:
    """Get multiple jobs by IDs."""
    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="Search artifacts not loaded")

    requested = set(body.job_ids)

    jobs: Dict[str, Dict[str, Any]] = {}
    for ch in retriever.chunks:
        jk = ch.get("job_key", "")
        if jk not in requested:
            continue
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
                "language": meta.get("language", ""),
                "n_chunks": 0,
            }
        jobs[jk]["n_chunks"] = jobs[jk].get("n_chunks", 0) + 1

    # Preserve requested order
    result = [jobs[jk] for jk in body.job_ids if jk in jobs]
    return {"total": len(result), "jobs": result}


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
    full_text = _remove_chunk_overlap([c.get("text", "") for c in sorted_chunks])

    # Parse sections using structural detection
    parsed_sections = detect_sections_structural(full_text)
    structured_sections = [
        {
            "name": s.name or "intro",
            "raw_name": s.raw_name,
            "text": s.text,
        }
        for s in parsed_sections
        if s.text.strip()
    ]

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
        "parsed_sections": structured_sections,
        "full_text": full_text,
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
