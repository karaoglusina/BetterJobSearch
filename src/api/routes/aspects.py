"""Aspect endpoints: distribution, values."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["aspects"])


AVAILABLE_ASPECTS = [
    "skills", "tools", "language", "remote_policy",
    "experience", "education", "benefits", "domain", "culture",
]


@router.get("/aspects")
async def list_aspects() -> Dict[str, Any]:
    """List all available aspects."""
    return {"aspects": AVAILABLE_ASPECTS}


@router.get("/aspects/{name}/distribution")
async def aspect_distribution(request: Request, name: str) -> Dict[str, Any]:
    """Get distribution of values for an aspect across all jobs.

    Runs the appropriate extractor on all job texts and aggregates.
    """
    if name not in AVAILABLE_ASPECTS:
        raise HTTPException(status_code=400, detail=f"Unknown aspect: {name}. Available: {AVAILABLE_ASPECTS}")

    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="Search artifacts not loaded")

    from ...nlp.aspect_extractor import AspectExtractor
    extractor = AspectExtractor()

    # Group chunks by job
    job_texts: Dict[str, str] = {}
    for ch in retriever.chunks:
        jk = ch.get("job_key", "")
        if jk not in job_texts:
            job_texts[jk] = ""
        job_texts[jk] += " " + ch.get("text", "")

    # Extract aspect for each job
    from collections import Counter
    value_counter: Counter = Counter()
    jobs_with_values = 0

    for jk, text in job_texts.items():
        aspects = extractor.extract_all_as_dict(text[:5000])
        values = aspects.get(name, [])
        if values:
            jobs_with_values += 1
            for v in values:
                value_counter[v] += 1

    total_jobs = len(job_texts)
    coverage = jobs_with_values / total_jobs if total_jobs > 0 else 0.0

    return {
        "aspect": name,
        "total_jobs": total_jobs,
        "jobs_with_values": jobs_with_values,
        "coverage": round(coverage, 3),
        "value_counts": dict(value_counter.most_common(50)),
    }


@router.get("/aspects/{name}/values")
async def aspect_values(request: Request, name: str) -> Dict[str, Any]:
    """Get all unique values for an aspect."""
    if name not in AVAILABLE_ASPECTS:
        raise HTTPException(status_code=400, detail=f"Unknown aspect: {name}")

    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="Search artifacts not loaded")

    from ...nlp.aspect_extractor import AspectExtractor
    extractor = AspectExtractor()

    all_values: set[str] = set()
    job_texts: Dict[str, str] = {}
    for ch in retriever.chunks:
        jk = ch.get("job_key", "")
        if jk not in job_texts:
            job_texts[jk] = ""
        job_texts[jk] += " " + ch.get("text", "")

    for text in job_texts.values():
        aspects = extractor.extract_all_as_dict(text[:5000])
        values = aspects.get(name, [])
        all_values.update(values)

    return {"aspect": name, "values": sorted(all_values)}
