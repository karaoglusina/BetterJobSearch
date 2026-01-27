"""Search endpoint."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=8, ge=1, le=100)
    alpha: float = Field(default=0.55, ge=0.0, le=1.0)
    location: Optional[str] = None
    company: Optional[str] = None


class SearchResult(BaseModel):
    job_id: str
    title: str = ""
    company: str = ""
    location: str = ""
    url: str = ""
    snippet: str = ""
    section: Optional[str] = None


@router.post("/search")
async def search_jobs(request: Request, body: SearchRequest) -> Dict[str, Any]:
    """Hybrid search with optional filters."""
    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="Search artifacts not loaded")

    # Build filter predicate
    def where(ch: Dict[str, Any]) -> bool:
        meta = ch.get("meta", {})
        if body.location:
            if body.location.lower() not in (meta.get("location") or "").lower():
                return False
        if body.company:
            if body.company.lower() not in (meta.get("company") or "").lower():
                return False
        return True

    has_filter = body.location or body.company
    chunks = retriever.search(
        body.query,
        k=body.k * 3 if has_filter else body.k,
        alpha=body.alpha,
        where=where if has_filter else None,
    )

    # Dedupe by job, keep best chunk per job
    seen_jobs: set[str] = set()
    results: List[Dict[str, Any]] = []
    for ch in chunks:
        jk = ch.get("job_key", "")
        if jk in seen_jobs:
            continue
        seen_jobs.add(jk)
        meta = ch.get("meta", {})
        results.append({
            "job_id": jk,
            "title": meta.get("title", ""),
            "company": meta.get("company", ""),
            "location": meta.get("location", ""),
            "url": meta.get("jobUrl", ""),
            "snippet": ch.get("text", "")[:200],
            "section": ch.get("section"),
        })
        if len(results) >= body.k:
            break

    return {
        "query": body.query,
        "total_results": len(results),
        "results": results,
    }
