"""NLP tools: extract_keywords, compare_aspects, find_similar."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .registry import ToolRegistry


def extract_keywords(text: str, n: int = 10) -> str:
    """Extract keywords from a piece of text."""
    from ...nlp.keyword_extractor import _fallback_keywords
    keywords = _fallback_keywords(text, n)
    result = [{"keyword": kw, "score": round(score, 4)} for kw, score in keywords]
    return json.dumps(result, ensure_ascii=False)


def compare_aspects(job_ids: List[str], aspect: str = "skills") -> str:
    """Compare a specific aspect across multiple jobs side-by-side."""
    from ...search.retriever import HybridRetriever

    try:
        retriever = HybridRetriever.from_artifacts()
    except Exception:
        return "Search artifacts not loaded."

    comparison: Dict[str, Any] = {}
    for job_id in job_ids[:5]:  # cap at 5 jobs
        chunks = retriever.get_chunks_for_job(job_id)
        if not chunks:
            comparison[job_id] = {"title": "Unknown", "values": []}
            continue

        meta = chunks[0].get("meta", {})
        full_text = " ".join(ch.get("text", "") for ch in chunks)

        # Quick aspect extraction
        from ...nlp.aspect_extractor import AspectExtractor
        extractor = AspectExtractor()
        aspects = extractor.extract_all_as_dict(full_text)

        comparison[job_id] = {
            "title": meta.get("title", "Unknown"),
            "company": meta.get("company", "Unknown"),
            aspect: aspects.get(aspect, []),
        }

    return json.dumps(comparison, ensure_ascii=False)


def find_similar_jobs(job_id: str, k: int = 5) -> str:
    """Find jobs similar to a given job."""
    from ...search.retriever import HybridRetriever

    try:
        retriever = HybridRetriever.from_artifacts()
    except Exception:
        return "Search artifacts not loaded."

    # Get representative text from job
    chunks = retriever.get_chunks_for_job(job_id)
    if not chunks:
        return f"No data for job_id: {job_id}"

    meta = chunks[0].get("meta", {})
    title = meta.get("title", "")
    # Use title + first chunk as query
    query_text = f"{title} {chunks[0].get('text', '')[:200]}"

    results = retriever.search(
        query_text,
        k=k * 3,
        where=lambda c: c.get("job_key") != job_id,
    )

    # Dedupe by job
    seen: set[str] = set()
    similar: List[Dict[str, str]] = []
    for ch in results:
        jk = ch.get("job_key", "")
        if jk in seen:
            continue
        seen.add(jk)
        m = ch.get("meta", {})
        similar.append({
            "job_id": jk,
            "title": m.get("title", "?"),
            "company": m.get("company", "?"),
            "location": m.get("location", ""),
        })
        if len(similar) >= k:
            break

    return json.dumps(similar, ensure_ascii=False)


def register_nlp_tools(registry: ToolRegistry) -> None:
    """Register NLP tools."""
    registry.register("extract_keywords", extract_keywords, {
        "name": "extract_keywords",
        "description": "Extract keywords from a piece of text. Returns keywords with relevance scores.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to extract keywords from"},
                "n": {"type": "integer", "description": "Number of keywords (default 10)", "default": 10},
            },
            "required": ["text"],
        },
    })

    registry.register("compare_aspects", compare_aspects, {
        "name": "compare_aspects",
        "description": "Compare a specific aspect (skills, tools, language, etc.) across multiple jobs side-by-side.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_ids": {"type": "array", "items": {"type": "string"}, "description": "List of job IDs to compare"},
                "aspect": {"type": "string", "description": "Aspect to compare (default: skills)", "default": "skills"},
            },
            "required": ["job_ids"],
        },
    })

    registry.register("find_similar_jobs", find_similar_jobs, {
        "name": "find_similar_jobs",
        "description": "Find jobs similar to a given job, based on content similarity.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID to find similar jobs for"},
                "k": {"type": "integer", "description": "Number of similar jobs (default 5)", "default": 5},
            },
            "required": ["job_id"],
        },
    })
