"""Search tools: semantic_search, keyword_search, hybrid_search, filter_jobs."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .registry import ToolRegistry

# Lazy-loaded retriever
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from ...search.retriever import HybridRetriever
        _retriever = HybridRetriever.from_artifacts()
    return _retriever


def _format_results(chunks: List[Dict[str, Any]], max_results: int = 10) -> str:
    """Format chunk results as concise text for the agent."""
    if not chunks:
        return "No results found."

    lines: List[str] = []
    seen_jobs: set[str] = set()
    for ch in chunks[:max_results * 3]:  # oversample to get unique jobs
        jk = ch.get("job_key", "")
        if jk in seen_jobs:
            continue
        seen_jobs.add(jk)
        meta = ch.get("meta", {})
        title = meta.get("title", "?")
        company = meta.get("company", "?")
        location = meta.get("location", "")
        snippet = ch.get("text", "")[:150]
        lines.append(f"- {title} @ {company} ({location}) | {snippet}...")
        if len(lines) >= max_results:
            break

    return f"Found {len(chunks)} chunks across {len(seen_jobs)} jobs:\n" + "\n".join(lines)


def semantic_search(query: str, k: int = 8) -> str:
    """Vector-only search."""
    retriever = _get_retriever()
    results = retriever.search_semantic(query, k=k)
    return _format_results(results, max_results=k)


def keyword_search(query: str, k: int = 8) -> str:
    """BM25-only search."""
    retriever = _get_retriever()
    results = retriever.search_keyword(query, k=k)
    return _format_results(results, max_results=k)


def hybrid_search(query: str, k: int = 8, alpha: float = 0.55) -> str:
    """Combined vector + BM25 search."""
    retriever = _get_retriever()
    results = retriever.search(query, k=k, alpha=alpha)
    return _format_results(results, max_results=k)


def filter_jobs(skills: Optional[List[str]] = None, location: Optional[str] = None, remote_policy: Optional[str] = None) -> str:
    """Filter jobs by deterministic metadata criteria."""
    retriever = _get_retriever()
    all_chunks = retriever.get_all_chunks()

    # Group by job
    jobs: Dict[str, Dict[str, Any]] = {}
    for ch in all_chunks:
        jk = ch.get("job_key", "")
        if jk not in jobs:
            meta = ch.get("meta", {})
            jobs[jk] = {
                "job_key": jk,
                "title": meta.get("title", ""),
                "company": meta.get("company", ""),
                "location": meta.get("location", ""),
                "text_sample": ch.get("text", "")[:200],
            }

    results = list(jobs.values())

    if location:
        loc_lower = location.lower()
        results = [j for j in results if loc_lower in j.get("location", "").lower()]

    if skills:
        skill_set = {s.lower() for s in skills}
        results = [
            j for j in results
            if any(s in j.get("text_sample", "").lower() for s in skill_set)
        ]

    if not results:
        return "No jobs match the given filters."

    lines = [f"- {j['title']} @ {j['company']} ({j['location']})" for j in results[:15]]
    return f"Found {len(results)} matching jobs:\n" + "\n".join(lines)


def register_search_tools(registry: ToolRegistry) -> None:
    """Register all search tools."""
    registry.register("semantic_search", semantic_search, {
        "name": "semantic_search",
        "description": "Search jobs using semantic similarity (vector embeddings). Best for conceptual/meaning-based queries.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "k": {"type": "integer", "description": "Number of results (default 8)", "default": 8},
            },
            "required": ["query"],
        },
    })

    registry.register("keyword_search", keyword_search, {
        "name": "keyword_search",
        "description": "Search jobs using exact keyword matching (BM25). Best for specific terms, tools, or company names.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "k": {"type": "integer", "description": "Number of results (default 8)", "default": 8},
            },
            "required": ["query"],
        },
    })

    registry.register("hybrid_search", hybrid_search, {
        "name": "hybrid_search",
        "description": "Combined semantic + keyword search. Best general-purpose search.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "k": {"type": "integer", "description": "Number of results (default 8)", "default": 8},
                "alpha": {"type": "number", "description": "Weight: 1.0=semantic only, 0.0=keyword only (default 0.55)", "default": 0.55},
            },
            "required": ["query"],
        },
    })

    registry.register("filter_jobs", filter_jobs, {
        "name": "filter_jobs",
        "description": "Filter jobs by metadata: skills, location, remote policy. Returns matching jobs without ranking.",
        "parameters": {
            "type": "object",
            "properties": {
                "skills": {"type": "array", "items": {"type": "string"}, "description": "Required skills to filter by"},
                "location": {"type": "string", "description": "Location filter (partial match)"},
                "remote_policy": {"type": "string", "description": "Remote policy: remote, hybrid, or onsite"},
            },
        },
    })
