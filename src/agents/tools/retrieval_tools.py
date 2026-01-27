"""Retrieval tools: get_job_summary, get_chunks, get_full_text, expand_context."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .registry import ToolRegistry


_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from ...search.retriever import HybridRetriever
        _retriever = HybridRetriever.from_artifacts()
    return _retriever


def get_job_summary(job_id: str) -> str:
    """Get lightweight summary: title, company, aspects, keywords (Tier 0)."""
    retriever = _get_retriever()
    chunks = retriever.get_chunks_for_job(job_id)
    if not chunks:
        return f"No data found for job_id: {job_id}"

    meta = chunks[0].get("meta", {})
    texts = [ch.get("text", "") for ch in chunks]
    full_text = " ".join(texts)[:500]

    summary = {
        "job_id": job_id,
        "title": meta.get("title", "Unknown"),
        "company": meta.get("company", "Unknown"),
        "location": meta.get("location", ""),
        "n_chunks": len(chunks),
        "text_preview": full_text,
    }
    return json.dumps(summary, ensure_ascii=False)


def get_job_chunks(job_id: str, section: Optional[str] = None) -> str:
    """Get chunks for a job, optionally filtered by section (Tier 1-2)."""
    retriever = _get_retriever()
    chunks = retriever.get_chunks_for_job(job_id)

    if section:
        chunks = [c for c in chunks if c.get("section", "").lower() == section.lower()]

    if not chunks:
        return f"No chunks found for job_id: {job_id}" + (f" section: {section}" if section else "")

    lines: List[str] = []
    for ch in chunks[:10]:
        sec = f" [{ch.get('section', '')}]" if ch.get("section") else ""
        lines.append(f"{sec} {ch.get('text', '')[:300]}")

    return "\n\n".join(lines)


def get_full_text(job_id: str) -> str:
    """Get full job description text (Tier 3)."""
    retriever = _get_retriever()
    chunks = retriever.get_chunks_for_job(job_id)
    if not chunks:
        return f"No data found for job_id: {job_id}"

    # Sort by order
    chunks_sorted = sorted(chunks, key=lambda c: c.get("order", 0))
    meta = chunks_sorted[0].get("meta", {})

    header = f"# {meta.get('title', '?')} @ {meta.get('company', '?')}\n"
    header += f"Location: {meta.get('location', '?')}\n\n"
    body = "\n\n".join(ch.get("text", "") for ch in chunks_sorted)

    return header + body


def expand_context(job_id: str, question: str) -> str:
    """Retrieve chunks from a specific job relevant to a question."""
    retriever = _get_retriever()
    results = retriever.search(
        question,
        k=15,
        where=lambda c: c.get("job_key") == job_id,
    )

    if not results:
        return f"No relevant chunks found for job_id: {job_id} and question: {question}"

    lines: List[str] = []
    for ch in results[:5]:
        sec = f" [{ch.get('section', '')}]" if ch.get("section") else ""
        lines.append(f"{sec} {ch.get('text', '')[:400]}")

    return "\n\n".join(lines)


def register_retrieval_tools(registry: ToolRegistry) -> None:
    """Register retrieval tools."""
    registry.register("get_job_summary", get_job_summary, {
        "name": "get_job_summary",
        "description": "Get a lightweight summary of a job: title, company, location, preview. Use for quick lookups.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID (URL or key)"},
            },
            "required": ["job_id"],
        },
    })

    registry.register("get_job_chunks", get_job_chunks, {
        "name": "get_job_chunks",
        "description": "Get text chunks from a job, optionally filtered by section (responsibilities, requirements, benefits, etc.).",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID"},
                "section": {"type": "string", "description": "Optional section filter (e.g., 'requirements', 'benefits')"},
            },
            "required": ["job_id"],
        },
    })

    registry.register("get_full_text", get_full_text, {
        "name": "get_full_text",
        "description": "Get the complete job description text. Use sparingly - only when you need all details.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID"},
            },
            "required": ["job_id"],
        },
    })

    registry.register("expand_context", expand_context, {
        "name": "expand_context",
        "description": "Search within a specific job for chunks relevant to a question. Useful for answering specific questions about a job.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID"},
                "question": {"type": "string", "description": "What you want to know about this job"},
            },
            "required": ["job_id", "question"],
        },
    })
