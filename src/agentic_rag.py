

"""Simple agentic retrieval loop.

This module lets an LLM iteratively refine a user's question into
sub‑queries and run retrieval for each hop. Results across hops are
aggregated and returned with the trace of queries issued.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple, TypedDict

from . import rag
from .rag import Chunk


class HopTrace(TypedDict):
    hop: int
    query: str

# Optional OpenAI client -----------------------------------------------------
OPENAI_AVAILABLE = False
try:  # pragma: no cover - dependency is optional
    import openai  # type: ignore

    OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    openai = None  # type: ignore

# Cache for sub-query results to avoid recomputing embeddings
_CACHE: Dict[str, List[Chunk]] = {}


def _call_llm(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> str:
    """Call OpenAI's ChatCompletion API with provided messages."""
    if not OPENAI_AVAILABLE or not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI API key not configured")

    openai.api_key = os.environ["OPENAI_API_KEY"]
    resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
    return resp["choices"][0]["message"]["content"].strip()


def _ensure_retrieval_ready() -> None:
    """Ensure retrieval artifacts are loaded before calling rag.retrieve().

    If artifacts are present on disk, load them into the rag cache. Otherwise raise
    a clear error guiding the user to build the index first.
    """
    # If already loaded, do nothing
    if getattr(rag, "_CACHE", {}).get("chunks"):
        return
    # Load from artifacts if available, else instruct to build
    if os.path.exists(rag.CHUNKS_PATH) and os.path.exists(rag.FAISS_PATH) and os.path.exists(rag.BM25_PATH):
        rag.load_cache()
        return
    raise RuntimeError(
        "Retrieval artifacts not found. Build the index first via the CLI in `src/rag.py` (--build)."
    )


def plan_and_retrieve(question: str, *, max_hops: int = 3, k: int = 8) -> Tuple[List[Chunk], List[HopTrace]]:
    """Plan searches for ``question`` and aggregate retrieval results.

    Args:
        question: Original user question.
        max_hops: Maximum number of search hops.
        k: Number of chunks to retrieve per hop.

    Returns:
        A tuple ``(chunks, trace)`` where ``chunks`` are aggregated unique
        results from all hops and ``trace`` is the list of executed sub‑queries.
    """

    system = (
        "Break the user's question into focused search queries. "
        "Return one query per turn. Reply with DONE when no more searches are needed."
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    aggregated: List[Chunk] = []
    seen: set[str] = set()
    trace: List[HopTrace] = []

    # If OpenAI is unavailable, do a single-hop retrieval using the original question
    if not (OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY")):
        _ensure_retrieval_ready()
        chunks = rag.retrieve(question, k=k)
        for c in chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                aggregated.append(c)
        trace.append({"hop": 1, "query": question})
        return aggregated, trace

    for hop in range(1, max_hops + 1):
        sub_query = _call_llm(messages)
        if sub_query.strip().upper() == "DONE":
            break

        trace.append({"hop": hop, "query": sub_query})
        if sub_query in _CACHE:
            chunks = _CACHE[sub_query]
        else:
            _ensure_retrieval_ready()
            chunks = rag.retrieve(sub_query, k=k)
            _CACHE[sub_query] = chunks

        for c in chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                aggregated.append(c)

        # Summarize top results for the LLM to decide next step
        preview = "\n".join(c.text[:200] for c in chunks[:3])
        messages.append({"role": "assistant", "content": sub_query})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Top results for '{sub_query}':\n{preview}\n\n"
                    "If more search is needed, provide the next query. Otherwise say DONE."
                ),
            }
        )

    return aggregated, trace


def _dedup_chunks(chs: List[Chunk]) -> List[Chunk]:
    seen: set[str] = set()
    out: List[Chunk] = []
    for c in chs:
        key = getattr(c, "chunk_id_v2", None) or getattr(c, "chunk_id", None)
        if key and key not in seen:
            seen.add(str(key))
            out.append(c)
    return out


def answer_job_question(job_id: str, question: str, k: int = 4, expand_k: int = 3, conf_floor: float = 0.7):
    """Expand-on-demand Q&A for one job.

    Returns QAResult-like dict; falls back to extractive if LLM not available.
    """
    from .rag import retrieve as retrieve_global
    from .rag import build_context_with_used
    from .llm_io import QAResult, call_llm_json, make_qa_messages

    _ensure_retrieval_ready()
    # Seed retrieval constrained to job via filtering
    seed = rag.retrieve_filtered(question, where=lambda c: getattr(c, "job_key", None) == job_id, k=k)
    if not seed:
        seed = rag.retrieve(question, k=k)

    # If no LLM, return extractive-like answer
    if not (OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY")):
        ctx, used = build_context_with_used(seed, max_chars=1800)
        return {"answer": ctx, "confidence": 0.0, "needed_keywords": [], "citations": []}

    # Ask LLM for JSON answer with confidence/needed_keywords
    msgs = make_qa_messages([
        {"chunk_id": getattr(c, "chunk_id_v2", getattr(c, "chunk_id", "")), "title": c.meta.get("title"), "company": c.meta.get("company"), "text": c.text}
        for c in seed
    ], question)
    ans: QAResult = call_llm_json(msgs, QAResult)
    if ans.confidence >= conf_floor or not ans.needed_keywords:
        return ans.model_dump()

    extra: List[Chunk] = []
    for kw in ans.needed_keywords[:3]:
        extra += rag.retrieve_filtered(kw, where=lambda c: getattr(c, "job_key", None) == job_id, k=expand_k)
    combined = _dedup_chunks(seed + extra)

    msgs2 = make_qa_messages([
        {"chunk_id": getattr(c, "chunk_id_v2", getattr(c, "chunk_id", "")), "title": c.meta.get("title"), "company": c.meta.get("company"), "text": c.text}
        for c in combined
    ], question)
    ans2: QAResult = call_llm_json(msgs2, QAResult)
    return ans2.model_dump()


__all__ = ["plan_and_retrieve", "answer_job_question"]