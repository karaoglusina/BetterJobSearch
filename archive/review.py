"""Review helpers for chunk-level labeling (v3.3).

Provides a simple interactive loop to annotate retrieved chunks with a
"fitness" label, and batch utilities to set the same fitness over a list
of chunks. Uses the content-based chunk id (chunk_id_v2) to persist
annotations via the feedback API.

Fitness scale:
  2 = perfect
  1 = yes
  0 = maybe
 -1 = not really
 -2 = impossible
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import hashlib
import re

from .feedback import (
    record_chunk_fitness,
    get_chunk_fitness,
    record_chunk_labels,
    upsert_chunk_fitness_bulk,
)


FITNESS_KEYS: Dict[str, int] = {
    "2": 2, "1": 1, "0": 0, "-1": -1, "-2": -2,
    "p": 2,  # perfect
    "y": 1,  # yes
    "m": 0,  # maybe
    "n": -1, # not really
    "i": -2, # impossible
}


def _normalize_for_id(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def content_chunk_id(job_key: str, text: str) -> str:
    """Compute content-based chunk id (md5(job_key + '|' + normalized_text))."""
    norm = _normalize_for_id(text)
    return hashlib.md5(f"{job_key}|{norm}".encode("utf-8")).hexdigest()


def ensure_chunk_id_v2_for(c: Any) -> str:
    """Return chunk_id_v2 for a chunk-like object, computing if missing.

    Expects attributes: job_key (str), text (str), and optionally chunk_id_v2.
    The function attaches chunk_id_v2 on the object if it was empty/missing.
    """
    cid = getattr(c, "chunk_id_v2", "") or ""
    if cid:
        return cid
    jid = getattr(c, "job_key", None)
    txt = getattr(c, "text", None)
    if not isinstance(jid, str) or not isinstance(txt, str):
        raise ValueError("Chunk must have string attributes job_key and text")
    cid2 = content_chunk_id(jid, txt)
    try:
        setattr(c, "chunk_id_v2", cid2)
    except Exception:
        pass
    return cid2


def load_existing_fitness(chunks: List[Any]) -> Dict[str, int]:
    """Fetch existing fitness scores for a list of chunks by chunk_id_v2."""
    ids = [ensure_chunk_id_v2_for(c) for c in chunks]
    return get_chunk_fitness(ids)


def annotate_chunks_cli(
    chunks: List[Any],
    *,
    user: str = "me",
    index_build_id: Optional[str] = None,
    preview_chars: int = 220,
) -> Dict[str, int]:
    """Interactive annotation loop in console / notebooks.

    For each chunk, prints company, title, url and a snippet, shows existing fitness
    if present, and prompts for a new fitness. Press ENTER to skip. Use keys:
    2/p, 1/y, 0/m, -1/n, -2/i. 'q' to quit early.

    Returns a mapping {chunk_id_v2 -> fitness} for all labels saved in this run.
    """
    saved: Dict[str, int] = {}
    existing = load_existing_fitness(chunks)
    print("\nAnnotation mode: [2/p]=perfect  [1/y]=yes  [0/m]=maybe  [-1/n]=not really  [-2/i]=impossible  [ENTER]=skip  [q]=quit")
    for idx, c in enumerate(chunks, 1):
        cid = ensure_chunk_id_v2_for(c)
        job_key = getattr(c, "job_key", "")
        meta = getattr(c, "meta", {}) or {}
        title = meta.get("title", "?")
        company = meta.get("company", "?")
        url = meta.get("jobUrl", "")
        snippet = (getattr(c, "text", "") or "").strip()[:preview_chars]
        prev = existing.get(cid)
        if prev is not None:
            print(f"\n[{idx}/{len(chunks)}] {company} — {title} | {url} | current fitness: {prev}")
        else:
            print(f"\n[{idx}/{len(chunks)}] {company} — {title} | {url} | current fitness: None")
        print("Snippet:", snippet, ("..." if len(getattr(c, "text", "")) > preview_chars else ""))
        try:
            raw = input("Fitness [2/1/0/-1/-2 or p/y/m/n/i, ENTER=skip, q=quit]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting annotation loop.")
            break
        if raw == "":
            pass
        if raw.lower() == "q":
            print("Quit.")
            break
        score = FITNESS_KEYS.get(raw.lower()) if raw else None
        if score is not None:
            res = record_chunk_fitness(
                cid,
                score,
                job_key=job_key,
                user=user,
                index_build_id=index_build_id,
            )
            saved[cid] = score
            print(f"Saved fitness={score} (matched={res.get('matched')}, modified={res.get('modified')})")
        # Facet / type prompts (optional)
        try:
            raw_facet = input("Facet (slug or free text, ENTER=skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting annotation loop.")
            break
        try:
            raw_type = input("Type within facet (slug or free text, ENTER=skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting annotation loop.")
            break
        if raw_facet or raw_type:
            record_chunk_labels(
                cid,
                chunk_facet=raw_facet or None,
                type_within_facet=raw_type or None,
                user=user,
            )
    print(f"\nSaved {len(saved)} labels in this session.")
    return saved


def batch_set_fitness(
    chunks: List[Any],
    fitness: int,
    *,
    user: str = "me",
    index_build_id: Optional[str] = None,
) -> int:
    """Apply the same fitness to all provided chunks using a bulk upsert."""
    records: List[Dict[str, Any]] = []
    for c in chunks:
        cid = ensure_chunk_id_v2_for(c)
        job_key = getattr(c, "job_key", None)
        records.append({"chunk_id_v2": cid, "job_key": job_key, "user": user})
    res = upsert_chunk_fitness_bulk(records, fitness=fitness, user=user, index_build_id=index_build_id)
    # Return number of affected docs as an approximation of saved labels
    return int(res.get("modified", 0)) + int(res.get("upserted", 0))

