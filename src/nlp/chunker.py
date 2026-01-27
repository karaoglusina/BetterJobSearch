"""Aspect-aware chunking for job descriptions.

Refactored from src/rag.py split_bullets_or_sentences with aspect tagging.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

from ..models.aspect import AspectExtraction
from ..models.chunk import ChunkWithAspects
from .cleaner import clean_html
from .section_detector import detect_sections, Section
from .aspect_extractor import AspectExtractor

BULLET_REGEX = re.compile(r"^\s*([\-\*\u2022\u25CF\u25E6]|\d+\.)\s+")
SENT_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _stable_chunk_id(job_key: str, order: int) -> str:
    return hashlib.md5(f"{job_key}|{order}".encode("utf-8")).hexdigest()


def _content_chunk_id(job_key: str, text: str) -> str:
    norm = re.sub(r"\s+", " ", (text or "").strip()).lower()
    return hashlib.md5(f"{job_key}|{norm}".encode("utf-8")).hexdigest()


def _split_bullets_or_sentences(block: str, char_limit: int = 900, overlap_chars: int = 150) -> List[str]:
    """Split a text block into bullet points or windowed sentences.

    Ported from rag.py with same logic for backward compatibility.
    """
    lines = [line.strip() for line in block.split("\n") if line.strip()]

    # If looks like a bullet list, keep bullets as chunks
    if sum(1 for line in lines if BULLET_REGEX.match(line)) >= max(2, len(lines) // 2):
        bullets: List[str] = []
        for line in lines:
            if BULLET_REGEX.match(line):
                bullets.append(BULLET_REGEX.sub("", line).strip())
            elif bullets:
                bullets[-1] += " " + line
        return [b for b in bullets if len(b) > 3]

    # Else split into sentences and window them
    text = " ".join(lines)
    sents = re.split(SENT_SPLIT_REGEX, text)
    sents = [s.strip() for s in sents if s.strip()]

    chunks: List[str] = []
    buf: List[str] = []
    for s in sents:
        if sum(len(x) for x in buf) + len(s) + len(buf) <= char_limit:
            buf.append(s)
        else:
            if buf:
                chunks.append(" ".join(buf))
            buf = [s]
    if buf:
        chunks.append(" ".join(buf))

    # Add overlap
    overlapped: List[str] = []
    prev_tail = ""
    for c in chunks:
        piece = (prev_tail + " " + c).strip() if prev_tail else c
        overlapped.append(piece)
        prev_tail = c[-overlap_chars:]
    return overlapped


def chunk_job(
    doc: Dict[str, Any],
    *,
    aspect_extractor: Optional[AspectExtractor] = None,
    order_start: int = 0,
) -> List[ChunkWithAspects]:
    """Convert a job document into aspect-aware chunks.

    Args:
        doc: Job document dict (flat or nested job_data).
        aspect_extractor: Optional extractor for tagging chunks with aspects.
        order_start: Starting order index.

    Returns:
        List of ChunkWithAspects objects.
    """
    job_data = doc.get("job_data", doc)
    description = job_data.get("description", "")
    if not description:
        return []

    job_key = job_data.get("jobUrl") or (
        f"{job_data.get('companyName', 'unknown')}|{job_data.get('title', 'unknown')}"
    )

    # Clean HTML preserving structure
    cleaned = clean_html(description)

    # Detect sections
    sections = detect_sections(cleaned)
    if not sections:
        sections = [Section(name=None, raw_name=None, text=cleaned)]

    meta = {
        "title": job_data.get("title"),
        "company": job_data.get("companyName"),
        "days_old": doc.get("meta", {}).get("days_old"),
        "location": job_data.get("location"),
        "sector": job_data.get("sector"),
        "contractType": job_data.get("contractType"),
        "experienceLevel": job_data.get("experienceLevel"),
        "applyType": job_data.get("applyType"),
        "workType": job_data.get("workType"),
        "salary": job_data.get("salary"),
        "jobUrl": job_data.get("jobUrl"),
        "applyUrl": job_data.get("applyUrl"),
        "applied_times": doc.get("meta", {}).get("applied_times"),
    }

    chunks: List[ChunkWithAspects] = []
    order = order_start

    for section in sections:
        pieces = _split_bullets_or_sentences(section.text)
        for piece in pieces:
            if len(piece) < 20:
                continue

            raw_text = piece.strip()

            # Extract aspects for this chunk
            aspects: List[AspectExtraction] = []
            if aspect_extractor is not None:
                aspects = aspect_extractor.extract_for_chunk(raw_text, section=section.name)

            chunk = ChunkWithAspects(
                chunk_id=_stable_chunk_id(job_key, order),
                chunk_id_v2=_content_chunk_id(job_key, raw_text),
                job_key=job_key,
                text=raw_text,
                section=section.name,
                order=order,
                meta=meta,
                aspects=aspects,
                keywords=[],  # Filled later by keyword extraction
            )
            chunks.append(chunk)
            order += 1

    return chunks
