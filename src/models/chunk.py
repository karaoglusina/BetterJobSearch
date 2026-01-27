"""Chunk data models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .aspect import AspectExtraction


class Chunk(BaseModel):
    """A single chunk of text from a job description (Pydantic version)."""

    chunk_id: str
    chunk_id_v2: str = ""
    job_key: str
    text: str
    section: Optional[str] = None
    order: int = 0
    meta: Dict[str, Any] = Field(default_factory=dict)


class ChunkWithAspects(Chunk):
    """Chunk enriched with extracted aspects and keywords."""

    aspects: List[AspectExtraction] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
