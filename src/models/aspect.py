"""Aspect extraction data models."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AspectValue(BaseModel):
    """A single extracted value for an aspect."""

    value: str
    evidence_span: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class AspectExtraction(BaseModel):
    """Extraction result for a single aspect from a job posting or chunk."""

    aspect: str  # e.g., "skills", "language", "remote_policy"
    values: List[str] = Field(default_factory=list)
    evidence_spans: List[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    method: str = "unknown"  # "phrase_matcher" | "regex" | "llm" | "keyword"


class DomainClassification(BaseModel):
    """LLM-based domain classification result."""

    domain: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str = ""


class CultureExtraction(BaseModel):
    """LLM-based culture/values extraction."""

    values: List[str] = Field(default_factory=list)
    evidence_spans: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SeniorityAssessment(BaseModel):
    """LLM-based seniority assessment."""

    level: Optional[str] = None  # "junior" | "mid" | "senior" | "lead" | "staff"
    years_experience: Optional[str] = None  # e.g., "3-5 years"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str = ""
