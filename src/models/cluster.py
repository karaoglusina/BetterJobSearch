"""Cluster data models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ClusterInfo(BaseModel):
    """Information about a single cluster."""

    cluster_id: int
    label: str = ""
    size: int = 0
    keywords: List[str] = Field(default_factory=list)
    sample_titles: List[str] = Field(default_factory=list)


class ClusterResult(BaseModel):
    """Full clustering result for an aspect or default projection."""

    aspect: str = "default"
    n_clusters: int = 0
    clusters: List[ClusterInfo] = Field(default_factory=list)
    # Per-job data for scatter plot
    job_ids: List[str] = Field(default_factory=list)
    x: List[float] = Field(default_factory=list)
    y: List[float] = Field(default_factory=list)
    cluster_ids: List[int] = Field(default_factory=list)
    noise_count: int = 0  # HDBSCAN noise points (label=-1)


class AspectDistribution(BaseModel):
    """Distribution of values for a given aspect across all jobs."""

    aspect: str
    total_jobs: int = 0
    value_counts: Dict[str, int] = Field(default_factory=dict)
    coverage: float = 0.0  # fraction of jobs with at least one value
