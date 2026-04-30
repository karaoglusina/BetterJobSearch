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


class ClusterPolygon(BaseModel):
    """Concave-hull polygon ('continent') for a single cluster."""

    cluster_id: int
    path: List[List[float]] = Field(default_factory=list)


class ClusterLabel(BaseModel):
    """A topic label placed on the scatter map.

    level=0 is one per top cluster at its centroid; level=1 is a sub-topic
    placed at a sub-cluster centroid (only shown when zoomed in).
    """

    text: str
    x: float
    y: float
    level: int = 0
    priority: float = 0.0
    cluster_id: int
    sub_id: int = -1


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
    # Atlas-style visualization layers
    polygons: List[ClusterPolygon] = Field(default_factory=list)
    labels: List[ClusterLabel] = Field(default_factory=list)
    palette: Dict[str, str] = Field(default_factory=dict)


class AspectDistribution(BaseModel):
    """Distribution of values for a given aspect across all jobs."""

    aspect: str
    total_jobs: int = 0
    value_counts: Dict[str, int] = Field(default_factory=dict)
    coverage: float = 0.0  # fraction of jobs with at least one value
