"""Job data models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobSummary(BaseModel):
    """Lightweight job summary for listings and search results."""

    job_id: str
    title: str = ""
    company: str = ""
    location: str = ""
    url: str = ""
    days_old: Optional[int] = None
    aspects: Dict[str, List[str]] = Field(default_factory=dict)
    keywords: List[str] = Field(default_factory=list)
    cluster_id: Optional[int] = None
    cluster_label: str = ""


class JobDetail(JobSummary):
    """Full job detail with description and embeddings."""

    description_raw: str = ""
    description_clean: str = ""
    sections: List[str] = Field(default_factory=list)
    contract_type: Optional[str] = None
    experience_level: Optional[str] = None
    work_type: Optional[str] = None
    salary: Optional[str] = None
    umap_x: Optional[float] = None
    umap_y: Optional[float] = None
    n_chunks: int = 0


class Job(BaseModel):
    """Internal job representation used during pipeline processing."""

    job_id: str
    title: str = ""
    company: str = ""
    location: str = ""
    url: str = ""
    description_raw: str = ""
    description_clean: str = ""
    days_old: Optional[int] = None
    contract_type: Optional[str] = None
    experience_level: Optional[str] = None
    work_type: Optional[str] = None
    salary: Optional[str] = None
    apply_url: str = ""
    sector: Optional[str] = None

    # NLP-extracted
    aspects: Dict[str, List[str]] = Field(default_factory=dict)
    keywords: List[str] = Field(default_factory=list)
    sections: List[str] = Field(default_factory=list)

    # Clustering
    umap_x: Optional[float] = None
    umap_y: Optional[float] = None
    cluster_id: Optional[int] = None
    cluster_label: str = ""

    # UI state
    job_label: List[str] = Field(default_factory=list)
    facets: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw_doc(cls, doc: Dict[str, Any]) -> "Job":
        """Create a Job from a raw JSON document (flat or nested job_data)."""
        job_data = doc.get("job_data", doc)
        meta = doc.get("meta", {})
        return cls(
            job_id=job_data.get("jobUrl") or f"{job_data.get('companyName', 'unknown')}|{job_data.get('title', 'unknown')}",
            title=job_data.get("title", ""),
            company=job_data.get("companyName", ""),
            location=job_data.get("location", ""),
            url=job_data.get("jobUrl", ""),
            description_raw=job_data.get("description", ""),
            days_old=meta.get("days_old"),
            contract_type=job_data.get("contractType"),
            experience_level=job_data.get("experienceLevel"),
            work_type=job_data.get("workType"),
            salary=job_data.get("salary"),
            apply_url=job_data.get("applyUrl", ""),
            sector=job_data.get("sector"),
        )
