"""Job labels endpoints: load and save user-defined labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...config import ARTIFACTS_DIR

router = APIRouter(tags=["labels"])

LABELS_FILE = ARTIFACTS_DIR / "job_labels.json"


def _load_labels() -> Dict[str, List[str]]:
    """Load labels from JSON file."""
    if LABELS_FILE.exists():
        try:
            with open(LABELS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_labels(labels: Dict[str, List[str]]) -> None:
    """Save labels to JSON file."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)


class LabelsUpdate(BaseModel):
    labels: Dict[str, List[str]]


class LabelAction(BaseModel):
    job_id: str
    label: str


@router.get("/labels")
async def get_labels() -> Dict[str, Any]:
    """Get all job labels."""
    labels = _load_labels()
    # Get all unique labels
    all_labels = sorted(set(l for labels_list in labels.values() for l in labels_list))
    return {
        "labels": labels,
        "all_labels": all_labels,
        "n_labeled_jobs": len(labels),
    }


@router.put("/labels")
async def save_labels(body: LabelsUpdate) -> Dict[str, Any]:
    """Replace all labels with new data."""
    # Filter out empty label lists
    labels = {k: v for k, v in body.labels.items() if v}
    _save_labels(labels)
    return {
        "success": True,
        "n_labeled_jobs": len(labels),
    }


@router.post("/labels/add")
async def add_label(body: LabelAction) -> Dict[str, Any]:
    """Add a label to a job."""
    labels = _load_labels()
    if body.job_id not in labels:
        labels[body.job_id] = []
    if body.label not in labels[body.job_id]:
        labels[body.job_id].append(body.label)
    _save_labels(labels)
    return {
        "success": True,
        "job_id": body.job_id,
        "labels": labels[body.job_id],
    }


@router.post("/labels/remove")
async def remove_label(body: LabelAction) -> Dict[str, Any]:
    """Remove a label from a job."""
    labels = _load_labels()
    if body.job_id in labels and body.label in labels[body.job_id]:
        labels[body.job_id].remove(body.label)
        if not labels[body.job_id]:
            del labels[body.job_id]
    _save_labels(labels)
    return {
        "success": True,
        "job_id": body.job_id,
        "labels": labels.get(body.job_id, []),
    }
