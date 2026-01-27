"""Parquet-based caching for cluster projections."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import ARTIFACTS_DIR


class ClusterCache:
    """Cache cluster projections and labels to parquet files.

    File naming: {cache_dir}/clusters_{aspect}.parquet
    Columns: job_id, x, y, cluster_id, cluster_label
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or (ARTIFACTS_DIR / "cluster_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, aspect: str) -> Path:
        safe_name = aspect.replace("/", "_").replace(" ", "_").lower()
        return self.cache_dir / f"clusters_{safe_name}.parquet"

    def has(self, aspect: str) -> bool:
        """Check if cached projection exists for an aspect."""
        return self._path(aspect).exists()

    def load(self, aspect: str) -> Optional[pd.DataFrame]:
        """Load cached projection DataFrame.

        Returns DataFrame with columns: job_id, x, y, cluster_id, cluster_label
        """
        path = self._path(aspect)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def save(
        self,
        aspect: str,
        job_ids: List[str],
        x: List[float],
        y: List[float],
        cluster_ids: List[int],
        cluster_labels: Optional[Dict[int, str]] = None,
    ) -> None:
        """Save projection to parquet.

        Args:
            aspect: Aspect name (used in filename).
            job_ids: Job identifiers.
            x: X coordinates.
            y: Y coordinates.
            cluster_ids: Cluster assignments.
            cluster_labels: Optional mapping of cluster_id -> label.
        """
        labels_map = cluster_labels or {}
        df = pd.DataFrame({
            "job_id": job_ids,
            "x": x,
            "y": y,
            "cluster_id": cluster_ids,
            "cluster_label": [labels_map.get(cid, f"Cluster {cid}") for cid in cluster_ids],
        })
        df.to_parquet(self._path(aspect), index=False)

    def invalidate(self, aspect: Optional[str] = None) -> None:
        """Remove cached projections.

        Args:
            aspect: Specific aspect to invalidate, or None for all.
        """
        if aspect:
            path = self._path(aspect)
            if path.exists():
                path.unlink()
        else:
            for f in self.cache_dir.glob("clusters_*.parquet"):
                f.unlink()

    def list_cached(self) -> List[str]:
        """List all cached aspect names."""
        return [
            f.stem.replace("clusters_", "")
            for f in self.cache_dir.glob("clusters_*.parquet")
        ]
