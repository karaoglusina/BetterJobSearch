"""Concave-hull / alpha-shape polygons per cluster for Atlas-style 'continents'."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _chaikin_smooth(points: List[List[float]], iterations: int = 2) -> List[List[float]]:
    """Chaikin's corner-cutting smoothing for closed polygons.

    Produces a softer, more blob-like outline than the raw hull.
    """
    if len(points) < 3:
        return points

    pts = np.asarray(points, dtype=float)
    for _ in range(iterations):
        n = len(pts)
        smoothed = np.empty((2 * n, 2), dtype=float)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            smoothed[2 * i] = 0.75 * p0 + 0.25 * p1
            smoothed[2 * i + 1] = 0.25 * p0 + 0.75 * p1
        pts = smoothed
    return pts.tolist()


def _convex_hull_path(xy: np.ndarray) -> List[List[float]]:
    """Fallback: convex hull ordered counter-clockwise."""
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return xy.tolist()
    if xy.shape[0] < 3:
        return xy.tolist()
    try:
        hull = ConvexHull(xy)
    except Exception:
        return xy.tolist()
    return xy[hull.vertices].tolist()


def _alpha_shape_path(xy: np.ndarray, alpha: float) -> Optional[List[List[float]]]:
    """Build a single-ring concave-hull path using alphashape.

    Returns None if alphashape is unavailable or the shape is degenerate.
    """
    try:
        import alphashape
        from shapely.geometry import Polygon, MultiPolygon
    except ImportError:
        return None

    if xy.shape[0] < 4:
        return None

    try:
        shape = alphashape.alphashape(xy, alpha)
    except Exception:
        return None

    if shape is None or shape.is_empty:
        return None

    # Pick the largest polygon if we got a MultiPolygon
    if isinstance(shape, MultiPolygon):
        shape = max(shape.geoms, key=lambda p: p.area)

    if not isinstance(shape, Polygon):
        return None

    # Buffer slightly to avoid razor-thin shapes; simplify to keep the payload small.
    try:
        buffered = shape.buffer(0.0)
        simplified = buffered.simplify(tolerance=max(buffered.area ** 0.5 * 0.02, 1e-3))
        if isinstance(simplified, MultiPolygon):
            simplified = max(simplified.geoms, key=lambda p: p.area)
        if isinstance(simplified, Polygon) and not simplified.is_empty:
            shape = simplified
    except Exception:
        pass

    coords = list(shape.exterior.coords)
    if len(coords) < 4:
        return None
    # Drop duplicate closing point
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    return [[float(x), float(y)] for x, y in coords]


def compute_cluster_hulls(
    x: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    alpha: Optional[float] = None,
    smooth_iterations: int = 2,
    min_points: int = 4,
) -> List[Dict[str, Any]]:
    """Compute a soft polygon per cluster.

    Args:
        x: X coordinates (n,).
        y: Y coordinates (n,).
        cluster_ids: Cluster labels (n,). -1 is noise (skipped).
        alpha: alphashape parameter. If None, auto-tune based on cluster spread.
        smooth_iterations: Chaikin smoothing passes.
        min_points: Minimum points required to compute a polygon.

    Returns:
        List of {cluster_id, path: [[x, y], ...]} records, ordered by cluster_id.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cluster_ids = np.asarray(cluster_ids)

    unique = sorted(int(c) for c in set(cluster_ids.tolist()) if c >= 0)
    results: List[Dict[str, Any]] = []

    for cid in unique:
        mask = cluster_ids == cid
        if mask.sum() < min_points:
            continue
        xy = np.column_stack([x[mask], y[mask]])

        # Auto-alpha: inverse of the mean pairwise spread; small alpha => convex-ish,
        # larger alpha => tighter concave.
        if alpha is None:
            span = float(np.sqrt(xy.var(axis=0).sum()) + 1e-6)
            auto_alpha = 1.0 / max(span, 0.25)
        else:
            auto_alpha = alpha

        path = _alpha_shape_path(xy, auto_alpha)
        if path is None:
            path = _convex_hull_path(xy)

        if len(path) < 3:
            continue

        if smooth_iterations > 0:
            path = _chaikin_smooth(path, iterations=smooth_iterations)

        results.append({"cluster_id": cid, "path": path})

    return results
