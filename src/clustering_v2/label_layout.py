"""Greedy overlap-avoidance for cluster labels at each zoom level."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


# Approximate character-width-in-data-units is impossible without knowing the
# viewport, but at server time we can work in data space: we estimate a bbox
# proportional to the text length and the overall plot span, then run greedy
# non-overlap in data units. The frontend can then render the labels as-is; a
# minor amount of viewport-dependent overlap is acceptable because Plotly will
# naturally separate text at different zoom levels.

_CHAR_WIDTH_RATIO = 0.011  # fraction of plot span per character (tuned)
_LINE_HEIGHT_RATIO = 0.025  # fraction of plot span per text line


def _estimate_bbox(
    text: str,
    cx: float,
    cy: float,
    *,
    span_x: float,
    span_y: float,
) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) for a label centered at (cx, cy)."""
    text_len = max(len(text), 1)
    half_w = 0.5 * text_len * _CHAR_WIDTH_RATIO * span_x
    half_h = 0.5 * _LINE_HEIGHT_RATIO * span_y
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _bboxes_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    *,
    padding: float = 0.0,
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (
        ax1 + padding < bx0
        or bx1 + padding < ax0
        or ay1 + padding < by0
        or by1 + padding < ay0
    )


def resolve_label_overlaps(
    labels: List[Dict[str, Any]],
    *,
    span_x: float,
    span_y: float,
    padding_ratio: float = 0.004,
) -> List[Dict[str, Any]]:
    """Greedy overlap resolution; drops lower-priority labels that collide.

    Labels are partitioned by `level` and resolved independently so level-1
    labels never compete with level-0 labels.

    Args:
        labels: Label records with keys `text, x, y, level, priority`.
        span_x: Full x-axis span of the plot.
        span_y: Full y-axis span of the plot.
        padding_ratio: Extra breathing room as a fraction of average span.

    Returns:
        Filtered list of label records (same keys, sorted by level then priority desc).
    """
    if not labels:
        return []

    padding = padding_ratio * max(span_x, span_y)

    by_level: Dict[int, List[Dict[str, Any]]] = {}
    for lbl in labels:
        by_level.setdefault(int(lbl.get("level", 0)), []).append(lbl)

    kept: List[Dict[str, Any]] = []

    for level in sorted(by_level.keys()):
        placed_bboxes: List[tuple[float, float, float, float]] = []
        ordered = sorted(
            by_level[level],
            key=lambda l: float(l.get("priority", 0)),
            reverse=True,
        )
        for lbl in ordered:
            bbox = _estimate_bbox(
                lbl["text"],
                float(lbl["x"]),
                float(lbl["y"]),
                span_x=span_x,
                span_y=span_y,
            )
            collides = any(
                _bboxes_overlap(bbox, placed, padding=padding)
                for placed in placed_bboxes
            )
            if collides:
                continue
            placed_bboxes.append(bbox)
            kept.append(lbl)

    kept.sort(key=lambda l: (int(l.get("level", 0)), -float(l.get("priority", 0))))
    return kept


def compute_plot_spans(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (span_x, span_y) with a tiny floor so division is safe."""
    if x.size == 0 or y.size == 0:
        return 1.0, 1.0
    span_x = float(np.ptp(x)) or 1.0
    span_y = float(np.ptp(y)) or 1.0
    return max(span_x, 1e-6), max(span_y, 1e-6)
