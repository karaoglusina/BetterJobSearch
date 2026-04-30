"""Glasbey-style perceptually-distinct palette for cluster ids."""

from __future__ import annotations

from typing import Dict, Iterable, List

# Vendored Glasbey-dark-like palette: 64 perceptually-distinct colors tuned for
# dark backgrounds. These were generated from colorcet's glasbey_bw_minc_20
# family as a one-off so we don't require colorcet at runtime; if colorcet is
# installed we prefer its palette.
_VENDORED_GLASBEY_DARK: List[str] = [
    "#d60000", "#018700", "#b500ff", "#05acc6", "#97ff00", "#ffa52f", "#ff8ec8",
    "#79525e", "#00fdcf", "#afa5ff", "#93ac83", "#9a6900", "#366962", "#d3008c",
    "#fdf490", "#c86e66", "#9ee2ff", "#00c846", "#a877ac", "#b8ba01", "#f4bfb1",
    "#ff28fd", "#f2cdff", "#009e7c", "#ff6200", "#56642a", "#953f1f", "#90318e",
    "#ff3464", "#a0e491", "#8c9ab1", "#829026", "#ae083f", "#77c6ba", "#bc9157",
    "#e48eff", "#72b8ff", "#c6a5c1", "#ff9070", "#d3c37c", "#bceddb", "#6b8567",
    "#916e56", "#f9ff00", "#bac1df", "#ac567c", "#ffcd03", "#ff49b1", "#c15603",
    "#5d8c8c", "#8c5b73", "#d4fff0", "#7c829a", "#6a43b9", "#70ff7c", "#ffb1c1",
    "#d14b37", "#0066a8", "#e39a9a", "#6a8400", "#b6a5d8", "#a7c1fa", "#5e6eff",
    "#ff2e1a",
]

# Noise / unclustered
NOISE_COLOR = "#555555"


def _load_colorcet_palette() -> List[str] | None:
    """Return colorcet glasbey_dark if available, else None."""
    try:
        import colorcet as cc
    except ImportError:
        return None

    # glasbey_dark is a list of hex strings in recent colorcet releases.
    for attr in ("glasbey_dark", "b_glasbey_bw_minc_20", "glasbey"):
        palette = getattr(cc, attr, None)
        if palette:
            return list(palette)
    return None


def build_palette(cluster_ids: Iterable[int]) -> Dict[str, str]:
    """Map each unique non-negative cluster id to a distinct hex color.

    Noise (id = -1) maps to `NOISE_COLOR`. Returns a dict keyed by *string*
    cluster id so it serializes cleanly to JSON.
    """
    palette = _load_colorcet_palette() or _VENDORED_GLASBEY_DARK
    n = len(palette)

    unique_positive = sorted({int(c) for c in cluster_ids if int(c) >= 0})

    result: Dict[str, str] = {"-1": NOISE_COLOR}
    for cid in unique_positive:
        # Modulo so we never run out even with huge cluster counts.
        result[str(cid)] = palette[cid % n]
    return result
