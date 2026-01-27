"""SBERT embedding utilities.

Uses MPS (Metal Performance Shaders) on Apple Silicon when available,
falls back to CPU otherwise.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

_model = None
_model_name = None


def _best_device() -> str:
    """Pick the best available device: MPS (Apple Silicon) > CPU."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model(model_name: str = DEFAULT_MODEL):
    """Return a cached SentenceTransformer instance."""
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from sentence_transformers import SentenceTransformer
        device = _best_device()
        _model = SentenceTransformer(model_name, device=device)
        _model_name = model_name
    return _model


def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    *,
    batch_size: int = 32,
    show_progress: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Embed a list of texts using sentence-transformers.

    Args:
        texts: List of text strings.
        model_name: SBERT model name.
        batch_size: Encoding batch size.
        show_progress: Show progress bar.
        normalize: L2-normalize embeddings (required for cosine similarity with FAISS IP).

    Returns:
        numpy array of shape (len(texts), dim), dtype float32.
    """
    model = get_model(model_name)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype="float32")


def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed a single query string. Returns shape (1, dim)."""
    return embed_texts([query], model_name=model_name, show_progress=False)
