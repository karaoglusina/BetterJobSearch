"""SBERT embedding utilities.

Forces CPU device to avoid segfaults on Apple Silicon (M1/M2/M3)
when multiple threads access the model concurrently.
"""

from __future__ import annotations

# Must be set BEFORE importing numpy/torch/sentence-transformers
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import threading
from typing import List

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

_model = None
_model_name = None
_model_lock = threading.Lock()


def get_model(model_name: str = DEFAULT_MODEL):
    """Return a cached SentenceTransformer instance (thread-safe)."""
    global _model, _model_name
    if _model is not None and _model_name == model_name:
        return _model
    with _model_lock:
        # Double-check after acquiring lock
        if _model is not None and _model_name == model_name:
            return _model
        from sentence_transformers import SentenceTransformer
        # Force CPU to avoid MPS segfaults with concurrent thread access
        _model = SentenceTransformer(model_name, device="cpu")
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
