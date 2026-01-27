"""Search module: FAISS + BM25 hybrid retrieval."""

from .embedder import embed_texts, get_model
from .indexer import build_index, IndexArtifacts
from .retriever import HybridRetriever

__all__ = [
    "embed_texts",
    "get_model",
    "build_index",
    "IndexArtifacts",
    "HybridRetriever",
]
