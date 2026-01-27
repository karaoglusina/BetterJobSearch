"""FAISS + BM25 index building.

Extracted from rag.py build_index.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .embedder import embed_texts

# Configure FAISS to use single thread
faiss.omp_set_num_threads(1)

ARTIFACT_DIR = Path(__file__).parent.parent.parent / "artifacts"
CHUNKS_PATH = ARTIFACT_DIR / "chunks.jsonl"
FAISS_PATH = ARTIFACT_DIR / "faiss.index"
BM25_PATH = ARTIFACT_DIR / "bm25.pkl"


@dataclass
class IndexArtifacts:
    """Loaded search index artifacts."""

    chunks: List[Dict[str, Any]]
    faiss_index: faiss.Index
    bm25: BM25Okapi


def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization for BM25."""
    return re.findall(r"[A-Za-z0-9_#+.\-]+", text.lower())


def build_faiss(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS IndexFlatIP from normalized embeddings."""
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def build_bm25(corpus_texts: List[str]) -> BM25Okapi:
    """Build a BM25 index from texts."""
    tokenized = [simple_tokenize(t) for t in corpus_texts]
    return BM25Okapi(tokenized)


def build_index(
    chunk_dicts: List[Dict[str, Any]],
    *,
    artifact_dir: Optional[Path] = None,
) -> IndexArtifacts:
    """Build FAISS and BM25 indexes from chunk dictionaries.

    Args:
        chunk_dicts: List of chunk dicts with at least 'text' key.
        artifact_dir: Directory to save artifacts (default: project/artifacts/).

    Returns:
        IndexArtifacts with loaded indexes.
    """
    out_dir = artifact_dir or ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = out_dir / "chunks.jsonl"
    faiss_path = out_dir / "faiss.index"
    bm25_path = out_dir / "bm25.pkl"

    if not chunk_dicts:
        raise ValueError("No chunks to index")

    # Persist chunks
    with open(chunks_path, "w", encoding="utf-8") as f:
        for ch in chunk_dicts:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    texts = [ch.get("text", "") for ch in chunk_dicts]
    print(f"Embedding {len(texts)} chunks...")
    emb = embed_texts(texts)

    index = build_faiss(emb)
    faiss.write_index(index, str(faiss_path))

    bm25 = build_bm25(texts)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    print(f"Index built. Chunks: {len(chunk_dicts)} | FAISS: {faiss_path} | BM25: {bm25_path}")
    return IndexArtifacts(chunks=chunk_dicts, faiss_index=index, bm25=bm25)


def load_artifacts(artifact_dir: Optional[Path] = None) -> IndexArtifacts:
    """Load artifacts from disk."""
    out_dir = artifact_dir or ARTIFACT_DIR
    chunks_path = out_dir / "chunks.jsonl"
    faiss_path = out_dir / "faiss.index"
    bm25_path = out_dir / "bm25.pkl"

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    index = faiss.read_index(str(faiss_path))

    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    return IndexArtifacts(chunks=chunks, faiss_index=index, bm25=bm25)
