#!/usr/bin/env python3
"""
RAG — Chunking + Hybrid Retrieval for Job Descriptions

Features:
- Clean + chunk descriptions by section/bullets/sentences with overlap
- Embed chunks (sentence-transformers, all-mpnet-base-v2)
- FAISS vector index (cosine) + simple BM25 (rank_bm25)
- Hybrid retrieval (BM25 + vectors) with optional re-ranking
- Minimal chat(): extractive answer or LLM synth if you set OPENAI_API_KEY

Files written/read:
- ./artifacts/chunks.jsonl  (all chunk metadata)
- ./artifacts/faiss.index   (vector index)
- ./artifacts/bm25.pkl      (BM25 corpus)

Usage:
    # Build index from sample data
    from src.rag import build_index, load_cache, retrieve
    
    jobs = [...]  # Load from JSON file
    build_index(docs=jobs)
    
    # Search
    load_cache()
    chunks = retrieve("machine learning engineer")
"""
from __future__ import annotations

# Set environment variables BEFORE imports to prevent segfaults on M1/M2 Macs
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import hashlib
import json
import pickle
import re
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Suppress BeautifulSoup URL warnings (we're parsing text that may contain URLs)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Configure FAISS to use single thread (avoid multiprocessing issues on M1/M2 Macs)
faiss.omp_set_num_threads(1)

# --- Paths ---
ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts"
CHUNKS_PATH = ARTIFACT_DIR / "chunks.jsonl"
FAISS_PATH = ARTIFACT_DIR / "faiss.index"
BM25_PATH = ARTIFACT_DIR / "bm25.pkl"

# --- Optional LLM (OpenAI) ---
OPENAI_AVAILABLE = False
try:
    import openai  # type: ignore

    OPENAI_AVAILABLE = True
except Exception:
    pass

# --- Text patterns ---
HEADER_REGEX = re.compile(
    r"^\s*(responsibilities|what you'?ll do|what you will do|requirements|qualifications|"
    r"must[- ]?have|nice[- ]?to[- ]?have|about you|about us|benefits|compensation|"
    r"culture|how to apply)\s*:?\s*$",
    re.IGNORECASE,
)
BULLET_REGEX = re.compile(r"^\s*([\-\*•\u2022\u25CF\u25E6]|\d+\.)\s+")
SENT_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class Chunk:
    """A single chunk of text from a job description."""

    chunk_id: str
    chunk_id_v2: str  # content-based stable id (job_key + normalized chunk text)
    job_key: str  # stable key for the job, use jobUrl if available
    text: str
    section: Optional[str]
    order: int
    meta: Dict[str, Any]


def _stable_chunk_id(job_key: str, order: int) -> str:
    """Return a stable per-chunk identifier across Python processes."""
    return hashlib.md5(f"{job_key}|{order}".encode("utf-8")).hexdigest()


def _normalize_for_id(text: str) -> str:
    """Normalize text for content-based chunk id generation."""
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _content_chunk_id(job_key: str, text: str) -> str:
    """Return a content-based identifier using job_key + normalized chunk text."""
    norm = _normalize_for_id(text)
    return hashlib.md5(f"{job_key}|{norm}".encode("utf-8")).hexdigest()


def clean_text(html_or_text: str) -> str:
    """Clean HTML and normalize whitespace."""
    if not html_or_text:
        return ""
    try:
        soup = BeautifulSoup(html_or_text, "html.parser")
        text = soup.get_text(" ")
    except Exception:
        text = html_or_text
    text = re.sub(r"\r|\t|\u00A0", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sections(lines: List[str]) -> List[Tuple[Optional[str], List[str]]]:
    """Split lines into named sections based on header patterns."""
    sections: List[Tuple[Optional[str], List[str]]] = []
    current_name: Optional[str] = None
    current_lines: List[str] = []
    for ln in lines:
        if HEADER_REGEX.match(ln):
            if current_lines:
                sections.append((current_name, current_lines))
                current_lines = []
            current_name = ln.strip().lower()
        else:
            current_lines.append(ln)
    if current_lines:
        sections.append((current_name, current_lines))
    return sections


def split_bullets_or_sentences(block: str) -> List[str]:
    """Split a text block into bullet points or windowed sentences."""
    lines = [line.strip() for line in block.split("\n") if line.strip()]
    # If looks like a bullet list, keep bullets as chunks
    if sum(1 for line in lines if BULLET_REGEX.match(line)) >= max(2, len(lines) // 2):
        bullets = []
        for line in lines:
            if BULLET_REGEX.match(line):
                bullets.append(BULLET_REGEX.sub("", line).strip())
            elif bullets:
                # append to previous bullet if continuation line
                bullets[-1] += " " + line
        return [b for b in bullets if len(b) > 3]

    # Else split into sentences and window them
    text = " ".join(lines)
    sents = re.split(SENT_SPLIT_REGEX, text)
    sents = [s.strip() for s in sents if s.strip()]

    # Windowing (approximate 80–200 tokens → ~500–900 chars)
    chunks = []
    buf = []
    char_limit = 900
    for s in sents:
        if sum(len(x) for x in buf) + len(s) + len(buf) <= char_limit:
            buf.append(s)
        else:
            if buf:
                chunks.append(" ".join(buf))
            buf = [s]
    if buf:
        chunks.append(" ".join(buf))

    # Add overlap
    overlapped = []
    prev_tail = ""
    overlap_chars = 150
    for c in chunks:
        piece = (prev_tail + " " + c).strip() if prev_tail else c
        overlapped.append(piece)
        prev_tail = c[-overlap_chars:]
    return overlapped


def make_chunks(doc: Dict[str, Any], order_start: int = 0) -> List[Chunk]:
    """Convert a job document into a list of chunks.

    Args:
        doc: Job document with job_data and optional meta fields, or flat structure
        order_start: Starting order index for chunks

    Returns:
        List of Chunk objects
    """
    # Support both flat structure and nested job_data structure
    job_data = doc.get("job_data", doc)
    
    desc = clean_text(job_data.get("description", ""))
    if not desc:
        return []

    job_key = job_data.get("jobUrl") or (
        f"{job_data.get('companyName', 'unknown')}|"
        f"{job_data.get('title', 'unknown')}"
    )

    # Try to respect line breaks for headers/bullets
    soft_lines = re.sub(r"\s*\n\s*", "\n", job_data.get("description", ""))
    text_lines = [BeautifulSoup(ln, "html.parser").get_text(" ") for ln in soft_lines.split("\n")]
    text_lines = [re.sub(r"\s+", " ", ln).strip() for ln in text_lines]
    text_lines = [ln for ln in text_lines if ln]

    sections = split_sections(text_lines) if text_lines else [(None, [desc])]

    chunks: List[Chunk] = []
    order = order_start
    for section_name, lines in sections:
        block = "\n".join(lines)
        pieces = split_bullets_or_sentences(block)
        for p in pieces:
            if len(p) < 20:
                continue
            raw_text = p.strip()
            ch = Chunk(
                chunk_id=_stable_chunk_id(job_key, order),
                chunk_id_v2=_content_chunk_id(job_key, raw_text),
                job_key=job_key,
                text=raw_text,
                section=section_name,
                order=order,
                meta={
                    "title": job_data.get("title"),
                    "company": job_data.get("companyName"),
                    "days_old": doc.get("meta", {}).get("days_old"),
                    "location": job_data.get("location"),
                    "sector": job_data.get("sector"),
                    "contractType": job_data.get("contractType"),
                    "experienceLevel": job_data.get("experienceLevel"),
                    "applyType": job_data.get("applyType"),
                    "workType": job_data.get("workType"),
                    "salary": job_data.get("salary"),
                    "jobUrl": job_data.get("jobUrl"),
                    "applyUrl": job_data.get("applyUrl"),
                    "applied_times": doc.get("meta", {}).get("applied_times"),
                },
            )
            chunks.append(ch)
            order += 1
    return chunks


# ---------- Index building ----------


def ensure_dir():
    """Ensure artifacts directory exists."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


_MODEL = None


def _get_model(model_name: str = "sentence-transformers/all-mpnet-base-v2") -> SentenceTransformer:
    """Return a cached SentenceTransformer instance for faster repeated calls."""
    global _MODEL
    if _MODEL is None or getattr(_MODEL, "_name", None) != model_name:
        # Force CPU device to avoid MPS issues on M1/M2 Macs
        m = SentenceTransformer(model_name, device='cpu')
        setattr(m, "_name", model_name)
        _MODEL = m
    return _MODEL  # type: ignore[return-value]


def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-mpnet-base-v2") -> np.ndarray:
    """Embed a list of texts using sentence-transformers."""
    model = _get_model(model_name)
    # Use smaller batch size and disable multi-process pool to avoid segfaults
    vecs = model.encode(
        texts, 
        batch_size=32, 
        show_progress_bar=True, 
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return np.asarray(vecs, dtype="float32")


def build_faiss(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS index from embeddings."""
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine if vectors are normalized
    index.add(embeddings)
    return index


def build_bm25(corpus_texts: List[str]) -> BM25Okapi:
    """Build a BM25 index from texts."""
    tokenized = [simple_tokenize(t) for t in corpus_texts]
    return BM25Okapi(tokenized)


def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization for BM25."""
    return re.findall(r"[A-Za-z0-9_#+\.\-]+", text.lower())


def build_index(docs: List[Dict[str, Any]]) -> None:
    """Build FAISS and BM25 indexes from job documents.

    Args:
        docs: List of job documents (each with job_data and optional meta fields)

    Artifacts written:
        - artifacts/chunks.jsonl
        - artifacts/faiss.index
        - artifacts/bm25.pkl
    """
    ensure_dir()
    print(f"Processing {len(docs)} jobs...")

    all_chunks: List[Chunk] = []
    for doc in tqdm(docs, desc="Chunking"):
        all_chunks.extend(make_chunks(doc))

    if not all_chunks:
        print("No chunks produced.")
        return

    # Persist chunks
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")

    texts = [c.text for c in all_chunks]
    print(f"Embedding {len(texts)} chunks…")
    emb = embed_texts(texts)
    index = build_faiss(emb)
    faiss.write_index(index, str(FAISS_PATH))

    bm25 = build_bm25(texts)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(f"Done. Chunks: {len(all_chunks)} | FAISS: {FAISS_PATH} | BM25: {BM25_PATH}")


# ---------- Retrieval & Chat ----------

_CACHE: Dict[str, Any] = {}


def load_artifacts() -> Tuple[List[Chunk], faiss.Index, BM25Okapi]:
    """Load artifacts from disk."""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f]
    index = faiss.read_index(str(FAISS_PATH))
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    return chunks, index, bm25


def load_cache() -> None:
    """Load artifacts into memory cache."""
    chunks, index, bm25 = load_artifacts()
    _CACHE["chunks"] = chunks
    _CACHE["faiss"] = index
    _CACHE["bm25"] = bm25


def is_loaded() -> bool:
    """Check if artifacts are loaded in cache."""
    return "chunks" in _CACHE and "faiss" in _CACHE and "bm25" in _CACHE


def get_all_chunks() -> List[Chunk]:
    """Return all chunks from cache."""
    return _CACHE.get("chunks", [])


def hybrid_search(query: str, k: int = 12, alpha: float = 0.55) -> List[int]:
    """Return indices of top chunks using hybrid scoring.

    score = alpha * vector_score + (1-alpha) * bm25_score_norm

    Args:
        query: Search query
        k: Number of results to return
        alpha: Weight for vector vs BM25 (1.0 = vectors only, 0.0 = BM25 only)
    """
    chunks, index, bm25 = _CACHE["chunks"], _CACHE["faiss"], _CACHE["bm25"]

    # Vector search
    qv = embed_texts([query])[0:1]
    sims, idxs = index.search(qv, k)
    vec_scores = sims[0]
    vec_idxs = idxs[0]

    # BM25 search
    bm_scores = bm25.get_scores(simple_tokenize(query))

    # Combine candidates
    candidates = set(vec_idxs.tolist()) | set(np.argsort(bm_scores)[-k:].tolist())
    candidates = list(candidates)

    # Normalize BM25 scores
    bm_sub = np.array([bm_scores[i] for i in candidates], dtype="float32")
    if bm_sub.max() > 0:
        bm_sub = bm_sub / (bm_sub.max() + 1e-6)

    # Build vector sub-scores aligned to candidates
    vec_map = {int(i): float(s) for i, s in zip(vec_idxs, vec_scores)}
    vec_sub = np.array([vec_map.get(int(i), 0.0) for i in candidates], dtype="float32")

    # Hybrid score
    hybrid = alpha * vec_sub + (1 - alpha) * bm_sub
    order = np.argsort(hybrid)[::-1][:k]
    ranked = [int(candidates[i]) for i in order]
    return ranked


def retrieve(query: str, k: int = 8, alpha: float = 0.55) -> List[Chunk]:
    """Retrieve top-k chunks for a query using hybrid search.

    Args:
        query: Query string
        k: Number of chunks to return
        alpha: Blend between vectors and BM25 (1.0 = vectors only, 0.0 = BM25 only)

    Returns:
        List of Chunk objects
    """
    ranked_idx = hybrid_search(query, k=k, alpha=alpha)
    chunks = [_CACHE["chunks"][i] for i in ranked_idx]
    _CACHE["last_retrieved_chunks"] = chunks
    return chunks


def retrieve_filtered(
    query: str,
    where,
    k: int = 8,
    oversample: int = 120,
    alpha: float = 0.55,
    require_phrase: bool = False,
) -> List[Chunk]:
    """Hybrid search with metadata filtering.

    Args:
        query: Text query searched in chunks (semantic + keyword)
        where: Predicate taking a Chunk and returning True/False
        k: Number of chunks to return after filtering
        oversample: Number of candidates to pull before filtering
        alpha: Hybrid mix (1.0 vector-only → 0.0 BM25-only)
        require_phrase: If True, only keep chunks containing exact query phrase

    Returns:
        List of filtered Chunk objects
    """
    idxs = hybrid_search(query, k=max(k, oversample), alpha=alpha)
    chs = [_CACHE["chunks"][i] for i in idxs]
    chs = [c for c in chs if where(c)]

    if require_phrase:
        qn = _normalize_for_id(query)
        if qn:
            chs = [c for c in chs if qn in _normalize_for_id(getattr(c, "text", ""))]
    return chs[:k]


# ---------- Utility functions ----------


def format_citations(chs: List[Chunk]) -> str:
    """Format chunks as numbered citations."""
    lines = []
    for i, c in enumerate(chs, 1):
        sec = f" [{c.section}]" if c.section else ""
        lines.append(
            f"[{i}] {c.meta.get('company', '?')} — {c.meta.get('title', '?')}{sec} — "
            f"{c.meta.get('jobUrl', '')} (d{c.meta.get('days_old', '?')})"
        )
    return "\n".join(lines)


def build_context(chs: List[Chunk], max_chars: int = 3500) -> str:
    """Build context string from chunks for LLM prompts."""
    ctx, total = [], 0
    for i, c in enumerate(chs, 1):
        snippet = c.text.strip()
        head = f"\n\n[Source {i}] {c.meta.get('company', '?')} — {c.meta.get('title', '?')}\n"
        head += f"URL: {c.meta.get('jobUrl', '')}\n"
        block = head + snippet
        if total + len(block) > max_chars and i > 1:
            break
        ctx.append(block)
        total += len(block)
    return "".join(ctx)


def build_context_with_used(chs: List[Chunk], max_chars: int = 3500) -> Tuple[str, List[Chunk]]:
    """Build context string and return the subset of chunks that fit."""
    used: List[Chunk] = []
    ctx_parts: List[str] = []
    total = 0
    for i, c in enumerate(chs, 1):
        snippet = c.text.strip()
        head = f"\n\n[Source {i}] {c.meta.get('company', '?')} — {c.meta.get('title', '?')}\n"
        head += f"URL: {c.meta.get('jobUrl', '')}\n"
        block = head + snippet
        if total + len(block) > max_chars and i > 1:
            break
        ctx_parts.append(block)
        used.append(c)
        total += len(block)
    return "".join(ctx_parts), used


def _parse_citation_indices(answer_text: str) -> List[int]:
    """Extract [n] citation indices from model output."""
    raw = re.findall(r"\[(\d+)\]", answer_text)
    seen: set[int] = set()
    ordered: List[int] = []
    for tok in raw:
        try:
            val = int(tok)
        except Exception:
            continue
        if val not in seen:
            seen.add(val)
            ordered.append(val)
    return ordered


def answer_extractive(query: str, top_chunks: List[Chunk]) -> str:
    """Simple extractive answer: concatenate cited snippets."""
    header = "Key snippets found (verbatim, lightly trimmed):\n"
    body = "\n\n".join([f"[{i + 1}] {c.text.strip()}" for i, c in enumerate(top_chunks)])
    return header + body


def answer_with_llm(
    query: str,
    top_chunks: List[Chunk],
    model: str = "gpt-4o-mini",
    *,
    max_context_chars: int = 3500,
) -> str:
    """Answer query using LLM with retrieved context."""
    if not OPENAI_AVAILABLE or not os.environ.get("OPENAI_API_KEY"):
        return answer_extractive(query, top_chunks)

    openai.api_key = os.environ["OPENAI_API_KEY"]
    system = (
        "You are a concise job-market analyst assistant. Use ONLY the provided sources. "
        "Cite using [n] markers that match 'Source n' blocks. If unsure, say so."
    )
    ctx, used_chunks = build_context_with_used(top_chunks, max_chars=max_context_chars)
    prompt = (
        f"Question: {query}\n\nContext from job postings:\n{ctx}\n\n"
        "Instructions: Answer the question using the sources. Include [n] citations."
    )

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = resp["choices"][0]["message"]["content"].strip()

        # Parse citations and keep only cited subset
        indices = _parse_citation_indices(text)
        if indices:
            cited = [used_chunks[i - 1] for i in indices if 1 <= i <= len(used_chunks)]
        else:
            cited = list(used_chunks)
        _CACHE["used_chunks"] = list(used_chunks)
        _CACHE["cited_chunks"] = cited
        return text
    except Exception as e:
        _CACHE["used_chunks"] = list(used_chunks)
        _CACHE["cited_chunks"] = list(used_chunks)
        return f"[LLM fallback to extractive due to error: {e}]\n\n" + answer_extractive(query, used_chunks)


# ---------- Deduplication utilities ----------


def dedupe_by_job(chunks: List[Chunk]) -> List[Chunk]:
    """Keep only first chunk per job."""
    seen = OrderedDict()
    for c in chunks:
        seen.setdefault(c.job_key, c)
    return list(seen.values())


def dedupe_by_chunk(chunks: List[Chunk]) -> List[Chunk]:
    """Dedupe chunks by chunk_id_v2 and merge matched_queries."""
    seen: OrderedDict[str, Chunk] = OrderedDict()
    for c in chunks:
        cid = getattr(c, "chunk_id_v2", None) or getattr(c, "chunk_id", None)
        key = str(cid) if cid is not None else str(id(c))
        if key not in seen:
            seen[key] = c
            continue
        # Merge matched_queries
        base = seen[key].meta.get("matched_queries") if hasattr(seen[key], "meta") else None
        add = c.meta.get("matched_queries") if hasattr(c, "meta") else None
        base_set = set(base) if isinstance(base, (set, list, tuple)) else set()
        add_set = set(add) if isinstance(add, (set, list, tuple)) else set()
        union = base_set | add_set
        if union:
            seen[key].meta["matched_queries"] = union
    return list(seen.values())


def expand_to_full_jobs(
    chunks: List[Chunk],
    *,
    universe: Optional[List[Chunk]] = None,
) -> List[Chunk]:
    """Expand chunks to include all chunks from the same jobs.

    Args:
        chunks: Seed chunks (their job_key determines which jobs to expand)
        universe: Optional source of all chunks. Defaults to cached artifacts.

    Returns:
        All chunks from jobs present in input, deduped by chunk_id_v2.
    """
    if not chunks:
        return []

    all_chunks = universe if universe is not None else _CACHE.get("chunks")
    if all_chunks is None:
        raise ValueError("Chunks universe not loaded. Call load_cache() or pass universe=...")

    # Preserve input job_key order
    job_order: List[str] = []
    seen_jobs: set[str] = set()
    no_job_chunks: List[Chunk] = []

    for c in chunks:
        jk = getattr(c, "job_key", None)
        if not jk:
            no_job_chunks.append(c)
            continue
        if jk not in seen_jobs:
            seen_jobs.add(jk)
            job_order.append(jk)

    # Build job_key -> chunks index from universe
    jobs_to_chunks: Dict[str, List[Chunk]] = {}
    for ac in all_chunks:
        jk = getattr(ac, "job_key", None)
        if not jk:
            continue
        jobs_to_chunks.setdefault(jk, []).append(ac)

    # Flatten in input job order, sort per job by `order`
    out: List[Chunk] = []
    seen_chunk_ids: set[str] = set()
    for jk in job_order:
        job_chunks = jobs_to_chunks.get(jk, [])
        job_chunks_sorted = sorted(job_chunks, key=lambda x: getattr(x, "order", 0))
        for ac in job_chunks_sorted:
            cid = getattr(ac, "chunk_id_v2", None) or getattr(ac, "chunk_id", None)
            key = str(cid) if cid is not None else str(id(ac))
            if key in seen_chunk_ids:
                continue
            seen_chunk_ids.add(key)
            out.append(ac)

    # Append input chunks that had no job_key
    for c in no_job_chunks:
        cid = getattr(c, "chunk_id_v2", None) or getattr(c, "chunk_id", None)
        key = str(cid) if cid is not None else str(id(c))
        if key in seen_chunk_ids:
            continue
        seen_chunk_ids.add(key)
        out.append(c)

    return out


# ---------- Chat loop ----------


def chat_loop(model: str = "gpt-4o-mini"):
    """Interactive chat loop for querying the job corpus."""
    load_cache()
    print("\nType your question (or 'exit'):\n")
    while True:
        try:
            q = input("» ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        top = retrieve(q, k=8)
        ans = answer_with_llm(q, top, model=model)
        print("\n—— ANSWER —————————————————————————————————\n")
        print(ans)
        print("\n—— SOURCES ———————————————————————————————\n")
        print(format_citations(top))
        print("\n")


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="RAG system for job descriptions")
    p.add_argument("--build", type=str, metavar="JSON_FILE", help="Build index from JSON file")
    p.add_argument("--chat", action="store_true", help="Start a simple chat loop")
    p.add_argument("--model", default="gpt-4o-mini", help="LLM model name (if using OpenAI)")
    args = p.parse_args()

    if args.build:
        json_path = Path(args.build)
        if not json_path.exists():
            print(f"Error: File not found: {json_path}")
            sys.exit(1)

        print(f"Loading jobs from {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both list and dict with 'jobs' key
        if isinstance(data, list):
            docs = data
        elif isinstance(data, dict) and "jobs" in data:
            docs = data["jobs"]
        else:
            print("Error: JSON must be a list of jobs or a dict with 'jobs' key")
            sys.exit(1)

        build_index(docs=docs)

    if args.chat:
        if not CHUNKS_PATH.exists():
            print("No artifacts found. Run with --build first.")
            sys.exit(1)
        chat_loop(model=args.model)

    if not args.build and not args.chat:
        p.print_help()
