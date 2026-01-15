"""Job and chunk clustering (v5 - standalone).

This module provides a dependency-light pipeline to:
- Vectorize chunk/job texts with TF-IDF or SBERT
- Reduce dimensionality with TruncatedSVD (LSA) for plotting and clustering
- Cluster with KMeans (auto-k via silhouette if not provided)
- Extract top keywords per cluster
- Produce DataFrames for visualization

Designed for interactive notebook use on the list of Chunk objects returned
from `src.rag.retrieve(...)` or `retrieve_filtered(...)`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def _prepare_texts(chunks: List[Any]) -> List[str]:
    """Extract text strings from chunk objects."""
    texts: List[str] = []
    for c in chunks:
        txt = getattr(c, "text", None)
        if isinstance(txt, str) and txt.strip():
            texts.append(txt)
        else:
            texts.append("")
    return texts


def _auto_k(svd_features: np.ndarray, k_min: int, k_max: int, random_state: int) -> int:
    """Automatically determine best k using silhouette score."""
    best_k = k_min
    best_score = -1.0
    for k in range(k_min, k_max + 1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            labels = km.fit_predict(svd_features)
            if len(set(labels)) <= 1:
                continue
            sc = silhouette_score(svd_features, labels)
            if sc > best_score:
                best_score = sc
                best_k = k
        except Exception:
            continue
    return best_k


def _top_terms_per_cluster(
    tfidf_matrix: Any,
    labels: np.ndarray,
    vectorizer: TfidfVectorizer,
    top_n: int = 10,
) -> Dict[int, List[str]]:
    """Compute top-N terms per cluster by averaging tf-idf weights."""
    terms = vectorizer.get_feature_names_out()
    keywords: Dict[int, List[str]] = {}
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if not np.any(mask):
            keywords[cluster_id] = []
            continue
        cluster_tfidf = tfidf_matrix[mask]
        mean_vector = np.asarray(cluster_tfidf.mean(axis=0)).ravel()
        top_idx = np.argsort(mean_vector)[::-1][:top_n]
        keywords[cluster_id] = [str(terms[i]) for i in top_idx if mean_vector[i] > 0]
    return keywords


def cluster_chunks(
    chunks: List[Any],
    n_clusters: Optional[int] = None,
    *,
    max_k: int = 12,
    min_k: int = 2,
    svd_components_features: int = 50,
    random_state: int = 42,
    top_terms: int = 10,
) -> Dict[str, Any]:
    """Cluster chunk texts and return labels, keywords, and plotting coords.

    Args:
        chunks: List of Chunk objects with .text and .meta attributes
        n_clusters: Number of clusters (auto-determined if None)
        max_k: Maximum k for auto-detection
        min_k: Minimum k for auto-detection
        svd_components_features: Number of SVD components for features
        random_state: Random seed for reproducibility
        top_terms: Number of top keywords to extract per cluster

    Returns:
        Dict with keys:
        - k: int - number of clusters
        - labels: np.ndarray of shape (N,)
        - coords_2d: np.ndarray of shape (N, 2) for plotting
        - keywords: Dict[int, List[str]] per cluster
        - df: pandas.DataFrame with columns [x, y, cluster, title, company, url, text]
    """
    if not chunks:
        return {
            "k": 0,
            "labels": np.array([]),
            "coords_2d": np.zeros((0, 2)),
            "keywords": {},
            "df": pd.DataFrame(),
        }

    texts = _prepare_texts(chunks)

    # Vectorize with TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
    )
    tfidf = vectorizer.fit_transform(texts)

    # Reduce to feature space for clustering
    feat_dim = max(2, min(svd_components_features, min(tfidf.shape) - 1))
    svd_feat = TruncatedSVD(n_components=feat_dim, random_state=random_state)
    features = svd_feat.fit_transform(tfidf)

    # 2D projection for plotting
    svd_2d = TruncatedSVD(n_components=2, random_state=random_state)
    coords_2d = svd_2d.fit_transform(tfidf)

    # Decide K automatically if not provided
    if n_clusters is None:
        n_samples = len(chunks)
        upper = min(max_k, max(min_k, n_samples // 10) or min_k)
        lower = min_k if upper >= min_k else 2
        n_clusters = _auto_k(features, lower, max(lower, upper), random_state)
        if n_clusters < 2:
            n_clusters = 2

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(features)

    # Keywords per cluster
    keywords = _top_terms_per_cluster(tfidf, labels, vectorizer, top_n=top_terms)

    # Build DataFrame
    rows: List[Dict[str, Any]] = []
    for i, c in enumerate(chunks):
        meta = getattr(c, "meta", {}) or {}
        rows.append(
            {
                "x": float(coords_2d[i, 0]),
                "y": float(coords_2d[i, 1]),
                "cluster": int(labels[i]),
                "title": meta.get("title"),
                "company": meta.get("company"),
                "url": meta.get("jobUrl"),
                "text": (getattr(c, "text", "") or "")[:240],
            }
        )
    df = pd.DataFrame(rows)

    return {
        "k": int(n_clusters),
        "labels": labels,
        "coords_2d": coords_2d,
        "keywords": keywords,
        "df": df,
    }


def build_job_texts_full() -> List[Dict[str, Any]]:
    """Build per-job texts by concatenating all chunk texts for each job.

    Uses RAG artifacts only; no external database required.

    Returns:
        List of dicts with keys: job_id, text, title, company, url, n_chunks
    """
    from . import rag  # lazy import

    if not rag.is_loaded():
        rag.load_cache()
    chunks = rag.get_all_chunks()

    by_job: Dict[str, List[Any]] = {}
    for c in chunks:
        jk = getattr(c, "job_key", None)
        if not jk:
            continue
        by_job.setdefault(jk, []).append(c)

    rows: List[Dict[str, Any]] = []
    for job_id, arr in by_job.items():
        arr_sorted = sorted(arr, key=lambda x: getattr(x, "order", 0))
        seen: set = set()
        texts: List[str] = []
        title = None
        company = None
        url = None
        for c in arr_sorted:
            cid = getattr(c, "chunk_id_v2", None)
            if cid in seen:
                continue
            seen.add(cid)
            if title is None:
                title = (getattr(c, "meta", {}) or {}).get("title")
            if company is None:
                company = (getattr(c, "meta", {}) or {}).get("company")
            if url is None:
                url = (getattr(c, "meta", {}) or {}).get("jobUrl")
            txt = getattr(c, "text", None) or ""
            if txt:
                texts.append(str(txt))
        description = "\n\n".join(texts)
        rows.append(
            {
                "job_id": str(job_id),
                "text": description,
                "title": title,
                "company": company,
                "url": url,
                "n_chunks": len(seen),
            }
        )
    return rows


def cluster_jobs(
    jobs: Optional[List[Dict[str, Any]]] = None,
    n_clusters: Optional[int] = None,
    *,
    max_k: int = 12,
    min_k: int = 2,
    random_state: int = 42,
    embed_mode: str = "sbert",  # "sbert" | "tfidf"
) -> Dict[str, Any]:
    """Cluster jobs using full text.

    Args:
        jobs: List of job dicts (if None, builds from RAG artifacts)
        n_clusters: Number of clusters (auto-determined if None)
        max_k: Maximum k for auto-detection
        min_k: Minimum k for auto-detection
        random_state: Random seed
        embed_mode: "sbert" for neural embeddings, "tfidf" for traditional

    Returns:
        Dict with keys: k, labels, coords_2d, keywords, df
    """
    if jobs is None:
        jobs = build_job_texts_full()

    if not jobs:
        return {
            "k": 0,
            "labels": np.array([]),
            "coords_2d": np.zeros((0, 2)),
            "keywords": {},
            "df": pd.DataFrame(),
        }

    texts = [r.get("text", "") or "" for r in jobs]

    if embed_mode == "sbert":
        # Neural embeddings using rag.embed_texts
        from . import rag

        emb = rag.embed_texts(texts)
        features = emb

        # 2D projection for plotting
        svd_2d = TruncatedSVD(n_components=2, random_state=random_state)
        coords_2d = svd_2d.fit_transform(emb)
        vectorizer = None
        tfidf = None
    else:
        # TF-IDF baseline
        vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True,
        )
        tfidf = vectorizer.fit_transform(texts)

        # Reduce to feature space for clustering
        feat_dim = max(2, min(50, min(tfidf.shape) - 1))
        svd_feat = TruncatedSVD(n_components=feat_dim, random_state=random_state)
        features = svd_feat.fit_transform(tfidf)

        # 2D projection for plotting
        svd_2d = TruncatedSVD(n_components=2, random_state=random_state)
        coords_2d = svd_2d.fit_transform(tfidf)

    # Decide K automatically if not provided
    if n_clusters is None:
        n_samples = len(jobs)
        upper = min(max_k, max(min_k, n_samples // 10) or min_k)
        lower = min_k if upper >= min_k else 2
        n_clusters = _auto_k(features, lower, max(lower, upper), random_state)
        if n_clusters < 2 and n_samples >= 2:
            n_clusters = 2
        if n_samples < 2:
            return {
                "k": 1,
                "labels": np.zeros((n_samples,), dtype=int),
                "coords_2d": coords_2d,
                "keywords": {},
                "df": pd.DataFrame(jobs),
            }

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(features)

    # Keywords per cluster (only for TF-IDF mode)
    keywords: Dict[int, List[str]] = {}
    if embed_mode == "tfidf" and tfidf is not None and vectorizer is not None:
        keywords = _top_terms_per_cluster(tfidf, labels, vectorizer, top_n=10)

    # Build DataFrame
    df = pd.DataFrame(
        [
            {
                "x": float(coords_2d[i, 0]),
                "y": float(coords_2d[i, 1]),
                "cluster": int(labels[i]),
                "job_id": jobs[i].get("job_id"),
                "title": jobs[i].get("title"),
                "company": jobs[i].get("company"),
                "url": jobs[i].get("url"),
                "text": (jobs[i].get("text") or "")[:400],
                "n_chunks": int(jobs[i].get("n_chunks", 0)),
            }
            for i in range(len(jobs))
        ]
    )

    return {
        "k": int(n_clusters),
        "labels": labels,
        "coords_2d": coords_2d,
        "keywords": keywords,
        "df": df,
    }


def compute_and_save_job_embeddings(
    out_path: str = "data/job_group_embeddings.parquet",
) -> str:
    """Compute SBERT embeddings per job and save to Parquet.

    Saves columns: [job_id, title, company, url, n_chunks, vector]
    """
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq

    from . import rag

    rows = build_job_texts_full()
    if not rows:
        # Write empty file
        table = pa.Table.from_pylist([])
        pq.write_table(table, out_path)
        return out_path

    texts = [r.get("text", "") or "" for r in rows]
    emb = rag.embed_texts(texts)

    all_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        all_rows.append(
            {
                "job_id": r.get("job_id"),
                "group": "full_text",
                "title": r.get("title"),
                "company": r.get("company"),
                "url": r.get("url"),
                "n_chunks": int(r.get("n_chunks", 0)),
                "vector": emb[i].tolist(),
            }
        )

    # Convert to arrow and save
    dim = len(all_rows[0]["vector"]) if all_rows else 0
    fields = [
        pa.field("job_id", pa.string()),
        pa.field("group", pa.string()),
        pa.field("title", pa.string()),
        pa.field("company", pa.string()),
        pa.field("url", pa.string()),
        pa.field("n_chunks", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), list_size=dim) if dim > 0 else pa.list_(pa.float32())),
    ]
    schema = pa.schema(fields)

    norm_rows: List[Dict[str, Any]] = []
    for r in all_rows:
        v = r.get("vector") or []
        if dim and len(v) != dim:
            if len(v) < dim:
                v = list(v) + [0.0] * (dim - len(v))
            else:
                v = list(v)[:dim]
        norm_rows.append(
            {
                "job_id": r.get("job_id"),
                "group": r.get("group"),
                "title": r.get("title"),
                "company": r.get("company"),
                "url": r.get("url"),
                "n_chunks": int(r.get("n_chunks", 0)),
                "vector": v,
            }
        )

    table = pa.Table.from_pylist(norm_rows, schema=schema)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pq.write_table(table, out_path)
    return out_path


def cluster_jobs_from_parquet(
    path: str,
    n_clusters: Optional[int] = None,
    *,
    max_k: int = 12,
    min_k: int = 2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Cluster jobs using precomputed SBERT vectors from Parquet file.

    Args:
        path: Path to Parquet file with job embeddings
        n_clusters: Number of clusters (auto-determined if None)
        max_k: Maximum k for auto-detection
        min_k: Minimum k for auto-detection
        random_state: Random seed

    Returns:
        Dict with keys: k, labels, coords_2d, keywords, df
    """
    # Load parquet
    try:
        dfv = pd.read_parquet(path)
    except Exception:
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            data = {name: table[name].to_pylist() for name in table.column_names}
            dfv = pd.DataFrame(data)
        except Exception as ee:
            raise ee

    if dfv.empty:
        return {
            "k": 0,
            "labels": np.array([]),
            "coords_2d": np.zeros((0, 2)),
            "keywords": {},
            "df": pd.DataFrame(),
        }

    # Build feature matrix from list vectors
    vectors = dfv["vector"].tolist()
    X = np.asarray([np.asarray(v, dtype="float32") for v in vectors], dtype="float32")

    # 2D projection for plotting
    svd_2d = TruncatedSVD(n_components=2, random_state=random_state)
    coords_2d = svd_2d.fit_transform(X)

    # Choose k if not provided
    if n_clusters is None:
        n_samples = len(X)
        upper = min(max_k, max(min_k, n_samples // 10) or min_k)
        lower = min_k if upper >= min_k else 2

        # Project to 50 dims for silhouette speed
        proj = TruncatedSVD(
            n_components=min(50, X.shape[1] - 1) if X.shape[1] > 2 else 2,
            random_state=random_state,
        )
        feat = proj.fit_transform(X)
        n_clusters = _auto_k(feat, lower, max(lower, upper), random_state)

        if n_clusters < 2 and n_samples >= 2:
            n_clusters = 2
        if n_samples < 2:
            out_df = dfv[["job_id", "title", "company", "url", "n_chunks"]].copy()
            out_df["x"], out_df["y"], out_df["cluster"], out_df["text"] = 0.0, 0.0, 0, ""
            return {
                "k": 1,
                "labels": np.zeros((n_samples,), dtype=int),
                "coords_2d": coords_2d,
                "keywords": {},
                "df": out_df,
            }

    # Cluster in vector space
    kmeans = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(X)

    # Build output DataFrame
    out_rows: List[Dict[str, Any]] = []
    for i, row in dfv.reset_index(drop=True).iterrows():
        out_rows.append(
            {
                "x": float(coords_2d[i, 0]),
                "y": float(coords_2d[i, 1]),
                "cluster": int(labels[i]),
                "job_id": row.get("job_id"),
                "title": row.get("title"),
                "company": row.get("company"),
                "url": row.get("url"),
                "n_chunks": int(row.get("n_chunks", 0)),
            }
        )
    df = pd.DataFrame(out_rows)

    return {
        "k": int(n_clusters),
        "labels": labels,
        "coords_2d": coords_2d,
        "keywords": {},
        "df": df,
    }


def keywords_summary(result: Dict[str, Any], top: int = 6) -> List[Tuple[int, List[str]]]:
    """Return list of (cluster_id, top_keywords) pairs for quick printing."""
    out: List[Tuple[int, List[str]]] = []
    kw = result.get("keywords", {}) or {}
    for cid, words in sorted(kw.items()):
        out.append((int(cid), list(words)[:top]))
    return out
