"""Cluster labeling using c-TF-IDF and optional LLM."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..models.cluster import ClusterInfo


def compute_ctfidf(
    documents_per_cluster: Dict[int, List[str]],
    *,
    top_n: int = 10,
    min_df: float = 0.0,
    max_df: float = 1.0,
    meaningful_phrases: Optional[List[str]] = None,
) -> Dict[int, List[str]]:
    """Compute class-based TF-IDF (c-TF-IDF) keywords per cluster.

    c-TF-IDF: concatenate all docs in a cluster into one "class document",
    then compute TF-IDF across classes to find distinctive terms.

    Args:
        documents_per_cluster: Mapping of cluster_id -> list of document texts.
        top_n: Number of top terms per cluster.
        min_df: Minimum document frequency ratio (0.0 to 1.0). Terms appearing
            in fewer than min_df * n_documents are ignored.
        max_df: Maximum document frequency ratio (0.0 to 1.0). Terms appearing
            in more than max_df * n_documents are ignored.
        meaningful_phrases: Optional whitelist of phrases to prefer.

    Returns:
        Dict of cluster_id -> list of top keywords.
    """
    if not documents_per_cluster:
        return {}

    cluster_ids = sorted(documents_per_cluster.keys())
    # Skip noise cluster (-1)
    cluster_ids = [c for c in cluster_ids if c >= 0]

    if not cluster_ids:
        return {}

    # Concatenate documents per cluster
    class_docs = [" ".join(documents_per_cluster.get(cid, [""])) for cid in cluster_ids]

    # Convert min_df/max_df from ratio to appropriate format
    # TfidfVectorizer accepts float (proportion) or int (absolute count)
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
        min_df=max(min_df, 0.0),  # Ensure non-negative
        max_df=min(max_df, 1.0),  # Ensure <= 1.0
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(class_docs)
    except ValueError:
        # Empty vocabulary, return empty results
        return {cid: [] for cid in cluster_ids}

    feature_names = vectorizer.get_feature_names_out()

    # Build phrase set for filtering if provided
    phrase_set = set(meaningful_phrases) if meaningful_phrases else None

    result: Dict[int, List[str]] = {}
    for i, cid in enumerate(cluster_ids):
        row = np.asarray(tfidf_matrix[i].todense()).ravel()
        top_indices = np.argsort(row)[::-1]

        keywords = []
        for j in top_indices:
            if row[j] <= 0:
                break
            term = str(feature_names[j])
            # If phrase whitelist is provided, prefer those terms
            if phrase_set is not None and len(keywords) < top_n:
                if term in phrase_set or any(term in p for p in phrase_set):
                    keywords.append(term)
                elif len(keywords) < top_n // 2:
                    # Allow some non-whitelisted terms
                    keywords.append(term)
            else:
                keywords.append(term)
            if len(keywords) >= top_n:
                break

        result[cid] = keywords

    return result


def label_clusters_with_llm(
    keywords_per_cluster: Dict[int, List[str]],
    titles_per_cluster: Dict[int, List[str]],
    *,
    model: str = "gpt-4o-mini",
) -> Dict[int, str]:
    """Generate human-readable labels for clusters using LLM.

    Args:
        keywords_per_cluster: c-TF-IDF top keywords per cluster.
        titles_per_cluster: Sample job titles per cluster.
        model: OpenAI model to use.

    Returns:
        Dict of cluster_id -> label string.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        # Fallback: use top 3 keywords as label
        return {
            cid: ", ".join(kws[:3]) if kws else f"Cluster {cid}"
            for cid, kws in keywords_per_cluster.items()
        }

    try:
        from openai import OpenAI
        client = OpenAI()
    except ImportError:
        return {
            cid: ", ".join(kws[:3]) if kws else f"Cluster {cid}"
            for cid, kws in keywords_per_cluster.items()
        }

    labels: Dict[int, str] = {}

    for cid in sorted(keywords_per_cluster.keys()):
        kws = keywords_per_cluster.get(cid, [])[:5]
        titles = titles_per_cluster.get(cid, [])[:3]

        prompt = (
            f"Given these top keywords for a job cluster: {json.dumps(kws)}\n"
            f"And sample job titles: {json.dumps(titles)}\n\n"
            "Generate a short (2-5 word) descriptive label for this cluster.\n"
            "Return only the label, no quotes or explanation."
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=30,
            )
            label = resp.choices[0].message.content.strip().strip('"').strip("'")
            labels[cid] = label
        except Exception:
            labels[cid] = ", ".join(kws[:3]) if kws else f"Cluster {cid}"

    return labels


def label_clusters(
    labels: np.ndarray,
    texts: List[str],
    titles: List[str],
    *,
    use_llm: bool = False,
) -> List[ClusterInfo]:
    """Full cluster labeling pipeline.

    Args:
        labels: Cluster labels array (n_samples,).
        texts: Document texts per sample.
        titles: Job titles per sample.
        use_llm: Whether to use LLM for label generation.

    Returns:
        List of ClusterInfo objects.
    """
    # Group documents and titles by cluster
    docs_per_cluster: Dict[int, List[str]] = {}
    titles_per_cluster: Dict[int, List[str]] = {}

    for i, label in enumerate(labels):
        cid = int(label)
        docs_per_cluster.setdefault(cid, []).append(texts[i] if i < len(texts) else "")
        titles_per_cluster.setdefault(cid, []).append(titles[i] if i < len(titles) else "")

    # c-TF-IDF keywords
    keywords = compute_ctfidf(docs_per_cluster)

    # Generate labels
    if use_llm:
        cluster_labels = label_clusters_with_llm(keywords, titles_per_cluster)
    else:
        cluster_labels = {
            cid: ", ".join(kws[:3]) if kws else f"Cluster {cid}"
            for cid, kws in keywords.items()
        }

    # Build ClusterInfo objects
    cluster_ids = sorted(set(int(l) for l in labels if l >= 0))
    result: List[ClusterInfo] = []

    for cid in cluster_ids:
        result.append(ClusterInfo(
            cluster_id=cid,
            label=cluster_labels.get(cid, f"Cluster {cid}"),
            size=len(docs_per_cluster.get(cid, [])),
            keywords=keywords.get(cid, [])[:10],
            sample_titles=titles_per_cluster.get(cid, [])[:5],
        ))

    return result
