"""Supervised classification for chunk facets/types (v4.3).

Baseline approach:
- Build labeled dataset from Mongo (chunk_annotations + chunk_feedback)
- Vectorize with TF-IDF (1–2 grams)
- Train Logistic Regression classifiers for:
  - facet (single-label)
  - type_within_facet (single-label, optional; can filter to a facet subset)
- Predict on new chunks and persist predictions via feedback APIs
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from pymongo import MongoClient

from .config import MONGODB_URI
from .feedback import (
    canonicalize_slug,
    record_model_predictions,
)


def _get_db(uri: Optional[str] = None, db_name: str = "job_rag_db"):
    uri = uri or MONGODB_URI
    if not uri:
        raise ValueError("MongoDB URI is required")
    client = MongoClient(uri)
    return client, client[db_name]


def load_labeled_chunks(
    *,
    uri: Optional[str] = None,
    db_name: str = "job_rag_db",
    min_chars: int = 20,
    facet_only: bool = False,
) -> List[Dict[str, Any]]:
    """Load chunks with human-provided facet/type labels for training.

    Returns list of dicts: {chunk_id_v2, text, job_key, facet, type}
    """
    client, db = _get_db(uri, db_name)
    try:
        ann = list(db["chunk_annotations"].find({}))
        # In artifacts, we keep chunk text in file; for training we assume you'll provide chunks from memory.
        # As an alternative, you can pre-join text externally. Here we only return annotations; the caller should attach text.
        out: List[Dict[str, Any]] = []
        for a in ann:
            facet = canonicalize_slug(a.get("chunk_facet"))
            t = canonicalize_slug(a.get("type_within_facet"))
            if facet_only and not facet:
                continue
            if not facet and not t:
                continue
            out.append({
                "chunk_id_v2": a.get("chunk_id_v2"),
                "job_key": a.get("job_key"),
                "facet": facet,
                "type": t,
            })
        return out
    finally:
        client.close()


def build_training_arrays(
    chunks: List[Any],
    annotations: List[Dict[str, Any]],
    *,
    target: str = "facet",  # "facet" or "type"
    min_chars: int = 20,
) -> Tuple[List[str], List[str], List[str]]:
    """Join in-memory chunks (with text) and annotations for training.

    Returns (texts, labels, chunk_ids)
    """
    id_to_ann: Dict[str, Dict[str, Any]] = {str(a.get("chunk_id_v2")): a for a in annotations}
    texts: List[str] = []
    labels: List[str] = []
    ids: List[str] = []
    for c in chunks:
        cid = str(getattr(c, "chunk_id_v2", "") or getattr(c, "chunk_id", ""))
        if cid not in id_to_ann:
            continue
        txt = (getattr(c, "text", "") or "").strip()
        if len(txt) < min_chars:
            continue
        lab = id_to_ann[cid].get("facet" if target == "facet" else "type")
        if not lab:
            continue
        texts.append(txt)
        labels.append(lab)
        ids.append(cid)
    return texts, labels, ids


def train_classifier(
    texts: List[str],
    labels: List[str],
    *,
    C: float = 2.0,
    max_features: int = 30000,
) -> Tuple[Pipeline, LabelEncoder, Dict[str, Any]]:
    """Train a simple TF-IDF + LogisticRegression pipeline and return metrics."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42, stratify=y)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, stop_words="english")),
        ("clf", LogisticRegression(max_iter=200, C=C, n_jobs=None)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=list(le.classes_), output_dict=True, zero_division=0)
    return pipe, le, report


def predict_and_persist(
    pipe: Pipeline,
    le: LabelEncoder,
    chunks: List[Any],
    *,
    target: str = "facet",
    model_name: str = "tfidf_logreg_v1",
) -> List[Tuple[str, str, float]]:
    """Run predictions on chunks and write to chunk_annotations predicted_* fields.

    Returns list of (chunk_id_v2, predicted_label, confidence)
    """
    texts: List[str] = []
    ids: List[str] = []
    for c in chunks:
        cid = str(getattr(c, "chunk_id_v2", "") or getattr(c, "chunk_id", ""))
        txt = (getattr(c, "text", "") or "").strip()
        if not cid or not txt:
            continue
        texts.append(txt)
        ids.append(cid)
    if not texts:
        return []
    # Predict probabilities for confidence
    proba = pipe.predict_proba(texts)
    pred_idx = np.argmax(proba, axis=1)
    pred_labels = le.inverse_transform(pred_idx)
    confidences = proba[np.arange(len(pred_idx)), pred_idx]
    out: List[Tuple[str, str, float]] = []
    for cid, lab, conf in zip(ids, pred_labels, confidences):
        if target == "facet":
            record_model_predictions(cid, predicted_facet=lab, predicted_confidence=float(conf), model_name=model_name)
        else:
            record_model_predictions(cid, predicted_type=lab, predicted_confidence=float(conf), model_name=model_name)
        out.append((cid, str(lab), float(conf)))
    return out

