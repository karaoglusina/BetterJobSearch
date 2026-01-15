"""Chunk-level feedback storage and APIs (v3.2).

Persists human annotations for chunks (e.g., fitness labels) in MongoDB,
separate from local artifacts (chunks.jsonl / FAISS / BM25).

Public functions:
- record_chunk_fitness
- get_chunk_fitness
- get_job_chunk_feedback

Notes:
- Uses content-based chunk ids: chunk_id_v2 = md5(job_key + '|' + normalized_text)
- Fitness scale: {-2: "impossible", -1: "not really", 0: "maybe", 1: "yes", 2: "perfect"}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
from datetime import datetime

from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database

from .config import MONGODB_URI


FEEDBACK_COLLECTION_NAME = "chunk_feedback"
ANNOTATIONS_COLLECTION_NAME = "chunk_annotations"
DEFAULT_DB_NAME = "job_rag_db"

FITNESS_LABEL_FOR_SCORE: Dict[int, str] = {
    -2: "impossible",
    -1: "not really",
    0: "maybe",
    1: "yes",
    2: "perfect",
}

VALID_FITNESS_SCORES = set(FITNESS_LABEL_FOR_SCORE.keys())


def canonicalize_slug(value: Optional[str]) -> Optional[str]:
    """Canonicalize a facet/type label into a lowercase slug (a-z0-9_).

    Robust to None/NaN and non-string inputs.
    """
    if value is None:
        return None
    # Handle NaN floats
    if isinstance(value, float) and math.isnan(value):
        return None
    # Coerce non-strings to strings
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception:
            return None
    s = value.strip().lower()
    import re as _re
    s = _re.sub(r"[^a-z0-9_]+", "_", s)
    s = _re.sub(r"_+", "_", s).strip("_")
    return s or None


def _get_client(uri: Optional[str] = None) -> MongoClient:
    if not uri:
        uri = MONGODB_URI
    if not uri:
        raise ValueError("MongoDB URI is required")
    return MongoClient(uri)


def _get_feedback_collection(
    client: MongoClient, db_name: str = DEFAULT_DB_NAME
) -> Collection:
    db: Database = client[db_name]
    col: Collection = db[FEEDBACK_COLLECTION_NAME]
    _ensure_indexes(col)
    return col


def _ensure_indexes(col: Collection) -> None:
    try:
        # Unique chunk_id_v2 ensures one feedback record per chunk (latest wins)
        col.create_index([("chunk_id_v2", ASCENDING)], unique=True, name="uniq_chunk_v2")
        # Job key for grouping/aggregation
        col.create_index([("job_key", ASCENDING)], name="job_key_idx")
        # Fitness to enable filtered scans
        col.create_index([("fitness", ASCENDING)], name="fitness_idx")
    except Exception:
        # Indexes may already exist; ignore errors
        pass


def _get_annotations_collection(
    client: MongoClient, db_name: str = DEFAULT_DB_NAME
) -> Collection:
    db: Database = client[db_name]
    col: Collection = db[ANNOTATIONS_COLLECTION_NAME]
    _ensure_ann_indexes(col)
    return col


def _ensure_ann_indexes(col: Collection) -> None:
    try:
        col.create_index([("chunk_id_v2", ASCENDING)], unique=True, name="uniq_chunk_v2")
        col.create_index([("job_key", ASCENDING)], name="job_key_idx")
        col.create_index([("chunk_facet", ASCENDING)], name="facet_idx")
        col.create_index([("chunk_facet", ASCENDING), ("type_within_facet", ASCENDING)], name="facet_type_idx")
    except Exception:
        pass


def _normalize_tags(tags: Optional[List[str]]) -> List[str]:
    if not tags:
        return []
    unique = sorted({t.strip() for t in tags if t and t.strip()})
    return unique


def _validate_fitness(fitness: int) -> None:
    if fitness not in VALID_FITNESS_SCORES:
        raise ValueError(
            f"Invalid fitness score: {fitness}. Allowed: {sorted(VALID_FITNESS_SCORES)}"
        )


def record_chunk_fitness(
    chunk_id_v2: str,
    fitness: int,
    *,
    job_key: Optional[str] = None,
    job_mongo_id: Optional[str] = None,
    label: Optional[str] = None,
    rationale: Optional[str] = None,
    tags: Optional[List[str]] = None,
    user: str = "me",
    index_build_id: Optional[str] = None,
    uri: Optional[str] = None,
    db_name: str = DEFAULT_DB_NAME,
) -> Dict[str, Any]:
    """Upsert a fitness label for a chunk.

    Overwrites previous fitness for the same chunk_id_v2.
    """
    _validate_fitness(fitness)
    final_label = label or FITNESS_LABEL_FOR_SCORE.get(fitness, str(fitness))
    now = datetime.utcnow()
    norm_tags = _normalize_tags(tags)

    client = _get_client(uri)
    try:
        col = _get_feedback_collection(client, db_name)
        update_doc: Dict[str, Any] = {
            "chunk_id_v2": chunk_id_v2,
            "fitness": fitness,
            "label": final_label,
            "user": user,
            "updated_at": now,
        }
        if rationale is not None:
            update_doc["rationale"] = rationale
        if norm_tags:
            update_doc["tags"] = norm_tags
        if job_key:
            update_doc["job_key"] = job_key
        if job_mongo_id:
            update_doc["job_mongo_id"] = job_mongo_id
        if index_build_id:
            update_doc["index_build_id"] = index_build_id

        result = col.update_one(
            {"chunk_id_v2": chunk_id_v2},
            {
                "$set": update_doc,
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

        return {
            "matched": result.matched_count,
            "modified": result.modified_count,
            "upserted_id": str(result.upserted_id) if result.upserted_id else None,
        }
    finally:
        client.close()


def upsert_chunk_fitness_bulk(
    records: List[Dict[str, Any]],
    *,
    fitness: int,
    user: str = "me",
    index_build_id: Optional[str] = None,
    uri: Optional[str] = None,
    db_name: str = DEFAULT_DB_NAME,
) -> Dict[str, Any]:
    """Bulk upsert fitness for many chunk_ids in a single round-trip.

    Each record in `records` must include `chunk_id_v2` and may include optional keys:
      - job_key, job_mongo_id, rationale, tags, label, user

    Returns summary counts.
    """
    _validate_fitness(fitness)
    now = datetime.utcnow()
    client = _get_client(uri)
    try:
        col = _get_feedback_collection(client, db_name)
        ops: List[UpdateOne] = []
        for r in records:
            cid = r.get("chunk_id_v2")
            if not cid:
                continue
            final_label = r.get("label") or FITNESS_LABEL_FOR_SCORE.get(fitness, str(fitness))
            update_doc: Dict[str, Any] = {
                "chunk_id_v2": cid,
                "fitness": fitness,
                "label": final_label,
                "user": r.get("user", user),
                "updated_at": now,
            }
            if r.get("rationale") is not None:
                update_doc["rationale"] = r.get("rationale")
            if r.get("tags"):
                update_doc["tags"] = _normalize_tags(r.get("tags"))
            if r.get("job_key"):
                update_doc["job_key"] = r.get("job_key")
            if r.get("job_mongo_id"):
                update_doc["job_mongo_id"] = r.get("job_mongo_id")
            if index_build_id:
                update_doc["index_build_id"] = index_build_id

            ops.append(
                UpdateOne(
                    {"chunk_id_v2": cid},
                    {"$set": update_doc, "$setOnInsert": {"created_at": now}},
                    upsert=True,
                )
            )

        if not ops:
            return {"matched": 0, "modified": 0, "upserted": 0}

        result = col.bulk_write(ops, ordered=False)
        upserted = getattr(result, "upserted_ids", {}) or {}
        return {
            "matched": result.matched_count,
            "modified": result.modified_count,
            "upserted": len(upserted),
        }
    finally:
        client.close()

def get_chunk_fitness(
    chunk_ids_v2: List[str],
    *,
    uri: Optional[str] = None,
    db_name: str = DEFAULT_DB_NAME,
) -> Dict[str, int]:
    """Return a mapping {chunk_id_v2 -> fitness} for provided chunk ids."""
    if not chunk_ids_v2:
        return {}
    client = _get_client(uri)
    try:
        col = _get_feedback_collection(client, db_name)
        cursor = col.find(
            {"chunk_id_v2": {"$in": list(set(chunk_ids_v2))}}, {"chunk_id_v2": 1, "fitness": 1}
        )
        out: Dict[str, int] = {}
        for doc in cursor:
            cid = doc.get("chunk_id_v2")
            fit = doc.get("fitness")
            if cid is not None and fit is not None:
                out[str(cid)] = int(fit)
        return out
    finally:
        client.close()


def record_chunk_labels(
    chunk_id_v2: str,
    *,
    chunk_facet: Optional[str] = None,
    type_within_facet: Optional[str] = None,
    user: str = "me",
    uri: Optional[str] = None,
    db_name: str = DEFAULT_DB_NAME,
) -> Dict[str, Any]:
    """Upsert non-fitness labels for a chunk (facet, type)."""
    if not chunk_id_v2:
        raise ValueError("chunk_id_v2 is required")
    now = datetime.utcnow()
    update_fields: Dict[str, Any] = {"user": user, "updated_at": now}
    cf = canonicalize_slug(chunk_facet) if chunk_facet is not None else None
    ctype = canonicalize_slug(type_within_facet) if type_within_facet is not None else None
    if cf is not None:
        update_fields["chunk_facet"] = cf
    if ctype is not None:
        update_fields["type_within_facet"] = ctype
    client = _get_client(uri)
    try:
        col = _get_annotations_collection(client, db_name)
        result = col.update_one(
            {"chunk_id_v2": chunk_id_v2},
            {"$set": update_fields, "$setOnInsert": {"created_at": now, "chunk_id_v2": chunk_id_v2}},
            upsert=True,
        )
        return {"matched": result.matched_count, "modified": result.modified_count}
    finally:
        client.close()


def upsert_chunk_labels_bulk(
    records: List[Dict[str, Any]],
    *,
    user: str = "me",
    uri: Optional[str] = None,
    db_name: str = DEFAULT_DB_NAME,
) -> Dict[str, Any]:
    """Bulk upsert facet/type annotations into chunk_annotations.

    Each record may include keys: chunk_id_v2 (required), chunk_facet, type_within_facet,
    types_within_facet (list[str] or comma-separated str), job_key, job_mongo_id.
    """
    if not records:
        return {"matched": 0, "modified": 0, "upserted": 0}
    now = datetime.utcnow()
    client = _get_client(uri)
    try:
        col = _get_annotations_collection(client, db_name)
        ops: List[UpdateOne] = []
        matched = modified = upserted = 0
        for r in records:
            cid = r.get("chunk_id_v2")
            if not cid:
                continue
            # Normalize fields
            cf = canonicalize_slug(r.get("chunk_facet")) if r.get("chunk_facet") is not None else None
            ctype = canonicalize_slug(r.get("type_within_facet")) if r.get("type_within_facet") is not None else None
            ty = r.get("types_within_facet")
            if isinstance(ty, str):
                ty = [t.strip() for t in ty.split(",") if t and t.strip()]
            if isinstance(ty, list):
                # Keep original text values (no slugging; snapshots will canonicalize as needed)
                types_norm = [str(x) for x in ty if x]
            else:
                types_norm = None
            set_fields: Dict[str, Any] = {
                "user": r.get("user", user),
                "updated_at": now,
            }
            if cf is not None:
                set_fields["chunk_facet"] = cf
            if ctype is not None:
                set_fields["type_within_facet"] = ctype
            if types_norm is not None:
                set_fields["types_within_facet"] = types_norm
            if r.get("job_key"):
                set_fields["job_key"] = r.get("job_key")
            if r.get("job_mongo_id"):
                set_fields["job_mongo_id"] = r.get("job_mongo_id")

            ops.append(
                UpdateOne(
                    {"chunk_id_v2": cid},
                    {"$set": set_fields, "$setOnInsert": {"created_at": now, "chunk_id_v2": cid}},
                    upsert=True,
                )
            )
        if not ops:
            return {"matched": 0, "modified": 0, "upserted": 0}
        result = col.bulk_write(ops, ordered=False)
        upserted = len(getattr(result, "upserted_ids", {}) or {})
        return {
            "matched": result.matched_count,
            "modified": result.modified_count,
            "upserted": upserted,
        }
    finally:
        client.close()


def get_chunk_annotations(
    chunk_ids_v2: List[str],
    *,
    uri: Optional[str] = None,
    db_name: str = DEFAULT_DB_NAME,
) -> Dict[str, Dict[str, Any]]:
    """Return mapping {chunk_id_v2: {fitness, chunk_facet, type_within_facet}}."""
    if not chunk_ids_v2:
        return {}
    client = _get_client(uri)
    try:
        ann_col = _get_annotations_collection(client, db_name)
        fb_col = _get_feedback_collection(client, db_name)
        ids = list(set(chunk_ids_v2))
        out: Dict[str, Dict[str, Any]] = {}
        # annotations (facet/type)
        ann_cur = ann_col.find(
            {"chunk_id_v2": {"$in": ids}}, {"chunk_id_v2": 1, "chunk_facet": 1, "type_within_facet": 1}
        )
        for doc in ann_cur:
            cid = str(doc.get("chunk_id_v2"))
            out.setdefault(cid, {})
            out[cid]["chunk_facet"] = doc.get("chunk_facet")
            out[cid]["type_within_facet"] = doc.get("type_within_facet")
        # fitness
        fb_cur = fb_col.find(
            {"chunk_id_v2": {"$in": ids}}, {"chunk_id_v2": 1, "fitness": 1, "label": 1}
        )
        for doc in fb_cur:
            cid = str(doc.get("chunk_id_v2"))
            out.setdefault(cid, {})
            out[cid]["fitness"] = doc.get("fitness")
            if doc.get("label") is not None:
                out[cid]["label"] = doc.get("label")
        return out
    finally:
        client.close()

def get_job_chunk_feedback(
    job_key: str,
    *,
    uri: Optional[str] = None,
    db_name: str = DEFAULT_DB_NAME,
) -> List[Dict[str, Any]]:
    """Return all feedback entries for a given job_key (job_url)."""
    if not job_key:
        return []
    client = _get_client(uri)
    try:
        col = _get_feedback_collection(client, db_name)
        cursor = col.find({"job_key": job_key}).sort("updated_at", -1)
        results: List[Dict[str, Any]] = []
        for d in cursor:
            d["_id"] = str(d.get("_id"))
            results.append(d)
        return results
    finally:
        client.close()


def record_model_predictions(
    chunk_id_v2: str,
    *,
    predicted_facet: Optional[str] = None,
    predicted_type: Optional[str] = None,
    predicted_confidence: Optional[float] = None,
    model_name: Optional[str] = None,
    uri: Optional[str] = None,
    db_name: str = DEFAULT_DB_NAME,
) -> Dict[str, Any]:
    """Upsert model predictions for facet/type with confidence into chunk_annotations."""
    if not chunk_id_v2:
        raise ValueError("chunk_id_v2 is required")
    now = datetime.utcnow()
    client = _get_client(uri)
    try:
        col = _get_annotations_collection(client, db_name)
        fields: Dict[str, Any] = {
            "source": "model",
            "updated_at": now,
        }
        pf = canonicalize_slug(predicted_facet) if predicted_facet is not None else None
        pt = canonicalize_slug(predicted_type) if predicted_type is not None else None
        if pf is not None:
            fields["predicted_facet"] = pf
        if pt is not None:
            fields["predicted_type"] = pt
        if predicted_confidence is not None:
            fields["predicted_confidence"] = float(predicted_confidence)
        if model_name:
            fields["model_name"] = model_name
        result = col.update_one(
            {"chunk_id_v2": chunk_id_v2},
            {"$set": fields, "$setOnInsert": {"created_at": now, "chunk_id_v2": chunk_id_v2}},
            upsert=True,
        )
        return {"matched": result.matched_count, "modified": result.modified_count}
    finally:
        client.close()



# ------------------------------ ONLY FOR EXPLORATION

"""

def _extract_field_paths(obj: Dict[str, Any], prefix: str = "") -> List[str]:

    paths: List[str] = []
    for key, value in obj.items():
        field_path = f"{prefix}.{key}" if prefix else key
        paths.append(field_path)
        if isinstance(value, dict) and key != "_id":
            paths.extend(_extract_field_paths(value, field_path))
    return paths


def get_chunk_feedback_field_names(
    sample_size: int = 100,
    *,
    uri: Optional[str] = None,
    db_name: str = DEFAULT_DB_NAME,
) -> Dict[str, List[str]]:

    client = _get_client(uri)
    try:
        col = _get_feedback_collection(client, db_name)
        sample_docs = list(col.aggregate([{"$sample": {"size": int(sample_size)}}]))
        if not sample_docs:
            return {"top_level": [], "job": [], "meta": []}

        all_fields: set[str] = set()
        for doc in sample_docs:
            all_fields.update(_extract_field_paths(doc))

        buckets: Dict[str, List[str]] = {"top_level": [], "job": [], "meta": []}
        for f in sorted(all_fields):
            if f.startswith("job_") or f.startswith("job."):
                buckets["job"].append(f)
            elif f.startswith("meta."):
                buckets["meta"].append(f)
            else:
                buckets["top_level"].append(f)
        return buckets
    finally:
        client.close()


def get_chunk_annotations_field_names(
    sample_size: int = 100,
    *,
    uri: Optional[str] = None,
    db_name: str = DEFAULT_DB_NAME,
) -> Dict[str, List[str]]:

    client = _get_client(uri)
    try:
        col = _get_annotations_collection(client, db_name)
        sample_docs = list(col.aggregate([{"$sample": {"size": int(sample_size)}}]))
        if not sample_docs:
            return {"identifiers": [], "labels": [], "predictions": [], "meta": []}

        all_fields: set[str] = set()
        for doc in sample_docs:
            all_fields.update(_extract_field_paths(doc))

        buckets: Dict[str, List[str]] = {
            "identifiers": [],
            "labels": [],
            "predictions": [],
            "meta": [],
        }
        for f in sorted(all_fields):
            if f in {"_id", "chunk_id_v2", "job_key", "job_mongo_id"} or f.startswith("ids."):
                buckets["identifiers"].append(f)
            elif f.startswith("chunk_facet") or f.startswith("type_within_facet"):
                buckets["labels"].append(f)
            elif f.startswith("predicted_") or f in {"model_name", "source"}:
                buckets["predictions"].append(f)
            elif f.startswith("created_at") or f.startswith("updated_at") or f.startswith("meta."):
                buckets["meta"].append(f)
            else:
                buckets["labels"].append(f)  # default to labels for simple fields
        return buckets
    finally:
        client.close()

"""