from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError


# Optional OpenAI client -----------------------------------------------------
OPENAI_AVAILABLE = False
try:  # pragma: no cover - dependency is optional
    import openai  # type: ignore

    OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    openai = None  # type: ignore


class FacetLabel(BaseModel):
    facet: Optional[str] = None
    type_within_facet: Optional[str] = None
    types_within_facet: Optional[List[str]] = []
    confidence: float = 0.0
    evidence_span: Optional[str] = None
    suggested_new_facet: Optional[str] = None
    suggested_new_type: Optional[str] = None
    notes: Optional[str] = ""


class QAResult(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    needed_keywords: List[str] = []
    citations: List[Dict[str, str]] = []  # {chunk_id, span_text}


class Snapshot(BaseModel):
    domain: Optional[str]
    domain_confidence: float = 0.0
    language_requirements: Dict[str, Any] = {}
    seniority: Optional[str] = None
    role_family: Optional[str] = None
    must_have_skills: List[str] = []
    nice_to_have_skills: List[str] = []
    remote_policy: Optional[str] = None
    sponsorship_or_language_blocker: bool = False
    salary: Optional[str] = None
    fitness_hint: Optional[str] = None
    evidence: List[Dict[str, Any]] = []  # {field, chunk_id, span_text}


T = TypeVar("T", bound=BaseModel)


def _strip_json_fences(text: str) -> str:
    if not text:
        return text
    # Remove ```json ... ``` or ``` ... ``` fences
    fenced = re.findall(r"```(?:json)?\n([\s\S]*?)```", text)
    if fenced:
        return fenced[0].strip()
    return text.strip()


def _extract_json_fragment(text: str) -> Optional[str]:
    # Try to find the first {...} or [...] block heuristically
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch in "[{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch in "]}":
            if not stack:
                continue
            last = stack.pop()
            if not stack and start is not None:
                frag = text[start : i + 1]
                return frag
    return None


def call_llm_json(messages: List[Dict[str, str]], schema_cls: Type[T], *, model: str = "gpt-4o-mini") -> T:
    """Call an LLM and parse validated JSON into schema_cls.

    Expects the assistant to return a JSON object. If parsing fails, attempts
    to recover a JSON fragment. Raises if no model is configured.
    """
    if not (OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY")):
        raise RuntimeError("OpenAI API key not configured; cannot run call_llm_json")

    openai.api_key = os.environ["OPENAI_API_KEY"]
    resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
    text = resp["choices"][0]["message"]["content"].strip()
    raw = _strip_json_fences(text)
    try:
        data = json.loads(raw)
    except Exception:
        frag = _extract_json_fragment(raw) or _extract_json_fragment(text) or "{}"
        data = json.loads(frag)
    # First attempt
    try:
        return schema_cls.model_validate(data)
    except ValidationError:
        # Try to coerce common shape issues
        data = _coerce_for_schema(schema_cls, data)
        try:
            return schema_cls.model_validate(data)
        except ValidationError as ve:
            # Try one more time: clamp confidence into [0,1]
            if isinstance(data, dict) and "confidence" in data:
                try:
                    c = float(data.get("confidence", 0.0))
                    data["confidence"] = max(0.0, min(1.0, c))
                    return schema_cls.model_validate(data)
                except Exception:
                    pass
            raise ve


def make_labeler_messages(facet_board: Dict[str, Any], chunk: Dict[str, Any]) -> List[Dict[str, str]]:
    system = (
        "You label a job-post CHUNK with a facet from a provided list.\n"
        "Return strict JSON with these keys: {facet, type_within_facet, types_within_facet, confidence, evidence_span, suggested_new_facet, suggested_new_type, notes}.\n"
        "Rules:\n"
        "- facet: string from the provided list, or null if nothing fits.\n"
        "- If the chosen facet supports multiple types (Tools/Stack, Job Responsibilities, Company Culture & Values), fill types_within_facet as an array of 1–5 strings and set type_within_facet to the most salient one.\n"
        "- Otherwise, leave types_within_facet empty and set type_within_facet as a single string.\n"
        "- Do not nest objects for facet or type(s).\n"
        "- Grounding: your chosen type(s) MUST be directly supported by exact words or close synonyms present in the evidence_span.\n"
        "  If you cannot find such words, set type_within_facet=null and leave types_within_facet empty.\n"
        "- If nothing fits, set facet=null and propose suggested_new_facet (2–3 words).\n"
        "- If facet is known but type does not fit any allowed types, set type_within_facet=null and propose suggested_new_type (2–3 words).\n"
        "- Include a short evidence_span copied verbatim (1–2 sentences)."
    )
    board_text = json.dumps(facet_board, ensure_ascii=False)
    user = (
        f"Facet Board (names + allowed types):\n{board_text}\n\n"
        f"CHUNK:\nchunk_id={chunk.get('chunk_id')} job_id={chunk.get('job_id')}\n"
        f"title={chunk.get('title','')} company={chunk.get('company','')}\n"
        f"text={chunk.get('text','')[:1200]}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def make_qa_messages(chunks: List[Dict[str, Any]], question: str, *, max_chars: int = 3200) -> List[Dict[str, str]]:
    system = (
        "Answer a specific question about ONE job using the given snippets.\n"
        "Return strict JSON with keys: {answer, confidence, needed_keywords, citations}.\n"
        "Rules:\n"
        "- answer: concise string; if unsure, say 'Unknown' or a brief best-effort.\n"
        "- confidence: number 0..1.\n"
        "- needed_keywords: array of up to 3 strings you would search for next if unsure (else empty).\n"
        "- citations: array of objects {chunk_id: string, span_text: string} copied verbatim from snippets.\n"
        "Only cite text present in snippets."
    )
    ctx = []
    total = 0
    for i, ch in enumerate(chunks, 1):
        txt = (ch.get("text") or "").strip()
        head = f"\n\n[Source {i}] {ch.get('title','?')} — {ch.get('company','?')}\n"
        block = head + txt
        if total + len(block) > max_chars and i > 1:
            break
        ctx.append(block)
        total += len(block)
    u = f"Question: {question}\n\nSnippets:{''.join(ctx)}\n\nReturn JSON."
    return [{"role": "system", "content": system}, {"role": "user", "content": u}]


__all__ = [
    "FacetLabel",
    "QAResult",
    "Snapshot",
    "call_llm_json",
    "make_labeler_messages",
    "make_qa_messages",
]


def _coerce_for_schema(schema_cls: Type[T], data: Any) -> Any:
    """Best-effort coercion for common model shape mistakes from LLMs.

    - If list is returned, take the first element
    - For FacetLabel:
        * If facet is an object, use its 'name' or 'facet' field
        * If type_within_facet missing, map from possible alias keys ('type', 'typeWithinFacet', 'type-within-facet')
        * Ensure strings where expected
    """
    try:
        if schema_cls.__name__ == "FacetLabel":
            # If a list of labels is returned, pick the first
            if isinstance(data, list) and data:
                data = data[0]
            if not isinstance(data, dict):
                return data
            # Some models wrap in {labels: [{...}]}
            if "labels" in data and isinstance(data["labels"], list) and data["labels"]:
                data = data["labels"][0]
                if not isinstance(data, dict):
                    return data

            facet_val = data.get("facet")
            if isinstance(facet_val, dict):
                # Try common keys
                nested_facet_name = (
                    facet_val.get("name")
                    or facet_val.get("facet")
                    or facet_val.get("label")
                )
                if nested_facet_name is not None:
                    data["facet"] = nested_facet_name
                # Pull nested type into top-level if missing
                if "type_within_facet" not in data:
                    for tk in ("type_within_facet", "typeWithinFacet", "type-within-facet", "type"):
                        if tk in facet_val and facet_val.get(tk) is not None:
                            data["type_within_facet"] = facet_val.get(tk)
                            break
                # Pull nested confidence if present
                if "confidence" not in data and "confidence" in facet_val:
                    data["confidence"] = facet_val.get("confidence")
            # Map type aliases
            if "type_within_facet" not in data:
                for k in ("type_within_facet", "typeWithinFacet", "type-within-facet", "type"):
                    if k in data and data.get(k) is not None:
                        data["type_within_facet"] = data.get(k)
                        break
            # Map possible plural forms to types_within_facet
            if "types_within_facet" not in data or data.get("types_within_facet") in (None, ""):
                for lk in ("types_within_facet", "types", "values", "items", "tools"):
                    val = data.get(lk)
                    if isinstance(val, list) and val:
                        data["types_within_facet"] = val
                        break
            # If type_within_facet is a list, move to types_within_facet
            if isinstance(data.get("type_within_facet"), list):
                data["types_within_facet"] = list(data["type_within_facet"]) or []
                data["type_within_facet"] = (data["types_within_facet"][0] if data["types_within_facet"] else None)
            # Coerce to strings
            if data.get("facet") is not None and not isinstance(data.get("facet"), str):
                data["facet"] = str(data.get("facet"))
            if data.get("type_within_facet") is not None and not isinstance(data.get("type_within_facet"), str):
                data["type_within_facet"] = str(data.get("type_within_facet"))
            # Ensure types_within_facet is a list of strings
            twf = data.get("types_within_facet")
            if isinstance(twf, list):
                data["types_within_facet"] = [str(x) for x in twf if x is not None]
            # Map suggested new type aliases
            if "suggested_new_type" not in data:
                for ak in ("suggested_new_type", "suggestedType", "suggested_type", "new_type"):
                    if ak in data and data.get(ak):
                        data["suggested_new_type"] = str(data.get(ak))
                        break
            # Clamp confidence if present
            if "confidence" in data:
                try:
                    c = float(data.get("confidence", 0.0))
                    data["confidence"] = max(0.0, min(1.0, c))
                except Exception:
                    data["confidence"] = 0.0
        # Handle QAResult coercions
        if schema_cls.__name__ == "QAResult":
            # If a list is returned, use the first element
            if isinstance(data, list) and data:
                data = data[0]
            # If scalar, coerce into an answer string
            if not isinstance(data, dict):
                return {
                    "answer": str(data),
                    "confidence": 0.0,
                    "needed_keywords": [],
                    "citations": [],
                }
            # Map answer aliases
            raw_ans = (
                data.get("answer")
                or data.get("response")
                or data.get("final_answer")
                or data.get("output")
                or data.get("text")
            )
            if isinstance(raw_ans, dict):
                raw_ans = raw_ans.get("text") or raw_ans.get("content") or str(raw_ans)
            if raw_ans is not None:
                data["answer"] = str(raw_ans)
            # Confidence
            if "confidence" in data:
                try:
                    c = float(data.get("confidence", 0.0))
                    data["confidence"] = max(0.0, min(1.0, c))
                except Exception:
                    data["confidence"] = 0.0
            else:
                data["confidence"] = 0.0
            # needed_keywords normalization
            kw = data.get("needed_keywords")
            if isinstance(kw, str):
                parts = [p.strip() for p in kw.split(",") if p.strip()]
                data["needed_keywords"] = parts[:3]
            elif isinstance(kw, list):
                data["needed_keywords"] = [str(x) for x in kw if x is not None][:3]
            else:
                data["needed_keywords"] = []
            # citations normalization
            cites = data.get("citations")
            out_cites: List[Dict[str, str]] = []
            if isinstance(cites, list):
                for it in cites:
                    if isinstance(it, str):
                        out_cites.append({"chunk_id": "", "span_text": it})
                    elif isinstance(it, dict):
                        cid = it.get("chunk_id") or it.get("id") or it.get("chunkId") or ""
                        span = it.get("span_text") or it.get("text") or it.get("snippet") or ""
                        out_cites.append({"chunk_id": str(cid), "span_text": str(span)})
            data["citations"] = out_cites
            return data
        return data
    except Exception:
        return data


