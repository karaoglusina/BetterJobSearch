"""Search endpoint with hybrid search and boolean keyword filtering."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(tags=["search"])


# ---------------------------------------------------------------------------
# Boolean query parser (ported from legacy v1 app)
# Supports: AND, OR, NOT, quoted phrases, parentheses
# Example: ("data engineer" AND python) OR spark
# ---------------------------------------------------------------------------

_TOK_RE = re.compile(r'"([^"]+)"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+', re.IGNORECASE)
_OPS = {"AND": (2, "L"), "OR": (1, "L"), "NOT": (3, "R")}


def compile_boolean_query(query: str) -> Optional[List[str]]:
    """Compile a boolean search query to postfix (RPN) notation.

    Returns None if the query is empty, or a list of tokens in postfix order.
    Raises ValueError on mismatched parentheses.
    """
    if not query or not query.strip():
        return None

    tokens = []
    for m in _TOK_RE.finditer(query.strip()):
        g = m.group(1)  # captured quoted content
        if g is not None:
            tokens.append(g)
        else:
            tokens.append(m.group(0))

    out: List[str] = []
    stack: List[str] = []

    for t in tokens:
        u = t.upper()
        if u in _OPS:
            while stack:
                top = stack[-1]
                if top.upper() in _OPS:
                    p1, a1 = _OPS[u]
                    p2, _ = _OPS[top.upper()]
                    if (a1 == "L" and p1 <= p2) or (a1 == "R" and p1 < p2):
                        out.append(stack.pop())
                        continue
                break
            stack.append(u)
        elif t == "(":
            stack.append(t)
        elif t == ")":
            while stack and stack[-1] != "(":
                out.append(stack.pop())
            if not stack:
                raise ValueError("Mismatched parentheses")
            stack.pop()
        else:
            out.append(t)

    while stack:
        if stack[-1] in ("(", ")"):
            raise ValueError("Mismatched parentheses")
        out.append(stack.pop())

    return out


def eval_boolean_postfix(text: str, postfix: List[str]) -> bool:
    """Evaluate a postfix boolean expression against a text string."""
    if not postfix:
        return True

    text_lower = text.lower()
    st: List[bool] = []

    for t in postfix:
        if t in ("AND", "OR", "NOT"):
            if t == "NOT":
                a = bool(st.pop()) if st else False
                st.append(not a)
            else:
                b = bool(st.pop()) if st else False
                a = bool(st.pop()) if st else False
                st.append((a and b) if t == "AND" else (a or b))
        else:
            st.append(t.lower() in text_lower)

    return bool(st[-1]) if st else True


def extract_search_terms(query: str) -> List[str]:
    """Extract individual search terms from a boolean query for highlighting."""
    if not query or not query.strip():
        return []
    try:
        postfix = compile_boolean_query(query)
    except Exception:
        return [query.strip()]

    terms: List[str] = []
    seen: set = set()
    for t in postfix or []:
        if isinstance(t, str) and t.upper() not in ("AND", "OR", "NOT"):
            k = t.lower()
            if k not in seen:
                seen.add(k)
                terms.append(t)
    terms.sort(key=len, reverse=True)
    return terms


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=8, ge=1, le=100)
    alpha: float = Field(default=0.55, ge=0.0, le=1.0)
    location: Optional[str] = None
    company: Optional[str] = None


class KeywordSearchRequest(BaseModel):
    query: str
    fields: List[str] = Field(default=["title", "description"])
    limit: int = Field(default=50, ge=1, le=200)
    location: Optional[str] = None
    company: Optional[str] = None


class SearchResult(BaseModel):
    job_id: str
    title: str = ""
    company: str = ""
    location: str = ""
    url: str = ""
    snippet: str = ""
    section: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/search")
async def search_jobs(request: Request, body: SearchRequest) -> Dict[str, Any]:
    """Hybrid search with optional filters."""
    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="Search artifacts not loaded")

    # Build filter predicate
    def where(ch: Dict[str, Any]) -> bool:
        meta = ch.get("meta", {})
        if body.location:
            if body.location.lower() not in (meta.get("location") or "").lower():
                return False
        if body.company:
            if body.company.lower() not in (meta.get("company") or "").lower():
                return False
        return True

    has_filter = body.location or body.company
    chunks = retriever.search(
        body.query,
        k=body.k * 3 if has_filter else body.k,
        alpha=body.alpha,
        where=where if has_filter else None,
    )

    # Dedupe by job, keep best chunk per job
    seen_jobs: set[str] = set()
    results: List[Dict[str, Any]] = []
    for ch in chunks:
        jk = ch.get("job_key", "")
        if jk in seen_jobs:
            continue
        seen_jobs.add(jk)
        meta = ch.get("meta", {})
        results.append({
            "job_id": jk,
            "title": meta.get("title", ""),
            "company": meta.get("company", ""),
            "location": meta.get("location", ""),
            "url": meta.get("jobUrl", ""),
            "snippet": ch.get("text", "")[:200],
            "section": ch.get("section"),
        })
        if len(results) >= body.k:
            break

    return {
        "query": body.query,
        "total_results": len(results),
        "results": results,
    }


@router.post("/search/keyword")
async def keyword_search(request: Request, body: KeywordSearchRequest) -> Dict[str, Any]:
    """Boolean keyword search with AND/OR/NOT operators.

    Supports:
        - Simple terms: python django
        - Quoted phrases: "data engineer"
        - Boolean operators: AND, OR, NOT
        - Parentheses: ("data engineer" AND python) OR spark
        - Implicit AND when no operator between terms

    Fields to search in: title, description (chunk text), location
    """
    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="Search artifacts not loaded")

    # Parse the boolean query
    try:
        postfix = compile_boolean_query(body.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if postfix is None:
        return {"query": body.query, "total_results": 0, "results": [], "highlight_terms": []}

    # Insert implicit AND between consecutive terms (no operator between them)
    # E.g., "python django" -> "python AND django"
    enhanced: List[str] = []
    for i, token in enumerate(postfix):
        enhanced.append(token)

    # Group chunks into jobs and evaluate the boolean query
    job_map: Dict[str, Dict[str, Any]] = {}
    for ch in retriever.chunks:
        jk = ch.get("job_key", "")
        if jk not in job_map:
            meta = ch.get("meta", {})
            job_map[jk] = {
                "job_id": jk,
                "title": meta.get("title", ""),
                "company": meta.get("company", ""),
                "location": meta.get("location", ""),
                "url": meta.get("jobUrl", ""),
                "texts": [],
            }
        job_map[jk]["texts"].append(ch.get("text", ""))

    results: List[Dict[str, Any]] = []
    for jk, job in job_map.items():
        # Build searchable text from selected fields
        haystacks = []
        if "title" in body.fields:
            haystacks.append(job["title"] or "")
        if "description" in body.fields:
            haystacks.append(" ".join(job["texts"]))
        if "location" in body.fields:
            haystacks.append(job["location"] or "")
        combined = " \n ".join(haystacks)

        if eval_boolean_postfix(combined, enhanced):
            # Apply additional filters
            if body.location and body.location.lower() not in (job["location"] or "").lower():
                continue
            if body.company and body.company.lower() not in (job["company"] or "").lower():
                continue

            # Find best matching snippet
            snippet = ""
            terms = extract_search_terms(body.query)
            if terms:
                for text in job["texts"]:
                    if any(t.lower() in text.lower() for t in terms):
                        snippet = text[:200]
                        break
            if not snippet and job["texts"]:
                snippet = job["texts"][0][:200]

            results.append({
                "job_id": jk,
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "url": job["url"],
                "snippet": snippet,
            })

            if len(results) >= body.limit:
                break

    highlight_terms = extract_search_terms(body.query)

    return {
        "query": body.query,
        "total_results": len(results),
        "results": results,
        "highlight_terms": highlight_terms,
    }
