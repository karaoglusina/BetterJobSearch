"""Company culture/values extraction using LLM (subjective, needs context)."""

from __future__ import annotations

import json
import os
from typing import Optional

from ...models.aspect import AspectExtraction, CultureExtraction

CULTURE_VALUES = [
    "mission-driven", "customer-centric", "data-driven", "design-led",
    "engineering-led", "quality/craftsmanship", "fast iteration", "process-heavy",
    "autonomy/ownership", "mentorship/coaching", "learning culture",
    "work-life balance", "async collaboration", "transparency",
    "flat hierarchy", "hierarchical", "risk-tolerant", "regulated",
    "sustainability", "diversity/inclusion", "open source", "innovation-focused",
]


class CultureExtractor:
    """Extract culture/values using LLM."""

    def extract(self, text: str, title: str = "", company: str = "") -> AspectExtraction:
        """Extract culture values. Falls back to empty if LLM unavailable."""
        try:
            result = self._extract_with_llm(text, title, company)
            if result and result.values:
                return AspectExtraction(
                    aspect="culture",
                    values=result.values,
                    evidence_spans=result.evidence_spans,
                    confidence=result.confidence,
                    method="llm",
                )
        except Exception:
            pass

        return AspectExtraction(
            aspect="culture",
            values=[],
            evidence_spans=[],
            confidence=0.0,
            method="llm",
        )

    def _extract_with_llm(self, text: str, title: str, company: str) -> Optional[CultureExtraction]:
        """Call LLM to extract culture values."""
        if not os.environ.get("OPENAI_API_KEY"):
            return None

        try:
            from openai import OpenAI
        except ImportError:
            return None

        client = OpenAI()

        system = (
            "Extract company culture and values from this job posting.\n"
            f"Choose from this list (pick 1-5 that apply):\n{json.dumps(CULTURE_VALUES)}\n\n"
            "Return JSON: {\"values\": [...], \"evidence_spans\": [\"brief quote 1\", ...], \"confidence\": 0.0-1.0}\n"
            "Only pick values with clear evidence in the text."
        )
        user = f"Title: {title}\nCompany: {company}\n\nDescription (first 1500 chars):\n{text[:1500]}"

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0,
                max_tokens=400,
            )
            raw = resp.choices[0].message.content or "{}"
            if "```" in raw:
                import re
                fenced = re.findall(r"```(?:json)?\n?([\s\S]*?)```", raw)
                if fenced:
                    raw = fenced[0]
            data = json.loads(raw)
            return CultureExtraction(**data)
        except Exception:
            return None
