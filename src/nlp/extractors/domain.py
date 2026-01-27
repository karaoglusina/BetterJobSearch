"""Domain/sector classification using LLM (ambiguous, needs context)."""

from __future__ import annotations

import json
import os
from typing import Optional

from ...models.aspect import AspectExtraction, DomainClassification

# Domain taxonomy from facets.yml
DOMAINS = [
    "finance", "health", "gov/public", "retail/e-com", "logistics/supply",
    "energy", "education", "gaming/media", "construction", "telecom",
    "insurance", "manufacturing", "pharma/biotech", "legal",
    "marketing/advertising", "mobility/transport", "real_estate",
    "consulting", "research", "startup/scaleup", "hospitality",
    "entertainment", "technology", "agriculture", "automotive",
    "aerospace/defense", "nonprofit", "cybersecurity",
]


class DomainExtractor:
    """Classify job domain/sector using LLM."""

    def extract(self, text: str, title: str = "", company: str = "") -> AspectExtraction:
        """Extract domain using LLM. Falls back to empty if LLM unavailable."""
        try:
            classification = self._classify_with_llm(text, title, company)
            if classification:
                return AspectExtraction(
                    aspect="domain",
                    values=[classification.domain],
                    evidence_spans=[classification.evidence] if classification.evidence else [],
                    confidence=classification.confidence,
                    method="llm",
                )
        except Exception:
            pass

        return AspectExtraction(
            aspect="domain",
            values=[],
            evidence_spans=[],
            confidence=0.0,
            method="llm",
        )

    def _classify_with_llm(self, text: str, title: str, company: str) -> Optional[DomainClassification]:
        """Call LLM to classify domain."""
        if not os.environ.get("OPENAI_API_KEY"):
            return None

        try:
            from openai import OpenAI
        except ImportError:
            return None

        client = OpenAI()

        system = (
            "Classify the job posting into exactly one domain/sector from this list:\n"
            f"{json.dumps(DOMAINS)}\n\n"
            "Return JSON: {\"domain\": \"...\", \"confidence\": 0.0-1.0, \"evidence\": \"brief quote\"}\n"
            "If uncertain, pick the closest match with lower confidence."
        )
        user = f"Title: {title}\nCompany: {company}\n\nDescription (first 1500 chars):\n{text[:1500]}"

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content or "{}"
            # Strip markdown fences
            if "```" in raw:
                import re
                fenced = re.findall(r"```(?:json)?\n?([\s\S]*?)```", raw)
                if fenced:
                    raw = fenced[0]
            data = json.loads(raw)
            return DomainClassification(**data)
        except Exception:
            return None
