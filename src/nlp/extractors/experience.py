"""Experience/seniority extraction using regex and NER patterns."""

from __future__ import annotations

import re
from typing import List, Set

from ...models.aspect import AspectExtraction

# Years of experience patterns
YEARS_PATTERNS = [
    re.compile(r"\b(\d+)\s*\+?\s*years?\s*(of\s+)?(professional\s+|relevant\s+|work\s+)?experience\b", re.I),
    re.compile(r"\b(\d+)\s*-\s*(\d+)\s*years?\s*(of\s+)?(professional\s+|relevant\s+|work\s+)?experience\b", re.I),
    re.compile(r"\bminimum\s+(\d+)\s*years?\b", re.I),
    re.compile(r"\bat\s+least\s+(\d+)\s*years?\b", re.I),
    re.compile(r"\b(\d+)\s*\+?\s*years?\s+(in|with|working)\b", re.I),
]

# Seniority level patterns
SENIORITY_PATTERNS = [
    (re.compile(r"\b(junior|entry[- ]level|graduate|starter|trainee)\b", re.I), "junior"),
    (re.compile(r"\b(mid[- ]?level|intermediate|medior)\b", re.I), "mid"),
    (re.compile(r"\b(senior|sr\.?|experienced|advanced)\b", re.I), "senior"),
    (re.compile(r"\b(lead|team\s+lead|tech\s+lead|principal)\b", re.I), "lead"),
    (re.compile(r"\b(staff|distinguished|fellow)\b", re.I), "staff"),
    (re.compile(r"\b(head\s+of|director|vp|vice\s+president|chief)\b", re.I), "executive"),
    (re.compile(r"\b(manager|managing)\b", re.I), "manager"),
]


class ExperienceExtractor:
    """Extract experience requirements and seniority level."""

    def extract(self, text: str) -> AspectExtraction:
        values: List[str] = []
        spans: List[str] = []
        seen: Set[str] = set()

        # Extract years of experience
        for pattern in YEARS_PATTERNS:
            for match in pattern.finditer(text):
                matched_text = match.group(0)
                key = matched_text.lower()
                if key not in seen:
                    seen.add(key)
                    values.append(matched_text)
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    spans.append(text[start:end].strip())
                    break  # one years match is enough

        # Extract seniority level
        for pattern, level in SENIORITY_PATTERNS:
            for match in pattern.finditer(text):
                if level not in seen:
                    seen.add(level)
                    values.append(level)
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    spans.append(text[start:end].strip())
                    break  # one match per level

        return AspectExtraction(
            aspect="experience",
            values=values,
            evidence_spans=spans,
            confidence=1.0 if values else 0.0,
            method="regex",
        )
