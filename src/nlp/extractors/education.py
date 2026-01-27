"""Education requirement extraction using regex patterns."""

from __future__ import annotations

import re
from typing import List, Set

from ...models.aspect import AspectExtraction

EDUCATION_PATTERNS = [
    # PhD
    (re.compile(r"\b(ph\.?d|doctorate|doctoral)\b", re.I), "PhD"),
    # Master's
    (re.compile(r"\b(master'?s?\s+degree|m\.?sc\.?|m\.?a\.?|m\.?eng\.?|m\.?b\.?a\.?|mba)\b", re.I), "Master"),
    (re.compile(r"\bmaster'?s?\s+(in|of)\b", re.I), "Master"),
    # Bachelor's
    (re.compile(r"\b(bachelor'?s?\s+degree|b\.?sc\.?|b\.?a\.?|b\.?eng\.?|b\.?b\.?a\.?)\b", re.I), "Bachelor"),
    (re.compile(r"\bbachelor'?s?\s+(in|of)\b", re.I), "Bachelor"),
    # Generic degree
    (re.compile(r"\b(university|college)\s+degree\b", re.I), "Degree"),
    (re.compile(r"\b(higher\s+education|post[- ]?graduate|postgraduate)\b", re.I), "Postgraduate"),
    (re.compile(r"\b(diploma|associate'?s?\s+degree|certification)\b", re.I), "Diploma/Certificate"),
]

# Field of study patterns
FIELD_PATTERNS = [
    re.compile(
        r"\b(computer\s+science|software\s+engineering|information\s+technology|"
        r"data\s+science|statistics|mathematics|physics|engineering|"
        r"business\s+administration|economics|"
        r"artificial\s+intelligence|machine\s+learning|"
        r"information\s+systems|quantitative\s+field|"
        r"stem|technical\s+field|related\s+field)\b",
        re.I,
    ),
]

# Preference indicators
PREFERRED_PATTERN = re.compile(r"\b(preferred|nice[- ]to[- ]have|advantage|bonus|plus|or\s+equivalent)\b", re.I)
REQUIRED_PATTERN = re.compile(r"\b(required|mandatory|must|essential|necessary)\b", re.I)


class EducationExtractor:
    """Extract education requirements from job descriptions."""

    def extract(self, text: str) -> AspectExtraction:
        values: List[str] = []
        spans: List[str] = []
        seen: Set[str] = set()

        for pattern, level in EDUCATION_PATTERNS:
            for match in pattern.finditer(text):
                if level not in seen:
                    seen.add(level)
                    # Check surrounding context for preferred/required
                    context_start = max(0, match.start() - 60)
                    context_end = min(len(text), match.end() + 60)
                    context = text[context_start:context_end]

                    qualifier = ""
                    if PREFERRED_PATTERN.search(context):
                        qualifier = " preferred"
                    elif REQUIRED_PATTERN.search(context):
                        qualifier = " required"

                    # Check for field of study in context
                    field = ""
                    for fp in FIELD_PATTERNS:
                        fm = fp.search(context)
                        if fm:
                            field = f" in {fm.group(0)}"
                            break

                    values.append(f"{level}{field}{qualifier}".strip())
                    spans.append(context.strip())

        return AspectExtraction(
            aspect="education",
            values=values,
            evidence_spans=spans,
            confidence=1.0 if values else 0.0,
            method="regex",
        )
