"""Language requirement extraction using regex patterns."""

from __future__ import annotations

import re
from typing import List, Set

from ...models.aspect import AspectExtraction

# Patterns for language requirements
LANGUAGE_PATTERNS = [
    # Dutch requirements
    (re.compile(r"\b(fluent|native|proficient|excellent)\s+(in\s+)?dutch\b", re.I), "Dutch required"),
    (re.compile(r"\bdutch\s+(is\s+)?(required|mandatory|essential|necessary|must)\b", re.I), "Dutch required"),
    (re.compile(r"\b(must|need to|should)\s+(speak|know|have)\s+dutch\b", re.I), "Dutch required"),
    (re.compile(r"\bnederlands\s+(is\s+)?(vereist|verplicht)\b", re.I), "Dutch required"),
    (re.compile(r"\bdutch\s+(is\s+)?(preferred|nice|advantage|plus|bonus)\b", re.I), "Dutch preferred"),
    (re.compile(r"\b(preferably|ideally)\s+(speak|know)\s+dutch\b", re.I), "Dutch preferred"),
    # English requirements
    (re.compile(r"\b(fluent|native|proficient|excellent)\s+(in\s+)?english\b", re.I), "English fluent"),
    (re.compile(r"\benglish\s+(is\s+)?(required|mandatory|essential|necessary|must)\b", re.I), "English required"),
    (re.compile(r"\b(professional|business|working)\s+(level\s+)?english\b", re.I), "English professional"),
    # German requirements
    (re.compile(r"\b(fluent|native|proficient|excellent)\s+(in\s+)?german\b", re.I), "German fluent"),
    (re.compile(r"\bgerman\s+(is\s+)?(required|mandatory|essential|preferred)\b", re.I), "German required"),
    # French requirements
    (re.compile(r"\b(fluent|native|proficient|excellent)\s+(in\s+)?french\b", re.I), "French fluent"),
    (re.compile(r"\bfrench\s+(is\s+)?(required|mandatory|essential|preferred)\b", re.I), "French required"),
    # Generic multilingual
    (re.compile(r"\b(multilingual|polyglot|multiple languages)\b", re.I), "Multilingual"),
    (re.compile(r"\b(bilingual)\b", re.I), "Bilingual"),
]


class LanguageExtractor:
    """Extract language requirements from job descriptions."""

    def extract(self, text: str) -> AspectExtraction:
        seen: Set[str] = set()
        values: List[str] = []
        spans: List[str] = []

        for pattern, label in LANGUAGE_PATTERNS:
            for match in pattern.finditer(text):
                if label not in seen:
                    seen.add(label)
                    values.append(label)
                    start = max(0, match.start() - 40)
                    end = min(len(text), match.end() + 40)
                    spans.append(text[start:end].strip())

        return AspectExtraction(
            aspect="language",
            values=values,
            evidence_spans=spans,
            confidence=1.0 if values else 0.0,
            method="regex",
        )
