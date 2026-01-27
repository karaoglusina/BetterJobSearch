"""Skills extraction using spaCy PhraseMatcher and vocabulary lists."""

from __future__ import annotations

from pathlib import Path
from typing import List, Set

from ...models.aspect import AspectExtraction

VOCAB_DIR = Path(__file__).parent.parent / "vocab"

# Lazy-loaded spaCy resources
_nlp = None
_skills_matcher = None
_tools_matcher = None


def _load_vocab(filename: str) -> List[str]:
    """Load vocabulary from a text file (one term per line)."""
    path = VOCAB_DIR / filename
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.blank("en")
        except ImportError:
            _nlp = None
    return _nlp


def _build_phrase_matcher(nlp, terms: List[str], label: str):
    """Build a spaCy PhraseMatcher from term list."""
    from spacy.matcher import PhraseMatcher

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(term) for term in terms]
    if patterns:
        matcher.add(label, patterns)
    return matcher


def _get_skills_matcher():
    global _skills_matcher
    if _skills_matcher is None:
        nlp = _get_nlp()
        if nlp is None:
            return None
        terms = _load_vocab("skills.txt")
        _skills_matcher = _build_phrase_matcher(nlp, terms, "SKILL")
    return _skills_matcher


def _get_tools_matcher():
    global _tools_matcher
    if _tools_matcher is None:
        nlp = _get_nlp()
        if nlp is None:
            return None
        terms = _load_vocab("tools.txt")
        _tools_matcher = _build_phrase_matcher(nlp, terms, "TOOL")
    return _tools_matcher


class SkillsExtractor:
    """Extract skills and tools from text using spaCy PhraseMatcher."""

    def extract_skills(self, text: str) -> AspectExtraction:
        """Extract technical skills from text."""
        values, spans = self._match(text, "skills")
        return AspectExtraction(
            aspect="skills",
            values=sorted(values),
            evidence_spans=spans[:10],
            confidence=1.0,
            method="phrase_matcher",
        )

    def extract_tools(self, text: str) -> AspectExtraction:
        """Extract tools/platforms from text."""
        values, spans = self._match(text, "tools")
        return AspectExtraction(
            aspect="tools",
            values=sorted(values),
            evidence_spans=spans[:10],
            confidence=1.0,
            method="phrase_matcher",
        )

    def _match(self, text: str, aspect: str) -> tuple[list[str], list[str]]:
        nlp = _get_nlp()
        if nlp is None:
            return self._fallback_match(text, aspect)

        matcher = _get_skills_matcher() if aspect == "skills" else _get_tools_matcher()
        if matcher is None:
            return self._fallback_match(text, aspect)

        doc = nlp(text)
        matches = matcher(doc)

        seen: Set[str] = set()
        values: List[str] = []
        spans: List[str] = []

        for match_id, start, end in matches:
            span_text = doc[start:end].text
            # Normalize: use the matched text with original casing
            key = span_text.lower()
            if key not in seen:
                seen.add(key)
                values.append(span_text)
                # Get context (±30 chars)
                char_start = max(0, doc[start].idx - 30)
                char_end = min(len(text), doc[end - 1].idx + len(doc[end - 1].text) + 30)
                spans.append(text[char_start:char_end].strip())

        return values, spans

    def _fallback_match(self, text: str, aspect: str) -> tuple[list[str], list[str]]:
        """Fallback when spaCy is not available: case-insensitive substring search."""
        filename = "skills.txt" if aspect == "skills" else "tools.txt"
        terms = _load_vocab(filename)
        text_lower = text.lower()

        seen: Set[str] = set()
        values: List[str] = []
        spans: List[str] = []

        for term in terms:
            term_lower = term.lower()
            idx = text_lower.find(term_lower)
            if idx >= 0:
                if term_lower not in seen:
                    seen.add(term_lower)
                    values.append(term)
                    start = max(0, idx - 30)
                    end = min(len(text), idx + len(term) + 30)
                    spans.append(text[start:end].strip())

        return values, spans
