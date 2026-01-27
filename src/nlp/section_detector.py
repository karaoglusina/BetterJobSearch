"""Section boundary detection for job descriptions using regex + spaCy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Section header patterns (expanded from original rag.py)
HEADER_PATTERNS = re.compile(
    r"^\s*("
    r"responsibilities|what you'?ll do|what you will do|your role|the role|"
    r"key responsibilities|main responsibilities|role description|"
    r"requirements|qualifications|what we'?re looking for|what you bring|"
    r"must[- ]?have|nice[- ]?to[- ]?have|preferred|desired|bonus points|"
    r"about you|your profile|who you are|ideal candidate|"
    r"about us|about the company|who we are|about the team|the team|"
    r"benefits|perks|what we offer|compensation|our offer|"
    r"culture|our values|how we work|"
    r"how to apply|application process|next steps|"
    r"skills|technical skills|required skills|"
    r"education|educational requirements|"
    r"experience|work experience|professional experience|"
    r"tools|technologies|tech stack|stack|"
    r"languages|language requirements"
    r")\s*:?\s*$",
    re.IGNORECASE,
)

# Map raw section names to canonical names
SECTION_CANONICAL = {
    "responsibilities": "responsibilities",
    "what you'll do": "responsibilities",
    "what youll do": "responsibilities",
    "what you will do": "responsibilities",
    "your role": "responsibilities",
    "the role": "responsibilities",
    "key responsibilities": "responsibilities",
    "main responsibilities": "responsibilities",
    "role description": "responsibilities",
    "requirements": "requirements",
    "qualifications": "requirements",
    "what we're looking for": "requirements",
    "what were looking for": "requirements",
    "what you bring": "requirements",
    "must-have": "requirements",
    "must have": "requirements",
    "nice-to-have": "nice_to_have",
    "nice to have": "nice_to_have",
    "preferred": "nice_to_have",
    "desired": "nice_to_have",
    "bonus points": "nice_to_have",
    "about you": "about_you",
    "your profile": "about_you",
    "who you are": "about_you",
    "ideal candidate": "about_you",
    "about us": "about_company",
    "about the company": "about_company",
    "who we are": "about_company",
    "about the team": "about_company",
    "the team": "about_company",
    "benefits": "benefits",
    "perks": "benefits",
    "what we offer": "benefits",
    "compensation": "benefits",
    "our offer": "benefits",
    "culture": "culture",
    "our values": "culture",
    "how we work": "culture",
    "how to apply": "how_to_apply",
    "application process": "how_to_apply",
    "next steps": "how_to_apply",
    "skills": "skills",
    "technical skills": "skills",
    "required skills": "skills",
    "education": "education",
    "educational requirements": "education",
    "experience": "experience",
    "work experience": "experience",
    "professional experience": "experience",
    "tools": "tools",
    "technologies": "tools",
    "tech stack": "tools",
    "stack": "tools",
    "languages": "languages",
    "language requirements": "languages",
}


@dataclass
class Section:
    """A detected section in a job description."""

    name: Optional[str]  # canonical name or None for intro
    raw_name: Optional[str]  # original header text
    text: str
    start_line: int = 0
    end_line: int = 0
    sentences: List[str] = field(default_factory=list)


def _canonicalize(raw_name: str) -> Optional[str]:
    """Map a raw section header to its canonical name."""
    cleaned = raw_name.strip().lower().rstrip(":")
    return SECTION_CANONICAL.get(cleaned)


def detect_sections(text: str) -> List[Section]:
    """Split job description text into labeled sections.

    Args:
        text: Job description text (may contain newlines).

    Returns:
        List of Section objects, each with canonical name and text content.
    """
    lines = text.split("\n")
    sections: List[Section] = []
    current_name: Optional[str] = None
    current_raw: Optional[str] = None
    current_lines: List[str] = []
    current_start: int = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if HEADER_PATTERNS.match(stripped):
            # Save previous section
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append(Section(
                        name=current_name,
                        raw_name=current_raw,
                        text=section_text,
                        start_line=current_start,
                        end_line=i - 1,
                    ))
            current_raw = stripped.rstrip(":")
            current_name = _canonicalize(stripped)
            current_lines = []
            current_start = i + 1
        else:
            current_lines.append(line)

    # Save final section
    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append(Section(
                name=current_name,
                raw_name=current_raw,
                text=section_text,
                start_line=current_start,
                end_line=len(lines) - 1,
            ))

    return sections


def detect_sections_with_spacy(text: str) -> List[Section]:
    """Enhanced section detection with spaCy sentence boundaries.

    Falls back to detect_sections() if spaCy is not available.
    """
    sections = detect_sections(text)

    try:
        import spacy
        try:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
        except Exception:
            return sections

        for section in sections:
            doc = nlp(section.text)
            section.sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except ImportError:
        # spaCy not available; split on sentence-ending punctuation
        import re
        sent_split = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
        for section in sections:
            section.sentences = [s.strip() for s in sent_split.split(section.text) if s.strip()]

    return sections
