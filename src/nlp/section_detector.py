"""Section boundary detection for job descriptions using structural + keyword matching."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Bullet point patterns
BULLET_PATTERN = re.compile(r"^\s*([•\-\*–]\s+|\d+\.\s+)")

# Section header patterns (keyword-based fallback)
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


def is_structural_header(line: str, next_lines: List[str]) -> bool:
    """Detect header by morphological features, not keywords.

    A line is likely a header if it:
    - Is short (< 60 chars, < 8 words)
    - Ends with colon
    - Is ALL CAPS or Title Case
    - Is followed by bullets or indented content
    """
    line = line.strip()

    # Reject if too long or empty
    if not line or len(line) > 60:
        return False

    # Feature 1: Ends with colon
    ends_with_colon = line.endswith(":")

    # Feature 2: Short (fewer than 8 words)
    words = line.rstrip(":").split()
    word_count = len(words)
    is_short = word_count <= 8

    # Reject very short lines (likely not headers)
    if word_count < 2 and not ends_with_colon:
        return False

    # Feature 3: ALL CAPS
    is_all_caps = line.isupper() and len(line) > 3

    # Feature 4: Title Case (most words capitalized)
    if words:
        cap_ratio = sum(1 for w in words if w and w[0].isupper()) / len(words)
        is_title_case = cap_ratio >= 0.6 and word_count >= 2
    else:
        is_title_case = False

    # Feature 5: Followed by bullets or indented content
    followed_by_bullets = any(
        BULLET_PATTERN.match(l) for l in next_lines[:3] if l.strip()
    )

    # Feature 6: Numbered prefix (1., 2., etc.)
    has_number_prefix = bool(re.match(r"^\d+\.\s+", line))

    # Scoring: combine features
    score = 0
    if ends_with_colon and is_short:
        score += 3  # Strong signal
    if is_all_caps:
        score += 2
    if is_title_case and is_short:
        score += 2
    if followed_by_bullets:
        score += 2
    if has_number_prefix:
        score += 1

    return score >= 3


def canonicalize_by_content(header_text: str, section_content: str) -> str:
    """Derive canonical aspect from header and content when keyword matching fails.

    Uses expanded pattern matching based on content analysis.
    """
    header_lower = header_text.lower().strip().rstrip(":")

    # First: try keyword matching (existing approach)
    if canonical := SECTION_CANONICAL.get(header_lower):
        return canonical

    # Second: content-based classification
    content_lower = section_content.lower()

    # Check for skill-like content (significantly expanded)
    skill_keywords = r"(python|java|sql|aws|docker|kubernetes|power\s*bi|tableau|qlik|excel|sap|erp|crm|salesforce|jira|confluence|postman|swagger|agile|scrum|git|azure|rest|api|json|xml|html|css|javascript|c#|\.net|dbt|nosql|mongodb|bigquery|hive|exago|erwin|sas\s*eg|kimball|aws\s*glue|bpmn|uml|lean|six\s*sigma|bizbok|bisl|isqi|bcs|one\s*identity|sailpoint|active\s*directory|iam|navision|dynamics\s*nav|murex|mxml|msmq|iis|siebel|sharepoint|visual\s*studio|soap|livebook|eod|gom)"
    if re.search(skill_keywords, content_lower):
        return "skills"

    # Check for responsibility-like content (expanded with new verbs)
    responsibility_patterns = r"(you will|you'll|responsible for|your role|your tasks|your mission|the challenge|what you'll do|tricks of the trade|develop|build|create|lead|manage|coordinate|facilitate|support|collaborate|translate|design|implement|maintain|analyze|conduct|engage|deliver|contribute|drive|establish|identify|assess|provide|work closely|act as|own|ensure|spearhead|partner|quantify|monitor|uncover|obtain|scope|supervise|capture|review and edit|be expert in|take the lead)"
    if re.search(responsibility_patterns, content_lower):
        return "responsibilities"

    # Check for requirement patterns (expanded)
    requirement_patterns = r"(\d+\+?\s*years?|experience|degree|bachelor|master|phd|hbo|wo|qualification|required|must have|essential|mandatory|proven track|expertise in|proficient|knowledge of|familiar with|understanding of|skilled at|fluent in|certification|ireb|babok|itil|togaf|competenc|self-starter|curious mindset|comfortable working in ambiguity|fast-paced|minimal oversight|portfolio|hands-on experience|conceptual understanding|exposure to|capability to manage|perseverance)"
    if re.search(requirement_patterns, content_lower):
        return "requirements"

    # Check for "nice to have" patterns (expanded)
    nice_to_have_patterns = r"(nice to have|bonus points|plus|advantage|preferred|desirable|beneficial|ideal|strong plus|is a plus|is advantageous|is a big advantage|is beneficial|knowledge of .* is a plus)"
    if re.search(nice_to_have_patterns, content_lower):
        return "nice_to_have"

    # Check for benefit patterns (expanded)
    benefit_patterns = r"(salary|gross monthly|compensation|bonus|vacation|holiday|health|insurance|pension|equity|stock|perks|benefits|working from home|hybrid|remote|wfh|allowance|leave|parental|travel|bike|lease|discount|development|training|academy|13th month|benefit budget|personal development budget|atv days|ns business card|lease car|mobility budget|laptop and phone|shuttle bus|end-of-year bonus|performance bonus)"
    if re.search(benefit_patterns, content_lower):
        return "benefits"

    # Check for company description (expanded)
    company_patterns = r"(we are|our company|our team|our mission|our purpose|our values|founded|established|headquartered|about us|organization|who we are|what we do|leading|global|international|employees|turnover|revenue)"
    if re.search(company_patterns, content_lower):
        return "about_company"

    # Check for role/position description (expanded)
    role_patterns = r"(about the role|position overview|job profile|the role|the position|the opportunity|introduction|in a nutshell|summary of the role|main purpose|your mission|the challenge|vacancy description|how you fit)"
    if re.search(role_patterns, content_lower):
        return "role_description"

    # Check for application/contact information (expanded)
    contact_patterns = r"(apply now|how to apply|contact us|get in touch|interested|recruitment|application process|want to know more|do you want more information|express your interest|questions about|contact the recruiter|we would like to meet you)"
    if re.search(contact_patterns, content_lower):
        return "application_info"

    # Check for team/culture information (expanded)
    culture_patterns = r"(our culture|work environment|team|atmosphere|values|diversity|inclusion|working at|join us|why work|what makes|open no-nonsense|international cooperation|connection|down to earth|collaborative mindset)"
    if re.search(culture_patterns, content_lower):
        return "culture_team"

    # Default: use cleaned header as aspect name
    return re.sub(r"[^a-z0-9]+", "_", header_lower)[:30] or "other"


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


def detect_sections_structural(text: str) -> List[Section]:
    """Detect sections using structural (form-based) analysis.

    Combines keyword matching with structural header detection:
    1. First tries keyword-based detection (existing approach)
    2. Then uses structural analysis (short line + colon + bullets)
    3. Canonicalizes by content when header keywords don't match

    Args:
        text: Job description text.

    Returns:
        List of Section objects with canonical names and content.
    """
    lines = text.split("\n")
    sections: List[Section] = []
    current_raw: Optional[str] = None
    current_lines: List[str] = []
    current_start: int = 0

    def save_section(end_line: int) -> None:
        """Save the current section if it has content."""
        nonlocal current_lines, current_raw
        if not current_lines:
            return

        section_text = "\n".join(current_lines).strip()
        if not section_text:
            return

        # Determine canonical name
        if current_raw:
            # Try keyword first, then content-based
            canonical = _canonicalize(current_raw) or canonicalize_by_content(
                current_raw, section_text
            )
        else:
            canonical = None  # Intro section

        sections.append(
            Section(
                name=canonical,
                raw_name=current_raw,
                text=section_text,
                start_line=current_start,
                end_line=end_line,
            )
        )

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check keyword-based header first
        is_keyword_header = bool(HEADER_PATTERNS.match(stripped))

        # Check structural header if not keyword match
        next_lines = lines[i + 1 : i + 5] if i < len(lines) - 1 else []
        is_structural = not is_keyword_header and is_structural_header(
            stripped, next_lines
        )

        if is_keyword_header or is_structural:
            # Save previous section
            save_section(i - 1)

            # Start new section
            current_raw = stripped.rstrip(":")
            current_lines = []
            current_start = i + 1
        else:
            current_lines.append(line)

    # Save final section
    save_section(len(lines) - 1)

    return sections


def detect_sections_with_spacy(text: str) -> List[Section]:
    """Enhanced section detection with spaCy sentence boundaries.

    Falls back to detect_sections_structural() if spaCy is not available.
    """
    sections = detect_sections_structural(text)

    try:
        import spacy

        try:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
        except Exception:
            return sections

        for section in sections:
            doc = nlp(section.text)
            section.sentences = [
                sent.text.strip() for sent in doc.sents if sent.text.strip()
            ]
    except ImportError:
        # spaCy not available; split on sentence-ending punctuation
        sent_split = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
        for section in sections:
            section.sentences = [
                s.strip() for s in sent_split.split(section.text) if s.strip()
            ]

    return sections
