"""Orchestrates all aspect extractors for a job posting."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models.aspect import AspectExtraction
from .cleaner import clean_html, clean_text
from .extractors.skills import SkillsExtractor
from .extractors.language import LanguageExtractor
from .extractors.remote_policy import RemotePolicyExtractor
from .extractors.experience import ExperienceExtractor
from .extractors.education import EducationExtractor
from .extractors.benefits import BenefitsExtractor
from .extractors.domain import DomainExtractor
from .extractors.culture import CultureExtractor


class AspectExtractor:
    """Orchestrates deterministic and LLM-based aspect extraction.

    Usage:
        extractor = AspectExtractor()
        aspects = extractor.extract_all(text, title="...", company="...")
        aspects_dict = extractor.extract_all_as_dict(text)
    """

    def __init__(self, *, enable_llm: bool = False):
        """Initialize extractors.

        Args:
            enable_llm: Whether to run LLM-based extractors (domain, culture).
                       Disabled by default for batch processing speed.
        """
        self.skills_extractor = SkillsExtractor()
        self.language_extractor = LanguageExtractor()
        self.remote_extractor = RemotePolicyExtractor()
        self.experience_extractor = ExperienceExtractor()
        self.education_extractor = EducationExtractor()
        self.benefits_extractor = BenefitsExtractor()
        self.domain_extractor = DomainExtractor()
        self.culture_extractor = CultureExtractor()
        self.enable_llm = enable_llm

    def extract_all(
        self,
        text: str,
        *,
        title: str = "",
        company: str = "",
    ) -> List[AspectExtraction]:
        """Run all extractors on text and return list of AspectExtractions.

        Args:
            text: Clean job description text.
            title: Job title for LLM context.
            company: Company name for LLM context.
        """
        results: List[AspectExtraction] = []

        # Deterministic extractors (fast, always run)
        results.append(self.skills_extractor.extract_skills(text))
        results.append(self.skills_extractor.extract_tools(text))
        results.append(self.language_extractor.extract(text))
        results.append(self.remote_extractor.extract(text))
        results.append(self.experience_extractor.extract(text))
        results.append(self.education_extractor.extract(text))
        results.append(self.benefits_extractor.extract(text))

        # LLM-based extractors (slow, optional)
        if self.enable_llm:
            results.append(self.domain_extractor.extract(text, title=title, company=company))
            results.append(self.culture_extractor.extract(text, title=title, company=company))

        return results

    def extract_all_as_dict(
        self,
        text: str,
        *,
        title: str = "",
        company: str = "",
    ) -> Dict[str, List[str]]:
        """Extract all aspects and return as {aspect_name: [values]}."""
        extractions = self.extract_all(text, title=title, company=company)
        result: Dict[str, List[str]] = {}
        for ext in extractions:
            if ext.values:
                result[ext.aspect] = ext.values
        return result

    def extract_for_chunk(
        self,
        chunk_text: str,
        section: Optional[str] = None,
    ) -> List[AspectExtraction]:
        """Extract aspects from a single chunk (deterministic only, no LLM)."""
        results: List[AspectExtraction] = []
        results.append(self.skills_extractor.extract_skills(chunk_text))
        results.append(self.skills_extractor.extract_tools(chunk_text))
        results.append(self.language_extractor.extract(chunk_text))
        results.append(self.remote_extractor.extract(chunk_text))
        results.append(self.experience_extractor.extract(chunk_text))
        results.append(self.education_extractor.extract(chunk_text))
        results.append(self.benefits_extractor.extract(chunk_text))
        # Filter to only non-empty
        return [r for r in results if r.values]
