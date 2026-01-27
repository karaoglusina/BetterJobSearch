"""Tiered context manager for agent token budget control."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# Token budget allocation
DEFAULT_BUDGET = {
    "system_prompt": 300,
    "conversation_history": 800,
    "retrieved_context": 4000,
    "user_query": 100,
    "response_reserve": 800,
}

# Approximate tokens per tier
TIER_TOKENS = {
    0: 50,   # Aspects + Keywords
    1: 200,  # Relevant Chunks
    2: 500,  # Full Sections
    3: 800,  # Full Job Description
}


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4 + 1


class ContextManager:
    """Manages tiered context expansion for agent conversations.

    Tiers:
        0: Aspects + Keywords (~50 tokens/job) - Pre-extracted, instant
        1: Relevant Chunks (~200 tokens/job) - Hybrid retrieval
        2: Full Sections (~500 tokens/job) - Section-level
        3: Full Job Description (~800 tokens/job) - Complete text

    Agent starts at Tier 0 and escalates based on task needs.
    """

    def __init__(self, budget: Optional[Dict[str, int]] = None):
        self.budget = budget or DEFAULT_BUDGET.copy()
        self.context_budget = self.budget["retrieved_context"]

    def build_tier0_context(
        self,
        job_summaries: List[Dict[str, Any]],
        max_jobs: int = 50,
    ) -> str:
        """Tier 0: Aspects + keywords only (~50 tokens/job).

        Args:
            job_summaries: List of {job_id, title, company, aspects, keywords}.
            max_jobs: Maximum jobs to include.
        """
        parts: List[str] = []
        tokens_used = 0

        for job in job_summaries[:max_jobs]:
            line = f"- {job.get('title', '?')} @ {job.get('company', '?')}"
            aspects = job.get("aspects", {})
            if aspects:
                aspect_str = "; ".join(
                    f"{k}: {', '.join(v[:3])}" for k, v in aspects.items() if v
                )
                line += f" | {aspect_str}"
            keywords = job.get("keywords", [])
            if keywords:
                line += f" | kw: {', '.join(keywords[:5])}"

            est = _estimate_tokens(line)
            if tokens_used + est > self.context_budget:
                break
            parts.append(line)
            tokens_used += est

        return "\n".join(parts)

    def build_tier1_context(
        self,
        chunks: List[Dict[str, Any]],
        max_chunks: int = 15,
    ) -> str:
        """Tier 1: Relevant chunks (~200 tokens/job).

        Args:
            chunks: Retrieved chunk dicts with text, meta.
            max_chunks: Maximum chunks to include.
        """
        parts: List[str] = []
        tokens_used = 0

        for i, ch in enumerate(chunks[:max_chunks], 1):
            meta = ch.get("meta", {})
            header = f"[{i}] {meta.get('title', '?')} — {meta.get('company', '?')}"
            section = ch.get("section", "")
            if section:
                header += f" [{section}]"
            text = ch.get("text", "")[:500]
            block = f"{header}\n{text}"

            est = _estimate_tokens(block)
            if tokens_used + est > self.context_budget:
                break
            parts.append(block)
            tokens_used += est

        return "\n\n".join(parts)

    def build_tier2_context(
        self,
        job_sections: List[Dict[str, Any]],
        max_sections: int = 5,
    ) -> str:
        """Tier 2: Full sections (~500 tokens/job).

        Args:
            job_sections: List of {job_id, title, company, section_name, text}.
        """
        parts: List[str] = []
        tokens_used = 0

        for sec in job_sections[:max_sections]:
            header = f"## {sec.get('title', '?')} — {sec.get('company', '?')} [{sec.get('section_name', '')}]"
            text = sec.get("text", "")[:2000]
            block = f"{header}\n{text}"

            est = _estimate_tokens(block)
            if tokens_used + est > self.context_budget:
                break
            parts.append(block)
            tokens_used += est

        return "\n\n".join(parts)

    def build_tier3_context(
        self,
        full_descriptions: List[Dict[str, Any]],
        max_jobs: int = 2,
    ) -> str:
        """Tier 3: Full job description (~800 tokens/job).

        Args:
            full_descriptions: List of {job_id, title, company, description}.
        """
        parts: List[str] = []
        tokens_used = 0

        for job in full_descriptions[:max_jobs]:
            header = f"## {job.get('title', '?')} — {job.get('company', '?')}"
            text = job.get("description", "")[:3000]
            block = f"{header}\n{text}"

            est = _estimate_tokens(block)
            if tokens_used + est > self.context_budget:
                break
            parts.append(block)
            tokens_used += est

        return "\n\n".join(parts)

    def select_tier(self, intent: str, n_jobs: int) -> int:
        """Select appropriate tier based on intent and job count.

        Args:
            intent: Classified intent (search, compare, explore, detail).
            n_jobs: Number of jobs being processed.

        Returns:
            Recommended tier (0-3).
        """
        if intent == "search" and n_jobs > 10:
            return 0
        elif intent == "search":
            return 1
        elif intent == "explore":
            return 0
        elif intent == "compare" and n_jobs <= 3:
            return 2
        elif intent == "detail":
            return 3
        else:
            return 1
