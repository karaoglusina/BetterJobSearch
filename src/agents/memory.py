"""Sliding window + summary memory for agent conversations."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..models.agent import ConversationTurn, MemoryState


class ConversationMemory:
    """Manages conversation history with sliding window and summary.

    - Keeps last N turns in full (sliding window)
    - Summarizes older turns into a brief summary
    - Tracks mentioned entities (job IDs, preferences)
    """

    def __init__(self, *, max_recent_turns: int = 5):
        self.max_recent_turns = max_recent_turns
        self.state = MemoryState()

    def add_turn(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a conversation turn."""
        turn = ConversationTurn(role=role, content=content, **kwargs)
        self.state.recent_turns.append(turn)

        # Extract mentioned job IDs
        self._extract_job_ids(content)

        # Compress if window exceeded
        if len(self.state.recent_turns) > self.max_recent_turns * 2:
            self._compress()

    def get_messages(self) -> List[Dict[str, str]]:
        """Return conversation history as OpenAI message format.

        Includes summary context if available, then recent turns.
        """
        messages: List[Dict[str, str]] = []

        if self.state.summary:
            messages.append({
                "role": "system",
                "content": f"Conversation summary so far:\n{self.state.summary}",
            })

        for turn in self.state.recent_turns[-self.max_recent_turns:]:
            msg: Dict[str, str] = {"role": turn.role, "content": turn.content}
            messages.append(msg)

        return messages

    def get_context_hint(self) -> str:
        """Return a brief context hint for the agent about conversation state."""
        parts: List[str] = []
        if self.state.mentioned_job_ids:
            parts.append(f"Previously discussed jobs: {', '.join(self.state.mentioned_job_ids[-10:])}")
        if self.state.user_preferences:
            parts.append(f"User preferences: {json.dumps(self.state.user_preferences)}")
        return "\n".join(parts) if parts else ""

    def set_preference(self, key: str, value: Any) -> None:
        """Set a user preference (e.g., preferred_location, skill_focus)."""
        self.state.user_preferences[key] = value

    def reset(self) -> None:
        """Clear all memory."""
        self.state = MemoryState()

    def _extract_job_ids(self, text: str) -> None:
        """Extract job IDs or URLs mentioned in text."""
        import re
        # Match URLs (common job ID format)
        urls = re.findall(r'https?://[^\s<>"]+', text)
        for url in urls:
            if url not in self.state.mentioned_job_ids:
                self.state.mentioned_job_ids.append(url)

    def _compress(self) -> None:
        """Compress older turns into summary."""
        if len(self.state.recent_turns) <= self.max_recent_turns:
            return

        old_turns = self.state.recent_turns[:-self.max_recent_turns]
        self.state.recent_turns = self.state.recent_turns[-self.max_recent_turns:]

        # Build summary from old turns
        summary_parts: List[str] = []
        if self.state.summary:
            summary_parts.append(self.state.summary)

        for turn in old_turns:
            if turn.role == "user":
                summary_parts.append(f"User asked: {turn.content[:100]}")
            elif turn.role == "assistant":
                summary_parts.append(f"Assistant: {turn.content[:100]}")

        # Try LLM summarization, fall back to concatenation
        new_summary = self._summarize_with_llm(summary_parts)
        if new_summary:
            self.state.summary = new_summary
        else:
            self.state.summary = "\n".join(summary_parts[-10:])

    def _summarize_with_llm(self, parts: List[str]) -> Optional[str]:
        """Summarize conversation history with LLM."""
        if not os.environ.get("OPENAI_API_KEY"):
            return None

        try:
            from openai import OpenAI
            client = OpenAI()

            text = "\n".join(parts)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Summarize this conversation history in 2-3 sentences. Focus on what the user was looking for and key findings."},
                    {"role": "user", "content": text[:2000]},
                ],
                temperature=0,
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return None
