"""Coordinator agent: intent classification + task routing."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from ..models.agent import AgentResult, IntentClassification
from .memory import ConversationMemory
from .tools.registry import get_default_registry, ToolRegistry
from .workers.search_worker import SearchWorker
from .workers.analysis_worker import AnalysisWorker
from .workers.exploration_worker import ExplorationWorker


CLASSIFY_SYSTEM = """Classify the user's intent into one of these categories:
- search: Finding jobs matching criteria (e.g., "find Python jobs", "show me remote positions")
- compare: Comparing specific jobs (e.g., "compare these 3 jobs", "how do X and Y differ")
- explore: Understanding the market landscape (e.g., "what kinds of AI jobs exist", "show me clusters by skills")
- detail: Deep dive into a specific job (e.g., "tell me about this job", "what skills does job X need")
- general: General questions not about specific jobs (e.g., "how does the search work", "what aspects are available")

Return JSON: {"intent": "...", "confidence": 0.0-1.0, "entities": {}, "sub_tasks": []}

In entities, extract:
- job_ids: any mentioned job IDs or URLs
- aspects: any mentioned aspects (skills, language, remote, etc.)
- skills: any mentioned specific skills
- location: any mentioned locations
"""


class Coordinator:
    """Top-level coordinator that classifies intent and delegates to workers.

    Usage:
        coordinator = Coordinator()
        result = coordinator.handle("Find Python jobs in Amsterdam")
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        registry: Optional[ToolRegistry] = None,
    ):
        self.model = model
        self.registry = registry or get_default_registry()
        self.memory = ConversationMemory()

        # Initialize workers
        self.search_worker = SearchWorker(self.registry, model=model)
        self.analysis_worker = AnalysisWorker(self.registry, model=model)
        self.exploration_worker = ExplorationWorker(self.registry, model=model)

    def handle(
        self,
        user_message: str,
        *,
        on_tool_call: Optional[Callable] = None,
        on_intent: Optional[Callable[[IntentClassification], None]] = None,
    ) -> AgentResult:
        """Handle a user message by classifying intent and delegating to the appropriate worker.

        Args:
            user_message: The user's question or request.
            on_tool_call: Optional callback for tool call events.
            on_intent: Optional callback when intent is classified.

        Returns:
            AgentResult with the answer and trace.
        """
        # Add to memory
        self.memory.add_turn("user", user_message)

        # Classify intent
        intent = self._classify_intent(user_message)
        if on_intent:
            on_intent(intent)

        # Build context hint from memory
        context = self.memory.get_context_hint()

        # Route to worker
        if intent.intent == "search":
            result = self.search_worker.run(user_message, context=context, on_tool_call=on_tool_call)
        elif intent.intent in ("compare", "detail"):
            result = self.analysis_worker.run(user_message, context=context, on_tool_call=on_tool_call)
        elif intent.intent == "explore":
            result = self.exploration_worker.run(user_message, context=context, on_tool_call=on_tool_call)
        elif intent.intent == "general":
            result = self._handle_general(user_message)
        else:
            result = self.search_worker.run(user_message, context=context, on_tool_call=on_tool_call)

        # Save assistant response to memory
        self.memory.add_turn("assistant", result.answer)

        return result

    def _classify_intent(self, message: str) -> IntentClassification:
        """Classify user intent using LLM."""
        if not os.environ.get("OPENAI_API_KEY"):
            return self._classify_heuristic(message)

        try:
            from openai import OpenAI
            client = OpenAI()

            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CLASSIFY_SYSTEM},
                    {"role": "user", "content": message},
                ],
                temperature=0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content or "{}"
            if "```" in raw:
                import re
                fenced = re.findall(r"```(?:json)?\n?([\s\S]*?)```", raw)
                if fenced:
                    raw = fenced[0]
            data = json.loads(raw)
            return IntentClassification(**data)
        except Exception:
            return self._classify_heuristic(message)

    def _classify_heuristic(self, message: str) -> IntentClassification:
        """Fallback heuristic intent classification."""
        msg_lower = message.lower()

        if any(w in msg_lower for w in ["find", "search", "show me", "looking for", "jobs with", "positions"]):
            return IntentClassification(intent="search", confidence=0.7)
        elif any(w in msg_lower for w in ["compare", "difference", "versus", "vs", "better"]):
            return IntentClassification(intent="compare", confidence=0.7)
        elif any(w in msg_lower for w in ["cluster", "overview", "landscape", "kinds of", "types of", "market", "trend"]):
            return IntentClassification(intent="explore", confidence=0.7)
        elif any(w in msg_lower for w in ["tell me about", "detail", "what does", "summarize", "describe"]):
            return IntentClassification(intent="detail", confidence=0.7)
        else:
            return IntentClassification(intent="search", confidence=0.5)

    def _handle_general(self, message: str) -> AgentResult:
        """Handle general questions without tool use."""
        if not os.environ.get("OPENAI_API_KEY"):
            return AgentResult(
                answer="I can help you search, compare, and explore job postings. Try asking something like 'Find Python jobs in Amsterdam' or 'What types of AI jobs exist?'",
                model=self.model,
            )

        try:
            from openai import OpenAI
            client = OpenAI()

            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a job market analyst assistant. Answer general questions about the system capabilities, available aspects (skills, tools, language, remote_policy, experience, education, benefits, domain, culture), and how to use the search and exploration features."},
                    {"role": "user", "content": message},
                ],
                temperature=0.3,
                max_tokens=400,
            )
            return AgentResult(
                answer=resp.choices[0].message.content or "",
                model=self.model,
            )
        except Exception as e:
            return AgentResult(answer=f"Error: {e}", model=self.model)

    def reset_memory(self) -> None:
        """Reset conversation memory."""
        self.memory.reset()
