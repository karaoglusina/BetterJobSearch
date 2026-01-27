"""Agent system data models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Record of a single tool invocation by an agent."""

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None


class IntentClassification(BaseModel):
    """Classified user intent for routing to the right worker."""

    intent: str  # "search" | "compare" | "explore" | "detail" | "general"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    entities: Dict[str, Any] = Field(default_factory=dict)  # extracted job_ids, aspects, etc.
    sub_tasks: List[str] = Field(default_factory=list)  # for complex multi-step queries


class AgentResult(BaseModel):
    """Result from an agent execution."""

    answer: str = ""
    tool_calls: List[ToolCall] = Field(default_factory=list)
    job_ids_referenced: List[str] = Field(default_factory=list)
    ui_actions: List[Dict[str, Any]] = Field(default_factory=list)  # e.g., highlight_jobs, recluster
    model: str = ""
    total_tokens: int = 0


class ConversationTurn(BaseModel):
    """A single turn in conversation history."""

    role: str  # "user" | "assistant" | "system" | "tool"
    content: str = ""
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_call_id: Optional[str] = None


class MemoryState(BaseModel):
    """Agent memory state for sliding window + summary."""

    recent_turns: List[ConversationTurn] = Field(default_factory=list)
    summary: str = ""
    mentioned_job_ids: List[str] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
