"""Analysis worker: job comparison, summarization, deep-dive analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...models.agent import AgentResult
from ..loop import react_loop
from ..tools.registry import ToolRegistry

SYSTEM_PROMPT = """You are a job analysis specialist. Your job is to provide detailed analysis of specific jobs, compare jobs, and answer detailed questions.

You have access to retrieval and analysis tools:
- get_selected_jobs: Get details about jobs the user has selected in the UI
- get_job_summary: Quick overview of a job (title, company, location)
- get_job_chunks: Get text sections from a job (requirements, responsibilities, benefits)
- get_full_text: Get complete job description (use sparingly)
- expand_context: Search within a job for specific info
- compare_aspects: Side-by-side comparison of an aspect across jobs
- find_similar_jobs: Find jobs similar to a given one
- extract_keywords: Extract keywords from text

Strategy:
1. If the user mentions "selected jobs", "these jobs", or has jobs selected, use get_selected_jobs first
2. Start with get_job_summary for quick context (Tier 0)
3. Use get_job_chunks for specific sections when needed (Tier 1-2)
4. Only use get_full_text when comprehensive analysis is needed (Tier 3)
5. Use compare_aspects for multi-job comparisons
6. Provide structured, clear analysis

Be thorough but concise. Cite specific details from the job descriptions."""


class AnalysisWorker:
    """Agent specialized in analyzing and comparing jobs."""

    def __init__(self, registry: ToolRegistry, *, model: str = "gpt-4o-mini"):
        self.registry = registry
        self.model = model
        self.tool_names = [
            "get_selected_jobs", "get_job_summary", "get_job_chunks", "get_full_text",
            "expand_context", "compare_aspects", "find_similar_jobs",
            "extract_keywords",
        ]

    def run(
        self,
        query: str,
        *,
        context: str = "",
        on_tool_call: Optional[Any] = None,
        selected_job_ids: Optional[List[str]] = None,
    ) -> AgentResult:
        """Run the analysis worker."""
        tools = self.registry.get_openai_tools(self.tool_names)

        user_msg = query
        if context:
            user_msg = f"{query}\n\nContext:\n{context}"

        # Create a tool executor that passes selected_job_ids to the get_selected_jobs tool
        def tool_executor(tool_name: str, **kwargs):
            if tool_name == "get_selected_jobs" and selected_job_ids:
                kwargs["_selected_job_ids"] = selected_job_ids
            return self.registry.execute(tool_name, **kwargs)

        return react_loop(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_msg,
            tools=tools,
            tool_executor=tool_executor,
            max_iterations=6,
            model=self.model,
            on_tool_call=on_tool_call,
        )
