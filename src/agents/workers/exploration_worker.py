"""Exploration worker: cluster browsing, market overview, trend discovery."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...models.agent import AgentResult
from ..loop import react_loop
from ..tools.registry import ToolRegistry

SYSTEM_PROMPT = """You are a job market exploration specialist. Your job is to help users understand the job market landscape through clustering, distributions, and overviews.

You have access to exploration and search tools:
- get_selected_jobs: Get details about jobs the user has selected in the UI
- cluster_by_aspect: Re-cluster the scatter plot by an aspect (skills, language, domain, etc.)
- browse_cluster: Show sample jobs from a cluster
- cluster_by_concept: Cluster by a free-text concept
- aspect_distribution: Show how values distribute across jobs
- hybrid_search: General search
- extract_keywords: Extract themes from text

Strategy:
1. If the user mentions "selected jobs", "these jobs", or has jobs selected, use get_selected_jobs first
2. For market overview questions, use aspect_distribution to show landscape
3. For "what kinds of X" questions, use cluster_by_aspect or cluster_by_concept
4. For browsing, use browse_cluster to show examples
5. Combine multiple tools to build a comprehensive picture
6. Summarize patterns and insights, not just raw data

Help users discover trends and patterns in the job market."""


class ExplorationWorker:
    """Agent specialized in exploring the job market landscape."""

    def __init__(self, registry: ToolRegistry, *, model: str = "gpt-4o-mini"):
        self.registry = registry
        self.model = model
        self.tool_names = [
            "get_selected_jobs", "cluster_by_aspect", "browse_cluster", "cluster_by_concept",
            "aspect_distribution", "hybrid_search", "extract_keywords",
        ]

    def run(
        self,
        query: str,
        *,
        context: str = "",
        on_tool_call: Optional[Any] = None,
        selected_job_ids: Optional[List[str]] = None,
    ) -> AgentResult:
        """Run the exploration worker."""
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
