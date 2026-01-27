"""Search worker: finds jobs matching criteria using hybrid search + query rewriting."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...models.agent import AgentResult
from ..loop import react_loop
from ..tools.registry import ToolRegistry

SYSTEM_PROMPT = """You are a job search specialist. Your job is to find relevant job postings based on user queries.

You have access to search tools:
- hybrid_search: Best general-purpose search (combines semantic + keyword)
- semantic_search: For conceptual queries ("jobs involving customer analytics")
- keyword_search: For specific terms ("Python", "Amsterdam", company names)
- filter_jobs: For metadata filtering (location, skills, remote policy)

Strategy:
1. Analyze the user's query to understand what they're looking for
2. Use the most appropriate search tool(s)
3. If initial results are insufficient, try reformulating the query
4. Synthesize findings into a clear, concise answer
5. Always mention specific job titles and companies in your response

Keep responses concise and actionable."""


class SearchWorker:
    """Agent specialized in finding jobs."""

    def __init__(self, registry: ToolRegistry, *, model: str = "gpt-4o-mini"):
        self.registry = registry
        self.model = model
        self.tool_names = ["hybrid_search", "semantic_search", "keyword_search", "filter_jobs"]

    def run(
        self,
        query: str,
        *,
        context: str = "",
        on_tool_call: Optional[Any] = None,
    ) -> AgentResult:
        """Run the search worker."""
        tools = self.registry.get_openai_tools(self.tool_names)

        user_msg = query
        if context:
            user_msg = f"{query}\n\nContext:\n{context}"

        return react_loop(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_msg,
            tools=tools,
            tool_executor=self.registry.execute,
            max_iterations=4,
            model=self.model,
            on_tool_call=on_tool_call,
        )
