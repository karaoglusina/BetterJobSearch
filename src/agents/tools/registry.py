"""Tool name -> handler mapping with OpenAI function schema generation."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional


class ToolRegistry:
    """Registry mapping tool names to handlers and OpenAI function schemas.

    Usage:
        registry = ToolRegistry()
        registry.register("semantic_search", handler_fn, schema)
        result = registry.execute("semantic_search", {"query": "python"})
        openai_tools = registry.get_openai_tools()
    """

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        handler: Callable[..., str],
        schema: Dict[str, Any],
    ) -> None:
        """Register a tool.

        Args:
            name: Tool name.
            handler: Function(**kwargs) -> str result.
            schema: OpenAI function schema (name, description, parameters).
        """
        self._handlers[name] = handler
        self._schemas[name] = schema

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool by name.

        Returns:
            String result from the tool handler.

        Raises:
            KeyError: If tool not registered.
        """
        if name not in self._handlers:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            result = self._handlers[name](**arguments)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_openai_tools(self, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Return OpenAI function-calling tool schemas.

        Args:
            names: Optional subset of tool names. If None, returns all.
        """
        tool_names = names or list(self._schemas.keys())
        return [
            {"type": "function", "function": self._schemas[name]}
            for name in tool_names
            if name in self._schemas
        ]

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._handlers.keys())


def get_default_registry() -> ToolRegistry:
    """Build the default tool registry with all available tools."""
    registry = ToolRegistry()

    from .search_tools import register_search_tools
    from .retrieval_tools import register_retrieval_tools
    from .cluster_tools import register_cluster_tools
    from .nlp_tools import register_nlp_tools

    register_search_tools(registry)
    register_retrieval_tools(registry)
    register_cluster_tools(registry)
    register_nlp_tools(registry)

    return registry
