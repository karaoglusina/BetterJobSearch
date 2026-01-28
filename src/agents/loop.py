"""Reusable ReAct (Reason + Act) loop for all agents."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

# Ensure .env is loaded by importing config
from ..config import OPENAI_API_KEY as CONFIG_OPENAI_API_KEY

from ..models.agent import AgentResult, ToolCall


def react_loop(
    system_prompt: str,
    user_message: str,
    tools: List[Dict[str, Any]],
    tool_executor: Callable[[str, Dict[str, Any]], str],
    *,
    messages_prefix: Optional[List[Dict[str, Any]]] = None,
    max_iterations: int = 8,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    on_tool_call: Optional[Callable[[ToolCall], None]] = None,
    on_response: Optional[Callable[[str], None]] = None,
) -> AgentResult:
    """Execute a ReAct loop: the LLM reasons about what to do, calls tools,
    then synthesizes a final answer.

    Args:
        system_prompt: System message defining agent behavior.
        user_message: User's query.
        tools: OpenAI function-calling tool schemas.
        tool_executor: Function(tool_name, arguments) -> result string.
        messages_prefix: Optional conversation history to prepend.
        max_iterations: Maximum tool-calling rounds.
        model: OpenAI model name.
        temperature: Sampling temperature.
        on_tool_call: Optional callback for each tool call (for streaming/UI).
        on_response: Optional callback for final text response.

    Returns:
        AgentResult with answer, tool call trace, and metadata.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return AgentResult(
            answer="OpenAI library not available. Install with: pip install openai",
            model=model,
        )

    # Check both os.environ and config (config loads from .env file)
    api_key = os.environ.get("OPENAI_API_KEY") or CONFIG_OPENAI_API_KEY
    if not api_key:
        return AgentResult(
            answer="OPENAI_API_KEY not set. Cannot run agent. Please set it in .env file or as environment variable.",
            model=model,
        )
    
    # Ensure OpenAI client can access the key
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key

    client = OpenAI()

    messages: List[Dict[str, Any]] = []
    if messages_prefix:
        messages.extend(messages_prefix)
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    all_tool_calls: List[ToolCall] = []
    total_tokens = 0

    for iteration in range(max_iterations):
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            return AgentResult(
                answer=f"LLM call failed: {e}",
                tool_calls=all_tool_calls,
                model=model,
                total_tokens=total_tokens,
            )

        msg = response.choices[0].message
        total_tokens += response.usage.total_tokens if response.usage else 0

        # No tool calls -> final answer
        if not msg.tool_calls:
            answer = msg.content or ""
            if on_response:
                on_response(answer)
            return AgentResult(
                answer=answer,
                tool_calls=all_tool_calls,
                model=model,
                total_tokens=total_tokens,
            )

        # Process tool calls
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            # Execute tool
            try:
                result = tool_executor(tool_name, arguments)
                error = None
            except Exception as e:
                result = f"Tool error: {e}"
                error = str(e)

            tool_call = ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                error=error,
            )
            all_tool_calls.append(tool_call)

            if on_tool_call:
                on_tool_call(tool_call)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)[:4000],  # Truncate large results
            })

    # Max iterations reached
    return AgentResult(
        answer="I've reached the maximum number of reasoning steps. Here's what I found so far based on the tool calls above.",
        tool_calls=all_tool_calls,
        model=model,
        total_tokens=total_tokens,
    )
