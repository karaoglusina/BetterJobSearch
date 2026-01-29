"""WebSocket chat endpoint for agentic conversation."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["chat"])


@router.websocket("/chat")
async def chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for agentic chat.

    Protocol:
        Client sends: {"type": "message", "content": "..."}
        Server sends:
            {"type": "intent", "intent": "search", "confidence": 0.9}
            {"type": "tool_call", "tool": "hybrid_search", "args": {...}}
            {"type": "tool_result", "tool": "hybrid_search", "result": "..."}
            {"type": "answer", "content": "...", "job_ids": [...]}
            {"type": "ui_action", "action": "highlight_jobs", "data": {...}}
            {"type": "error", "message": "..."}
    """
    await websocket.accept()

    # Lazy-init coordinator per connection
    coordinator = None

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            if msg.get("type") == "reset":
                if coordinator:
                    coordinator.reset_memory()
                await websocket.send_json({"type": "reset_ack"})
                continue

            if msg.get("type") != "message":
                continue

            content = msg.get("content", "").strip()
            if not content:
                continue

            # Extract context (selected jobs, filters, etc.)
            context = msg.get("context", {})
            selected_job_ids = context.get("selectedJobIds", [])

            # Initialize coordinator lazily
            if coordinator is None:
                try:
                    from ...agents.coordinator import Coordinator
                    coordinator = Coordinator()
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Failed to initialize chat agent: {e}",
                    })
                    continue

            # Run coordinator in thread pool to avoid blocking the event loop.
            # Tool call callbacks are collected and sent after completion since
            # the coordinator runs synchronously in a separate thread.
            loop = asyncio.get_event_loop()
            tool_call_log = []

            def on_tool_call(tc):
                tool_call_log.append({
                    "tool": tc.tool_name,
                    "args": tc.arguments,
                    "result": str(tc.result)[:500] if tc.result else "",
                    "error": tc.error,
                })

            def _run():
                return coordinator.handle(
                    content,
                    on_tool_call=on_tool_call,
                    selected_job_ids=selected_job_ids,
                )

            try:
                result = await loop.run_in_executor(None, _run)
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                continue

            # Extract job IDs from tool call results
            if not result.job_ids_referenced:
                extracted_ids: List[str] = []
                for tc in result.tool_calls:
                    if tc.result:
                        m = re.search(r"\[JOB_IDS:([^\]]+)\]", str(tc.result))
                        if m:
                            extracted_ids.extend(m.group(1).split(","))
                if extracted_ids:
                    result.job_ids_referenced = extracted_ids

            # Send tool call events that were collected during execution
            for tc_data in tool_call_log:
                await websocket.send_json({"type": "tool_call", **tc_data})

            # Send answer
            await websocket.send_json({
                "type": "answer",
                "content": result.answer,
                "job_ids": result.job_ids_referenced,
                "tool_calls": [
                    {"tool": tc.tool_name, "args": tc.arguments}
                    for tc in result.tool_calls
                ],
                "model": result.model,
                "total_tokens": result.total_tokens,
            })

            # Send UI actions
            for action in result.ui_actions:
                await websocket.send_json({
                    "type": "ui_action",
                    **action,
                })

            # Auto-emit set_jobs action when agent references specific jobs
            if result.job_ids_referenced:
                await websocket.send_json({
                    "type": "ui_action",
                    "action": "set_jobs",
                    "job_ids": result.job_ids_referenced,
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
