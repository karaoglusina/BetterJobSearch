"""WebSocket chat endpoint for agentic conversation."""

from __future__ import annotations

import json
from typing import Any, Dict

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

            # Initialize coordinator lazily
            if coordinator is None:
                from ...agents.coordinator import Coordinator
                coordinator = Coordinator()

            # Define callbacks for streaming events
            async def on_tool_call(tc):
                await websocket.send_json({
                    "type": "tool_call",
                    "tool": tc.tool_name,
                    "args": tc.arguments,
                    "result": str(tc.result)[:500] if tc.result else "",
                    "error": tc.error,
                })

            async def on_intent(intent):
                await websocket.send_json({
                    "type": "intent",
                    "intent": intent.intent,
                    "confidence": intent.confidence,
                })

            # Run coordinator (sync — runs in thread pool)
            import asyncio

            def _run():
                return coordinator.handle(
                    content,
                    on_tool_call=lambda tc: asyncio.run(on_tool_call(tc)) if False else None,
                )

            result = await asyncio.get_event_loop().run_in_executor(None, _run)

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

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
