"""FastAPI application entry point."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import ARTIFACTS_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle events."""
    # Startup: load search artifacts if they exist
    chunks_path = ARTIFACTS_DIR / "chunks.jsonl"
    if chunks_path.exists():
        try:
            from ..search.retriever import HybridRetriever
            app.state.retriever = HybridRetriever.from_artifacts()
            print(f"Loaded search artifacts: {len(app.state.retriever.chunks)} chunks")
        except Exception as e:
            print(f"Warning: Could not load search artifacts: {e}")
            app.state.retriever = None
    else:
        print("No search artifacts found. Run 'python -m src.pipeline build' first.")
        app.state.retriever = None

    # Initialize coordinator (lazy - only if OpenAI available)
    app.state.coordinator = None

    yield

    # Shutdown
    print("Shutting down BetterJobSearch API.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="BetterJobSearch API",
        description="Job market analysis with NLP, clustering, and agentic search.",
        version="2.0.0",
        lifespan=lifespan,
    )

    # CORS - allow React dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",  # Vite dev server
            "http://localhost:3000",  # Alternative
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from .routes import jobs, search, clusters, aspects, chat
    app.include_router(jobs.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(clusters.router, prefix="/api")
    app.include_router(aspects.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "artifacts_loaded": app.state.retriever is not None,
            "n_chunks": len(app.state.retriever.chunks) if app.state.retriever else 0,
        }

    return app


# For uvicorn direct run
app = create_app()
