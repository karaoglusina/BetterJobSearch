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

    # Pre-load embedding model so first requests don't trigger concurrent loads
    if app.state.retriever is not None:
        try:
            from ..search.embedder import get_model
            print("Pre-loading embedding model...")
            get_model()
            print("Embedding model ready.")
        except Exception as e:
            print(f"Warning: Could not pre-load embedding model: {e}")

    # Load aspect data for aspect-based clustering
    aspects_path = ARTIFACTS_DIR / "aspects.jsonl"
    if aspects_path.exists():
        import json
        aspect_data = {}
        with open(aspects_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                aspect_data[rec["job_id"]] = rec.get("aspects", {})
        app.state.aspect_data = aspect_data
        print(f"Loaded aspect data for {len(aspect_data)} jobs")
    else:
        app.state.aspect_data = {}

    # Load meaningful phrases for cluster labeling (if available)
    phrases_path = ARTIFACTS_DIR / "meaningful_phrases.json"
    if phrases_path.exists():
        import json
        with open(phrases_path, "r", encoding="utf-8") as f:
            phrases_data = json.load(f)
        app.state.meaningful_phrases = phrases_data.get("phrases", [])
        print(f"Loaded {len(app.state.meaningful_phrases)} meaningful phrases for cluster labeling")
    else:
        app.state.meaningful_phrases = []

    # Load phrase scores for NPMI/effect_size filtering (if available)
    phrase_scores_path = ARTIFACTS_DIR / "phrase_scores.parquet"
    if phrase_scores_path.exists():
        try:
            import pandas as pd
            phrase_scores_df = pd.read_parquet(phrase_scores_path)
            # Convert to dict for fast lookup: ngram -> {npmi, effect_size}
            app.state.phrase_scores = {
                row['ngram']: {'npmi': row['npmi'], 'effect_size': row['effect_size']}
                for _, row in phrase_scores_df.iterrows()
            }
            print(f"Loaded phrase scores for {len(app.state.phrase_scores)} phrases (NPMI/effect_size filtering enabled)")
        except Exception as e:
            print(f"Warning: Could not load phrase scores: {e}")
            app.state.phrase_scores = {}
    else:
        app.state.phrase_scores = {}

    # Summarize languages from chunk metadata (language is now stored during build)
    if app.state.retriever is not None:
        seen_jobs: set = set()
        lang_counts: dict[str, int] = {}
        for ch in app.state.retriever.chunks:
            jk = ch.get("job_key", "")
            if jk in seen_jobs:
                continue
            seen_jobs.add(jk)
            lang = ch.get("meta", {}).get("language", "")
            if lang:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
        if lang_counts:
            print(f"Languages in corpus ({len(seen_jobs)} jobs): {dict(sorted(lang_counts.items(), key=lambda x: -x[1])[:10])}")

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
    from .routes import jobs, search, clusters, aspects, chat, labels
    app.include_router(jobs.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(clusters.router, prefix="/api")
    app.include_router(aspects.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(labels.router, prefix="/api")

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "BetterJobSearch API",
            "version": "2.0.0",
            "status": "running",
            "docs": "/docs",
            "health": "/api/health",
            "endpoints": {
                "jobs": "/api/jobs",
                "search": "/api/search",
                "clusters": "/api/clusters",
                "aspects": "/api/aspects",
                "chat": "/api/chat (WebSocket)",
            },
        }

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
