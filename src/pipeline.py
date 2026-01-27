"""
BetterJobSearch Pipeline v2 - Main Entry Point

This module provides the primary interface for:
- Loading job data from JSON files
- Building RAG indexes with NLP aspect extraction
- Running the FastAPI backend + React UI
- Interactive agentic chat

Example usage:
    # Build search index with NLP pipeline
    python -m src.pipeline build --data data/sample_jobs.json

    # Start the FastAPI backend
    python -m src.pipeline serve --port 8000

    # Interactive search
    python -m src.pipeline search "machine learning engineer"

    # Agentic chat (requires OpenAI API key)
    python -m src.pipeline chat
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DATA_DIR, DEFAULT_SAMPLE_DATA


def load_jobs(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load job data from a JSON file.

    Args:
        path: Path to JSON file. If None, uses default sample data.

    Returns:
        List of job documents

    The JSON can be either:
    - A list of job documents directly
    - A dict with a 'jobs' key containing the list
    """
    json_path = Path(path) if path else DEFAULT_SAMPLE_DATA

    if not json_path.exists():
        raise FileNotFoundError(f"Job data file not found: {json_path}")

    print(f"Loading jobs from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both list and dict with 'jobs' key
    if isinstance(data, list):
        jobs = data
    elif isinstance(data, dict) and "jobs" in data:
        jobs = data["jobs"]
    else:
        raise ValueError("JSON must be a list of jobs or a dict with 'jobs' key")

    print(f"Loaded {len(jobs)} jobs")
    return jobs


def build_index(
    jobs: Optional[List[Dict[str, Any]]] = None,
    data_path: Optional[str] = None,
    *,
    extract_aspects: bool = True,
    extract_keywords: bool = False,
    enable_llm_aspects: bool = False,
) -> None:
    """Build RAG indexes (FAISS + BM25) from job data with NLP pipeline.

    Args:
        jobs: List of job documents. If None, loads from data_path.
        data_path: Path to JSON file to load jobs from.
        extract_aspects: Run deterministic aspect extraction on chunks.
        extract_keywords: Run keyword extraction (slower).
        enable_llm_aspects: Run LLM-based aspect extraction (domain, culture).

    Creates:
        - artifacts/chunks.jsonl
        - artifacts/faiss.index
        - artifacts/bm25.pkl
    """
    from . import rag

    if jobs is None:
        jobs = load_jobs(data_path)

    # Run legacy build first (chunking + embedding + indexing)
    rag.build_index(docs=jobs)

    # Optionally run NLP aspect extraction
    if extract_aspects:
        print("\nRunning NLP aspect extraction...")
        _run_aspect_extraction(jobs, enable_llm=enable_llm_aspects)

    if extract_keywords:
        print("\nExtracting keywords...")
        _run_keyword_extraction()


def _run_aspect_extraction(jobs: List[Dict[str, Any]], *, enable_llm: bool = False) -> None:
    """Run aspect extraction pipeline on all jobs."""
    try:
        from .nlp.aspect_extractor import AspectExtractor
        from .nlp.cleaner import clean_html
    except ImportError as e:
        print(f"Skipping aspect extraction (missing dependency: {e})")
        return

    extractor = AspectExtractor(enable_llm=enable_llm)
    results: List[Dict[str, Any]] = []

    for i, doc in enumerate(jobs):
        job_data = doc.get("job_data", doc)
        description = job_data.get("description", "")
        if not description:
            continue

        cleaned = clean_html(description)
        title = job_data.get("title", "")
        company = job_data.get("companyName", "")

        aspects = extractor.extract_all_as_dict(cleaned, title=title, company=company)

        job_key = job_data.get("jobUrl") or f"{company}|{title}"
        results.append({
            "job_id": job_key,
            "title": title,
            "company": company,
            "aspects": aspects,
        })

        if (i + 1) % 50 == 0:
            print(f"  Extracted aspects for {i + 1}/{len(jobs)} jobs")

    # Save aspects to artifacts
    from .config import ARTIFACTS_DIR
    aspects_path = ARTIFACTS_DIR / "aspects.jsonl"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(aspects_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Saved aspects for {len(results)} jobs to {aspects_path}")


def _run_keyword_extraction() -> None:
    """Run keyword extraction on all job texts."""
    try:
        from .nlp.keyword_extractor import extract_keywords_tfidf
    except ImportError as e:
        print(f"Skipping keyword extraction (missing dependency: {e})")
        return

    from . import rag

    if not rag.is_loaded():
        rag.load_cache()
    chunks = rag.get_all_chunks()

    # Group chunks by job
    by_job: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        jk = getattr(c, "job_key", None)
        if not jk:
            continue
        if jk not in by_job:
            meta = getattr(c, "meta", {}) or {}
            by_job[jk] = {"job_id": str(jk), "texts": [], "title": meta.get("title", "")}
        by_job[jk]["texts"].append(getattr(c, "text", "") or "")

    jobs = [{"job_id": v["job_id"], "text": "\n\n".join(v["texts"])} for v in by_job.values()]
    texts = [j["text"] for j in jobs]

    if not texts:
        print("  No job texts found.")
        return

    keywords_per_doc = extract_keywords_tfidf(texts, top_n=15)

    from .config import ARTIFACTS_DIR
    kw_path = ARTIFACTS_DIR / "keywords.jsonl"
    with open(kw_path, "w", encoding="utf-8") as f:
        for i, (job, kws) in enumerate(zip(jobs, keywords_per_doc)):
            f.write(json.dumps({
                "job_id": job.get("job_id", ""),
                "keywords": [kw for kw, _ in kws],
            }, ensure_ascii=False) + "\n")

    print(f"  Saved keywords for {len(jobs)} jobs to {kw_path}")


def search(
    query: str,
    k: int = 8,
    alpha: float = 0.55,
) -> None:
    """Search for jobs matching a query.

    Args:
        query: Search query string
        k: Number of results to return
        alpha: Hybrid search weight (1.0 = vectors only, 0.0 = BM25 only)
    """
    from . import rag

    if not rag.is_loaded():
        print("Loading search index...")
        rag.load_cache()

    print(f"\nSearching for: {query}\n")
    chunks = rag.retrieve(query, k=k, alpha=alpha)

    # Group by job
    seen_jobs = set()
    for i, chunk in enumerate(chunks, 1):
        job_key = chunk.job_key
        if job_key in seen_jobs:
            continue
        seen_jobs.add(job_key)

        meta = chunk.meta
        print(f"[{i}] {meta.get('title', 'Unknown')} @ {meta.get('company', 'Unknown')}")
        print(f"    Location: {meta.get('location', 'N/A')}")
        print(f"    URL: {meta.get('jobUrl', 'N/A')}")
        print(f"    Snippet: {chunk.text[:150]}...")
        print()


def serve(port: int = 8000, reload: bool = False) -> None:
    """Start the FastAPI backend server.

    Args:
        port: Port to run on
        reload: Enable auto-reload for development
    """
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install 'better-job-search[api]'")
        sys.exit(1)

    print(f"Starting BetterJobSearch API on http://localhost:{port}")
    print(f"  React UI: http://localhost:5173 (run 'npm run dev' in frontend/)")
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
    )


def chat(model: str = "gpt-4o-mini") -> None:
    """Start an interactive agentic chat session.

    Uses the multi-agent system with tool-based context expansion.
    Falls back to simple RAG chat if OpenAI is not available.

    Args:
        model: OpenAI model to use
    """
    import os

    if os.environ.get("OPENAI_API_KEY"):
        _agentic_chat(model)
    else:
        # Fallback to legacy chat
        from . import rag
        rag.chat_loop(model=model)


def _agentic_chat(model: str = "gpt-4o-mini") -> None:
    """Interactive agentic chat using the coordinator."""
    from .agents.coordinator import Coordinator

    coordinator = Coordinator(model=model)
    print("\nBetterJobSearch Agentic Chat")
    print("Type your question (or 'exit'):\n")

    while True:
        try:
            q = input("\u00bb ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not q or q.lower() in {"exit", "quit"}:
            break

        if q.lower() == "/reset":
            coordinator.reset_memory()
            print("Memory reset.\n")
            continue

        def on_tool(tc):
            print(f"  [tool] {tc.tool_name}({json.dumps(tc.arguments)[:80]})")

        result = coordinator.handle(q, on_tool_call=on_tool)

        print(f"\n--- Answer ({result.model}, {result.total_tokens} tokens) ---\n")
        print(result.answer)
        if result.tool_calls:
            print(f"\n  [{len(result.tool_calls)} tool calls used]")
        print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BetterJobSearch v2 - Job Market Analysis Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build search index from sample data
  python -m src.pipeline build

  # Build with NLP aspects + keywords
  python -m src.pipeline build --data my_jobs.json --keywords

  # Start FastAPI backend
  python -m src.pipeline serve

  # Search for jobs
  python -m src.pipeline search "machine learning engineer"

  # Agentic chat (requires OpenAI API key)
  python -m src.pipeline chat
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build RAG index from job data")
    build_parser.add_argument(
        "--data", "-d", type=str, default=None,
        help="Path to JSON file with job data (default: data/sample_jobs.json)",
    )
    build_parser.add_argument(
        "--no-aspects", action="store_true",
        help="Skip NLP aspect extraction",
    )
    build_parser.add_argument(
        "--keywords", action="store_true",
        help="Run TF-IDF keyword extraction (slower)",
    )
    build_parser.add_argument(
        "--llm-aspects", action="store_true",
        help="Enable LLM-based aspect extraction (domain, culture)",
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for jobs")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--k", "-k", type=int, default=8, help="Number of results (default: 8)")
    search_parser.add_argument("--alpha", "-a", type=float, default=0.55, help="Hybrid search weight (default: 0.55)")

    # FastAPI serve command
    serve_parser = subparsers.add_parser("serve", help="Start FastAPI backend (for React UI)")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive agentic chat")
    chat_parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "build":
            build_index(
                data_path=args.data,
                extract_aspects=not args.no_aspects,
                extract_keywords=args.keywords,
                enable_llm_aspects=args.llm_aspects,
            )
        elif args.command == "search":
            search(args.query, k=args.k, alpha=args.alpha)
        elif args.command == "serve":
            serve(port=args.port, reload=args.reload)
        elif args.command == "chat":
            chat(model=args.model)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
