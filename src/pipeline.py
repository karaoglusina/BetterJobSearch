"""
BetterJobSearch Pipeline - Main Entry Point

This module provides the primary interface for:
- Loading job data from JSON files
- Building RAG indexes for semantic search
- Running the interactive web UI

Example usage:
    # Build index from sample data
    python -m src.pipeline build --data data/sample_jobs.json
    
    # Start the web UI
    python -m src.pipeline ui --port 8050
    
    # Interactive search
    python -m src.pipeline search "machine learning engineer"
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
) -> None:
    """Build RAG indexes (FAISS + BM25) from job data.
    
    Args:
        jobs: List of job documents. If None, loads from data_path.
        data_path: Path to JSON file to load jobs from.
    
    Creates:
        - artifacts/chunks.jsonl
        - artifacts/faiss.index  
        - artifacts/bm25.pkl
    """
    from . import rag
    
    if jobs is None:
        jobs = load_jobs(data_path)
    
    rag.build_index(docs=jobs)


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


def run_ui(port: int = 8050, debug: bool = False) -> None:
    """Start the interactive web UI.
    
    Args:
        port: Port to run the server on
        debug: Enable Dash debug mode
    """
    from .ui.app import create_app
    
    print(f"Starting BetterJobSearch UI on http://localhost:{port}")
    app = create_app()
    app.run(debug=debug, host="0.0.0.0", port=port)


def chat(model: str = "gpt-4o-mini") -> None:
    """Start an interactive chat session for job Q&A.
    
    Requires OPENAI_API_KEY to be set.
    
    Args:
        model: OpenAI model to use
    """
    from . import rag
    
    rag.chat_loop(model=model)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BetterJobSearch - Job Market Analysis Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build search index from sample data
  python -m src.pipeline build
  
  # Build from custom data file
  python -m src.pipeline build --data my_jobs.json
  
  # Start the web UI
  python -m src.pipeline ui
  
  # Search for jobs
  python -m src.pipeline search "machine learning engineer"
  
  # Interactive Q&A (requires OpenAI API key)
  python -m src.pipeline chat
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build RAG index from job data")
    build_parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Path to JSON file with job data (default: data/sample_jobs.json)",
    )
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for jobs")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--k", "-k",
        type=int,
        default=8,
        help="Number of results (default: 8)",
    )
    search_parser.add_argument(
        "--alpha", "-a",
        type=float,
        default=0.55,
        help="Hybrid search weight (default: 0.55)",
    )
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Start the web UI")
    ui_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8050,
        help="Port to run on (default: 8050)",
    )
    ui_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive Q&A chat")
    chat_parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    try:
        if args.command == "build":
            build_index(data_path=args.data)
        elif args.command == "search":
            search(args.query, k=args.k, alpha=args.alpha)
        elif args.command == "ui":
            run_ui(port=args.port, debug=args.debug)
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
