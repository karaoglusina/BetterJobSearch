"""
Configuration settings for BetterJobSearch.

All API keys are OPTIONAL - the core functionality works without them.
LLM features require OPENAI_API_KEY to be set.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env file in project root (parent of src/)
_project_root = Path(__file__).parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)

# ===================
# API Keys (All Optional)
# ===================

# OpenAI API key - required only for LLM features
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Google Custom Search - optional, used for logo fetching fallback
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# ===================
# Project Paths
# ===================

PROJECT_ROOT = _project_root
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_DIR = PROJECT_ROOT / "config"

# ===================
# Default Settings
# ===================

DEFAULT_SAMPLE_DATA = DATA_DIR / "sample_jobs.json"
DEFAULT_OFFLINE_STORE = DATA_DIR / "offline_store.json"
DEFAULT_OFFLINE_CHANGES = DATA_DIR / "offline_changes.jsonl"

# Default Search Parameters (for reference)
DEFAULT_SEARCH_PARAMS = {
    "location": "Netherlands",
    "date_posted": "week",
    "limit": 10,
    "sort": "relevant"
}

# ===================
# Feature Flags
# ===================

def has_openai() -> bool:
    """Check if OpenAI API key is configured."""
    return bool(OPENAI_API_KEY)

def has_google_search() -> bool:
    """Check if Google Custom Search is configured."""
    return bool(GOOGLE_API_KEY and GOOGLE_CSE_ID)
