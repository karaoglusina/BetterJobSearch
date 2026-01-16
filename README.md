# BetterJobSearch

A job market analysis toolkit with RAG-powered semantic search, clustering visualization, and an interactive web UI.

## Features

- **Hybrid Search**: Combines semantic (SBERT) and keyword (BM25) search for accurate job discovery
- **Interactive Web UI**: 3-panel Dash application for exploring, filtering, and analyzing jobs
- **Clustering**: Visualize job market segments using TF-IDF/SBERT embeddings
- **LLM Integration**: Optional GPT-powered company summaries, job summaries, and intelligent job selection
- **Boolean Search**: Full support for AND/OR/NOT queries with phrase matching
- **Offline-First**: All data stored locally - no external database required

## Installation

### Using uv (Recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/karaoglusina/better-job-search.git
cd better-job-search

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
uv pip install -e .

# Install with UI support
uv pip install -e ".[ui]"

# Install with LLM features
uv pip install -e ".[llm]"

# Install everything
uv pip install -e ".[all]"
```

### Using pip

```bash
pip install -e ".[all]"
```

## Quick Start

### 1. Build the Search Index

```bash
# Using the included sample data
python -m src.pipeline build

# Or specify your own data file
python -m src.pipeline build --data path/to/your/jobs.json
```

### 2. Start the Web UI

```bash
python -m src.pipeline ui
```

Then open http://localhost:8050 in your browser.

### 3. Search from Command Line

```bash
python -m src.pipeline search "machine learning engineer"
```

## Data Format

The system expects job data in JSON format. Each job should have:

```json
{
  "job_data": {
    "title": "Senior Data Engineer",
    "companyName": "TechCorp",
    "description": "Full job description...",
    "location": "Amsterdam, Netherlands",
    "jobUrl": "https://linkedin.com/jobs/view/...",
    "applyType": "EasyApply",
    "contractType": "Full-time",
    "workType": "Remote"
  },
  "meta": {
    "days_old": 3,
    "applied_times": 42
  }
}
```

The JSON file can be either:
- A list of job objects: `[{job1}, {job2}, ...]`
- A dict with a "jobs" key: `{"jobs": [{job1}, {job2}, ...]}`

## Web UI Features

### Filters Panel
- Title, location, city dropdowns
- Boolean search with AND/OR/NOT and quoted phrases
- Semantic search for ranking by relevance
- Applied times and days old ranges
- Label filtering (yes/no/custom tags)

### Table View
- Sortable, filterable job grid
- Company logos (auto-fetched)
- LLM-generated summaries (requires OpenAI API key)
- Excel export

### Clusters View
- Domain and Role scatter plots
- Click to select jobs
- Adjust cluster count (k)

### Details Panel
- Full job description
- Company and job summaries
- Quick labeling (yes/no buttons)
- Custom tagging

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Required for LLM features
OPENAI_API_KEY=sk-your-key-here

# Optional: for logo search fallback
GOOGLE_API_KEY=your-google-api-key
GOOGLE_CSE_ID=your-custom-search-id
```

## Project Structure

```
better-job-search/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration and paths
│   ├── pipeline.py         # Main CLI entry point
│   ├── rag.py              # RAG pipeline (chunking, indexing, retrieval)
│   ├── retrieval.py        # Retrieval helpers
│   ├── clustering.py       # Job/chunk clustering
│   ├── agentic_rag.py      # Multi-hop LLM retrieval
│   ├── llm_io.py           # LLM utilities
│   └── ui/
│       ├── app.py          # Dash web application
│       └── assets/
│           └── styles.css
├── config/
│   ├── facets.yml          # Facet taxonomy
│   └── scoring.yml         # Scoring weights
├── data/
│   └── sample_jobs.json    # Sample dataset
├── artifacts/              # Generated search indexes (gitignored)
├── archive/                # Archived code for reference
├── pyproject.toml
├── .env.example
└── README.md
```

## API Reference

### Pipeline Functions

```python
from src.pipeline import load_jobs, build_index, search, run_ui

# Load job data
jobs = load_jobs("data/sample_jobs.json")

# Build search index
build_index(jobs=jobs)

# Search
search("data engineer python", k=10)

# Start UI
run_ui(port=8050)
```

### RAG Module

```python
from src import rag

# Load artifacts
rag.load_cache()

# Retrieve chunks
chunks = rag.retrieve("machine learning", k=8, alpha=0.55)

# Filtered retrieval
chunks = rag.retrieve_filtered(
    query="python",
    where=lambda c: c.meta.get("location") == "Amsterdam",
    k=10
)
```

### Clustering

```python
from src.clustering import cluster_chunks, cluster_jobs

# Cluster retrieved chunks
result = cluster_chunks(chunks, n_clusters=5)
print(result["keywords"])  # Top terms per cluster

# Cluster all jobs
result = cluster_jobs(embed_mode="sbert")
df = result["df"]  # DataFrame with x, y, cluster columns
```

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run linter
ruff check src/

# Run tests
pytest
```
