# BetterJobSearch

Job market analysis toolkit with hybrid RAG search, deterministic NLP aspect extraction, UMAP/HDBSCAN clustering, a multi-agent system, and a React + FastAPI interface.

## Features

- **Hybrid Search** -- SBERT semantic + BM25 keyword search with tunable alpha blending
- **NLP Aspect Extraction** -- deterministic extraction of skills, tools, language requirements, remote policy, experience, education, and benefits using spaCy PhraseMatcher and regex. Optional LLM-based domain and culture classification
- **UMAP + HDBSCAN Clustering** -- non-linear 2D projections with density-based clustering, dynamic re-clustering by aspect or free-text concept
- **Multi-Agent System** -- coordinator with intent classification routing to search, analysis, and exploration workers via a ReAct loop with OpenAI function calling
- **Tiered Context Management** -- 4-tier context strategy (aspects only ~50 tokens/job up to full text ~800 tokens/job) for efficient LLM usage
- **FastAPI Backend** -- REST API for jobs, search, clusters, aspects, plus WebSocket agentic chat
- **React Frontend** -- Plotly.js scatter plot (WebGL), AG Grid job table, live chat panel with tool call visibility
- **Offline-First** -- all data stored locally as JSON/JSONL, FAISS index, and BM25 pickle. No external database required

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/karaoglusina/better-job-search.git
cd better-job-search

# Create virtual environment with a compatible Python version
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[all]"

# Download spaCy model (requires pip inside the venv)
python -m spacy download en_core_web_sm
```

### Using pip (alternative)

```bash
pip install -e ".[all]"
python -m spacy download en_core_web_sm
```

## Quick Start

### 1. Build the Search Index

```bash
# Build from included sample data
python -m src.pipeline build

# Build from your own data
python -m src.pipeline build --data path/to/jobs.json

# Build with keyword extraction (slower, enables keyword tools)
python -m src.pipeline build --keywords

# Build with LLM-based domain/culture classification
python -m src.pipeline build --llm-aspects
```

This creates `artifacts/faiss.index`, `artifacts/bm25.pkl`, `artifacts/chunks.jsonl`, and optionally `artifacts/aspects.jsonl`.

### 2. Start the API + React UI

The application consists of two separate servers that need to run simultaneously:

**Terminal 1: FastAPI Backend (Port 8000)**

```bash
# Start the FastAPI backend server
python -m src.pipeline serve
```

This starts the backend API server on `http://localhost:8000` that provides:

- REST endpoints for jobs, search, clusters, and aspects (`/api/jobs`, `/api/search`, etc.)
- WebSocket endpoint for agentic chat (`/api/chat`)
- Health check endpoint (`/api/health`)

**Terminal 2: React Frontend (Port 5173)**

```bash
# Navigate to the frontend directory
cd frontend

# Install Node.js dependencies (only needed once, or when package.json changes)
npm install

# Start the Vite development server
npm run dev
```

This starts the React development server on `http://localhost:5173` that:

- Serves the React UI built with Vite
- Automatically reloads when you make code changes
- Connects to the FastAPI backend via API calls

**Access the Application:**

Open http://localhost:5173 in your browser. The React frontend will automatically communicate with the FastAPI backend running on port 8000.

The frontend makes HTTP requests to the backend API, and the backend serves the data and handles search/clustering operations.

### 3. Search from the CLI

```bash
python -m src.pipeline search "machine learning engineer"
python -m src.pipeline search "remote python" --k 15 --alpha 0.7
```

### 4. Agentic Chat (requires OpenAI API key)

```bash
export OPENAI_API_KEY=sk-...
python -m src.pipeline chat
```

The agent classifies your intent and routes to specialized workers:

- **Search**: "Find Python jobs in Amsterdam"
- **Compare**: "Compare these 3 jobs"
- **Explore**: "What kinds of AI jobs exist?"
- **Detail**: "Tell me about this job"

## Data Format

The system expects job data as a JSON file. Each job document:

```json
{
  "job_data": {
    "title": "Senior Data Engineer",
    "companyName": "TechCorp",
    "description": "Full job description HTML or text...",
    "location": "Amsterdam, Netherlands",
    "jobUrl": "https://linkedin.com/jobs/view/...",
    "contractType": "Full-time",
    "workType": "Remote"
  },
  "meta": {
    "days_old": 3
  }
}
```

The JSON file can be a list `[{job1}, {job2}]` or a dict `{"jobs": [{job1}, {job2}]}`.

## Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable         | Required         | Purpose                                                   |
| ---------------- | ---------------- | --------------------------------------------------------- |
| `OPENAI_API_KEY` | For LLM features | Agentic chat, domain/culture extraction, cluster labeling |
| `GOOGLE_API_KEY` | No               | Company logo search fallback                              |
| `GOOGLE_CSE_ID`  | No               | Custom search engine ID for logos                         |

## Architecture

### NLP Pipeline (`src/nlp/`)

Deterministic aspect extraction runs at index time:

| Aspect        | Method              | Example Output                            |
| ------------- | ------------------- | ----------------------------------------- |
| Skills        | spaCy PhraseMatcher | `["Python", "SQL", "Power BI"]`           |
| Tools         | spaCy PhraseMatcher | `["Docker", "Kubernetes", "AWS"]`         |
| Language      | Regex patterns      | `["Dutch required", "English fluent"]`    |
| Remote Policy | Regex + priority    | `"hybrid"`                                |
| Experience    | Regex + NER         | `"3-5 years"`, `"senior"`                 |
| Education     | Regex patterns      | `["BSc Computer Science"]`                |
| Benefits      | Keyword matching    | `["pension", "equity", "flexible hours"]` |
| Domain        | LLM (optional)      | `"FinTech"`                               |
| Culture       | LLM (optional)      | `["innovative", "fast-paced"]`            |

Entity normalization with rapidfuzz ensures `"JS"` / `"Javascript"` / `"JavaScript"` map to the same canonical form.

### Search (`src/search/`)

- **FAISS** (`IndexFlatIP`) for dense vector retrieval using SBERT embeddings
- **BM25** for sparse keyword retrieval
- **Hybrid** blending with configurable alpha (0.55 default)
- **Cross-encoder reranking** (optional, `ms-marco-MiniLM-L-6-v2`)

### Clustering (`src/clustering_v2/`)

- **UMAP** for 2D projection (non-linear, preserves topology)
- **HDBSCAN** for density-based clustering (auto-k, noise detection)
- **c-TF-IDF** for cluster label extraction
- **Aspect-based re-clustering**: select an aspect and the scatter plot re-clusters using binary feature matrices
- **Concept-based clustering**: type any concept (e.g., "customer-facing") and jobs are re-projected weighted by concept similarity
- Results cached to parquet files

### Agent System (`src/agents/`)

```
User Query
    |
Coordinator (intent classification)
    |
    +-> SearchWorker      (hybrid_search, semantic_search, keyword_search, filter_jobs)
    +-> AnalysisWorker    (get_job_summary, get_chunks, compare_aspects, find_similar)
    +-> ExplorationWorker (cluster_by_aspect, browse_cluster, aspect_distribution)
    |
Result synthesis
```

All agents use a shared ReAct loop (`src/agents/loop.py`) with OpenAI function calling. Context is managed in 4 tiers to stay within token budgets.

### API (`src/api/`)

| Method | Path                               | Description                             |
| ------ | ---------------------------------- | --------------------------------------- |
| `GET`  | `/api/jobs`                        | List jobs with filtering and pagination |
| `GET`  | `/api/jobs/{id}`                   | Job detail with chunks                  |
| `POST` | `/api/search`                      | Hybrid search with filters              |
| `GET`  | `/api/clusters/{aspect}`           | UMAP + HDBSCAN clusters for an aspect   |
| `POST` | `/api/clusters/concept`            | Cluster by free-text concept            |
| `GET`  | `/api/aspects`                     | List available aspects                  |
| `GET`  | `/api/aspects/{name}/distribution` | Value distribution for an aspect        |
| `WS`   | `/api/chat`                        | WebSocket agentic chat                  |
| `GET`  | `/api/health`                      | Health check                            |

### Frontend (`frontend/`)

React 18 + TypeScript + Vite. Key components:

- **ScatterPlot** -- Plotly.js WebGL scatter with cluster coloring and job highlighting
- **JobTable** -- AG Grid with sortable columns, pagination, row selection
- **ChatPanel** -- WebSocket chat with streaming responses and tool call display
- **AspectSelector** -- Pill buttons for predefined aspects + free-text concept input
- **FilterPanel** -- Location, company, title text filters
- **JobDetail** -- Expandable panel with full job description

## Project Structure

```
better-job-search/
├── src/
│   ├── pipeline.py                 # CLI entry point (build, serve, search, chat)
│   ├── config.py                   # Paths, API keys, feature flags
│   ├── rag.py                      # Core RAG engine (chunking, indexing, retrieval)
│   ├── models/                     # Pydantic data models
│   │   ├── aspect.py               #   AspectExtraction, DomainClassification
│   │   ├── chunk.py                #   Chunk, ChunkWithAspects
│   │   ├── job.py                  #   Job, JobSummary, JobDetail
│   │   ├── cluster.py              #   ClusterResult, ClusterInfo
│   │   └── agent.py                #   AgentResult, ToolCall, IntentClassification
│   ├── nlp/                        # Deterministic NLP pipeline
│   │   ├── aspect_extractor.py     #   Orchestrates all extractors
│   │   ├── cleaner.py              #   HTML cleaning, text normalization
│   │   ├── section_detector.py     #   Section boundary detection
│   │   ├── chunker.py              #   Aspect-aware chunking
│   │   ├── keyword_extractor.py    #   KeyBERT + TF-IDF
│   │   ├── entity_normalizer.py    #   rapidfuzz deduplication
│   │   ├── extractors/             #   Per-aspect extractors
│   │   │   ├── skills.py           #     spaCy PhraseMatcher
│   │   │   ├── language.py         #     Regex patterns
│   │   │   ├── remote_policy.py    #     Regex + priority
│   │   │   ├── experience.py       #     Regex + NER
│   │   │   ├── education.py        #     Regex patterns
│   │   │   ├── benefits.py         #     Keyword matching
│   │   │   ├── domain.py           #     LLM classification
│   │   │   └── culture.py          #     LLM extraction
│   │   └── vocab/                  #   Vocabulary files
│   │       ├── skills.txt
│   │       ├── tools.txt
│   │       ├── languages.txt
│   │       └── degrees.txt
│   ├── search/                     # Hybrid search system
│   │   ├── embedder.py             #   SBERT embedding (M1/M2 compatible)
│   │   ├── indexer.py              #   FAISS + BM25 index building
│   │   ├── retriever.py            #   HybridRetriever class
│   │   └── reranker.py             #   Cross-encoder reranking
│   ├── clustering_v2/              # UMAP + HDBSCAN clustering
│   │   ├── projector.py            #   UMAP 2D projection
│   │   ├── clusterer.py            #   HDBSCAN clustering
│   │   ├── labeler.py              #   c-TF-IDF + LLM labels
│   │   ├── aspect_clustering.py    #   Per-aspect re-clustering
│   │   └── cache.py                #   Parquet caching
│   ├── agents/                     # Multi-agent system
│   │   ├── coordinator.py          #   Intent classification + routing
│   │   ├── loop.py                 #   Reusable ReAct loop
│   │   ├── context.py              #   4-tier context management
│   │   ├── memory.py               #   Sliding window + summary memory
│   │   ├── workers/
│   │   │   ├── search_worker.py
│   │   │   ├── analysis_worker.py
│   │   │   └── exploration_worker.py
│   │   └── tools/
│   │       ├── registry.py         #   Tool name -> handler mapping
│   │       ├── search_tools.py
│   │       ├── retrieval_tools.py
│   │       ├── cluster_tools.py
│   │       └── nlp_tools.py
│   └── api/                        # FastAPI backend
│       ├── main.py                 #   App factory, CORS, lifespan
│       └── routes/
│           ├── jobs.py
│           ├── search.py
│           ├── clusters.py
│           ├── aspects.py
│           └── chat.py             #   WebSocket endpoint
├── frontend/                       # React + Vite + TypeScript
│   ├── src/
│   │   ├── App.tsx                 #   Main layout
│   │   ├── api/client.ts           #   Typed API client
│   │   ├── components/
│   │   │   ├── ScatterPlot.tsx
│   │   │   ├── JobTable.tsx
│   │   │   ├── ChatPanel.tsx
│   │   │   ├── AspectSelector.tsx
│   │   │   ├── JobDetail.tsx
│   │   │   └── FilterPanel.tsx
│   │   └── hooks/
│   │       ├── useWebSocket.ts
│   │       ├── useJobs.ts
│   │       └── useClusters.ts
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
├── config/
│   ├── facets.yml                  # 14-facet taxonomy
│   ├── facet_synonyms.yml
│   └── scoring.yml
├── data/
│   └── sample_jobs.json            # Sample dataset
├── artifacts/                      # Generated at build time (gitignored)
├── meta/
│   ├── docs/                       # Reference documentation
│   └── legacy/                     # Archived v1 code
├── pyproject.toml
├── .env.example
└── README.md
```

## CLI Reference

```
python -m src.pipeline <command> [options]

Commands:
  build    Build RAG index from job data
  serve    Start FastAPI backend (for React UI)
  search   Search for jobs from the command line
  chat     Interactive agentic chat (requires OpenAI API key)

build options:
  --data, -d PATH     Path to JSON file (default: data/sample_jobs.json)
  --no-aspects        Skip NLP aspect extraction
  --keywords          Run TF-IDF keyword extraction
  --llm-aspects       Enable LLM-based domain/culture extraction

serve options:
  --port, -p PORT     Port number (default: 8000)
  --reload            Enable auto-reload for development

search options:
  QUERY               Search query string
  --k, -k N           Number of results (default: 8)
  --alpha, -a FLOAT   Hybrid weight, 1.0=vectors 0.0=BM25 (default: 0.55)

chat options:
  --model, -m MODEL   OpenAI model (default: gpt-4o-mini)
```

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Lint
ruff check src/

# Tests
pytest

# Start API with auto-reload
python -m src.pipeline serve --reload
```

## License

MIT