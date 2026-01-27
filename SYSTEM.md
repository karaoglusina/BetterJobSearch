# BetterJobSearch -- System Documentation

Detailed technical documentation of the BetterJobSearch system architecture, data flows, and component design.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Data Pipeline](#data-pipeline)
4. [Search System](#search-system)
5. [NLP Pipeline](#nlp-pipeline)
6. [Clustering System](#clustering-system)
7. [Multi-Agent System](#multi-agent-system)
8. [API Layer](#api-layer)
9. [Frontend](#frontend)
10. [Data Models](#data-models)
11. [Configuration](#configuration)
12. [Storage & Artifacts](#storage--artifacts)
13. [Fallback Strategies](#fallback-strategies)

---

## System Overview

BetterJobSearch is a job market analysis toolkit combining hybrid RAG search, deterministic NLP aspect extraction, UMAP/HDBSCAN clustering, a multi-agent system, and a React + FastAPI interface.

**Key design principles:**

- **Offline-first** -- all data stored locally as JSON/JSONL, FAISS index, and BM25 pickle; no external database required
- **Modular extras** -- core search works standalone; NLP, clustering, agents, and API are optional install groups
- **M1/M2 Mac safe** -- thread environment variables set before numpy imports, FAISS single-threaded, CPU-only embeddings
- **Token-efficient** -- 4-tier context strategy keeps LLM costs low while preserving answer quality

**Technology stack:**

| Layer | Technology |
|-------|-----------|
| Search | FAISS (dense), rank-bm25 (sparse), sentence-transformers (embedding) |
| NLP | spaCy PhraseMatcher, KeyBERT, rapidfuzz, regex |
| Clustering | UMAP, HDBSCAN, c-TF-IDF |
| Agents | OpenAI function calling, ReAct loop |
| Backend | FastAPI, Uvicorn, WebSockets |
| Frontend | React 18, TypeScript, Vite, Plotly.js (WebGL), AG Grid |
| Data | Pydantic, pandas, pyarrow (parquet), JSONL |

---

## Architecture Diagram

```
                          ┌──────────────────────────────────────┐
                          │           USER INTERFACES            │
                          ├──────────────────┬───────────────────┤
                          │  CLI             │  React UI         │
                          │  (pipeline.py)   │  (localhost:5173) │
                          └────────┬─────────┴─────────┬─────────┘
                                   │                   │
                                   v                   v
                          ┌──────────────────────────────────────┐
                          │       FastAPI  (localhost:8000)       │
                          ├──────────────────────────────────────┤
                          │  /api/jobs      GET    list + filter │
                          │  /api/jobs/{id} GET    detail        │
                          │  /api/search    POST   hybrid search │
                          │  /api/clusters  GET    UMAP scatter  │
                          │  /api/aspects   GET    aspect data   │
                          │  /api/chat      WS     agentic chat  │
                          │  /api/health    GET    status check  │
                          └──────┬─────────────────────┬─────────┘
                                 │                     │
               ┌─────────────────┤                     │
               │                 │                     │
               v                 v                     v
    ┌─────────────────┐  ┌──────────────┐   ┌──────────────────┐
    │   Retriever      │  │  Clustering  │   │  Agent System    │
    │                  │  │              │   │                  │
    │  FAISS + BM25    │  │  UMAP        │   │  Coordinator     │
    │  hybrid blend    │  │  HDBSCAN     │   │  SearchWorker    │
    │  cross-encoder   │  │  c-TF-IDF    │   │  AnalysisWorker  │
    │  reranking       │  │  cache layer │   │  ExplorationWkr  │
    └────────┬─────────┘  └──────┬───────┘   └────────┬─────────┘
             │                   │                     │
             └───────────────────┼─────────────────────┘
                                 │
                                 v
                    ┌────────────────────────┐
                    │   Disk Artifacts       │
                    │   chunks.jsonl         │
                    │   faiss.index          │
                    │   bm25.pkl             │
                    │   aspects.jsonl        │
                    │   cluster_*.parquet    │
                    └────────────────────────┘
```

---

## Data Pipeline

The `build` command (`python -m src.pipeline build`) processes raw job data into searchable artifacts.

### Steps

1. **Load** -- Read JSON file (list or `{"jobs": [...]}` format)
2. **Clean** -- Strip HTML via BeautifulSoup, normalize whitespace (`src/nlp/cleaner.py`)
3. **Section detection** -- Identify boundaries (Responsibilities, Requirements, Benefits, etc.) using header regex (`src/nlp/section_detector.py`)
4. **Chunking** -- Split into overlapping chunks respecting section and bullet boundaries (`src/nlp/chunker.py`)
   - Bullet lists: each bullet is a chunk
   - Prose: sentence-windowed chunks (80--200 tokens) with 150-char overlap
5. **Embedding** -- Encode chunks with `all-mpnet-base-v2` via sentence-transformers (`src/search/embedder.py`)
   - L2-normalized for cosine similarity via inner product
   - Batch size 32
6. **Indexing** -- Build FAISS `IndexFlatIP` and BM25 corpus (`src/search/indexer.py`)
7. **Aspect extraction** (default, skip with `--no-aspects`) -- Run deterministic extractors per job (`src/nlp/aspect_extractor.py`)
8. **Keyword extraction** (opt-in with `--keywords`) -- TF-IDF + KeyBERT (`src/nlp/keyword_extractor.py`)
9. **LLM aspects** (opt-in with `--llm-aspects`) -- Domain classification and culture extraction via OpenAI
10. **Save** -- Write `chunks.jsonl`, `faiss.index`, `bm25.pkl`, `aspects.jsonl`, `keywords.jsonl` to `artifacts/`

### Input format

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

---

## Search System

Located in `src/search/`.

### Components

| File | Role |
|------|------|
| `embedder.py` | SBERT embedding with M1/M2 thread safety |
| `indexer.py` | FAISS + BM25 index construction |
| `retriever.py` | `HybridRetriever` -- loads artifacts and runs hybrid search |
| `reranker.py` | Optional cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) |

### Hybrid search algorithm

```
score = alpha * vector_score + (1 - alpha) * bm25_score
```

- **alpha = 1.0** -- pure semantic (FAISS)
- **alpha = 0.0** -- pure keyword (BM25)
- **alpha = 0.55** -- default balanced blend

### Flow

1. Encode query with SBERT
2. FAISS inner-product search -> top-k candidates with vector scores
3. BM25 search -> top-k candidates with BM25 scores
4. Normalize both score sets to [0, 1]
5. Blend with alpha weighting
6. Deduplicate by `job_key` (keep highest-scoring chunk per job)
7. Optional cross-encoder reranking on top results

---

## NLP Pipeline

Located in `src/nlp/`. Runs deterministic aspect extraction at index time.

### Orchestrator

`aspect_extractor.py` runs all extractors in sequence and returns `Dict[str, List[str]]` per job.

### Extractors (`src/nlp/extractors/`)

| Extractor | File | Method | Output Example |
|-----------|------|--------|----------------|
| Skills | `skills.py` | spaCy PhraseMatcher against `vocab/skills.txt` | `["Python", "SQL", "Power BI"]` |
| Tools | (shared with skills) | spaCy PhraseMatcher against `vocab/tools.txt` | `["Docker", "Kubernetes", "AWS"]` |
| Language | `language.py` | Regex patterns | `["Dutch required", "English fluent"]` |
| Remote Policy | `remote_policy.py` | Regex + priority ranking | `"hybrid"` |
| Experience | `experience.py` | Regex + NER | `"3-5 years"`, `"senior"` |
| Education | `education.py` | Regex patterns against `vocab/degrees.txt` | `["BSc Computer Science"]` |
| Benefits | `benefits.py` | Keyword matching | `["pension", "equity", "flexible hours"]` |
| Domain | `domain.py` | LLM classification (optional) | `"FinTech"` |
| Culture | `culture.py` | LLM extraction (optional) | `["innovative", "fast-paced"]` |

### Entity normalization

`entity_normalizer.py` uses rapidfuzz for fuzzy deduplication. Ensures `"JS"`, `"Javascript"`, and `"JavaScript"` resolve to the same canonical form.

### Vocabulary files (`src/nlp/vocab/`)

- `skills.txt` -- programming languages, frameworks, methodologies
- `tools.txt` -- infrastructure, platforms, software
- `languages.txt` -- spoken languages and proficiency markers
- `degrees.txt` -- academic degree patterns

---

## Clustering System

Located in `src/clustering_v2/`.

### Components

| File | Role |
|------|------|
| `projector.py` | UMAP 2D projection (n_neighbors=15, min_dist=0.1, metric=cosine) |
| `clusterer.py` | HDBSCAN density-based clustering (min_cluster_size=10) |
| `labeler.py` | c-TF-IDF label extraction, optional LLM label generation |
| `aspect_clustering.py` | Aspect-based and concept-based re-clustering |
| `cache.py` | Parquet file caching keyed by aspect name |

### Clustering modes

**Default clustering:**
1. Embed all job texts with SBERT
2. Project to 2D with UMAP
3. Cluster with HDBSCAN (auto-k, noise detection as label -1)
4. Label clusters with c-TF-IDF top terms

**Aspect-based re-clustering:**
1. Build binary feature matrix for the selected aspect (e.g., skills)
2. Project using aspect-weighted similarity
3. Re-cluster and re-label

**Concept-based clustering:**
1. User types a free-text concept (e.g., "customer-facing")
2. Compute semantic similarity of the concept to all job embeddings
3. Weight UMAP projection by similarity scores
4. Cluster the concept-weighted projection

### Fallbacks

- UMAP unavailable -> `TruncatedSVD` (linear, from scikit-learn)
- HDBSCAN unavailable -> `KMeans` with silhouette score for optimal k

---

## Multi-Agent System

Located in `src/agents/`.

### Architecture

```
User Query
    │
    v
Coordinator (intent classification)
    │
    ├── search    -> SearchWorker
    ├── compare   -> AnalysisWorker
    ├── explore   -> ExplorationWorker
    ├── detail    -> AnalysisWorker
    └── general   -> Direct LLM response
    │
    v
Result synthesis -> AgentResult
```

### Intent classification (`coordinator.py`)

- **LLM-based** (when `OPENAI_API_KEY` is set): gpt-4o-mini with JSON-mode output
- **Heuristic fallback**: keyword pattern matching on the query

### Workers

Each worker runs the shared ReAct loop (`loop.py`) with OpenAI function calling.

| Worker | File | Purpose | Tools |
|--------|------|---------|-------|
| SearchWorker | `workers/search_worker.py` | Find jobs matching criteria | `hybrid_search`, `semantic_search`, `keyword_search`, `filter_jobs` |
| AnalysisWorker | `workers/analysis_worker.py` | Compare and analyze specific jobs | `get_job_summary`, `get_job_chunks`, `compare_aspects`, `find_similar_jobs` |
| ExplorationWorker | `workers/exploration_worker.py` | Market overview and trends | `cluster_by_aspect`, `browse_cluster`, `cluster_by_concept`, `aspect_distribution` |

### ReAct loop (`loop.py`)

```
1. Send conversation + tool schemas to LLM
2. If LLM returns tool calls:
   a. Execute each tool via registry
   b. Append results to conversation
   c. Go to 1
3. If LLM returns text: return as final answer
4. If max iterations reached: return partial result
```

### Tool registry (`tools/registry.py`)

Maps tool names to handler functions. Generates OpenAI function-calling schemas from Python type hints. Handles execution and error capture.

**Tool categories:**

| Category | File | Tools |
|----------|------|-------|
| Search | `search_tools.py` | `hybrid_search`, `semantic_search`, `keyword_search`, `filter_jobs` |
| Retrieval | `retrieval_tools.py` | `get_job_summary`, `get_job_chunks`, `get_full_text`, `expand_context` |
| Cluster | `cluster_tools.py` | `cluster_by_aspect`, `browse_cluster`, `cluster_by_concept`, `aspect_distribution` |
| NLP | `nlp_tools.py` | `extract_keywords`, `extract_entities` |

### Context management (`context.py`)

4-tier strategy to control token usage:

| Tier | Tokens/Job | Content | When Used |
|------|-----------|---------|-----------|
| 0 | ~50 | Aspects + keywords only | Search results overview, many jobs |
| 1 | ~200 | Relevant chunks (hybrid retrieval) | Default for most queries |
| 2 | ~500 | Full sections | Comparison, moderate detail |
| 3 | ~800 | Complete job text | Single-job deep dive |

Tier selection is based on intent type and number of jobs in context.

### Conversation memory (`memory.py`)

- **Sliding window**: last N turns (default 5) kept in full
- **Summary**: older turns compressed via LLM summarization
- **Entity tracking**: remembers job IDs mentioned in conversation
- **User preferences**: extracts and stores location, skill, and role preferences

---

## API Layer

Located in `src/api/`.

### App factory (`main.py`)

- FastAPI app with lifespan manager
- On startup: loads `HybridRetriever` from artifacts
- Coordinator initialized lazily on first chat request
- CORS configured for `localhost:5173` (Vite) and `localhost:3000`

### Routes (`src/api/routes/`)

| Method | Path | File | Description |
|--------|------|------|-------------|
| GET | `/api/jobs` | `jobs.py` | List jobs with filtering (location, company, title) and pagination |
| GET | `/api/jobs/{id}` | `jobs.py` | Job detail with all chunks and aspects |
| POST | `/api/search` | `search.py` | Hybrid search with alpha, k, and filter parameters |
| GET | `/api/clusters/{aspect}` | `clusters.py` | UMAP + HDBSCAN scatter data for an aspect |
| POST | `/api/clusters/concept` | `clusters.py` | Concept-weighted clustering from free text |
| GET | `/api/aspects` | `aspects.py` | List all available aspects |
| GET | `/api/aspects/{name}/distribution` | `aspects.py` | Value frequency distribution for an aspect |
| WS | `/api/chat` | `chat.py` | WebSocket agentic chat |
| GET | `/api/health` | `main.py` | Health check with artifact counts |

### WebSocket chat protocol (`routes/chat.py`)

```
Client -> Server:
  {"type": "message", "content": "Find Python jobs in Amsterdam"}

Server -> Client (sequence):
  {"type": "intent", "intent": "search", "confidence": 0.95}
  {"type": "tool_call", "tool": "hybrid_search", "args": {...}, "result": {...}}
  {"type": "answer", "content": "I found 5 Python jobs in Amsterdam..."}
  {"type": "ui_action", "action": "highlight_jobs", "job_ids": [...]}
```

The frontend uses these messages to show tool call progress and trigger UI updates (e.g., highlighting jobs in the scatter plot or table).

---

## Frontend

Located in `frontend/`. Built with React 18 + TypeScript + Vite.

### Components (`frontend/src/components/`)

| Component | Purpose |
|-----------|---------|
| `ScatterPlot.tsx` | Plotly.js WebGL 2D scatter plot with cluster coloring, hover tooltips, click-to-select |
| `JobTable.tsx` | AG Grid with sortable/filterable columns, pagination, row selection synced with scatter |
| `ChatPanel.tsx` | WebSocket chat with streaming messages, tool call display, markdown rendering |
| `AspectSelector.tsx` | Pill buttons for predefined aspects (skills, tools, language, etc.) plus free-text concept input |
| `FilterPanel.tsx` | Text filters for location, company, title |
| `JobDetail.tsx` | Expandable panel showing full job description, aspects, and metadata |

### Hooks (`frontend/src/hooks/`)

| Hook | Purpose |
|------|---------|
| `useJobs.ts` | Fetch and search jobs via the REST API |
| `useClusters.ts` | Fetch cluster data, handle aspect and concept clustering |
| `useWebSocket.ts` | Manage WebSocket connection, message queue, reconnection |

### API client (`frontend/src/api/client.ts`)

Typed wrapper around `fetch` for all REST endpoints. Base URL: `/api` (proxied by Vite in development).

---

## Data Models

Located in `src/models/`. All models use Pydantic v2.

### Job models (`models/job.py`)

- **Job** -- internal representation with all fields (title, company, location, description, url, contract type, work type, days old)
- **JobSummary** -- lightweight version for listings and search results
- **JobDetail** -- full version including description, chunks, embeddings, and aspects

### Chunk models (`models/chunk.py`)

- **Chunk** -- base unit: `chunk_id`, `text`, `section`, `order`, `meta` (job_key, title, company)
- **ChunkWithAspects** -- enriched chunk with extracted aspects and keywords

### Aspect models (`models/aspect.py`)

- **AspectValue** -- single extracted value with evidence span and confidence
- **AspectExtraction** -- complete extraction result: aspect name, values, method (deterministic/llm), confidence
- **DomainClassification** -- LLM domain result with confidence and reasoning
- **CultureExtraction** -- LLM culture values
- **SeniorityAssessment** -- experience level assessment

### Cluster models (`models/cluster.py`)

- **ClusterResult** -- full scatter data: list of points with x, y, cluster_id, job metadata
- **ClusterInfo** -- per-cluster metadata: label, size, top terms

### Agent models (`models/agent.py`)

- **ToolCall** -- tool invocation record (name, arguments, result, duration)
- **IntentClassification** -- classified intent with confidence score
- **AgentResult** -- final output: answer text, tool calls, referenced job IDs
- **ConversationTurn** -- single turn in memory (role, content, tool calls)
- **MemoryState** -- full memory snapshot (turns, summary, entities, preferences)

---

## Configuration

### Environment variables (`.env`)

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | For LLM features | Agentic chat, domain/culture extraction, cluster labeling |
| `GOOGLE_API_KEY` | No | Company logo search fallback |
| `GOOGLE_CSE_ID` | No | Custom search engine ID for logos |

### Facet taxonomy (`config/facets.yml`)

Defines 14 facets for job classification:

1. Domain / Sector
2. Role Family
3. Job Responsibilities
4. Seniority
5. Language Requirement
6. Remote Policy
7. Tools / Stack
8. AI Focus
9. Company Culture & Values
10. Stakeholders
11. Hiring Constraints
12. Methodology / Frameworks
13. Company Size
14. Degree Requirement / Personality & Soft Skills

### Facet synonyms (`config/facet_synonyms.yml`)

Maps variant terms to canonical facet values for normalization.

### Scoring (`config/scoring.yml`)

Configures aspect weighting for ranking and comparison operations.

### Python configuration (`src/config.py`)

Central configuration module that reads environment variables, sets default paths (`artifacts/`, `data/`), and exposes feature flags.

---

## Storage & Artifacts

All generated data is stored in `artifacts/` (gitignored).

| File | Format | Contents |
|------|--------|----------|
| `chunks.jsonl` | JSONL (streaming) | All chunks with metadata, one per line |
| `faiss.index` | FAISS binary | Dense vector index (IndexFlatIP) |
| `bm25.pkl` | Pickle | BM25 corpus for sparse retrieval |
| `aspects.jsonl` | JSONL | Extracted aspects per job |
| `keywords.jsonl` | JSONL | TF-IDF keywords per job |
| `cluster_*.parquet` | Parquet (columnar) | Cached cluster results keyed by aspect |

Source data lives in `data/`:

| File | Contents |
|------|----------|
| `sample_jobs.json` | Included sample dataset for quick start |
| `offline_store.json` | User-provided job data |

---

## Fallback Strategies

The system degrades gracefully when optional dependencies or services are unavailable.

| Component | Condition | Fallback |
|-----------|-----------|----------|
| UMAP | `umap-learn` not installed | `TruncatedSVD` from scikit-learn (linear projection) |
| HDBSCAN | `hdbscan` not installed | `KMeans` with silhouette scoring for optimal k |
| OpenAI agents | `OPENAI_API_KEY` not set | Heuristic intent classification + direct RAG search |
| spaCy NLP | `spacy` not installed | Regex-only extraction (reduced coverage) |
| LLM aspects | `--llm-aspects` not passed | Skip domain/culture extraction |
| Cross-encoder | Model not available | Skip reranking, use hybrid scores directly |
| Company logos | Google API keys not set | No logo display |

---

## Project Structure

```
better-job-search/
├── src/
│   ├── __init__.py
│   ├── pipeline.py                 # CLI entry point (build, serve, search, chat)
│   ├── config.py                   # Paths, API keys, feature flags
│   ├── rag.py                      # Core RAG engine (chunking, indexing, retrieval)
│   ├── models/                     # Pydantic data models
│   │   ├── __init__.py
│   │   ├── job.py                  #   Job, JobSummary, JobDetail
│   │   ├── chunk.py                #   Chunk, ChunkWithAspects
│   │   ├── aspect.py              #   AspectExtraction, DomainClassification
│   │   ├── cluster.py             #   ClusterResult, ClusterInfo
│   │   └── agent.py               #   AgentResult, ToolCall, IntentClassification
│   ├── nlp/                        # Deterministic NLP pipeline
│   │   ├── __init__.py
│   │   ├── aspect_extractor.py    #   Orchestrates all extractors
│   │   ├── cleaner.py             #   HTML cleaning, text normalization
│   │   ├── section_detector.py    #   Section boundary detection
│   │   ├── chunker.py             #   Aspect-aware chunking
│   │   ├── keyword_extractor.py   #   KeyBERT + TF-IDF
│   │   ├── entity_normalizer.py   #   rapidfuzz deduplication
│   │   ├── extractors/            #   Per-aspect extractors
│   │   │   ├── __init__.py
│   │   │   ├── skills.py          #     spaCy PhraseMatcher
│   │   │   ├── language.py        #     Regex patterns
│   │   │   ├── remote_policy.py   #     Regex + priority
│   │   │   ├── experience.py      #     Regex + NER
│   │   │   ├── education.py       #     Regex patterns
│   │   │   ├── benefits.py        #     Keyword matching
│   │   │   ├── domain.py          #     LLM classification
│   │   │   └── culture.py         #     LLM extraction
│   │   └── vocab/                 #   Vocabulary files
│   │       ├── skills.txt
│   │       ├── tools.txt
│   │       ├── languages.txt
│   │       └── degrees.txt
│   ├── search/                     # Hybrid search system
│   │   ├── __init__.py
│   │   ├── embedder.py            #   SBERT embedding (M1/M2 compatible)
│   │   ├── indexer.py             #   FAISS + BM25 index building
│   │   ├── retriever.py           #   HybridRetriever class
│   │   └── reranker.py            #   Cross-encoder reranking
│   ├── clustering_v2/             # UMAP + HDBSCAN clustering
│   │   ├── __init__.py
│   │   ├── projector.py           #   UMAP 2D projection
│   │   ├── clusterer.py           #   HDBSCAN clustering
│   │   ├── labeler.py             #   c-TF-IDF + LLM labels
│   │   ├── aspect_clustering.py   #   Per-aspect re-clustering
│   │   └── cache.py               #   Parquet caching
│   ├── agents/                     # Multi-agent system
│   │   ├── __init__.py
│   │   ├── coordinator.py         #   Intent classification + routing
│   │   ├── loop.py                #   Reusable ReAct loop
│   │   ├── context.py             #   4-tier context management
│   │   ├── memory.py              #   Sliding window + summary memory
│   │   ├── workers/
│   │   │   ├── __init__.py
│   │   │   ├── search_worker.py
│   │   │   ├── analysis_worker.py
│   │   │   └── exploration_worker.py
│   │   └── tools/
│   │       ├── __init__.py
│   │       ├── registry.py        #   Tool name -> handler mapping
│   │       ├── search_tools.py
│   │       ├── retrieval_tools.py
│   │       ├── cluster_tools.py
│   │       └── nlp_tools.py
│   └── api/                        # FastAPI backend
│       ├── __init__.py
│       ├── main.py                #   App factory, CORS, lifespan
│       └── routes/
│           ├── __init__.py
│           ├── jobs.py
│           ├── search.py
│           ├── clusters.py
│           ├── aspects.py
│           └── chat.py            #   WebSocket endpoint
├── frontend/                       # React + Vite + TypeScript
│   ├── src/
│   │   ├── App.tsx                #   Main layout
│   │   ├── main.tsx               #   Entry point
│   │   ├── api/client.ts          #   Typed API client
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
│   ├── facet_synonyms.yml          # Synonym mappings
│   └── scoring.yml                 # Aspect weights
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
