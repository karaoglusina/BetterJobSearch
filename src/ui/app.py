"""
BetterJobSearch Web UI - Dash Application

Interactive 3-panel interface for exploring, filtering, and analyzing job postings.
All data is stored locally - no external database required.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
from pydantic import BaseModel

# Import LLM utilities (optional - gracefully handles missing OPENAI_API_KEY)
try:
    from ..llm_io import call_llm_json
except ImportError:
    call_llm_json = None  # type: ignore

# Optional: dash_resizable_panels for better layout
try:
    from dash_resizable_panels import Panel, PanelGroup, PanelResizeHandle  # type: ignore
except Exception:
    PanelGroup = None  # type: ignore
    Panel = None  # type: ignore
    PanelResizeHandle = None  # type: ignore

# Fallback to dash_split_pane
try:
    from dash_split_pane import SplitPane  # type: ignore
except Exception:
    SplitPane = None  # type: ignore


# =============================================================================
# Path Utilities
# =============================================================================


def _get_repo_root() -> str:
    """Get the repository root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _offline_paths() -> Dict[str, str]:
    """Get paths for offline data storage."""
    root = _get_repo_root()
    data_dir = os.path.join(root, "data")
    return {
        "data_dir": data_dir,
        "store": os.path.join(data_dir, "offline_store.json"),
        "changes": os.path.join(data_dir, "offline_changes.jsonl"),
        "sample": os.path.join(data_dir, "sample_jobs.json"),
    }


def _ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


# =============================================================================
# Data Storage Functions
# =============================================================================


def _read_offline_store() -> Dict[str, Any]:
    """Read the offline job store."""
    paths = _offline_paths()
    p = paths["store"]
    try:
        if not os.path.exists(p):
            return {}
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def _atomic_write_json(path: str, data: Any) -> None:
    """Atomically write JSON data to file."""
    _ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _append_change_log(change: Dict[str, Any]) -> None:
    """Append a change to the offline change log."""
    paths = _offline_paths()
    _ensure_dir(paths["data_dir"])
    change = dict(change)
    change["ts"] = change.get("ts") or datetime.now(timezone.utc).isoformat()
    try:
        with open(paths["changes"], "a", encoding="utf-8") as f:
            f.write(json.dumps(change) + "\n")
    except Exception:
        pass


def _initialize_from_sample() -> Dict[str, Any]:
    """Initialize offline store from sample data if it doesn't exist."""
    paths = _offline_paths()
    
    # If offline store already exists, use it
    if os.path.exists(paths["store"]):
        return _read_offline_store()
    
    # Try to load from sample data
    sample_path = paths["sample"]
    if not os.path.exists(sample_path):
        # Try alternative names
        alt_paths = [
            os.path.join(paths["data_dir"], "full_batch_scrape_32_keywords.json"),
            os.path.join(paths["data_dir"], "jobs.json"),
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                sample_path = alt
                break
    
    if not os.path.exists(sample_path):
        return {}
    
    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        
        # Handle both list and dict with 'jobs' key
        if isinstance(raw, list):
            jobs_list = raw
        elif isinstance(raw, dict) and "jobs" in raw:
            jobs_list = raw["jobs"]
        else:
            return {}
        
        # Convert to offline store format
        # Handle both nested structure (job_data/meta) and flat structure
        out: Dict[str, Any] = {}
        for j in jobs_list:
            # Check if data is nested (has job_data key) or flat (jobUrl at top level)
            if "job_data" in j:
                # Nested structure
                jd = j.get("job_data", {})
                meta = j.get("meta", {})
            else:
                # Flat structure - use the job dict directly
                jd = j
                meta = {}
            
            job_id = jd.get("jobUrl") or jd.get("id")
            if not job_id:
                continue
            
            out[job_id] = {
                "job_id": job_id,
                "title": jd.get("title"),
                "company": jd.get("companyName"),
                "location": jd.get("location"),
                "applyType": jd.get("applyType"),
                "contractType": jd.get("contractType"),
                "workType": jd.get("workType"),
                "url": job_id,
                "companyUrl": jd.get("companyUrl"),
                "description": jd.get("description") or jd.get("descriptionHtml", ""),
                "applied_times": meta.get("applied_times") or jd.get("applicationsCount"),
                "days_old": meta.get("days_old"),
                "city": meta.get("city"),
                "isNew": meta.get("isNew"),
                "domain_cluster": meta.get("domain_cluster"),
                "role_cluster": meta.get("role_cluster"),
                "facets": {},
                "job_label": [],
                "job_disqualified": False,
                "logo_url": None,
                "logo_thumb_data_uri": None,
                "LLMaboutCompany": None,
                "LLMjobSummary": None,
            }
        
        # Save to offline store
        if out:
            _atomic_write_json(paths["store"], out)
            print(f"Initialized offline store with {len(out)} jobs from {sample_path}")
        else:
            print(f"Warning: No jobs were loaded from {sample_path}. Check the data format.")
        
        return out
    except Exception as e:
        print(f"Error initializing from sample data: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {}


def _load_snapshots_and_meta() -> Dict[str, Any]:
    """Load job data from offline store, initializing from sample if needed."""
    return _initialize_from_sample()


def _load_embeddings_coords(parquet_path: str) -> Dict[str, Dict[str, List[float]]]:
    """Load precomputed embeddings and project to 2D for visualization."""
    import numpy as np
    from sklearn.decomposition import TruncatedSVD

    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
        except Exception:
            return {}
    
    groups = {}
    for g in df["group"].unique():
        gdf = df[df["group"] == g]
        if gdf.empty:
            continue
        vecs = np.asarray(
            [np.asarray(v, dtype="float32") for v in gdf["vector"].tolist()],
            dtype="float32",
        )
        proj = TruncatedSVD(n_components=2, random_state=42)
        coords = proj.fit_transform(vecs)
        job_ids = gdf["job_id"].astype(str).tolist()
        groups[g] = {
            jid: [float(coords[i, 0]), float(coords[i, 1])]
            for i, jid in enumerate(job_ids)
        }
    return groups


# =============================================================================
# Logo Fetching Utilities
# =============================================================================


def _find_company_logo_url(company: str) -> str | None:
    """Find a company logo URL using free APIs."""
    if not company or not str(company).strip():
        return None
    
    # Try DuckDuckGo Images
    try:
        from ddgs import DDGS
        query = f"{company} logo"
        with DDGS() as ddg:
            for res in ddg.images(keywords=query, max_results=6, safesearch="moderate"):
                try:
                    url = (res.get("image") or res.get("thumbnail") or "").strip()
                    if not url:
                        continue
                    low = url.lower()
                    if any(low.endswith(ext) for ext in (".svg", ".png", ".jpg", ".jpeg", ".webp")):
                        return url
                except Exception:
                    continue
    except Exception:
        pass
    
    # Try to find official domain
    def _domain_from_url(u: str) -> str | None:
        try:
            from urllib.parse import urlparse
            netloc = urlparse(u).netloc
            if not netloc:
                return None
            return netloc.split(":")[0].lstrip("www.")
        except Exception:
            return None
    
    domain: str | None = None
    try:
        from ddgs import DDGS
        q = f"{company} official site"
        with DDGS() as ddg:
            for res in ddg.text(q, max_results=5, safesearch="moderate"):
                href = (res.get("href") or res.get("link") or "").strip()
                d = _domain_from_url(href)
                if d and not any(x in d for x in ("linkedin.com", "facebook.com", "twitter.com", "x.com")):
                    domain = d
                    break
    except Exception:
        domain = None
    
    if domain:
        return f"https://logo.clearbit.com/{domain}"
    
    return None


def _make_logo_thumbnail_data_uri(url: str, size: int = 54) -> str | None:
    """Download image and create a small PNG data URI."""
    if not url:
        return None
    try:
        from PIL import Image, ImageOps

        resp = requests.get(url, stream=True, timeout=(5, 10))
        if resp.status_code != 200:
            return None
        
        buf = io.BytesIO()
        max_bytes = 2 * 1024 * 1024
        read_bytes = 0
        for chunk in resp.iter_content(chunk_size=65536):
            if not chunk:
                break
            read_bytes += len(chunk)
            if read_bytes > max_bytes:
                return None
            buf.write(chunk)
        buf.seek(0)

        with Image.open(buf) as im:
            try:
                im.seek(0)
            except Exception:
                pass
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA")
            im_fitted = ImageOps.contain(im, (size, size))
            canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            off_x = (size - im_fitted.width) // 2
            off_y = (size - im_fitted.height) // 2
            canvas.paste(im_fitted, (off_x, off_y))

            out = io.BytesIO()
            canvas.save(out, format="PNG")
            b64 = base64.b64encode(out.getvalue()).decode("ascii")
            return f"data:image/png;base64,{b64}"
    except Exception:
        return None


# =============================================================================
# UI Components
# =============================================================================


def _sidebar_filters() -> html.Div:
    """Build the sidebar filters panel."""
    return html.Div(
        [
            html.H5("Filters"),
            html.H6("Title"),
            dcc.Dropdown(id="title-dd", options=[], multi=True, placeholder="Title"),
            html.H6("Location"),
            dcc.Dropdown(id="location-dd", options=[], multi=True, placeholder="Location"),
            html.H6("City"),
            dcc.Dropdown(id="city-dd", options=[], multi=True, placeholder="City"),
            html.H6("Row lines"),
            dcc.Dropdown(
                id="row-lines",
                options=[
                    {"label": "1 line", "value": 1},
                    {"label": "2 lines", "value": 2},
                    {"label": "3 lines", "value": 3},
                ],
                value=2,
                clearable=False,
            ),
            html.H6("Search"),
            dcc.Textarea(
                id="search-query",
                value="",
                placeholder='e.g. ("data engineer" AND python) OR spark',
                style={"width": "100%", "height": "96px"},
            ),
            html.H6("Semantic (optional)"),
            dcc.Input(
                id="semantic-query",
                type="text",
                placeholder="semantic query",
                debounce=True,
                style={"width": "100%", "marginBottom": "6px"},
            ),
            dcc.Checklist(
                id="search-fields",
                options=[
                    {"label": "Title", "value": "title"},
                    {"label": "Description", "value": "description"},
                    {"label": "Location", "value": "location"},
                    {"label": "About Company", "value": "about_company"},
                    {"label": "Job Summary", "value": "job_summary"},
                ],
                value=["title", "description", "location", "about_company", "job_summary"],
                inputClassName="chk",
            ),
            html.Div(id="search-error", style={"color": "#b00", "fontSize": "12px", "marginTop": "4px"}),
            html.H6("Applied Times"),
            dbc.Row(
                [
                    dbc.Col(dcc.Input(id="applied-min", type="number", placeholder="min", value=0), width=6),
                    dbc.Col(dcc.Input(id="applied-max", type="number", placeholder="max", value=100), width=6),
                ],
                className="g-1",
                align="center",
            ),
            html.H6("Days Old"),
            dbc.Row(
                [
                    dbc.Col(dcc.Input(id="days-min", type="number", placeholder="min", value=0), width=6),
                    dbc.Col(dcc.Input(id="days-max", type="number", placeholder="max", value=14), width=6),
                ],
                className="g-1",
                align="center",
            ),
            html.H6("Other"),
            dcc.Dropdown(id="apply-type", multi=True, placeholder="applyType"),
            dcc.Dropdown(id="contract-type", multi=True, placeholder="contractType"),
            dcc.Dropdown(id="work-type", multi=True, placeholder="workType"),
            html.Div([dbc.Checkbox(id="filter-isnew", value=False, label="Show only new (isNew)")]),
            html.H6("Cluster filters"),
            dcc.Dropdown(id="domain-cluster-dd", multi=True, placeholder="Domain cluster", optionHeight=64),
            dcc.Dropdown(id="role-cluster-dd", multi=True, placeholder="Role cluster", optionHeight=64),
            dcc.Dropdown(id="label-dd-filter", multi=True, placeholder="Labels include", options=[]),
            dcc.Dropdown(id="label-dd-exclude", multi=True, placeholder="Labels exclude", options=[], value=["no"]),
            html.Hr(),
            html.H6("Columns"),
            dcc.Checklist(id="column-toggles", options=[], value=[]),
        ],
        id="sidebar",
        style={
            "overflowY": "auto",
            "height": "100vh",
            "padding": "12px",
            "borderRight": "1px solid #ddd",
        },
    )


def _sidebar_llm_selector() -> html.Div:
    """Build the LLM selector panel."""
    return html.Div(
        [
            html.H5("LLM Selector"),
            html.Div("Write a prompt describing which jobs you want to keep."),
            dcc.Textarea(
                id="llmsel-prompt",
                placeholder="e.g., Which of these jobs require Dutch?",
                style={"width": "100%", "height": "140px"},
            ),
            html.Div("Batch size"),
            dcc.Input(
                id="llmsel-batch",
                type="number",
                min=5,
                max=200,
                step=5,
                value=30,
                style={"width": "120px", "marginBottom": "6px"},
            ),
            dbc.Button("Run LLM selection", id="llmsel-run", color="primary", size="sm", style={"marginLeft": 6}),
            html.Div(id="llmsel-status", style={"marginTop": 4, "color": "#555", "fontSize": "12px"}),
            html.Hr(),
            dcc.Markdown(id="llmsel-output", link_target="_blank", dangerously_allow_html=True),
        ],
        style={"overflowY": "auto", "height": "100%"},
    )


def _sidebar_panel() -> html.Div:
    """Build the complete sidebar with tabs."""
    return html.Div(
        dcc.Tabs(
            id="sidebar-tabs",
            value="filters",
            children=[
                dcc.Tab(label="Filters", value="filters", children=[_sidebar_filters()]),
                dcc.Tab(label="LLM Selector", value="llm", children=[_sidebar_llm_selector()]),
            ],
        ),
        id="sidebar",
        style={
            "overflowY": "auto",
            "height": "100vh",
            "padding": "0px",
            "borderRight": "1px solid #ddd",
        },
    )


def _charts_panel() -> html.Div:
    """Build the charts panel for cluster visualization."""
    def card(title: str, graph_id: str) -> dbc.Card:
        return dbc.Card(
            [
                dbc.CardHeader(title),
                dbc.CardBody(
                    dcc.Graph(
                        id=graph_id,
                        figure={},
                        config={
                            "scrollZoom": True,
                            "doubleClick": "reset",
                            "modeBarButtonsToAdd": ["pan2d", "zoom2d", "resetScale2d"],
                            "displaylogo": False,
                        },
                    )
                ),
            ],
            style={"height": "100%"},
        )

    actions = dbc.Row(
        [
            dbc.Col(dbc.Button("Deselect all", id="clear-selection-clusters", size="sm", color="secondary"), width="auto"),
            dbc.Col(html.Div("k (clusters)"), width="auto"),
            dbc.Col(
                dcc.Input(id="chart-k", type="number", min=2, max=30, step=1, debounce=True, placeholder="auto", style={"width": "90px"}),
                width="auto",
            ),
        ],
        style={"marginBottom": 6, "alignItems": "center", "gap": "6px"},
    )
    domain_row = dbc.Row([dbc.Col(card("Domain", "chart-domain"), width=12)], style={"rowGap": "8px"})
    role_row = dbc.Row([dbc.Col(card("Role", "chart-role"), width=12)], style={"rowGap": "8px"})
    return html.Div(
        [actions, domain_row, role_row],
        id="charts-panel",
        style={"height": "calc(100vh - 24px)", "overflow": "auto"},
    )


def _table_panel() -> html.Div:
    """Build the main job table panel."""
    grid = dag.AgGrid(
        id="job-grid",
        columnDefs=[
            {
                "field": "logo_md",
                "headerName": "",
                "pinned": "left",
                "minWidth": 64,
                "width": 64,
                "resizable": True,
                "sortable": False,
                "filter": False,
                "cellRenderer": "markdown",
                "cellClass": "logo-cell",
            },
        ],
        rowData=[],
        defaultColDef={"resizable": True, "sortable": True, "filter": True, "cellClass": "wrap-text"},
        dashGridOptions={
            "rowSelection": "multiple",
            "animateRows": False,
            "rowHeight": 40,
            "tooltipShowDelay": 0,
            "suppressScrollOnNewData": True,
        },
        style={"height": "90%"},
    )
    return html.Div(
        [
            html.Div(id="job-count", style={"margin": "6px 0", "fontWeight": 600}),
            dbc.Button("Fetch logos for selection", id="fetch-logos", size="sm", color="secondary", style={"marginBottom": 6}),
            dbc.Button("Save changes", id="sync-now", size="sm", color="primary", style={"marginBottom": 6, "marginLeft": 6}),
            dbc.Button("Export Excel", id="export-excel", size="sm", color="secondary", style={"marginBottom": 6, "marginLeft": 6}),
            dbc.Button("LLM: About Company", id="llm-about-company", size="sm", color="warning", style={"marginBottom": 6, "marginLeft": 6}),
            dbc.Button("LLM: Summarise Job", id="llm-summarise-job", size="sm", color="warning", style={"marginBottom": 6, "marginLeft": 6}),
            dbc.Button("Deselect all", id="clear-selection", size="sm", color="secondary", style={"marginBottom": 6, "marginLeft": 6}),
            dbc.Checkbox(id="show-selected-only", value=False, label="Show selected only", style={"marginLeft": 12, "marginBottom": 6}),
            dcc.Download(id="download-excel"),
            grid,
        ],
        id="table-panel",
        style={"height": "calc(100vh - 24px)", "overflow": "auto", "borderTop": "1px solid #eee"},
    )


def _details_panel() -> html.Div:
    """Build the job details panel."""
    return html.Div(
        [
            html.H5("Job Details"),
            html.H6("About Company (LLM)"),
            html.Div(id="company-about", style={"marginBottom": 8}),
            html.H6("Job Summary (LLM)"),
            html.Div(id="company-job-summary", style={"marginBottom": 8}),
            html.H6("Labels"),
            dbc.Row(
                [
                    dbc.Col(dcc.Dropdown(id="job-label-dd", options=[], multi=True, placeholder="Select tags"), width=7),
                    dbc.Col(dcc.Input(id="job-label-new", type="text", placeholder="new tag(s), comma-separated"), width=3),
                    dbc.Col(dbc.Button("Save", id="save-job-label", color="primary", size="sm"), width=2),
                ],
                align="center",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Button("yes", id="save-job-label-yes", color="success", size="sm"), width="auto"),
                    dbc.Col(dbc.Button("no", id="save-job-label-no", color="danger", size="sm", style={"marginLeft": 6}), width="auto"),
                ],
                align="center",
                style={"marginTop": 6},
            ),
            html.Div(id="save-status", style={"marginTop": 6, "color": "#2c7"}),
            html.Hr(),
            html.Div(id="job-meta"),
            html.H6("Facets"),
            html.Div(id="job-facets"),
            html.H6("Description"),
            html.Div(id="job-desc", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace"}),
        ],
        id="details",
        style={
            "overflowY": "auto",
            "height": "100vh",
            "padding": "12px",
            "borderLeft": "1px solid #ddd",
        },
    )


# =============================================================================
# Search and Filter Utilities
# =============================================================================


def _compile_query(query: str):
    """Compile a boolean search query to postfix notation."""
    if not query or not str(query).strip():
        return None
    s = query.strip()
    tok_re = re.compile(r'"([^"]+)"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+', re.IGNORECASE)
    tokens = []
    for m in tok_re.finditer(s):
        g = m.group(0)
        if g.startswith('"') and g.endswith('"'):
            tokens.append(g[1:-1])
        else:
            tokens.append(g)
    
    ops = {"AND": (2, "L"), "OR": (1, "L"), "NOT": (3, "R")}
    out = []
    stack = []
    for t in tokens:
        u = t.upper()
        if u in ops:
            while stack:
                top = stack[-1]
                if top in ops:
                    p1, a1 = ops[u]
                    p2, _ = ops[top]
                    if (a1 == "L" and p1 <= p2) or (a1 == "R" and p1 < p2):
                        out.append(stack.pop())
                        continue
                break
            stack.append(u)
        elif t == "(":
            stack.append(t)
        elif t == ")":
            while stack and stack[-1] != "(":
                out.append(stack.pop())
            if not stack:
                raise ValueError("Mismatched parentheses")
            stack.pop()
        else:
            out.append(t)
    while stack:
        if stack[-1] in ("(", ")"):
            raise ValueError("Mismatched parentheses")
        out.append(stack.pop())
    return out


def _extract_search_terms(query: str | None) -> List[str]:
    """Extract search terms for highlighting."""
    if not query or not str(query).strip():
        return []
    try:
        postfix = _compile_query(query)
    except Exception:
        postfix = None
    terms: List[str] = []
    ops = {"AND", "OR", "NOT"}
    for t in postfix or []:
        if isinstance(t, str) and t.upper() not in ops:
            tt = t.strip()
            if tt:
                terms.append(tt)
    seen = set()
    uniq: List[str] = []
    for t in terms:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    uniq.sort(key=lambda x: len(x), reverse=True)
    return uniq


def _highlight_terms_markdown(text: str | None, terms: List[str]) -> str:
    """Highlight search terms with <mark> tags."""
    if not text or not terms:
        return text or ""
    try:
        pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)
    except re.error:
        return text
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)


def _highlight_terms_code(text: str | None, terms: List[str]) -> str:
    """Highlight search terms with backticks for code styling."""
    if not text or not terms:
        return text or ""
    try:
        pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)
    except re.error:
        return text
    return pattern.sub(lambda m: f"`{m.group(0)}`", text)


def _eval_postfix(job: Dict[str, Any], postfix, fields: List[str]) -> bool:
    """Evaluate a postfix boolean expression against a job."""
    if not postfix:
        return True
    haystacks = []
    if not fields:
        fields = ["title", "description"]
    for f in fields:
        if f == "title":
            haystacks.append((job.get("title") or "").lower())
        elif f == "description":
            haystacks.append((job.get("description") or "").lower())
        elif f == "location":
            haystacks.append((job.get("location") or "").lower())
        elif f == "labels":
            haystacks.append(" ".join([str(x) for x in (job.get("job_label") or [])]).lower())
        elif f == "about_company":
            haystacks.append((job.get("LLMaboutCompany") or "").lower())
        elif f == "job_summary":
            haystacks.append((job.get("LLMjobSummary") or "").lower())
    combined = " \n ".join(haystacks)
    st = []
    for t in postfix:
        if t in ("AND", "OR", "NOT"):
            if t == "NOT":
                a = bool(st.pop()) if st else False
                st.append(not a)
            else:
                b = bool(st.pop()) if st else False
                a = bool(st.pop()) if st else False
                st.append((a and b) if t == "AND" else (a or b))
        else:
            term = str(t).lower()
            st.append(term in combined)
    return bool(st[-1]) if st else True


def _filter_jobs(
    data: Dict[str, Any],
    emb: Dict[str, Any],
    group: str,
    facet_values: Dict[str, List[str]],
    hide_disq: bool,
    titles: List[str],
    locs: List[str],
    app_types: List[str],
    contract_types: List[str],
    work_types: List[str],
    applied_rng: List[int],
    days_rng: List[int],
    search_query: str | None,
    search_fields: List[str] | None,
    *,
    search_error_out: List[str] | None = None,
    label_filter: List[str] | None = None,
    label_exclude: List[str] | None = None,
    cities: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Filter jobs based on various criteria."""
    postfix = None
    if search_query and str(search_query).strip():
        try:
            postfix = _compile_query(search_query)
            if search_error_out is not None:
                search_error_out.clear()
        except Exception as e:
            if search_error_out is not None:
                search_error_out.clear()
                search_error_out.append(f"Query error: {e}")
            postfix = None
    
    rows = []
    for job_id, v in data.items():
        if hide_disq and v.get("job_disqualified"):
            continue
        if titles and (v.get("title") not in titles):
            continue
        if locs and (v.get("location") not in locs):
            continue
        if app_types and (v.get("applyType") not in app_types):
            continue
        if contract_types and (v.get("contractType") not in contract_types):
            continue
        if work_types and (v.get("workType") not in work_types):
            continue
        if cities:
            vc = v.get("city") or (v.get("meta") or {}).get("city")
            if vc not in set(cities):
                continue
        if label_filter:
            labels = set(str(x) for x in (v.get("job_label") or []))
            if not labels.issuperset(set(label_filter)):
                continue
        if label_exclude:
            labels = set(str(x) for x in (v.get("job_label") or []))
            if set(label_exclude) & labels:
                continue
        at = v.get("applied_times")
        if applied_rng is not None and isinstance(applied_rng, (list, tuple)) and len(applied_rng) == 2 and at is not None:
            if at < applied_rng[0] or at > applied_rng[1]:
                continue
        d = v.get("days_old")
        if days_rng is not None and isinstance(days_rng, (list, tuple)) and len(days_rng) == 2 and d is not None:
            if d < days_rng[0] or d > days_rng[1]:
                continue
        if postfix is not None:
            if not _eval_postfix(v, postfix, search_fields or []):
                continue
        rows.append(v)
    return rows


# =============================================================================
# Visualization Utilities
# =============================================================================


def _make_scatter_for(
    group: str,
    emb: Dict[str, Dict[str, List[float]]],
    job_ids: List[str],
    job_info: Dict[str, Dict[str, Any]],
    k_override: int | None = None,
) -> go.Figure:
    """Create a scatter plot for cluster visualization."""
    import numpy as np

    gmap = emb.get(group) or {}
    xs, ys, jids = [], [], []
    for jid in job_ids:
        c = gmap.get(jid)
        if not c:
            continue
        xs.append(c[0])
        ys.append(c[1])
        jids.append(jid)

    def _shorten(s: str, n: int = 260) -> str:
        try:
            s2 = (s or "").replace("\r\n", "\n").strip()
        except Exception:
            s2 = str(s or "")
        s2 = " ".join([t for t in s2.split("\n") if t.strip()])
        return s2 if len(s2) <= n else (s2[:n].rstrip() + "…")

    def _wrap(s: str, width: int = 70, max_lines: int = 8) -> str:
        try:
            words = (s or "").split()
        except Exception:
            words = str(s or "").split()
        lines = []
        cur = []
        cur_len = 0
        for w in words:
            wl = len(w)
            if cur_len + (1 if cur else 0) + wl > width:
                lines.append(" ".join(cur))
                if len(lines) >= max_lines:
                    break
                cur = [w]
                cur_len = wl
            else:
                cur.append(w)
                cur_len += (1 if cur_len > 0 else 0) + wl
        if len(lines) < max_lines and cur:
            lines.append(" ".join(cur))
        if len(lines) >= max_lines and lines:
            lines[-1] = lines[-1].rstrip() + "…"
        return "<br>".join(lines)

    hover_title = [
        f"{(job_info.get(j) or {}).get('title', '')} — {(job_info.get(j) or {}).get('company', '')}" for j in jids
    ]
    abouts = [_wrap(_shorten((job_info.get(j) or {}).get("LLMaboutCompany") or "")) for j in jids]
    sums = [_wrap(_shorten((job_info.get(j) or {}).get("LLMjobSummary") or "")) for j in jids]
    hover_full = [
        f"<span style='display:block;max-width:420px;white-space:normal'>{t}<br><br><b>About</b>:<br>{a or '-'}<br><b>Summary</b>:<br>{s or '-'}</span>"
        for t, a, s in zip(hover_title, abouts, sums)
    ]

    fig_height = 700
    if len(jids) < 2:
        fig = go.Figure(
            data=[
                go.Scattergl(
                    x=xs,
                    y=ys,
                    mode="markers",
                    customdata=jids,
                    hovertext=hover_full,
                    hovertemplate="%{hovertext}<extra></extra>",
                    marker=dict(size=6, opacity=0.8),
                )
            ]
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=fig_height,
            clickmode="event+select",
            dragmode="pan",
            hoverlabel=dict(align="left"),
        )
        return fig

    coords = np.column_stack([np.array(xs, dtype=float), np.array(ys, dtype=float)])
    n = len(jids)
    if isinstance(k_override, int) and k_override >= 2:
        k = min(max(2, k_override), max(2, n))
    else:
        k = max(2, min(8, max(2, n // 80)))

    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(coords)
    except Exception:
        labels = np.zeros(n, dtype=int)
        k = 1

    cluster_names = {cid: f"C{cid}" for cid in sorted(set(labels.tolist()))}

    if group == "role":
        palette = [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
        ]
    else:
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

    fig = go.Figure()
    for cid in sorted(set(labels.tolist())):
        m = labels == cid
        xs_c = np.array(xs)[m]
        ys_c = np.array(ys)[m]
        jids_c = np.array(jids)[m]
        hovers_c = np.array(hover_full)[m]
        fig.add_trace(
            go.Scattergl(
                x=xs_c,
                y=ys_c,
                mode="markers",
                customdata=jids_c,
                hovertext=hovers_c,
                name=cluster_names.get(cid, f"C{cid}"),
                hovertemplate="%{hovertext}<extra></extra>",
                marker=dict(size=6, opacity=0.85, color=palette[cid % len(palette)]),
            )
        )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=fig_height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        clickmode="event+select",
        dragmode="pan",
        hoverlabel=dict(align="left"),
    )
    return fig


# =============================================================================
# Main App Creation
# =============================================================================


def create_app() -> Dash:
    """Create and configure the Dash application."""
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
    )

    center_tabs = dcc.Tabs(
        id="main-tabs",
        value="table",
        children=[
            dcc.Tab(label="Table", value="table", children=[_table_panel()]),
            dcc.Tab(label="Clusters", value="clusters", children=[_charts_panel()]),
        ],
    )

    # Build layout based on available panel libraries
    if PanelGroup is not None and Panel is not None and PanelResizeHandle is not None:
        def handle():
            return PanelResizeHandle(className="resize-handle")

        main_layout = html.Div(
            [
                PanelGroup(
                    id="panels",
                    direction="horizontal",
                    children=[
                        Panel(
                            id="panel-left",
                            children=[html.Div(_sidebar_panel(), style={"height": "100%", "overflowY": "auto"})],
                            style={"minWidth": "220px"},
                        ),
                        handle(),
                        Panel(
                            id="panel-center",
                            children=[html.Div(center_tabs, style={"height": "100%", "overflow": "hidden"})],
                            style={"minWidth": "300px"},
                        ),
                        handle(),
                        Panel(
                            id="panel-right",
                            children=[html.Div(_details_panel(), style={"height": "100%", "overflowY": "auto"})],
                            style={"minWidth": "240px"},
                        ),
                    ],
                    style={"height": "100vh"},
                ),
                dcc.Interval(id="init", interval=500, n_intervals=0, max_intervals=1),
                dcc.Store(id="hovered-job"),
                dcc.Store(id="selected-job"),
                dcc.Store(id="store-data"),
                dcc.Store(id="store-emb"),
                dcc.Store(id="store-all-labels"),
                dcc.Store(id="store-selected", data=[]),
            ]
        )
    elif SplitPane is not None:
        main_layout = html.Div(
            [
                SplitPane(
                    id="outer-split",
                    split="vertical",
                    minSize=220,
                    defaultSize=280,
                    children=[
                        html.Div(_sidebar_panel(), style={"height": "100vh", "overflowY": "auto", "borderRight": "1px solid #ddd"}),
                        SplitPane(
                            id="inner-split",
                            split="vertical",
                            minSize=300,
                            defaultSize="75%",
                            children=[
                                html.Div(center_tabs, style={"height": "100vh", "overflow": "hidden"}),
                                html.Div(_details_panel(), style={"height": "100vh", "overflowY": "auto", "borderLeft": "1px solid #ddd"}),
                            ],
                            style={"height": "100vh"},
                        ),
                    ],
                    style={"height": "100vh"},
                ),
                dcc.Interval(id="init", interval=500, n_intervals=0, max_intervals=1),
                dcc.Store(id="hovered-job"),
                dcc.Store(id="selected-job"),
                dcc.Store(id="store-data"),
                dcc.Store(id="store-emb"),
                dcc.Store(id="store-all-labels"),
                dcc.Store(id="store-selected", data=[]),
            ]
        )
    else:
        main_layout = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(_sidebar_panel(), width=2),
                        dbc.Col(center_tabs, width=8, style={"overflowY": "auto", "height": "100vh", "padding": 0}),
                        dbc.Col(_details_panel(), width=2),
                    ],
                    style={"height": "100vh"},
                ),
                dcc.Interval(id="init", interval=500, n_intervals=0, max_intervals=1),
                dcc.Store(id="hovered-job"),
                dcc.Store(id="selected-job"),
                dcc.Store(id="store-data"),
                dcc.Store(id="store-emb"),
                dcc.Store(id="store-all-labels"),
                dcc.Store(id="store-selected", data=[]),
            ],
            fluid=True,
            style={"padding": 0},
        )

    app.layout = main_layout

    # ==========================================================================
    # Callbacks
    # ==========================================================================

    @app.callback(
        Output("store-data", "data"),
        Output("store-emb", "data"),
        Output("apply-type", "options"),
        Output("contract-type", "options"),
        Output("work-type", "options"),
        Output("save-status", "children", allow_duplicate=True),
        Input("init", "n_intervals"),
        prevent_initial_call="initial_duplicate",
    )
    def _init_load(_):
        # Load RAG artifacts if available
        try:
            from .. import rag as _rag
            if not _rag.is_loaded():
                _rag.load_cache()
        except Exception:
            pass

        data = _load_snapshots_and_meta()
        emb_path = os.path.join(_get_repo_root(), "data", "job_group_embeddings.parquet")
        emb = _load_embeddings_coords(emb_path)

        apply_opts = sorted({v.get("applyType") for v in data.values() if v.get("applyType")})
        contract_opts = sorted({v.get("contractType") for v in data.values() if v.get("contractType")})
        work_opts = sorted({v.get("workType") for v in data.values() if v.get("workType")})

        return (
            data,
            emb,
            [{"label": x, "value": x} for x in apply_opts],
            [{"label": x, "value": x} for x in contract_opts],
            [{"label": x, "value": x} for x in work_opts],
            f"Loaded {len(data)} jobs",
        )

    @app.callback(Output("title-dd", "options"), Input("store-data", "data"))
    def _title_options(data):
        if not data:
            return []
        titles = sorted({(v.get("title") or "").strip() for v in data.values() if v.get("title")})
        return [{"label": t, "value": t} for t in titles]

    @app.callback(Output("location-dd", "options"), Input("store-data", "data"))
    def _location_options(data):
        if not data:
            return []
        locs = sorted({(v.get("location") or "").strip() for v in data.values() if v.get("location")})
        return [{"label": loc, "value": loc} for loc in locs]

    @app.callback(Output("city-dd", "options"), Input("store-data", "data"))
    def _city_options(data):
        if not data:
            return []
        cities = sorted({
            (v.get("city") or v.get("meta", {}).get("city") or "").strip()
            for v in data.values()
            if (v.get("city") or v.get("meta", {}).get("city"))
        })
        return [{"label": c, "value": c} for c in cities if c]

    @app.callback(
        Output("domain-cluster-dd", "options"),
        Output("role-cluster-dd", "options"),
        Input("store-data", "data"),
    )
    def _cluster_filter_options(data):
        if not data:
            return [], []
        dom = sorted({
            str((v.get("meta") or {}).get("domain_cluster") or v.get("domain_cluster") or "").strip()
            for v in data.values()
            if ((v.get("meta") or {}).get("domain_cluster") or v.get("domain_cluster"))
        })
        rol = sorted({
            str((v.get("meta") or {}).get("role_cluster") or v.get("role_cluster") or "").strip()
            for v in data.values()
            if ((v.get("meta") or {}).get("role_cluster") or v.get("role_cluster"))
        })
        return ([{"label": x, "value": x} for x in dom if x], [{"label": x, "value": x} for x in rol if x])

    @app.callback(
        Output("label-dd-filter", "options"),
        Output("label-dd-exclude", "options"),
        Input("store-data", "data"),
    )
    def _label_filter_options(data):
        tags = set()
        for v in (data or {}).values():
            for t in v.get("job_label") or []:
                tags.add(str(t))
        opts = [{"label": t, "value": t} for t in sorted(tags)]
        if "no" not in tags:
            exclude_opts = [{"label": "no", "value": "no"}] + opts
        else:
            exclude_opts = opts
        return opts, exclude_opts

    @app.callback(
        Output("label-dd-exclude", "value"),
        Input("label-dd-exclude", "options"),
        State("label-dd-exclude", "value"),
    )
    def _default_exclude_value(opts, val):
        try:
            values = {o.get("value") for o in (opts or [])}
        except Exception:
            values = set()
        if (val is None or (isinstance(val, (list, tuple)) and len(val) == 0)) and ("no" in values):
            return ["no"]
        return no_update

    @app.callback(
        Output("column-toggles", "options"),
        Output("column-toggles", "value"),
        Output("applied-min", "value"),
        Output("applied-max", "value"),
        Output("days-min", "value"),
        Output("days-max", "value"),
        Input("store-data", "data"),
        State("column-toggles", "value"),
        State("applied-min", "value"),
        State("applied-max", "value"),
        State("days-min", "value"),
        State("days-max", "value"),
    )
    def _column_toggle_options(data, current_cols, cur_applied_min, cur_applied_max, cur_days_min, cur_days_max):
        static_opts = [
            {"label": "Title", "value": "title"},
            {"label": "About Company", "value": "about_company"},
            {"label": "Job Summary", "value": "job_summary"},
            {"label": "Company", "value": "company"},
            {"label": "City", "value": "city"},
            {"label": "Labels", "value": "job_label"},
            {"label": "Domain Cluster", "value": "domain_cluster"},
            {"label": "Role Cluster", "value": "role_cluster"},
            {"label": "Days Old", "value": "days_old"},
            {"label": "Location", "value": "location"},
            {"label": "URL", "value": "url"},
            {"label": "Applied Times", "value": "applied_times"},
        ]
        max_applied = 0
        max_days = 0
        if data:
            for v in data.values():
                at = v.get("applied_times")
                if isinstance(at, (int, float)):
                    max_applied = max(max_applied, int(at))
                d = v.get("days_old")
                if isinstance(d, (int, float)):
                    max_days = max(max_days, int(d))
        options = static_opts
        default_selected = [
            "title", "about_company", "job_summary", "company", "city",
            "job_label", "domain_cluster", "role_cluster", "days_old",
        ]
        applied_min = cur_applied_min if cur_applied_min is not None else 0
        applied_max = cur_applied_max if cur_applied_max is not None else (max_applied or 0)
        days_min = cur_days_min if cur_days_min is not None else 0
        days_max = cur_days_max if cur_days_max is not None else min(14, max_days or 0)
        selected = default_selected if not current_cols else current_cols
        return options, selected, applied_min, applied_max, days_min, days_max

    @app.callback(
        Output("job-grid", "columnDefs"),
        Output("job-grid", "rowData"),
        Output("chart-role", "figure"),
        Output("chart-domain", "figure"),
        Output("job-count", "children"),
        Output("job-grid", "dashGridOptions", allow_duplicate=True),
        Output("download-excel", "data", allow_duplicate=True),
        Input("store-data", "data"),
        Input("store-emb", "data"),
        Input("title-dd", "value"),
        Input("location-dd", "value"),
        Input("apply-type", "value"),
        Input("contract-type", "value"),
        Input("work-type", "value"),
        Input("label-dd-filter", "value"),
        Input("label-dd-exclude", "value"),
        Input("city-dd", "value"),
        Input("domain-cluster-dd", "value"),
        Input("role-cluster-dd", "value"),
        Input("row-lines", "value"),
        Input("applied-min", "value"),
        Input("applied-max", "value"),
        Input("days-min", "value"),
        Input("days-max", "value"),
        Input("column-toggles", "value"),
        Input("search-query", "value"),
        Input("semantic-query", "value"),
        Input("search-fields", "value"),
        Input("filter-isnew", "value"),
        Input("show-selected-only", "value"),
        Input("chart-k", "value"),
        Input("export-excel", "n_clicks"),
        State("store-selected", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _apply_filters(
        data, emb, titles, locs, app_types, contract_types, work_types,
        label_filter, label_exclude, cities, domain_clusters, role_clusters,
        row_lines, applied_min, applied_max, days_min, days_max, cols,
        search_query, semantic_query, search_fields, only_new, show_selected_only,
        chart_k, export_clicks, selected_ids_store
    ):
        if not data:
            empty_fig = go.Figure()
            default_grid_opts = {
                "rowSelection": "multiple",
                "animateRows": False,
                "rowHeight": 40,
                "tooltipShowDelay": 0,
                "suppressScrollOnNewData": True,
            }
            return [], [], empty_fig, empty_fig, "Jobs: 0", default_grid_opts, no_update

        _err: List[str] = []
        filtered = _filter_jobs(
            data, emb, group="full_text", facet_values={}, hide_disq=False,
            titles=titles or [], locs=locs or [], app_types=app_types or [],
            contract_types=contract_types or [], work_types=work_types or [],
            applied_rng=[(applied_min or 0), (applied_max if applied_max is not None else 10**9)],
            days_rng=[(days_min or 0), (days_max if days_max is not None else 10**9)],
            search_query=search_query, search_fields=search_fields or [],
            search_error_out=_err, label_filter=label_filter or [],
            label_exclude=label_exclude or [],
        )

        if only_new:
            filtered = [v for v in filtered if (v.get("isNew") is True or v.get("isNew") == 1)]
        if domain_clusters:
            filtered = [
                v for v in filtered
                if (v.get("domain_cluster") or (v.get("meta") or {}).get("domain_cluster")) in set(domain_clusters)
            ]
        if role_clusters:
            filtered = [
                v for v in filtered
                if (v.get("role_cluster") or (v.get("meta") or {}).get("role_cluster")) in set(role_clusters)
            ]

        highlight_terms = _extract_search_terms(search_query)

        base_cols = [
            {
                "field": "logo_md",
                "headerName": "",
                "pinned": "left",
                "minWidth": 64,
                "width": 64,
                "resizable": True,
                "sortable": False,
                "filter": False,
                "cellRenderer": "markdown",
                "cellClass": "logo-cell",
            },
        ]
        optional = []
        if cols:
            if "title" in cols:
                optional.append({"field": "title", "headerName": "Title", "minWidth": 140, "width": 180})
            if "about_company" in cols:
                optional.append({
                    "field": "about_company",
                    "headerName": "About Company",
                    "minWidth": 240,
                    "width": 340,
                    "cellStyle": {"whiteSpace": "normal", "overflow": "hidden"},
                    "cellClass": "wrap-text",
                    "cellRenderer": "markdown",
                })
            if "job_summary" in cols:
                optional.append({
                    "field": "job_summary",
                    "headerName": "Job Summary",
                    "minWidth": 360,
                    "width": 540,
                    "cellStyle": {"whiteSpace": "normal", "overflow": "hidden"},
                    "cellClass": "wrap-text",
                    "cellRenderer": "markdown",
                })
            if "company" in cols:
                optional.append({"field": "company", "headerName": "Company", "minWidth": 100, "width": 100})
            if "city" in cols:
                optional.append({"field": "city", "headerName": "City", "minWidth": 100, "width": 100})
            if "job_label" in cols:
                optional.append({"field": "job_label", "headerName": "Labels", "minWidth": 140})
            if "domain_cluster" in cols:
                optional.append({"field": "domain_cluster", "headerName": "Domain Cluster", "minWidth": 140})
            if "role_cluster" in cols:
                optional.append({"field": "role_cluster", "headerName": "Role Cluster", "minWidth": 140})
            if "location" in cols:
                optional.append({"field": "location", "headerName": "Location", "minWidth": 100, "width": 100})
            if "url" in cols:
                optional.append({"field": "url", "headerName": "URL", "minWidth": 220})
            if "days_old" in cols:
                optional.append({"field": "days_old", "headerName": "Days", "minWidth": 80})
            if "applied_times" in cols:
                optional.append({"field": "applied_times", "headerName": "Applied", "minWidth": 90, "sort": "asc"})

        col_defs = base_cols + optional

        row_data = []
        selected_ids: set = set(selected_ids_store or []) if show_selected_only else set()
        for v in filtered:
            row = {k: v.get(k) for k in {"title", "company", "url", "location", "applied_times", "days_old", "job_label"}}
            row["job_id"] = v.get("job_id")
            thumb = v.get("logo_thumb_data_uri")
            url = v.get("logo_url")
            img_src = thumb or url
            row["logo_md"] = f"![]({img_src})" if img_src else ""
            row["city"] = v.get("city") or (v.get("meta") or {}).get("city")
            row["domain_cluster"] = v.get("domain_cluster") or (v.get("meta") or {}).get("domain_cluster")
            row["role_cluster"] = v.get("role_cluster") or (v.get("meta") or {}).get("role_cluster")
            about_txt = v.get("LLMaboutCompany")
            if highlight_terms:
                about_txt = _highlight_terms_code(about_txt or "", highlight_terms)
            row["about_company"] = about_txt
            job_sum_txt = v.get("LLMjobSummary")
            if highlight_terms:
                job_sum_txt = _highlight_terms_code(job_sum_txt or "", highlight_terms)
            row["job_summary"] = job_sum_txt
            row_data.append(row)

        # Semantic rerank
        if semantic_query and isinstance(semantic_query, str) and semantic_query.strip():
            try:
                from .. import rag as _rag
                if not _rag.is_loaded():
                    _rag.load_cache()
                candidate_ids = set([r.get("job_id") for r in row_data if r.get("job_id")])
                if candidate_ids:
                    k_candidates = min(2000, max(400, len(candidate_ids) * 10))
                    chs = _rag.retrieve_filtered(
                        semantic_query,
                        where=lambda c: getattr(c, "job_key", None) in candidate_ids,
                        k=k_candidates,
                        oversample=max(800, k_candidates),
                        alpha=0.55,
                        require_phrase=False,
                    )
                    job_score: Dict[str, float] = {}
                    for rank, c in enumerate(chs, start=1):
                        jk = getattr(c, "job_key", None)
                        if not jk:
                            continue
                        score = 1.0 / (1.0 + rank)
                        if jk not in job_score or score > job_score[jk]:
                            job_score[jk] = score
                    for r in row_data:
                        jid = r.get("job_id")
                        r["semantic_score"] = float(job_score.get(jid, 0.0)) if jid else 0.0
                    row_data.sort(key=lambda r: -(r.get("semantic_score") or 0.0))
            except Exception:
                pass

        jids_all = [v.get("job_id") for v in filtered]
        info_map_all = {v.get("job_id"): v for v in filtered}

        if show_selected_only and selected_ids:
            row_data = [r for r in row_data if r.get("job_id") in selected_ids]

        k_override = None
        try:
            if chart_k is not None:
                k_override = int(chart_k)
        except Exception:
            k_override = None

        role_fig = _make_scatter_for("role", emb or {}, jids_all, info_map_all, k_override)
        domain_fig = _make_scatter_for("domain", emb or {}, jids_all, info_map_all, k_override)
        count_text = f"Jobs: {len(row_data)}"

        try:
            lines = int(row_lines or 1)
        except Exception:
            lines = 1
        lines = max(1, min(3, lines))
        row_height = 40 * (lines * 2)
        grid_opts = {
            "rowSelection": "multiple",
            "animateRows": False,
            "rowHeight": row_height,
            "tooltipShowDelay": 0,
            "suppressScrollOnNewData": True,
        }

        download = no_update
        trig = getattr(callback_context, "triggered", [])
        trig_id = trig[0]["prop_id"].split(".")[0] if trig else None
        if trig_id == "export-excel":
            try:
                df = pd.DataFrame(row_data)
                visible_fields = [c.get("field") for c in (col_defs or []) if isinstance(c, dict) and c.get("field")]
                cols_present = [c for c in visible_fields if c in df.columns]
                if cols_present:
                    df = df[cols_present]
                download = dcc.send_data_frame(df.to_excel, "jobs_export.xlsx", sheet_name="Jobs", index=False)
            except Exception:
                download = no_update

        return col_defs, row_data, role_fig, domain_fig, count_text, grid_opts, download

    @app.callback(
        Output("job-grid", "selectedRows", allow_duplicate=True),
        Input("chart-role", "clickData"),
        Input("chart-domain", "clickData"),
        Input("clear-selection", "n_clicks"),
        Input("clear-selection-clusters", "n_clicks"),
        State("job-grid", "rowData"),
        State("job-grid", "selectedRows"),
        State("chart-role", "figure"),
        State("chart-domain", "figure"),
        prevent_initial_call=True,
    )
    def _click_chart_to_grid(c1, c2, n_clear_table, n_clear_clusters, rows, selected, f_role, f_domain):
        trig = getattr(callback_context, "triggered_id", None)
        if trig in ("clear-selection", "clear-selection-clusters"):
            return []
        if not trig:
            return no_update
        if trig == "chart-role":
            cd = c1
        elif trig == "chart-domain":
            cd = c2
        else:
            return no_update
        if not (cd and rows):
            return no_update
        p = (cd.get("points") or [{}])[0]
        jid = p.get("customdata") or p.get("text")
        if not jid:
            return no_update
        row_map = {r.get("job_id"): r for r in (rows or [])}
        if jid not in row_map:
            return no_update
        selected = selected or []
        existing_ids = {r.get("job_id") for r in selected}
        if jid in existing_ids:
            new_sel = [r for r in selected if r.get("job_id") != jid]
        else:
            new_sel = selected + [row_map[jid]]
        return new_sel

    @app.callback(
        Output("chart-role", "figure", allow_duplicate=True),
        Output("chart-domain", "figure", allow_duplicate=True),
        Input("job-grid", "selectedRows"),
        State("chart-role", "figure"),
        State("chart-domain", "figure"),
        prevent_initial_call=True,
    )
    def _highlight_selection(sel, fig_role, fig_domain):
        def apply_selectedpoints(fig):
            if not fig:
                return fig
            if hasattr(fig, "to_plotly_json"):
                fig = fig.to_plotly_json()
            ids = {r.get("job_id") for r in (sel or []) if r.get("job_id")}
            data = fig.get("data", [])
            for tr in data:
                if tr.get("type") not in ("scattergl", "scatter"):
                    continue
                cds = tr.get("customdata") or []
                idxs = [i for i, jid in enumerate(cds) if jid in ids]
                tr["selectedpoints"] = idxs if idxs else None
                tr["selected"] = {"marker": {"size": 16, "line": {"color": "black", "width": 6}}}
                tr["unselected"] = {"marker": {"opacity": 0.8}}
            fig["data"] = data
            return fig
        return apply_selectedpoints(fig_role or {}), apply_selectedpoints(fig_domain or {})

    @app.callback(
        Output("store-selected", "data"),
        Input("job-grid", "selectedRows"),
    )
    def _persist_selection(sel):
        if not sel:
            return no_update
        return [r.get("job_id") for r in sel if r.get("job_id")]

    @app.callback(
        Output("job-grid", "selectedRows", allow_duplicate=True),
        Input("store-data", "data"),
        State("store-selected", "data"),
        State("job-grid", "rowData"),
        prevent_initial_call=True,
    )
    def _restore_selection(_data, sel_ids, rows):
        if not rows or not sel_ids:
            return no_update
        idset = set(sel_ids)
        restored = [r for r in rows if r.get("job_id") in idset]
        return restored

    @app.callback(
        Output("job-grid", "selectedRows", allow_duplicate=True),
        Input("job-grid", "rowData"),
        State("store-selected", "data"),
        prevent_initial_call=True,
    )
    def _restore_on_rowdata(rows, sel_ids):
        if not rows or not sel_ids:
            return no_update
        idset = set(sel_ids)
        restored = [r for r in rows if r.get("job_id") in idset]
        return restored

    @app.callback(
        Output("hovered-job", "data"),
        Input("job-grid", "virtualRowData"),
        prevent_initial_call=True,
    )
    def _grid_hover(_):
        return None

    @app.callback(
        Output("job-meta", "children"),
        Output("company-about", "children"),
        Output("company-job-summary", "children"),
        Output("job-facets", "children"),
        Output("job-desc", "children"),
        Output("job-label-dd", "value"),
        Output("job-meta", "style"),
        Output("job-facets", "style"),
        Output("job-desc", "style"),
        Input("job-grid", "selectedRows"),
        Input("store-data", "data"),
        Input("search-query", "value"),
        prevent_initial_call=True,
    )
    def _details_from_selection(sel, data, search_query):
        if not sel:
            return "", "", "", "", "", [], {}, {}, {}
        if len(sel) > 1:
            return "", "", "", "", "", [], {"display": "none"}, {"display": "none"}, {"display": "none"}
        row = sel[0]
        jid = row.get("job_id")
        v = data.get(jid) if data else None
        if not v:
            return "", "", "", "", "", [], {}, {}, {}
        meta = [
            html.Div([html.B("Title:"), f" {v.get('title', '')}"]),
            html.Div([
                html.B("Company:"),
                f" {v.get('company', '')}",
                html.Img(src=v.get("logo_url") or "", style={"height": "24px", "marginLeft": "8px"}) if v.get("logo_url") else html.Span(),
            ]),
            html.Div([html.B("URL:"), html.A(v.get("url", ""), href=v.get("url", ""), target="_blank")]),
            html.Div([html.B("Location:"), f" {v.get('location', '')}"]),
            html.Div([html.B("Applied Times:"), f" {v.get('applied_times', '')}"]),
            html.Div([html.B("Days Old:"), f" {v.get('days_old', '')}"]),
        ]
        facets_map = v.get("facets") or {}
        facet_blocks = []
        for fslug, fval in facets_map.items():
            types = ", ".join(fval.get("types") or [])
            facet_blocks.append(html.Div([html.B(fslug + ":"), f" {types}"]))
        highlight_terms = _extract_search_terms(search_query)
        desc_text = (v.get("description") or "").replace("\r\n", "\n")
        desc_text = re.sub(r"\n\s*[\*•·]\s+", "\n- ", desc_text)
        desc_text = re.sub(r"(\S)(\n)([A-Z][A-Za-z ]+:)", r"\1\n\n\3", desc_text)
        if highlight_terms:
            desc_text = _highlight_terms_markdown(desc_text, highlight_terms)
        desc_node = dcc.Markdown(desc_text, link_target="_blank", dangerously_allow_html=True)
        labels_val = list(v.get("job_label") or [])
        about = (v.get("LLMaboutCompany") or "").strip()
        if highlight_terms and about:
            about = _highlight_terms_markdown(about, highlight_terms)
        about_node = dcc.Markdown(about, dangerously_allow_html=True) if about else ""
        job_sum = (v.get("LLMjobSummary") or "").strip()
        sum_node = ""
        if job_sum:
            js_text = job_sum.replace("\r\n", "\n")
            js_text = re.sub(r"\n\s*[\*•·]\s+", "\n- ", js_text)
            js_text = re.sub(r"(\S)(\n)([A-Z][A-Za-z ]+:)", r"\1\n\n\3", js_text)
            if highlight_terms:
                js_text = _highlight_terms_markdown(js_text, highlight_terms)
            sum_node = dcc.Markdown(js_text, link_target="_blank", dangerously_allow_html=True)
        return meta, about_node, sum_node, facet_blocks, desc_node, labels_val, {}, {}, {}

    @app.callback(Output("job-label-dd", "options"), Input("store-data", "data"))
    def _label_suggestions(data):
        tags = set()
        for v in (data or {}).values():
            for t in v.get("job_label") or []:
                if t:
                    tags.add(str(t))
        return [{"label": t, "value": t} for t in sorted(tags)]

    @app.callback(
        Output("save-status", "children"),
        Output("job-label-new", "value"),
        Output("store-data", "data", allow_duplicate=True),
        Input("save-job-label", "n_clicks"),
        State("job-grid", "selectedRows"),
        State("job-label-dd", "value"),
        State("job-label-new", "value"),
        State("store-data", "data"),
        prevent_initial_call=True,
    )
    def _save_labels(n, sel, dd_values, new_values, data_store):
        if not n or not sel:
            return "", None, data_store
        dd_labels = [str(t).strip() for t in (dd_values or []) if str(t).strip()]
        extra = []
        if new_values:
            for part in str(new_values).split(","):
                t = part.strip()
                if t:
                    extra.append(t)
        labels = sorted(set(dd_labels + extra))
        if isinstance(data_store, dict):
            for row in sel:
                jid = row.get("job_id")
                if not jid:
                    continue
                if jid in data_store:
                    data_store[jid]["job_label"] = list(labels)
                _append_change_log({"job_id": jid, "op": "set", "fields": {"job_label": list(labels)}})
        paths = _offline_paths()
        _atomic_write_json(paths["store"], data_store)
        return "Saved", None, data_store

    @app.callback(
        Output("save-status", "children", allow_duplicate=True),
        Output("store-data", "data", allow_duplicate=True),
        Input("save-job-label-yes", "n_clicks"),
        Input("save-job-label-no", "n_clicks"),
        State("job-grid", "selectedRows"),
        State("store-data", "data"),
        prevent_initial_call=True,
    )
    def _quick_assign(yes_clicks, no_clicks, sel, data_store):
        trig = getattr(callback_context, "triggered", [])
        trig_id = trig[0]["prop_id"].split(".")[0] if trig else None
        if not sel or not data_store or trig_id not in ("save-job-label-yes", "save-job-label-no"):
            return no_update, no_update
        label = "yes" if trig_id == "save-job-label-yes" else "no"
        for row in sel:
            jid = row.get("job_id")
            if not jid or jid not in data_store:
                continue
            cur = list(data_store[jid].get("job_label") or [])
            if label not in cur:
                cur.append(label)
            data_store[jid]["job_label"] = sorted(set(cur))
            _append_change_log({"job_id": jid, "op": "set", "fields": {"job_label": list(data_store[jid]["job_label"])}})
        paths = _offline_paths()
        _atomic_write_json(paths["store"], data_store)
        return f"Assigned '{label}' to {len(sel)} job(s)", data_store

    @app.callback(
        Output("store-data", "data", allow_duplicate=True),
        Output("save-status", "children", allow_duplicate=True),
        Input("fetch-logos", "n_clicks"),
        Input("sync-now", "n_clicks"),
        Input("llm-about-company", "n_clicks"),
        Input("llm-summarise-job", "n_clicks"),
        State("job-grid", "selectedRows"),
        State("store-data", "data"),
        prevent_initial_call=True,
    )
    def _fetch_and_save_logos(n_fetch, n_sync, n_llm_about, n_llm_sum, selected, data_store):
        trig = getattr(callback_context, "triggered", [])
        trig_id = trig[0]["prop_id"].split(".")[0] if trig else None

        if trig_id == "sync-now":
            # Save changes locally
            if isinstance(data_store, dict):
                paths = _offline_paths()
                _atomic_write_json(paths["store"], data_store)
            return data_store, f"Saved {len(data_store or {})} jobs locally"

        if trig_id == "llm-about-company":
            if not selected or not isinstance(data_store, dict):
                return data_store, "Select at least one row"
            if not os.environ.get("OPENAI_API_KEY"):
                return data_store, "OPENAI_API_KEY not set"
            if call_llm_json is None:
                return data_store, "LLM module not available"

            class CompanyAbout(BaseModel):
                summary: str

            prompt_tpl = (
                "Briefly explain what this company does: what is its domain or industry, what is its specialization, "
                "what is its main product, service or offering. Keep it as short as possible; no extra info. "
                "Do not mention company name. Use markdown as a mini cheat sheet, one item per line.\n\n"
                "Company info:\n{company_url}"
            )
            updated = 0
            failed = 0
            local_store = data_store
            for row in selected or []:
                jid = row.get("job_id")
                v = local_store.get(jid) if jid else None
                if not v:
                    continue
                company_url = v.get("companyUrl") or v.get("url") or v.get("job_id") or ""
                if not company_url:
                    failed += 1
                    continue
                user_prompt = prompt_tpl.format(company_url=company_url)
                messages = [
                    {"role": "system", "content": 'Return strict JSON with one key: {"summary": string}.'},
                    {"role": "user", "content": user_prompt},
                ]
                try:
                    res = call_llm_json(messages, CompanyAbout)
                    summary = (res.summary or "").strip()
                    if summary:
                        local_store[jid]["LLMaboutCompany"] = summary
                        _append_change_log({"job_id": jid, "op": "set", "fields": {"LLMaboutCompany": summary}})
                        updated += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
            paths = _offline_paths()
            _atomic_write_json(paths["store"], local_store)
            return local_store, f"LLM updated: {updated} ok, {failed} failed"

        if trig_id == "llm-summarise-job":
            if not selected or not isinstance(data_store, dict):
                return data_store, "Select at least one row"
            if not os.environ.get("OPENAI_API_KEY"):
                return data_store, "OPENAI_API_KEY not set"
            if call_llm_json is None:
                return data_store, "LLM module not available"

            class JobSummary(BaseModel):
                summary: str

            prompt_tpl = (
                "Briefly summarise this job description.\n"
                'Return STRICT JSON with one key only: {{"summary": string}}.\n'
                "Put your result in summary as MARKDOWN numbered list with bold labels:\n"
                "1. **Main Responsibilities**: ...\n"
                "2. **Domain/Industry Focus**: ...\n"
                "3. **Key Skills/Qualifications**: ...\n"
                "Keep it concise.\n\nJob description:\n{desc}"
            )
            updated = 0
            failed = 0
            local_store = data_store
            for row in selected or []:
                jid = row.get("job_id")
                v = local_store.get(jid) if jid else None
                if not v:
                    continue
                desc = (v.get("description") or "").strip()
                if not desc:
                    failed += 1
                    continue
                user_prompt = prompt_tpl.format(desc=desc[:6000])
                messages = [
                    {"role": "system", "content": 'Return strict JSON with one key: {"summary": string}.'},
                    {"role": "user", "content": user_prompt},
                ]
                try:
                    res = call_llm_json(messages, JobSummary)
                    summary = (res.summary or "").strip()
                    if summary:
                        local_store[jid]["LLMjobSummary"] = summary
                        _append_change_log({"job_id": jid, "op": "set", "fields": {"LLMjobSummary": summary}})
                        updated += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
            paths = _offline_paths()
            _atomic_write_json(paths["store"], local_store)
            return local_store, f"Summaries: {updated} ok, {failed} failed"

        # Handle fetch logos
        if not n_fetch or not selected or not isinstance(data_store, dict):
            return data_store, no_update
        updated = 0
        for row in selected:
            jid = row.get("job_id")
            v = data_store.get(jid) if jid else None
            if not v:
                continue
            company = v.get("company") or ""
            if not v.get("logo_url") and company:
                url = _find_company_logo_url(company)
                if url:
                    thumb = _make_logo_thumbnail_data_uri(url, size=26)
                    data_store[jid]["logo_url"] = url
                    if thumb:
                        data_store[jid]["logo_thumb_data_uri"] = thumb
                    fields = {"logo_url": url}
                    if thumb:
                        fields["logo_thumb_data_uri"] = thumb
                    _append_change_log({"job_id": jid, "op": "set", "fields": fields})
                    updated += 1
        paths = _offline_paths()
        _atomic_write_json(paths["store"], data_store)
        return data_store, f"Fetched logos: {updated}" if updated else "No logos fetched"

    @app.callback(
        Output("job-grid", "selectedRows", allow_duplicate=True),
        Output("llmsel-output", "children", allow_duplicate=True),
        Output("llmsel-status", "children", allow_duplicate=True),
        Input("llmsel-run", "n_clicks"),
        State("llmsel-prompt", "value"),
        State("llmsel-batch", "value"),
        State("job-grid", "selectedRows"),
        State("store-data", "data"),
        prevent_initial_call=True,
    )
    def _llm_selector(n, prompt, batch_size, selected_rows, data_store):
        if not n:
            return no_update, no_update, no_update
        prompt = (prompt or "").strip()
        if not prompt:
            return no_update, "", "Enter a prompt first."
        sel = list(selected_rows or [])
        if not sel:
            return no_update, "", "Select at least one row in the Table first."
        if not os.environ.get("OPENAI_API_KEY"):
            return no_update, "", "OPENAI_API_KEY not set"
        if call_llm_json is None:
            return no_update, "", "LLM module not available"
        try:
            bs = int(batch_size or 30)
        except Exception:
            bs = 30
        bs = max(5, min(200, bs))

        jobs = []
        for r in sel:
            jid = r.get("job_id")
            v = (data_store or {}).get(jid)
            if not jid or not v:
                continue
            jobs.append({
                "job_id": jid,
                "title": v.get("title") or "",
                "company": v.get("company") or "",
                "about": v.get("LLMaboutCompany") or "",
                "summary": v.get("LLMjobSummary") or "",
                "url": v.get("url") or jid,
            })
        if not jobs:
            return no_update, "", "No valid selected jobs found."

        class LLMSelectOut(BaseModel):
            selected_job_ids: List[str]
            explanation: str

        out_ids: List[str] = []
        explanations: List[str] = []

        def fmt_job(j: Dict[str, Any]) -> str:
            def cut(s: str, n: int = 900) -> str:
                s2 = s or ""
                return s2 if len(s2) <= n else (s2[:n] + "…")
            return (
                f"- id: {j['job_id']}\n"
                f"  Title: {j['title']}\n"
                f"  Company: {j['company']}\n"
                f"  About: {cut(j['about'], 700)}\n"
                f"  Summary: {cut(j['summary'], 1400)}\n"
                f"  URL: {j['url']}\n"
            )

        i = 0
        total = len(jobs)
        while i < total:
            batch = jobs[i:i + bs]
            user = (
                "You are selecting a subset of the provided jobs based on the instruction.\n"
                "Return STRICT JSON: {\"selected_job_ids\": string[], \"explanation\": string}.\n\n"
                f"Instruction: {prompt}\n\n"
                "Jobs:\n" + "\n".join(fmt_job(j) for j in batch)
            )
            msgs = [
                {"role": "system", "content": "Return strict JSON only with keys selected_job_ids (string[]) and explanation (string)."},
                {"role": "user", "content": user},
            ]
            try:
                res = call_llm_json(msgs, LLMSelectOut)
                ids = [str(x) for x in (res.selected_job_ids or [])]
                out_ids.extend(ids)
                if res.explanation:
                    explanations.append(res.explanation)
            except Exception as e:
                explanations.append(f"[Batch {i // bs + 1}] LLM error: {type(e).__name__}: {e}")
            i += len(batch)

        idset = []
        seen = set()
        for jid in out_ids:
            if jid and jid not in seen:
                seen.add(jid)
                idset.append(jid)
        selected_rows_new = [
            {"job_id": jid, "title": (data_store or {}).get(jid, {}).get("title"), "company": (data_store or {}).get(jid, {}).get("company")}
            for jid in idset if (data_store or {}).get(jid)
        ]
        md = "\n\n".join([f"### Batch {idx + 1}\n\n" + exp for idx, exp in enumerate(explanations) if exp])
        status = f"Selected {len(selected_rows_new)} jobs from {len(jobs)} input."
        return selected_rows_new, md, status

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 8050)))
