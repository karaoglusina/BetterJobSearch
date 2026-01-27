const API_BASE = '/api';

export interface Job {
  job_id: string;
  title: string;
  company: string;
  location: string;
  url: string;
  days_old?: number;
  contract_type?: string;
  work_type?: string;
  salary?: string;
  n_chunks?: number;
}

export interface SearchResult {
  job_id: string;
  title: string;
  company: string;
  location: string;
  url: string;
  snippet: string;
  section?: string;
}

export interface ClusterPoint {
  job_id: string;
  title: string;
  company: string;
  x: number;
  y: number;
  cluster_id: number;
  cluster_label: string;
}

export interface HealthResponse {
  status: string;
  artifacts_loaded: boolean;
  n_chunks: number;
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`API error ${res.status}: ${error}`);
  }
  return res.json();
}

export const api = {
  health: () => fetchJson<HealthResponse>('/health'),

  listJobs: (params?: { skip?: number; limit?: number; location?: string; company?: string; title_contains?: string }) => {
    const searchParams = new URLSearchParams();
    if (params?.skip) searchParams.set('skip', String(params.skip));
    if (params?.limit) searchParams.set('limit', String(params.limit));
    if (params?.location) searchParams.set('location', params.location);
    if (params?.company) searchParams.set('company', params.company);
    if (params?.title_contains) searchParams.set('title_contains', params.title_contains);
    const qs = searchParams.toString();
    return fetchJson<{ total: number; jobs: Job[] }>(`/jobs${qs ? `?${qs}` : ''}`);
  },

  getJob: (jobId: string) => fetchJson<Job & { full_text: string; chunks: any[]; sections: string[] }>(`/jobs/${encodeURIComponent(jobId)}`),

  search: (query: string, k = 8, alpha = 0.55) =>
    fetchJson<{ query: string; total_results: number; results: SearchResult[] }>('/search', {
      method: 'POST',
      body: JSON.stringify({ query, k, alpha }),
    }),

  getClusters: (aspect = 'default') => fetchJson<{ aspect: string; n_jobs: number; data: ClusterPoint[] }>(`/clusters/${aspect}`),

  clusterByConcept: (concept: string) =>
    fetchJson<{ concept: string; n_jobs: number; n_clusters: number; data: ClusterPoint[] }>('/clusters/concept', {
      method: 'POST',
      body: JSON.stringify({ concept }),
    }),

  getAspects: () => fetchJson<{ aspects: string[] }>('/aspects'),

  getAspectDistribution: (name: string) =>
    fetchJson<{ aspect: string; total_jobs: number; coverage: number; value_counts: Record<string, number> }>(`/aspects/${name}/distribution`),
};
