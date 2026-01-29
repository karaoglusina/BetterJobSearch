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
  language?: string;
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
  cluster_keywords?: string[];
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

  listJobs: (params?: { skip?: number; limit?: number; location?: string; company?: string; title_contains?: string; language?: string }) => {
    const searchParams = new URLSearchParams();
    if (params?.skip) searchParams.set('skip', String(params.skip));
    if (params?.limit) searchParams.set('limit', String(params.limit));
    if (params?.location) searchParams.set('location', params.location);
    if (params?.company) searchParams.set('company', params.company);
    if (params?.title_contains) searchParams.set('title_contains', params.title_contains);
    if (params?.language) searchParams.set('language', params.language);
    const qs = searchParams.toString();
    return fetchJson<{ total: number; jobs: Job[] }>(`/jobs${qs ? `?${qs}` : ''}`);
  },

  getLanguages: () => fetchJson<{ languages: { code: string; count: number }[] }>('/languages'),

  getJob: (jobId: string) => fetchJson<Job & { full_text: string; chunks: any[]; sections: string[] }>(`/jobs/${encodeURIComponent(jobId)}`),

  getJobsBatch: (jobIds: string[]) =>
    fetchJson<{ total: number; jobs: Job[] }>('/jobs/batch', {
      method: 'POST',
      body: JSON.stringify({ job_ids: jobIds }),
    }),

  search: (query: string, k = 8, alpha = 0.55) =>
    fetchJson<{ query: string; total_results: number; results: SearchResult[] }>('/search', {
      method: 'POST',
      body: JSON.stringify({ query, k, alpha }),
    }),

  keywordSearch: (query: string, fields: string[] = ['title', 'description'], limit = 50) =>
    fetchJson<{ query: string; total_results: number; results: SearchResult[]; highlight_terms: string[] }>('/search/keyword', {
      method: 'POST',
      body: JSON.stringify({ query, fields, limit }),
    }),

  getClusters: (aspect = 'default', params?: {
    n_neighbors?: number;
    min_dist?: number;
    min_cluster_size?: number;
    tfidf_min_df?: number;
    tfidf_max_df?: number;
    npmi_min?: number;
    npmi_max?: number;
    effect_size_min?: number;
    effect_size_max?: number;
  }) => {
    const searchParams = new URLSearchParams();
    if (params?.n_neighbors !== undefined) searchParams.set('n_neighbors', String(params.n_neighbors));
    if (params?.min_dist !== undefined) searchParams.set('min_dist', String(params.min_dist));
    if (params?.min_cluster_size !== undefined) searchParams.set('min_cluster_size', String(params.min_cluster_size));
    if (params?.tfidf_min_df !== undefined) searchParams.set('tfidf_min_df', String(params.tfidf_min_df));
    if (params?.tfidf_max_df !== undefined) searchParams.set('tfidf_max_df', String(params.tfidf_max_df));
    if (params?.npmi_min !== undefined) searchParams.set('npmi_min', String(params.npmi_min));
    if (params?.npmi_max !== undefined) searchParams.set('npmi_max', String(params.npmi_max));
    if (params?.effect_size_min !== undefined) searchParams.set('effect_size_min', String(params.effect_size_min));
    if (params?.effect_size_max !== undefined) searchParams.set('effect_size_max', String(params.effect_size_max));
    const qs = searchParams.toString();
    return fetchJson<{ aspect: string; n_jobs: number; data: ClusterPoint[] }>(`/clusters/${aspect}${qs ? `?${qs}` : ''}`);
  },

  clusterByConcept: (concept: string) =>
    fetchJson<{ concept: string; n_jobs: number; n_clusters: number; data: ClusterPoint[] }>('/clusters/concept', {
      method: 'POST',
      body: JSON.stringify({ concept }),
    }),

  getAspects: () => fetchJson<{ aspects: string[] }>('/aspects'),

  getAspectDistribution: (name: string) =>
    fetchJson<{ aspect: string; total_jobs: number; coverage: number; value_counts: Record<string, number> }>(`/aspects/${name}/distribution`),

  // Labels API
  getLabels: () =>
    fetchJson<{ labels: Record<string, string[]>; all_labels: string[]; n_labeled_jobs: number }>('/labels'),

  saveLabels: (labels: Record<string, string[]>) =>
    fetchJson<{ success: boolean; n_labeled_jobs: number }>('/labels', {
      method: 'PUT',
      body: JSON.stringify({ labels }),
    }),

  addLabel: (jobId: string, label: string) =>
    fetchJson<{ success: boolean; job_id: string; labels: string[] }>('/labels/add', {
      method: 'POST',
      body: JSON.stringify({ job_id: jobId, label }),
    }),

  removeLabel: (jobId: string, label: string) =>
    fetchJson<{ success: boolean; job_id: string; labels: string[] }>('/labels/remove', {
      method: 'POST',
      body: JSON.stringify({ job_id: jobId, label }),
    }),
};
