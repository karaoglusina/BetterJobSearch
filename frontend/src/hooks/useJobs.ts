import { useCallback, useEffect, useState } from 'react';
import { api, Job, SearchResult } from '../api/client';

export type JobSource = 'default' | 'keyword' | 'agent' | 'filter';

export function useJobs() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [totalJobs, setTotalJobs] = useState(0);
  const [loading, setLoading] = useState(false);
  const [keywordSearchLoading, setKeywordSearchLoading] = useState(false);
  const [keywordSearchActive, setKeywordSearchActive] = useState(false);
  const [highlightTerms, setHighlightTerms] = useState<string[]>([]);
  const [jobSource, setJobSource] = useState<JobSource>('default');
  const [error, setError] = useState<string | null>(null);

  const fetchJobs = useCallback(async (params?: { skip?: number; limit?: number; location?: string; company?: string; title_contains?: string; language?: string }) => {
    setLoading(true);
    setError(null);
    setKeywordSearchActive(false);
    setHighlightTerms([]);
    setJobSource(params && Object.values(params).some(v => v !== undefined && v !== 50) ? 'filter' : 'default');
    try {
      const data = await api.listJobs(params);
      setJobs(data.jobs);
      setTotalJobs(data.total);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch jobs');
    } finally {
      setLoading(false);
    }
  }, []);

  const searchJobs = useCallback(async (query: string, k = 20): Promise<SearchResult[]> => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.search(query, k);
      return data.results;
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Search failed');
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  const keywordSearch = useCallback(async (query: string) => {
    setKeywordSearchLoading(true);
    setError(null);
    try {
      const data = await api.keywordSearch(query);
      const jobResults: Job[] = data.results.map((r) => ({
        job_id: r.job_id,
        title: r.title,
        company: r.company,
        location: r.location,
        url: r.url,
      }));
      setJobs(jobResults);
      setTotalJobs(data.total_results);
      setKeywordSearchActive(true);
      setHighlightTerms(data.highlight_terms || []);
      setJobSource('keyword');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Keyword search failed');
    } finally {
      setKeywordSearchLoading(false);
    }
  }, []);

  const setJobsFromExternal = useCallback((externalJobs: Job[], source: JobSource = 'agent') => {
    setJobs(externalJobs);
    setTotalJobs(externalJobs.length);
    setJobSource(source);
    setKeywordSearchActive(false);
    setHighlightTerms([]);
  }, []);

  useEffect(() => {
    fetchJobs({ limit: 50 });
  }, [fetchJobs]);

  return {
    jobs, totalJobs, loading, error, fetchJobs, searchJobs,
    keywordSearch, keywordSearchLoading, keywordSearchActive, highlightTerms,
    jobSource, setJobsFromExternal,
  };
}
