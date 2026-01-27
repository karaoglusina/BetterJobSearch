import { useCallback, useEffect, useState } from 'react';
import { api, Job, SearchResult } from '../api/client';

export function useJobs() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [totalJobs, setTotalJobs] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchJobs = useCallback(async (params?: { skip?: number; limit?: number; location?: string; company?: string; title_contains?: string }) => {
    setLoading(true);
    setError(null);
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

  useEffect(() => {
    fetchJobs({ limit: 50 });
  }, [fetchJobs]);

  return { jobs, totalJobs, loading, error, fetchJobs, searchJobs };
}
