import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ClusterPoint } from '../api/client';

export interface ClusterParams {
  n_neighbors?: number;
  min_dist?: number;
  min_cluster_size?: number;
}

export function useClusters() {
  const [clusterData, setClusterData] = useState<ClusterPoint[]>([]);
  const [currentAspect, setCurrentAspect] = useState('default');
  const [clusterParams, setClusterParams] = useState<ClusterParams>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Track request generation to discard stale responses
  const requestGen = useRef(0);
  const initialFetched = useRef(false);

  const fetchClusters = useCallback(async (aspect = 'default', params?: ClusterParams) => {
    const gen = ++requestGen.current;
    setLoading(true);
    setError(null);
    setCurrentAspect(aspect);
    if (params) setClusterParams(params);
    try {
      const data = await api.getClusters(aspect, params);
      if (gen === requestGen.current) {
        setClusterData(data.data);
      }
    } catch (e) {
      if (gen === requestGen.current) {
        setError(e instanceof Error ? e.message : 'Failed to fetch clusters');
      }
    } finally {
      if (gen === requestGen.current) {
        setLoading(false);
      }
    }
  }, []);

  // Load default clusters on mount (only once)
  useEffect(() => {
    if (!initialFetched.current) {
      initialFetched.current = true;
      fetchClusters('default');
    }
  }, [fetchClusters]);

  const clusterByConcept = useCallback(async (concept: string) => {
    const gen = ++requestGen.current;
    setLoading(true);
    setError(null);
    setCurrentAspect(`concept: ${concept}`);
    try {
      const data = await api.clusterByConcept(concept);
      if (gen === requestGen.current) {
        setClusterData(data.data);
      }
    } catch (e) {
      if (gen === requestGen.current) {
        setError(e instanceof Error ? e.message : 'Concept clustering failed');
      }
    } finally {
      if (gen === requestGen.current) {
        setLoading(false);
      }
    }
  }, []);

  return { clusterData, currentAspect, clusterParams, loading, error, fetchClusters, clusterByConcept };
}
