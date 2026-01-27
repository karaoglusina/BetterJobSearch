import { useCallback, useState } from 'react';
import { api, ClusterPoint } from '../api/client';

export function useClusters() {
  const [clusterData, setClusterData] = useState<ClusterPoint[]>([]);
  const [currentAspect, setCurrentAspect] = useState('default');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchClusters = useCallback(async (aspect = 'default') => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getClusters(aspect);
      setClusterData(data.data);
      setCurrentAspect(aspect);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch clusters');
    } finally {
      setLoading(false);
    }
  }, []);

  const clusterByConcept = useCallback(async (concept: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.clusterByConcept(concept);
      setClusterData(data.data);
      setCurrentAspect(`concept: ${concept}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Concept clustering failed');
    } finally {
      setLoading(false);
    }
  }, []);

  return { clusterData, currentAspect, loading, error, fetchClusters, clusterByConcept };
}
