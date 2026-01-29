import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { api } from '../api/client';

const LABELS_KEY = 'bjs_job_labels';

export interface JobLabels {
  [jobId: string]: string[];
}

function loadLabelsFromStorage(): JobLabels {
  try {
    const stored = localStorage.getItem(LABELS_KEY);
    return stored ? JSON.parse(stored) : {};
  } catch {
    return {};
  }
}

function saveLabelsToStorage(labels: JobLabels): void {
  localStorage.setItem(LABELS_KEY, JSON.stringify(labels));
}

export function useLabels() {
  const [jobLabels, setJobLabels] = useState<JobLabels>(loadLabelsFromStorage);
  const [loading, setLoading] = useState(true);
  const initializedRef = useRef(false);

  // Load labels from API on mount
  useEffect(() => {
    if (initializedRef.current) return;
    initializedRef.current = true;

    api.getLabels()
      .then(data => {
        // Merge API labels with localStorage (API takes precedence)
        const localLabels = loadLabelsFromStorage();
        const mergedLabels = { ...localLabels, ...data.labels };
        setJobLabels(mergedLabels);
        saveLabelsToStorage(mergedLabels);
      })
      .catch(err => {
        console.warn('Failed to load labels from API, using localStorage:', err);
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  // Get all unique labels across all jobs
  const allLabels = useMemo(() => {
    const labels = new Set<string>();
    Object.values(jobLabels).forEach(arr => arr.forEach(l => labels.add(l)));
    return Array.from(labels).sort();
  }, [jobLabels]);

  // Add a label to a job
  const addLabel = useCallback((jobId: string, label: string) => {
    setJobLabels(prev => {
      const current = prev[jobId] || [];
      if (current.includes(label)) return prev;
      const updated = { ...prev, [jobId]: [...current, label] };
      // Save to localStorage immediately
      saveLabelsToStorage(updated);
      return updated;
    });

    // Sync to API (fire and forget)
    api.addLabel(jobId, label).catch(err => {
      console.warn('Failed to sync label to API:', err);
    });
  }, []);

  // Remove a label from a job
  const removeLabel = useCallback((jobId: string, label: string) => {
    setJobLabels(prev => {
      const current = prev[jobId] || [];
      const updatedList = current.filter(l => l !== label);
      let updated: JobLabels;
      if (updatedList.length === 0) {
        const { [jobId]: _, ...rest } = prev;
        updated = rest;
      } else {
        updated = { ...prev, [jobId]: updatedList };
      }
      // Save to localStorage immediately
      saveLabelsToStorage(updated);
      return updated;
    });

    // Sync to API (fire and forget)
    api.removeLabel(jobId, label).catch(err => {
      console.warn('Failed to sync label removal to API:', err);
    });
  }, []);

  // Get labels for a specific job
  const getLabels = useCallback((jobId: string): string[] => {
    return jobLabels[jobId] || [];
  }, [jobLabels]);

  // Get jobs with a specific label
  const getJobsWithLabel = useCallback((label: string): string[] => {
    return Object.entries(jobLabels)
      .filter(([_, labels]) => labels.includes(label))
      .map(([jobId]) => jobId);
  }, [jobLabels]);

  // Check if all given jobs have a specific label
  const allJobsHaveLabel = useCallback((jobIds: Set<string>, label: string): boolean => {
    return Array.from(jobIds).every(id => (jobLabels[id] || []).includes(label));
  }, [jobLabels]);

  // Check if any of the given jobs have a specific label
  const anyJobHasLabel = useCallback((jobIds: Set<string>, label: string): boolean => {
    return Array.from(jobIds).some(id => (jobLabels[id] || []).includes(label));
  }, [jobLabels]);

  return {
    jobLabels,
    allLabels,
    loading,
    addLabel,
    removeLabel,
    getLabels,
    getJobsWithLabel,
    allJobsHaveLabel,
    anyJobHasLabel,
  };
}
