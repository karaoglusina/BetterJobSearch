import React, { useState, useCallback } from 'react';
import ScatterPlot from './components/ScatterPlot';
import JobTable from './components/JobTable';
import ChatPanel from './components/ChatPanel';
import AspectSelector from './components/AspectSelector';
import JobDetail from './components/JobDetail';
import FilterPanel from './components/FilterPanel';
import { useJobs } from './hooks/useJobs';
import { useClusters } from './hooks/useClusters';

type View = 'scatter' | 'table';

export default function App() {
  const { jobs, totalJobs, loading: jobsLoading, fetchJobs } = useJobs();
  const { clusterData, currentAspect, loading: clusterLoading, fetchClusters, clusterByConcept } = useClusters();
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [highlightedJobs, setHighlightedJobs] = useState<Set<string>>(new Set());
  const [view, setView] = useState<View>('scatter');

  const handleAspectSelect = useCallback((aspect: string) => {
    fetchClusters(aspect);
  }, [fetchClusters]);

  const handlePointClick = useCallback((jobId: string) => {
    setSelectedJobId(jobId);
    setHighlightedJobs(new Set([jobId]));
  }, []);

  const handleJobSelect = useCallback((jobId: string) => {
    setSelectedJobId(jobId);
    setHighlightedJobs(new Set([jobId]));
  }, []);

  const handleFilterChange = useCallback((filters: { location?: string; company?: string; title_contains?: string }) => {
    fetchJobs({ limit: 50, ...filters });
  }, [fetchJobs]);

  const handleCloseDetail = useCallback(() => {
    setSelectedJobId(null);
    setHighlightedJobs(new Set());
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Header */}
      <header style={{
        padding: '10px 20px', borderBottom: '1px solid var(--border)',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        background: 'var(--bg-secondary)',
      }}>
        <h1 style={{ fontSize: 18, fontWeight: 700, margin: 0 }}>BetterJobSearch</h1>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
          {totalJobs} jobs loaded
        </div>
      </header>

      {/* Main layout */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Left sidebar */}
        <aside style={{
          width: 220, borderRight: '1px solid var(--border)',
          background: 'var(--bg-secondary)', overflowY: 'auto',
          display: 'flex', flexDirection: 'column',
        }}>
          <AspectSelector
            currentAspect={currentAspect}
            onSelect={handleAspectSelect}
            onConceptSubmit={clusterByConcept}
            loading={clusterLoading}
          />
          <div style={{ borderTop: '1px solid var(--border)' }} />
          <FilterPanel onFilterChange={handleFilterChange} />
        </aside>

        {/* Center content */}
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* View toggle */}
          <div style={{
            padding: '8px 16px', borderBottom: '1px solid var(--border)',
            display: 'flex', gap: 8,
          }}>
            <button
              onClick={() => setView('scatter')}
              style={{
                padding: '4px 12px', borderRadius: 6, fontSize: 12,
                border: `1px solid ${view === 'scatter' ? 'var(--accent)' : 'var(--border)'}`,
                background: view === 'scatter' ? 'var(--accent)' : 'transparent',
                color: view === 'scatter' ? 'white' : 'var(--text-secondary)',
                cursor: 'pointer',
              }}
            >
              Clusters
            </button>
            <button
              onClick={() => setView('table')}
              style={{
                padding: '4px 12px', borderRadius: 6, fontSize: 12,
                border: `1px solid ${view === 'table' ? 'var(--accent)' : 'var(--border)'}`,
                background: view === 'table' ? 'var(--accent)' : 'transparent',
                color: view === 'table' ? 'white' : 'var(--text-secondary)',
                cursor: 'pointer',
              }}
            >
              Table
            </button>
            {clusterLoading && (
              <span style={{ color: 'var(--text-secondary)', fontSize: 12, marginLeft: 8, lineHeight: '28px' }}>
                Loading clusters...
              </span>
            )}
          </div>

          {/* Main view */}
          <div style={{ flex: 1, overflow: 'hidden' }}>
            {view === 'scatter' ? (
              <ScatterPlot
                data={clusterData}
                aspect={currentAspect}
                onPointClick={handlePointClick}
                highlightedJobs={highlightedJobs}
              />
            ) : (
              <JobTable
                jobs={jobs}
                onJobSelect={handleJobSelect}
                selectedJobId={selectedJobId}
              />
            )}
          </div>

          {/* Job detail panel */}
          <JobDetail jobId={selectedJobId} onClose={handleCloseDetail} />
        </main>

        {/* Right chat panel */}
        <aside style={{ width: 340 }}>
          <ChatPanel />
        </aside>
      </div>
    </div>
  );
}
