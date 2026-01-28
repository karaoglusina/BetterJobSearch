import { useState, useCallback, useMemo } from 'react';
import ScatterPlot from './components/ScatterPlot';
import JobTable from './components/JobTable';
import ChatPanel from './components/ChatPanel';
import AspectSelector from './components/AspectSelector';
import JobDetail from './components/JobDetail';
import FilterPanel from './components/FilterPanel';
import ClusterSettings from './components/ClusterSettings';
import PresetBar, { UIPreset } from './components/PresetBar';
import { useJobs } from './hooks/useJobs';
import { useClusters, ClusterParams } from './hooks/useClusters';

type View = 'scatter' | 'table';

export default function App() {
  const {
    jobs, totalJobs, fetchJobs,
    keywordSearch, keywordSearchLoading, keywordSearchActive,
    jobSource, setJobsFromExternal,
  } = useJobs();
  const { clusterData, currentAspect, loading: clusterLoading, fetchClusters, clusterByConcept } = useClusters();
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [highlightedJobs, setHighlightedJobs] = useState<Set<string>>(new Set());
  const [view, setView] = useState<View>('scatter');
  const [clusterFilterEnabled, setClusterFilterEnabled] = useState(true);

  const filterJobIds = useMemo(() => {
    if (!keywordSearchActive || !clusterFilterEnabled) return undefined;
    return new Set(jobs.map((j) => j.job_id));
  }, [jobs, keywordSearchActive, clusterFilterEnabled]);

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

  const handleFilterChange = useCallback((filters: { location?: string; company?: string; title_contains?: string; language?: string }) => {
    fetchJobs({ limit: 50, ...filters });
  }, [fetchJobs]);

  const handleKeywordSearch = useCallback((query: string) => {
    keywordSearch(query);
    // Switch to table view to see keyword search results
    setView('table');
  }, [keywordSearch]);

  const handleClusterParamsApply = useCallback((params: ClusterParams) => {
    fetchClusters(currentAspect, params);
  }, [fetchClusters, currentAspect]);

  const handleAgentSetJobs = useCallback(async (jobIds: string[]) => {
    if (!jobIds.length) return;
    try {
      const { api } = await import('./api/client');
      const data = await api.getJobsBatch(jobIds);
      setJobsFromExternal(data.jobs, 'agent');
      setView('table');
    } catch (e) {
      console.error('Failed to load agent jobs:', e);
    }
  }, [setJobsFromExternal]);

  const handleCloseDetail = useCallback(() => {
    setSelectedJobId(null);
    setHighlightedJobs(new Set());
  }, []);

  const handleRestorePreset = useCallback((preset: UIPreset) => {
    fetchClusters(preset.aspect);
    setView(preset.view);
    setClusterFilterEnabled(preset.clusterFilterEnabled);
  }, [fetchClusters]);

  const currentPresetState = useMemo(() => ({
    aspect: currentAspect,
    view,
    clusterFilterEnabled,
  }), [currentAspect, view, clusterFilterEnabled]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Header */}
      <header style={{
        padding: '10px 20px', borderBottom: '1px solid var(--border)',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        background: 'var(--bg-secondary)',
      }}>
        <h1 style={{ fontSize: 18, fontWeight: 700, margin: 0 }}>BetterJobSearch</h1>
        <PresetBar currentState={currentPresetState} onRestore={handleRestorePreset} />
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 8 }}>
          {totalJobs} jobs
          {jobSource === 'keyword' && <span style={{ color: 'var(--accent)', fontSize: 11 }}>keyword search</span>}
          {jobSource === 'agent' && <span style={{ color: '#22c55e', fontSize: 11 }}>agent results</span>}
          {jobSource === 'filter' && <span style={{ color: 'var(--text-secondary)', fontSize: 11 }}>filtered</span>}
          {jobSource !== 'default' && (
            <button
              onClick={() => fetchJobs({ limit: 50 })}
              style={{
                background: 'transparent', border: '1px solid var(--border)',
                color: 'var(--text-secondary)', padding: '2px 6px', borderRadius: 4,
                cursor: 'pointer', fontSize: 10,
              }}
            >
              Show all
            </button>
          )}
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
          <ClusterSettings onApply={handleClusterParamsApply} loading={clusterLoading} />
          <div style={{ borderTop: '1px solid var(--border)' }} />
          <FilterPanel
            onFilterChange={handleFilterChange}
            onKeywordSearch={handleKeywordSearch}
            keywordSearchLoading={keywordSearchLoading}
            jobSource={jobSource}
          />
        </aside>

        {/* Center content */}
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* View toggle */}
          <div style={{
            padding: '8px 16px', borderBottom: '1px solid var(--border)',
            display: 'flex', gap: 8, alignItems: 'center',
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
              <span style={{ color: 'var(--text-secondary)', fontSize: 12, marginLeft: 8 }}>
                Loading clusters...
              </span>
            )}
            {keywordSearchActive && (
              <label style={{ display: 'flex', alignItems: 'center', gap: 4, marginLeft: 8, fontSize: 12, color: 'var(--accent)', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={clusterFilterEnabled}
                  onChange={(e) => setClusterFilterEnabled(e.target.checked)}
                  style={{ accentColor: 'var(--accent)' }}
                />
                Filter clusters by search
              </label>
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
                filterJobIds={filterJobIds}
                filterActive={keywordSearchActive && clusterFilterEnabled}
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
          <ChatPanel onSetJobs={handleAgentSetJobs} />
        </aside>
      </div>
    </div>
  );
}
