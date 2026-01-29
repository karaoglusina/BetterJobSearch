import { useState, useCallback, useMemo, useEffect } from 'react';
import ScatterPlot from './components/ScatterPlot';
import JobTable from './components/JobTable';
import ChatPanel from './components/ChatPanel';
import AspectSelector from './components/AspectSelector';
import JobDetail from './components/JobDetail';
import FilterPanel from './components/FilterPanel';
import ClusterSettings, { DEFAULT_CLUSTER_SETTINGS } from './components/ClusterSettings';
import KeywordSettings, { DEFAULT_KEYWORD_SETTINGS, KeywordSettingsState } from './components/KeywordSettings';
import PresetBar, { UIPreset, ClusterSettingsState } from './components/PresetBar';
import LabelDialog from './components/LabelDialog';
import { useJobs } from './hooks/useJobs';
import { useClusters, ClusterParams } from './hooks/useClusters';
import { useLabels } from './hooks/useLabels';

type View = 'scatter' | 'table';

export default function App() {
  const {
    jobs, totalJobs, fetchJobs,
    keywordSearch, keywordSearchLoading, keywordSearchActive,
    jobSource, setJobsFromExternal,
  } = useJobs();
  const { clusterData, currentAspect, loading: clusterLoading, fetchClusters, clusterByConcept } = useClusters();
  const { jobLabels, allLabels, addLabel, removeLabel } = useLabels();
  const [selectedJobIds, setSelectedJobIds] = useState<Set<string>>(new Set());
  const [view, setView] = useState<View>('scatter');
  const [clusterFilterEnabled, setClusterFilterEnabled] = useState(true);
  const [labelDialogOpen, setLabelDialogOpen] = useState(false);
  const [labelFilter, setLabelFilter] = useState<string[]>([]);
  const [clusterSettings, setClusterSettings] = useState<ClusterSettingsState>(DEFAULT_CLUSTER_SETTINGS);
  const [keywordSettings, setKeywordSettings] = useState<KeywordSettingsState>(DEFAULT_KEYWORD_SETTINGS);

  // Get first selected job for detail panel
  const selectedJobId = useMemo(() => {
    const arr = Array.from(selectedJobIds);
    return arr.length > 0 ? arr[arr.length - 1] : null;
  }, [selectedJobIds]);

  // Filter jobs by label (client-side)
  const filteredJobs = useMemo(() => {
    if (labelFilter.length === 0) return jobs;
    return jobs.filter(job => {
      const jobLabelsList = jobLabels[job.job_id] || [];
      return labelFilter.some(label => jobLabelsList.includes(label));
    });
  }, [jobs, jobLabels, labelFilter]);

  const filterJobIds = useMemo(() => {
    if (!keywordSearchActive || !clusterFilterEnabled) return undefined;
    return new Set(filteredJobs.map((j) => j.job_id));
  }, [filteredJobs, keywordSearchActive, clusterFilterEnabled]);

  const handleAspectSelect = useCallback((aspect: string) => {
    fetchClusters(aspect);
  }, [fetchClusters]);

  // Single-click handler (for scatter plot click)
  const handlePointClick = useCallback((jobId: string) => {
    setSelectedJobIds(new Set([jobId]));
  }, []);

  // Multi-select handler (for both table and scatter lasso/box)
  const handleSelectionChange = useCallback((ids: Set<string>) => {
    setSelectedJobIds(ids);
  }, []);

  // Legacy single-select handler for table row click
  const handleJobSelect = useCallback((jobId: string) => {
    setSelectedJobIds(new Set([jobId]));
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
    // Merge current cluster settings, keyword settings, and any overrides
    const mergedParams: ClusterParams = {
      n_neighbors: clusterSettings.n_neighbors,
      min_dist: clusterSettings.min_dist,
      min_cluster_size: clusterSettings.min_cluster_size > 0 ? clusterSettings.min_cluster_size : undefined,
      tfidf_min_df: keywordSettings.tfidf_min_df > 0 ? keywordSettings.tfidf_min_df / 100 : undefined,
      tfidf_max_df: keywordSettings.tfidf_max_df < 100 ? keywordSettings.tfidf_max_df / 100 : undefined,
      npmi_min: keywordSettings.npmi_min > 0 ? keywordSettings.npmi_min : undefined,
      npmi_max: keywordSettings.npmi_max < 1.0 ? keywordSettings.npmi_max : undefined,
      effect_size_min: keywordSettings.effect_size_min > 0 ? keywordSettings.effect_size_min : undefined,
      effect_size_max: keywordSettings.effect_size_max < 20 ? keywordSettings.effect_size_max : undefined,
      ...params,
    };
    fetchClusters(currentAspect, mergedParams);
  }, [fetchClusters, currentAspect, clusterSettings, keywordSettings]);

  const handleKeywordParamsApply = useCallback((params: ClusterParams) => {
    // Merge keyword params with cluster settings
    const mergedParams: ClusterParams = {
      n_neighbors: clusterSettings.n_neighbors,
      min_dist: clusterSettings.min_dist,
      min_cluster_size: clusterSettings.min_cluster_size > 0 ? clusterSettings.min_cluster_size : undefined,
      ...params,
    };
    fetchClusters(currentAspect, mergedParams);
  }, [fetchClusters, currentAspect, clusterSettings]);

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
    setSelectedJobIds(new Set());
  }, []);

  // Label handlers
  const handleApplyLabel = useCallback((label: string) => {
    selectedJobIds.forEach(jobId => addLabel(jobId, label));
  }, [selectedJobIds, addLabel]);

  const handleRemoveLabel = useCallback((label: string) => {
    selectedJobIds.forEach(jobId => removeLabel(jobId, label));
  }, [selectedJobIds, removeLabel]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Check if user is typing in an input
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
        return;
      }

      // 'L' to open label dialog when jobs are selected
      if (e.key === 'l' || e.key === 'L') {
        if (selectedJobIds.size > 0) {
          e.preventDefault();
          setLabelDialogOpen(true);
        }
      }

      // Escape to clear selection
      if (e.key === 'Escape') {
        setSelectedJobIds(new Set());
        setLabelDialogOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedJobIds]);

  const handleRestorePreset = useCallback((preset: UIPreset) => {
    // Restore view state
    setView(preset.view);
    setClusterFilterEnabled(preset.clusterFilterEnabled);

    // Restore cluster settings if present
    if (preset.clusterSettings) {
      setClusterSettings(preset.clusterSettings);
    }

    // Restore label filter if present
    if (preset.labelFilter) {
      setLabelFilter(preset.labelFilter);
    }

    // Build cluster params from preset settings
    const settings = preset.clusterSettings || clusterSettings;
    const params: ClusterParams = {
      n_neighbors: settings.n_neighbors,
      min_dist: settings.min_dist,
      min_cluster_size: settings.min_cluster_size > 0 ? settings.min_cluster_size : undefined,
      tfidf_min_df: settings.tfidf_min_df > 0 ? settings.tfidf_min_df / 100 : undefined,
      tfidf_max_df: settings.tfidf_max_df < 100 ? settings.tfidf_max_df / 100 : undefined,
    };

    fetchClusters(preset.aspect, params);
  }, [fetchClusters, clusterSettings]);

  const currentPresetState = useMemo(() => ({
    aspect: currentAspect,
    view,
    clusterFilterEnabled,
    clusterSettings,
    labelFilter,
  }), [currentAspect, view, clusterFilterEnabled, clusterSettings, labelFilter]);

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
          {labelFilter.length > 0 ? `${filteredJobs.length} / ${totalJobs} jobs` : `${totalJobs} jobs`}
          {labelFilter.length > 0 && <span style={{ color: '#f59e0b', fontSize: 11 }}>label filter</span>}
          {jobSource === 'keyword' && <span style={{ color: 'var(--accent)', fontSize: 11 }}>keyword search</span>}
          {jobSource === 'agent' && <span style={{ color: '#22c55e', fontSize: 11 }}>agent results</span>}
          {jobSource === 'filter' && <span style={{ color: 'var(--text-secondary)', fontSize: 11 }}>filtered</span>}
          {(jobSource !== 'default' || labelFilter.length > 0) && (
            <button
              onClick={() => {
                fetchJobs({ limit: 50 });
                setLabelFilter([]);
              }}
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
          <ClusterSettings
            onApply={handleClusterParamsApply}
            loading={clusterLoading}
            settings={clusterSettings}
            onSettingsChange={setClusterSettings}
          />
          <div style={{ borderTop: '1px solid var(--border)' }} />
          <KeywordSettings
            onApply={handleKeywordParamsApply}
            loading={clusterLoading}
            settings={keywordSettings}
            onSettingsChange={setKeywordSettings}
          />
          <div style={{ borderTop: '1px solid var(--border)' }} />
          <FilterPanel
            onFilterChange={handleFilterChange}
            onKeywordSearch={handleKeywordSearch}
            keywordSearchLoading={keywordSearchLoading}
            jobSource={jobSource}
            availableLabels={allLabels}
            selectedLabels={labelFilter}
            onLabelFilterChange={setLabelFilter}
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
                onSelectionChange={handleSelectionChange}
                selectedJobIds={selectedJobIds}
                filterJobIds={filterJobIds}
                filterActive={keywordSearchActive && clusterFilterEnabled}
              />
            ) : (
              <JobTable
                jobs={filteredJobs}
                onJobSelect={handleJobSelect}
                onSelectionChange={handleSelectionChange}
                selectedJobIds={selectedJobIds}
                jobLabels={jobLabels}
              />
            )}
          </div>

          {/* Selection info bar */}
          {selectedJobIds.size > 0 && (
            <div style={{
              padding: '6px 16px',
              borderTop: '1px solid var(--border)',
              background: 'var(--bg-secondary)',
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              fontSize: 12,
            }}>
              <span style={{ color: 'var(--text-secondary)' }}>
                {selectedJobIds.size} job{selectedJobIds.size > 1 ? 's' : ''} selected
              </span>
              <button
                onClick={() => setLabelDialogOpen(true)}
                style={{
                  padding: '3px 8px',
                  background: 'var(--accent)',
                  border: 'none',
                  borderRadius: 4,
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: 11,
                }}
              >
                Label (L)
              </button>
              <button
                onClick={() => setSelectedJobIds(new Set())}
                style={{
                  padding: '3px 8px',
                  background: 'transparent',
                  border: '1px solid var(--border)',
                  borderRadius: 4,
                  color: 'var(--text-secondary)',
                  cursor: 'pointer',
                  fontSize: 11,
                }}
              >
                Clear
              </button>
            </div>
          )}

          {/* Job detail panel */}
          <JobDetail jobId={selectedJobId} onClose={handleCloseDetail} />
        </main>

        {/* Label dialog */}
        <LabelDialog
          isOpen={labelDialogOpen}
          onClose={() => setLabelDialogOpen(false)}
          selectedJobIds={selectedJobIds}
          existingLabels={allLabels}
          jobLabels={jobLabels}
          onApplyLabel={handleApplyLabel}
          onRemoveLabel={handleRemoveLabel}
        />

        {/* Right chat panel */}
        <aside style={{ width: 340 }}>
          <ChatPanel
            onSetJobs={handleAgentSetJobs}
            selectedJobIds={selectedJobIds}
            currentAspect={currentAspect}
          />
        </aside>
      </div>
    </div>
  );
}
