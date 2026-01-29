import React, { useState } from 'react';
import type { ClusterParams } from '../hooks/useClusters';
import type { ClusterSettingsState } from './PresetBar';

export const DEFAULT_CLUSTER_SETTINGS: ClusterSettingsState = {
  n_neighbors: 15,
  min_dist: 0.1,
  min_cluster_size: 0,
  tfidf_min_df: 0,
  tfidf_max_df: 100,
};

interface ClusterSettingsProps {
  onApply: (params: ClusterParams) => void;
  loading?: boolean;
  // Controlled mode
  settings?: ClusterSettingsState;
  onSettingsChange?: (settings: ClusterSettingsState) => void;
}

export default function ClusterSettings({
  onApply,
  loading,
  settings,
  onSettingsChange,
}: ClusterSettingsProps) {
  const [expanded, setExpanded] = useState(false);

  // Use controlled state if provided, otherwise local state
  const isControlled = settings !== undefined && onSettingsChange !== undefined;

  const [localNNeighbors, setLocalNNeighbors] = useState(DEFAULT_CLUSTER_SETTINGS.n_neighbors);
  const [localMinDist, setLocalMinDist] = useState(DEFAULT_CLUSTER_SETTINGS.min_dist);
  const [localMinClusterSize, setLocalMinClusterSize] = useState(DEFAULT_CLUSTER_SETTINGS.min_cluster_size);
  const [localTfidfMinDf, setLocalTfidfMinDf] = useState(DEFAULT_CLUSTER_SETTINGS.tfidf_min_df);
  const [localTfidfMaxDf, setLocalTfidfMaxDf] = useState(DEFAULT_CLUSTER_SETTINGS.tfidf_max_df);

  // Current values (from props or local)
  const nNeighbors = isControlled ? settings.n_neighbors : localNNeighbors;
  const minDist = isControlled ? settings.min_dist : localMinDist;
  const minClusterSize = isControlled ? settings.min_cluster_size : localMinClusterSize;
  const tfidfMinDf = isControlled ? settings.tfidf_min_df : localTfidfMinDf;
  const tfidfMaxDf = isControlled ? settings.tfidf_max_df : localTfidfMaxDf;

  // Setters that work in both controlled and uncontrolled modes
  const setNNeighbors = (v: number) => {
    if (isControlled) onSettingsChange({ ...settings, n_neighbors: v });
    else setLocalNNeighbors(v);
  };
  const setMinDist = (v: number) => {
    if (isControlled) onSettingsChange({ ...settings, min_dist: v });
    else setLocalMinDist(v);
  };
  const setMinClusterSize = (v: number) => {
    if (isControlled) onSettingsChange({ ...settings, min_cluster_size: v });
    else setLocalMinClusterSize(v);
  };
  const setTfidfMinDf = (v: number) => {
    if (isControlled) onSettingsChange({ ...settings, tfidf_min_df: v });
    else setLocalTfidfMinDf(v);
  };
  const setTfidfMaxDf = (v: number) => {
    if (isControlled) onSettingsChange({ ...settings, tfidf_max_df: v });
    else setLocalTfidfMaxDf(v);
  };

  const handleApply = () => {
    onApply({
      n_neighbors: nNeighbors,
      min_dist: minDist,
      min_cluster_size: minClusterSize > 0 ? minClusterSize : undefined,
      tfidf_min_df: tfidfMinDf > 0 ? tfidfMinDf / 100 : undefined,
      tfidf_max_df: tfidfMaxDf < 100 ? tfidfMaxDf / 100 : undefined,
    });
  };

  const handleReset = () => {
    if (isControlled) {
      onSettingsChange(DEFAULT_CLUSTER_SETTINGS);
    } else {
      setLocalNNeighbors(DEFAULT_CLUSTER_SETTINGS.n_neighbors);
      setLocalMinDist(DEFAULT_CLUSTER_SETTINGS.min_dist);
      setLocalMinClusterSize(DEFAULT_CLUSTER_SETTINGS.min_cluster_size);
      setLocalTfidfMinDf(DEFAULT_CLUSTER_SETTINGS.tfidf_min_df);
      setLocalTfidfMaxDf(DEFAULT_CLUSTER_SETTINGS.tfidf_max_df);
    }
    onApply({});
  };

  return (
    <div style={{ padding: '8px 12px' }}>
      <button
        onClick={() => setExpanded(!expanded)}
        style={{
          background: 'transparent', border: 'none', color: 'var(--text-secondary)',
          fontSize: 12, cursor: 'pointer', padding: 0, width: '100%', textAlign: 'left',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          textTransform: 'uppercase', fontWeight: 600, letterSpacing: '0.5px',
        }}
      >
        Cluster Settings
        <span style={{ fontSize: 10 }}>{expanded ? '\u25B2' : '\u25BC'}</span>
      </button>

      {expanded && (
        <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div>
            <label style={labelStyle}>
              Density (n_neighbors): {nNeighbors}
            </label>
            <input
              type="range" min={2} max={50} value={nNeighbors}
              onChange={(e) => setNNeighbors(Number(e.target.value))}
              style={sliderStyle}
            />
          </div>

          <div>
            <label style={labelStyle}>
              Spread (min_dist): {minDist.toFixed(2)}
            </label>
            <input
              type="range" min={0} max={100} value={Math.round(minDist * 100)}
              onChange={(e) => setMinDist(Number(e.target.value) / 100)}
              style={sliderStyle}
            />
          </div>

          <div>
            <label style={labelStyle}>
              Min cluster size: {minClusterSize === 0 ? 'auto' : minClusterSize}
            </label>
            <input
              type="range" min={0} max={50} value={minClusterSize}
              onChange={(e) => setMinClusterSize(Number(e.target.value))}
              style={sliderStyle}
            />
          </div>

          <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid var(--border)' }}>
            <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 6, textTransform: 'uppercase' }}>
              Label Filters (TF-IDF)
            </div>
          </div>

          <div>
            <label style={labelStyle}>
              Min doc freq: {tfidfMinDf === 0 ? 'off' : `${tfidfMinDf}%`}
            </label>
            <input
              type="range" min={0} max={20} step={1} value={tfidfMinDf}
              onChange={(e) => setTfidfMinDf(Number(e.target.value))}
              style={sliderStyle}
            />
            <div style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.7 }}>
              Ignore rare words (appearing in fewer docs)
            </div>
          </div>

          <div>
            <label style={labelStyle}>
              Max doc freq: {tfidfMaxDf === 100 ? 'off' : `${tfidfMaxDf}%`}
            </label>
            <input
              type="range" min={50} max={100} step={5} value={tfidfMaxDf}
              onChange={(e) => setTfidfMaxDf(Number(e.target.value))}
              style={sliderStyle}
            />
            <div style={{ fontSize: 9, color: 'var(--text-secondary)', opacity: 0.7 }}>
              Ignore common words (appearing in many docs)
            </div>
          </div>

          <div style={{ display: 'flex', gap: 6 }}>
            <button onClick={handleApply} disabled={loading} style={{
              flex: 1, padding: '5px', borderRadius: 6, border: 'none',
              background: 'var(--accent)', color: 'white', fontSize: 11,
              cursor: 'pointer', opacity: loading ? 0.5 : 1,
            }}>
              Apply
            </button>
            <button onClick={handleReset} style={{
              padding: '5px 10px', borderRadius: 6, border: '1px solid var(--border)',
              background: 'transparent', color: 'var(--text-secondary)', fontSize: 11,
              cursor: 'pointer',
            }}>
              Reset
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

const labelStyle: React.CSSProperties = {
  fontSize: 11, color: 'var(--text-secondary)', display: 'block', marginBottom: 2,
};

const sliderStyle: React.CSSProperties = {
  width: '100%', accentColor: 'var(--accent)', height: 4,
};
