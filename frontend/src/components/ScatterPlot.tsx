import React, { useMemo, useRef, useCallback, useState } from 'react';
import Plot from 'react-plotly.js';
import type { ClusterPoint } from '../api/client';

interface ScatterPlotProps {
  data: ClusterPoint[];
  aspect: string;
  onPointClick?: (jobId: string) => void;
  onSelectionChange?: (ids: Set<string>) => void;
  selectedJobIds?: Set<string>;
  filterJobIds?: Set<string>;
  filterActive?: boolean;
}

interface ClusterInfo {
  clusterId: number;
  label: string;
  keywords: string[];
  color: string;
  size: number;
}

const COLORS = [
  '#6366f1', '#f59e0b', '#22c55e', '#ef4444', '#3b82f6',
  '#ec4899', '#14b8a6', '#f97316', '#8b5cf6', '#06b6d4',
  '#84cc16', '#e11d48', '#0ea5e9', '#d946ef', '#10b981',
];

const zoomBtnStyle: React.CSSProperties = {
  width: 28, height: 28, borderRadius: 4,
  border: '1px solid var(--border)', background: 'var(--bg-secondary)',
  color: 'var(--text-primary)', cursor: 'pointer',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  fontSize: 16, fontWeight: 600, lineHeight: 1,
};

export default function ScatterPlot({ data, aspect, onPointClick, onSelectionChange, selectedJobIds, filterJobIds, filterActive }: ScatterPlotProps) {
  const plotRef = useRef<any>(null);
  const [hoveredCluster, setHoveredCluster] = useState<number | null>(null);

  // Extract cluster info for custom legend
  const clusterInfos = useMemo((): ClusterInfo[] => {
    if (!data.length) return [];
    const clusterMap = new Map<number, { label: string; keywords: string[]; size: number }>();
    for (const pt of data) {
      const cid = pt.cluster_id;
      if (!clusterMap.has(cid)) {
        clusterMap.set(cid, {
          label: pt.cluster_label || (cid === -1 ? 'Noise' : `Cluster ${cid}`),
          keywords: pt.cluster_keywords || [],
          size: 0,
        });
      }
      clusterMap.get(cid)!.size++;
    }
    return Array.from(clusterMap.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([clusterId, info]) => ({
        clusterId,
        label: info.label,
        keywords: info.keywords,
        color: clusterId === -1 ? '#555' : COLORS[clusterId % COLORS.length],
        size: info.size,
      }));
  }, [data]);

  const traces = useMemo(() => {
    if (!data.length) return [];

    const showFilter = filterActive && filterJobIds && filterJobIds.size > 0;
    const hasSelection = selectedJobIds && selectedJobIds.size > 0;

    // Group by cluster
    const clusters = new Map<number, ClusterPoint[]>();
    for (const pt of data) {
      const cid = pt.cluster_id;
      if (!clusters.has(cid)) clusters.set(cid, []);
      clusters.get(cid)!.push(pt);
    }

    return Array.from(clusters.entries()).map(([clusterId, points]) => {
      const isNoise = clusterId === -1;
      const color = isNoise ? '#555' : COLORS[clusterId % COLORS.length];
      const label = points[0]?.cluster_label || (isNoise ? 'Noise' : `Cluster ${clusterId}`);

      return {
        type: 'scattergl' as const,
        mode: 'markers' as const,
        name: label,
        x: points.map((p) => p.x),
        y: points.map((p) => p.y),
        text: points.map((p) => `${p.title}\n${p.company}`),
        customdata: points.map((p) => p.job_id),
        marker: {
          size: points.map((p) => {
            if (selectedJobIds?.has(p.job_id)) return 12;
            if (showFilter && !filterJobIds!.has(p.job_id)) return 3;
            return isNoise ? 4 : 6;
          }),
          color,
          opacity: points.map((p) => {
            if (showFilter && !filterJobIds!.has(p.job_id)) return 0.05;
            if (hasSelection)
              return selectedJobIds!.has(p.job_id) ? 1 : 0.2;
            return isNoise ? 0.3 : 0.7;
          }),
          line: {
            width: points.map((p) =>
              selectedJobIds?.has(p.job_id) ? 2 : 0
            ),
            color: '#fff',
          },
        },
        hoverinfo: 'text' as const,
      };
    });
  }, [data, selectedJobIds, filterJobIds, filterActive]);

  // Handle lasso/box selection
  const handleSelected = useCallback((event: any) => {
    if (!onSelectionChange) return;
    const points = event?.points;
    if (!points || points.length === 0) {
      // Selection cleared
      return;
    }
    const ids = new Set<string>(points.map((p: any) => p.customdata as string));
    onSelectionChange(ids);
  }, [onSelectionChange]);

  // Handle selection cleared (double-click or deselect)
  const handleDeselect = useCallback(() => {
    // Don't clear on deselect - keep current selection
  }, []);

  const zoom = useCallback((factor: number) => {
    const el = plotRef.current?.el;
    if (!el) return;

    const layout = el._fullLayout;
    const xRange = layout.xaxis.range;
    const yRange = layout.yaxis.range;
    const xMid = (xRange[0] + xRange[1]) / 2;
    const yMid = (yRange[0] + yRange[1]) / 2;
    const xHalf = (xRange[1] - xRange[0]) / 2 * factor;
    const yHalf = (yRange[1] - yRange[0]) / 2 * factor;

    const Plotly = (window as any).Plotly || el._context?._plotly;
    if (Plotly) {
      Plotly.relayout(el, {
        'xaxis.range': [xMid - xHalf, xMid + xHalf],
        'yaxis.range': [yMid - yHalf, yMid + yHalf],
      });
    }
  }, []);

  const resetZoom = useCallback(() => {
    const el = plotRef.current?.el;
    if (!el) return;
    const Plotly = (window as any).Plotly || el._context?._plotly;
    if (Plotly) {
      Plotly.relayout(el, {
        'xaxis.autorange': true,
        'yaxis.autorange': true,
      });
    }
  }, []);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <Plot
        ref={plotRef}
        data={traces}
        layout={{
          uirevision: aspect,
          title: { text: `Clusters by ${aspect}`, font: { color: '#e4e6eb', size: 14 } },
          paper_bgcolor: '#1a1d28',
          plot_bgcolor: '#1a1d28',
          xaxis: { visible: false },
          yaxis: { visible: false },
          showlegend: false,  // Hide default legend, use custom
          margin: { l: 10, r: 160, t: 40, b: 10 },  // Make room for custom legend
          hovermode: 'closest',
          dragmode: 'lasso',  // Enable lasso by default
        }}
        config={{
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: ['autoScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines'],
          modeBarButtonsToAdd: ['select2d', 'lasso2d'],
          displaylogo: false,
        }}
        style={{ width: '100%', height: '100%' }}
        onClick={(event: any) => {
          const point = event.points?.[0];
          if (point && onPointClick) {
            onPointClick(point.customdata as string);
          }
        }}
        onSelected={handleSelected}
        onDeselect={handleDeselect}
      />

      {/* Custom legend with keyword tooltips */}
      <div style={{
        position: 'absolute', top: 40, right: 8, width: 150,
        maxHeight: 'calc(100% - 80px)', overflowY: 'auto',
        background: 'rgba(26, 29, 40, 0.9)', borderRadius: 4,
        padding: 8, fontSize: 11,
      }}>
        {clusterInfos.map(info => (
          <div
            key={info.clusterId}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '3px 4px', borderRadius: 3, cursor: 'pointer',
              background: hoveredCluster === info.clusterId ? 'rgba(255,255,255,0.1)' : 'transparent',
              position: 'relative',
            }}
            onMouseEnter={() => setHoveredCluster(info.clusterId)}
            onMouseLeave={() => setHoveredCluster(null)}
          >
            <div style={{
              width: 10, height: 10, borderRadius: 2,
              background: info.color, flexShrink: 0,
            }} />
            <span style={{
              color: '#a0a4b0', overflow: 'hidden', textOverflow: 'ellipsis',
              whiteSpace: 'nowrap', flex: 1,
            }}>
              {info.label}
            </span>
            <span style={{ color: '#666', fontSize: 10 }}>({info.size})</span>

            {/* Keyword tooltip */}
            {hoveredCluster === info.clusterId && info.keywords.length > 0 && (
              <div style={{
                position: 'absolute', left: '100%', top: 0, marginLeft: 8,
                background: '#2a2d38', border: '1px solid var(--border)',
                borderRadius: 4, padding: 8, zIndex: 100,
                minWidth: 180, maxWidth: 250, boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
              }}>
                <div style={{ fontWeight: 600, marginBottom: 4, color: info.color }}>
                  Keywords
                </div>
                <div style={{ color: '#ccc', lineHeight: 1.5 }}>
                  {info.keywords.slice(0, 10).map((kw, i) => (
                    <span key={i} style={{
                      display: 'inline-block', background: 'rgba(255,255,255,0.1)',
                      borderRadius: 3, padding: '1px 5px', margin: '2px 3px 2px 0',
                      fontSize: 10,
                    }}>
                      {kw}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Zoom controls */}
      <div style={{
        position: 'absolute', bottom: 16, right: 16,
        display: 'flex', flexDirection: 'column', gap: 4,
      }}>
        <button style={zoomBtnStyle} onClick={() => zoom(0.5)} title="Zoom in">+</button>
        <button style={zoomBtnStyle} onClick={() => zoom(2)} title="Zoom out">−</button>
        <button style={{ ...zoomBtnStyle, fontSize: 12 }} onClick={resetZoom} title="Reset zoom">⟲</button>
      </div>
    </div>
  );
}
