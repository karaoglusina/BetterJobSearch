import React, { useMemo, useRef, useCallback } from 'react';
import Plot from 'react-plotly.js';
import type { ClusterPoint } from '../api/client';

interface ScatterPlotProps {
  data: ClusterPoint[];
  aspect: string;
  onPointClick?: (jobId: string) => void;
  highlightedJobs?: Set<string>;
  filterJobIds?: Set<string>;
  filterActive?: boolean;
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

export default function ScatterPlot({ data, aspect, onPointClick, highlightedJobs, filterJobIds, filterActive }: ScatterPlotProps) {
  const plotRef = useRef<any>(null);
  const traces = useMemo(() => {
    if (!data.length) return [];

    const showFilter = filterActive && filterJobIds && filterJobIds.size > 0;

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
            if (highlightedJobs?.has(p.job_id)) return 10;
            if (showFilter && !filterJobIds!.has(p.job_id)) return 3;
            return isNoise ? 4 : 6;
          }),
          color,
          opacity: points.map((p) => {
            if (showFilter && !filterJobIds!.has(p.job_id)) return 0.05;
            if (highlightedJobs?.size)
              return highlightedJobs.has(p.job_id) ? 1 : 0.2;
            return isNoise ? 0.3 : 0.7;
          }),
          line: {
            width: points.map((p) =>
              highlightedJobs?.has(p.job_id) ? 2 : 0
            ),
            color: '#fff',
          },
        },
        hoverinfo: 'text' as const,
      };
    });
  }, [data, highlightedJobs, filterJobIds, filterActive]);

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
          showlegend: true,
          legend: { font: { color: '#a0a4b0', size: 11 }, bgcolor: 'transparent' },
          margin: { l: 10, r: 10, t: 40, b: 10 },
          hovermode: 'closest',
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%', height: '100%' }}
        onClick={(event: any) => {
          const point = event.points?.[0];
          if (point && onPointClick) {
            onPointClick(point.customdata as string);
          }
        }}
      />

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
