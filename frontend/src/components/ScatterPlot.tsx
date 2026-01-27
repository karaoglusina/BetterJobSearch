import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import type { ClusterPoint } from '../api/client';

interface ScatterPlotProps {
  data: ClusterPoint[];
  aspect: string;
  onPointClick?: (jobId: string) => void;
  highlightedJobs?: Set<string>;
}

const COLORS = [
  '#6366f1', '#f59e0b', '#22c55e', '#ef4444', '#3b82f6',
  '#ec4899', '#14b8a6', '#f97316', '#8b5cf6', '#06b6d4',
  '#84cc16', '#e11d48', '#0ea5e9', '#d946ef', '#10b981',
];

export default function ScatterPlot({ data, aspect, onPointClick, highlightedJobs }: ScatterPlotProps) {
  const traces = useMemo(() => {
    if (!data.length) return [];

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
          size: points.map((p) =>
            highlightedJobs?.has(p.job_id) ? 10 : isNoise ? 4 : 6
          ),
          color,
          opacity: points.map((p) =>
            highlightedJobs?.size
              ? highlightedJobs.has(p.job_id) ? 1 : 0.2
              : isNoise ? 0.3 : 0.7
          ),
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
  }, [data, highlightedJobs]);

  return (
    <Plot
      data={traces}
      layout={{
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
      onClick={(event) => {
        const point = event.points?.[0];
        if (point && onPointClick) {
          onPointClick(point.customdata as string);
        }
      }}
    />
  );
}
