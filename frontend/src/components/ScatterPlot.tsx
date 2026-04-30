import React, { useMemo, useRef, useCallback, useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import type { ClusterPoint, ClusterPolygon, ClusterLabel } from '../api/client';

interface ScatterPlotProps {
  data: ClusterPoint[];
  aspect: string;
  polygons?: ClusterPolygon[];
  atlasLabels?: ClusterLabel[];
  palette?: Record<string, string>;
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

const FALLBACK_COLORS = [
  '#6366f1', '#f59e0b', '#22c55e', '#ef4444', '#3b82f6',
  '#ec4899', '#14b8a6', '#f97316', '#8b5cf6', '#06b6d4',
  '#84cc16', '#e11d48', '#0ea5e9', '#d946ef', '#10b981',
];

const NOISE_COLOR = '#555555';

const zoomBtnStyle: React.CSSProperties = {
  width: 28, height: 28, borderRadius: 4,
  border: '1px solid var(--border)', background: 'var(--bg-secondary)',
  color: 'var(--text-primary)', cursor: 'pointer',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  fontSize: 16, fontWeight: 600, lineHeight: 1,
};

function hexToRgba(hex: string, alpha: number): string {
  const cleaned = hex.replace('#', '');
  if (cleaned.length !== 6) return `rgba(100,100,100,${alpha})`;
  const r = parseInt(cleaned.slice(0, 2), 16);
  const g = parseInt(cleaned.slice(2, 4), 16);
  const b = parseInt(cleaned.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function polygonToPath(path: [number, number][] | number[][]): string {
  if (!path || path.length === 0) return '';
  const pts = path as number[][];
  const parts: string[] = [`M ${pts[0][0]} ${pts[0][1]}`];
  for (let i = 1; i < pts.length; i++) {
    parts.push(`L ${pts[i][0]} ${pts[i][1]}`);
  }
  parts.push('Z');
  return parts.join(' ');
}

export default function ScatterPlot({
  data,
  aspect,
  polygons = [],
  atlasLabels = [],
  palette = {},
  onPointClick,
  onSelectionChange,
  selectedJobIds,
  filterJobIds,
  filterActive,
}: ScatterPlotProps) {
  const plotRef = useRef<any>(null);
  const [hoveredCluster, setHoveredCluster] = useState<number | null>(null);
  const [zoomSpanRatio, setZoomSpanRatio] = useState(1);
  const initialSpanRef = useRef<{ x: number; y: number } | null>(null);

  const colorFor = useCallback(
    (clusterId: number): string => {
      if (clusterId === -1) return palette['-1'] || NOISE_COLOR;
      return palette[String(clusterId)] || FALLBACK_COLORS[clusterId % FALLBACK_COLORS.length];
    },
    [palette]
  );

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
        color: colorFor(clusterId),
        size: info.size,
      }));
  }, [data, colorFor]);

  const traces = useMemo(() => {
    if (!data.length) return [];

    const showFilter = filterActive && filterJobIds && filterJobIds.size > 0;
    const hasSelection = selectedJobIds && selectedJobIds.size > 0;
    const hasHoverFocus = hoveredCluster !== null;

    const clusters = new Map<number, ClusterPoint[]>();
    for (const pt of data) {
      const cid = pt.cluster_id;
      if (!clusters.has(cid)) clusters.set(cid, []);
      clusters.get(cid)!.push(pt);
    }

    const allTraces: any[] = [];

    Array.from(clusters.entries()).forEach(([clusterId, points]) => {
      const isNoise = clusterId === -1;
      const color = colorFor(clusterId);
      const label = points[0]?.cluster_label || (isNoise ? 'Noise' : `Cluster ${clusterId}`);

      const isDimmed = hasHoverFocus && hoveredCluster !== clusterId;

      const coreOpacity = points.map((p) => {
        if (showFilter && !filterJobIds!.has(p.job_id)) return 0.05;
        if (hasSelection && !selectedJobIds!.has(p.job_id)) return 0.18;
        if (isDimmed) return 0.12;
        return isNoise ? 0.35 : 0.85;
      });

      const coreSizes = points.map((p) => {
        if (selectedJobIds?.has(p.job_id)) return 12;
        if (showFilter && !filterJobIds!.has(p.job_id)) return 3;
        return isNoise ? 3 : 5;
      });

      // Soft glow layer (non-interactive, larger + transparent) - only for real clusters.
      if (!isNoise) {
        allTraces.push({
          type: 'scattergl' as const,
          mode: 'markers' as const,
          name: `${label}__glow`,
          x: points.map((p) => p.x),
          y: points.map((p) => p.y),
          hoverinfo: 'skip',
          showlegend: false,
          marker: {
            size: coreSizes.map((s) => s * 3.2),
            color,
            opacity: isDimmed ? 0.04 : (hasSelection ? 0.06 : 0.12),
            line: { width: 0 },
          },
          legendgroup: String(clusterId),
        });
      }

      // Core hit-test layer (interactive).
      allTraces.push({
        type: 'scattergl' as const,
        mode: 'markers' as const,
        name: label,
        x: points.map((p) => p.x),
        y: points.map((p) => p.y),
        text: points.map((p) => `${p.title}\n${p.company}`),
        customdata: points.map((p) => p.job_id),
        marker: {
          size: coreSizes,
          color,
          opacity: coreOpacity,
          line: {
            width: points.map((p) => (selectedJobIds?.has(p.job_id) ? 2 : 0)),
            color: '#fff',
          },
        },
        hoverinfo: 'text' as const,
        legendgroup: String(clusterId),
      });
    });

    return allTraces;
  }, [data, selectedJobIds, filterJobIds, filterActive, hoveredCluster, colorFor]);

  // Split server-computed labels into levels, gate level 1 behind zoom.
  const labelTraces = useMemo(() => {
    if (!atlasLabels || atlasLabels.length === 0) return [];

    const level0 = atlasLabels.filter((l) => l.level === 0);
    const level1 = atlasLabels.filter((l) => l.level === 1);

    const textColor = '#ffffff';

    const level0Trace = level0.length
      ? {
          type: 'scatter' as const,
          mode: 'text' as const,
          x: level0.map((l) => l.x),
          y: level0.map((l) => l.y),
          text: level0.map((l) => l.text),
          hoverinfo: 'skip' as const,
          showlegend: false,
          textfont: {
            size: 13,
            color: textColor,
            family: 'Inter, ui-sans-serif, system-ui, sans-serif',
          },
          textposition: 'middle center' as const,
          name: '__labels_l0',
        }
      : null;

    // Show level-1 labels only when user has zoomed past ~50% of the initial span.
    const showLevel1 = zoomSpanRatio < 0.55;
    const level1Trace = showLevel1 && level1.length
      ? {
          type: 'scatter' as const,
          mode: 'text' as const,
          x: level1.map((l) => l.x),
          y: level1.map((l) => l.y),
          text: level1.map((l) => l.text),
          hoverinfo: 'skip' as const,
          showlegend: false,
          textfont: {
            size: 10,
            color: 'rgba(255,255,255,0.72)',
            family: 'Inter, ui-sans-serif, system-ui, sans-serif',
          },
          textposition: 'middle center' as const,
          name: '__labels_l1',
        }
      : null;

    return [level0Trace, level1Trace].filter(Boolean) as any[];
  }, [atlasLabels, zoomSpanRatio]);

  // Cluster continent polygons as Plotly layout.shapes.
  const shapes = useMemo(() => {
    if (!polygons || polygons.length === 0) return [] as any[];
    const hasHoverFocus = hoveredCluster !== null;
    return polygons.map((poly) => {
      const color = colorFor(poly.cluster_id);
      const dimmed = hasHoverFocus && hoveredCluster !== poly.cluster_id;
      return {
        type: 'path' as const,
        path: polygonToPath(poly.path),
        fillcolor: hexToRgba(color, dimmed ? 0.04 : 0.11),
        line: {
          color: hexToRgba(color, dimmed ? 0.1 : 0.35),
          width: 1,
        },
        layer: 'below' as const,
        xref: 'x' as const,
        yref: 'y' as const,
      };
    });
  }, [polygons, hoveredCluster, colorFor]);

  const handleSelected = useCallback((event: any) => {
    if (!onSelectionChange) return;
    const points = event?.points;
    if (!points || points.length === 0) return;
    const ids = new Set<string>(points.map((p: any) => p.customdata as string).filter(Boolean));
    if (ids.size > 0) onSelectionChange(ids);
  }, [onSelectionChange]);

  const handleDeselect = useCallback(() => {}, []);

  const handleRelayout = useCallback(() => {
    const el = plotRef.current?.el;
    if (!el) return;
    const layout = el._fullLayout;
    if (!layout?.xaxis?.range || !layout?.yaxis?.range) return;

    const xRange = layout.xaxis.range;
    const yRange = layout.yaxis.range;
    const spanX = Math.abs(xRange[1] - xRange[0]);
    const spanY = Math.abs(yRange[1] - yRange[0]);

    if (!initialSpanRef.current) {
      initialSpanRef.current = { x: spanX, y: spanY };
      return;
    }

    const ratio = Math.max(
      spanX / initialSpanRef.current.x,
      spanY / initialSpanRef.current.y
    );
    setZoomSpanRatio(ratio);
  }, []);

  // Reset captured initial span when aspect changes (fresh projection).
  useEffect(() => {
    initialSpanRef.current = null;
    setZoomSpanRatio(1);
  }, [aspect]);

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

  const allTraces = useMemo(() => [...traces, ...labelTraces], [traces, labelTraces]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <Plot
        ref={plotRef}
        data={allTraces}
        layout={{
          uirevision: aspect,
          title: { text: `Clusters by ${aspect}`, font: { color: '#e4e6eb', size: 14 } },
          paper_bgcolor: '#0f111a',
          plot_bgcolor: '#0f111a',
          xaxis: { visible: false, scaleanchor: 'y', scaleratio: 1 },
          yaxis: { visible: false },
          showlegend: false,
          margin: { l: 10, r: 160, t: 40, b: 10 },
          hovermode: 'closest',
          dragmode: 'lasso',
          shapes,
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
          if (point && onPointClick && point.customdata) {
            onPointClick(point.customdata as string);
          }
        }}
        onSelected={handleSelected}
        onDeselect={handleDeselect}
        onRelayout={handleRelayout}
        onInitialized={handleRelayout}
      />

      {/* Custom legend with keyword tooltips and on-map cross-filter */}
      <div style={{
        position: 'absolute', top: 40, right: 8, width: 150,
        maxHeight: 'calc(100% - 80px)', overflowY: 'auto',
        background: 'rgba(15, 17, 26, 0.85)', borderRadius: 4,
        padding: 8, fontSize: 11,
        backdropFilter: 'blur(6px)',
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

            {hoveredCluster === info.clusterId && info.keywords.length > 0 && (
              <div style={{
                position: 'absolute', right: '100%', top: 0, marginRight: 8,
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
