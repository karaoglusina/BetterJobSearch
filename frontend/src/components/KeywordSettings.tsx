import React, { useState, useCallback } from 'react';
import type { ClusterParams } from '../hooks/useClusters';

export interface KeywordSettingsState {
  // TF-IDF settings
  tfidf_min_df: number; // 0-50, represents percentage
  tfidf_max_df: number; // 50-100, represents percentage
  // Coherence (NPMI) settings
  npmi_min: number; // 0-1.0
  npmi_max: number; // 0-2.0
  // Domain specificity (effect size) settings
  effect_size_min: number; // 0-20
  effect_size_max: number; // 0-20
}

export const DEFAULT_KEYWORD_SETTINGS: KeywordSettingsState = {
  tfidf_min_df: 0,
  tfidf_max_df: 100,
  npmi_min: 0,
  npmi_max: 1.0,
  effect_size_min: 0,
  effect_size_max: 20,
};

interface KeywordSettingsProps {
  onApply: (params: ClusterParams) => void;
  loading?: boolean;
  settings?: KeywordSettingsState;
  onSettingsChange?: (settings: KeywordSettingsState) => void;
}

const sliderContainerStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: 2, marginBottom: 10,
};

const sliderLabelStyle: React.CSSProperties = {
  display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-secondary)',
};

const sliderStyle: React.CSSProperties = {
  width: '100%', accentColor: 'var(--accent)',
};

const sectionHeaderStyle: React.CSSProperties = {
  fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)',
  textTransform: 'uppercase', letterSpacing: '0.5px',
  marginTop: 8, marginBottom: 6, paddingTop: 8,
  borderTop: '1px solid var(--border)',
};

export default function KeywordSettings({
  onApply,
  loading = false,
  settings: externalSettings,
  onSettingsChange,
}: KeywordSettingsProps) {
  const [localSettings, setLocalSettings] = useState<KeywordSettingsState>(DEFAULT_KEYWORD_SETTINGS);
  const [collapsed, setCollapsed] = useState(true);

  // Use external settings if provided (controlled mode), otherwise use local state
  const settings = externalSettings ?? localSettings;
  const setSettings = onSettingsChange ?? setLocalSettings;

  const updateSetting = useCallback((key: keyof KeywordSettingsState, value: number) => {
    setSettings({ ...settings, [key]: value });
  }, [settings, setSettings]);

  const handleApply = useCallback(() => {
    onApply({
      tfidf_min_df: settings.tfidf_min_df > 0 ? settings.tfidf_min_df / 100 : undefined,
      tfidf_max_df: settings.tfidf_max_df < 100 ? settings.tfidf_max_df / 100 : undefined,
      npmi_min: settings.npmi_min > 0 ? settings.npmi_min : undefined,
      npmi_max: settings.npmi_max < 1.0 ? settings.npmi_max : undefined,
      effect_size_min: settings.effect_size_min > 0 ? settings.effect_size_min : undefined,
      effect_size_max: settings.effect_size_max < 20 ? settings.effect_size_max : undefined,
    });
  }, [onApply, settings]);

  const handleReset = useCallback(() => {
    setSettings(DEFAULT_KEYWORD_SETTINGS);
  }, [setSettings]);

  return (
    <div style={{ padding: '8px 12px' }}>
      <div
        style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          cursor: 'pointer', userSelect: 'none',
        }}
        onClick={() => setCollapsed(!collapsed)}
      >
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
          Keyword Settings
        </span>
        <span style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
          {collapsed ? '+ expand' : '- collapse'}
        </span>
      </div>

      {!collapsed && (
        <div style={{ marginTop: 10 }}>
          {/* TF-IDF Section */}
          <div style={{ ...sectionHeaderStyle, marginTop: 0, borderTop: 'none', paddingTop: 0 }}>
            TF-IDF Frequency
          </div>

          <div style={sliderContainerStyle}>
            <div style={sliderLabelStyle}>
              <span>Min Doc Freq</span>
              <span>{settings.tfidf_min_df}%</span>
            </div>
            <input
              type="range"
              min={0} max={50} step={1}
              value={settings.tfidf_min_df}
              onChange={(e) => updateSetting('tfidf_min_df', Number(e.target.value))}
              style={sliderStyle}
            />
          </div>

          <div style={sliderContainerStyle}>
            <div style={sliderLabelStyle}>
              <span>Max Doc Freq</span>
              <span>{settings.tfidf_max_df}%</span>
            </div>
            <input
              type="range"
              min={50} max={100} step={1}
              value={settings.tfidf_max_df}
              onChange={(e) => updateSetting('tfidf_max_df', Number(e.target.value))}
              style={sliderStyle}
            />
          </div>

          {/* Coherence (NPMI) Section */}
          <div style={sectionHeaderStyle}>
            Coherence (NPMI)
          </div>

          <div style={sliderContainerStyle}>
            <div style={sliderLabelStyle}>
              <span>Min Coherence</span>
              <span>{settings.npmi_min.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0} max={1} step={0.05}
              value={settings.npmi_min}
              onChange={(e) => updateSetting('npmi_min', Number(e.target.value))}
              style={sliderStyle}
            />
          </div>

          <div style={sliderContainerStyle}>
            <div style={sliderLabelStyle}>
              <span>Max Coherence</span>
              <span>{settings.npmi_max.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0} max={2} step={0.05}
              value={settings.npmi_max}
              onChange={(e) => updateSetting('npmi_max', Number(e.target.value))}
              style={sliderStyle}
            />
          </div>

          {/* Domain Specificity Section */}
          <div style={sectionHeaderStyle}>
            Domain Specificity
          </div>

          <div style={sliderContainerStyle}>
            <div style={sliderLabelStyle}>
              <span>Min Specificity</span>
              <span>{settings.effect_size_min.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min={0} max={15} step={0.5}
              value={settings.effect_size_min}
              onChange={(e) => updateSetting('effect_size_min', Number(e.target.value))}
              style={sliderStyle}
            />
          </div>

          <div style={sliderContainerStyle}>
            <div style={sliderLabelStyle}>
              <span>Max Specificity</span>
              <span>{settings.effect_size_max.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min={5} max={20} step={0.5}
              value={settings.effect_size_max}
              onChange={(e) => updateSetting('effect_size_max', Number(e.target.value))}
              style={sliderStyle}
            />
          </div>

          {/* Action buttons */}
          <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
            <button
              onClick={handleApply}
              disabled={loading}
              style={{
                flex: 1, padding: '6px 12px', borderRadius: 4,
                background: 'var(--accent)', border: 'none',
                color: 'white', fontSize: 11, fontWeight: 600,
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.6 : 1,
              }}
            >
              {loading ? 'Applying...' : 'Apply'}
            </button>
            <button
              onClick={handleReset}
              disabled={loading}
              style={{
                padding: '6px 12px', borderRadius: 4,
                background: 'transparent', border: '1px solid var(--border)',
                color: 'var(--text-secondary)', fontSize: 11,
                cursor: loading ? 'not-allowed' : 'pointer',
              }}
            >
              Reset
            </button>
          </div>

          <div style={{ marginTop: 8, fontSize: 10, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
            Higher coherence = more meaningful phrases.
            Higher specificity = more job-domain unique terms.
          </div>
        </div>
      )}
    </div>
  );
}
