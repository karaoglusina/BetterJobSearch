import React, { useState } from 'react';

export interface UIPreset {
  name: string;
  aspect: string;
  view: 'scatter' | 'table';
  clusterFilterEnabled: boolean;
}

const STORAGE_KEY = 'bjs_presets';

function loadPresets(): UIPreset[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function savePresets(presets: UIPreset[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(presets));
}

interface PresetBarProps {
  currentState: Omit<UIPreset, 'name'>;
  onRestore: (preset: UIPreset) => void;
}

export default function PresetBar({ currentState, onRestore }: PresetBarProps) {
  const [presets, setPresets] = useState<UIPreset[]>(loadPresets);
  const [showSave, setShowSave] = useState(false);
  const [newName, setNewName] = useState('');

  const handleSave = () => {
    if (!newName.trim()) return;
    const preset: UIPreset = { ...currentState, name: newName.trim() };
    const updated = [...presets.filter(p => p.name !== preset.name), preset];
    setPresets(updated);
    savePresets(updated);
    setNewName('');
    setShowSave(false);
  };

  const handleDelete = (name: string) => {
    const updated = presets.filter(p => p.name !== name);
    setPresets(updated);
    savePresets(updated);
  };

  if (presets.length === 0 && !showSave) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        <button onClick={() => setShowSave(true)} style={addBtnStyle} title="Save current view as preset">
          + Save view
        </button>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexWrap: 'wrap' }}>
      {presets.map((p) => (
        <div key={p.name} style={{ display: 'inline-flex', alignItems: 'center' }}>
          <button
            onClick={() => onRestore(p)}
            style={presetBtnStyle}
            title={`Restore: ${p.aspect} / ${p.view}`}
          >
            {p.name}
          </button>
          <button
            onClick={() => handleDelete(p.name)}
            style={deleteBtnStyle}
            title="Delete preset"
          >
            x
          </button>
        </div>
      ))}
      {showSave ? (
        <form onSubmit={(e) => { e.preventDefault(); handleSave(); }} style={{ display: 'flex', gap: 2 }}>
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="Name..."
            autoFocus
            style={{
              padding: '2px 6px', borderRadius: 4, border: '1px solid var(--border)',
              background: 'var(--bg-tertiary)', color: 'var(--text-primary)',
              fontSize: 11, width: 80, outline: 'none',
            }}
          />
          <button type="submit" style={{ ...addBtnStyle, padding: '2px 6px' }}>OK</button>
          <button type="button" onClick={() => setShowSave(false)} style={{ ...addBtnStyle, padding: '2px 6px' }}>X</button>
        </form>
      ) : (
        <button onClick={() => setShowSave(true)} style={addBtnStyle}>+</button>
      )}
    </div>
  );
}

const presetBtnStyle: React.CSSProperties = {
  padding: '2px 8px', borderRadius: '4px 0 0 4px', fontSize: 11,
  border: '1px solid var(--border)', borderRight: 'none',
  background: 'var(--bg-tertiary)', color: 'var(--text-secondary)',
  cursor: 'pointer',
};

const deleteBtnStyle: React.CSSProperties = {
  padding: '2px 4px', borderRadius: '0 4px 4px 0', fontSize: 9,
  border: '1px solid var(--border)',
  background: 'var(--bg-tertiary)', color: 'var(--text-secondary)',
  cursor: 'pointer', opacity: 0.6,
};

const addBtnStyle: React.CSSProperties = {
  padding: '2px 8px', borderRadius: 4, fontSize: 11,
  border: '1px solid var(--border)',
  background: 'transparent', color: 'var(--text-secondary)',
  cursor: 'pointer',
};
