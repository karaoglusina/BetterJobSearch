import React, { useState } from 'react';

const ASPECTS = [
  'default', 'skills', 'tools', 'language', 'remote_policy',
  'experience', 'education', 'domain', 'benefits', 'culture',
];

interface AspectSelectorProps {
  currentAspect: string;
  onSelect: (aspect: string) => void;
  onConceptSubmit: (concept: string) => void;
  loading?: boolean;
}

export default function AspectSelector({ currentAspect, onSelect, onConceptSubmit, loading }: AspectSelectorProps) {
  const [concept, setConcept] = useState('');

  const handleConceptSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (concept.trim()) {
      onConceptSubmit(concept.trim());
      setConcept('');
    }
  };

  return (
    <div style={{ padding: 12 }}>
      <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
        Cluster By
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
        {ASPECTS.map((aspect) => (
          <button
            key={aspect}
            onClick={() => onSelect(aspect)}
            disabled={loading}
            style={{
              padding: '4px 10px', borderRadius: 12, fontSize: 12,
              border: `1px solid ${currentAspect === aspect ? 'var(--accent)' : 'var(--border)'}`,
              background: currentAspect === aspect ? 'var(--accent)' : 'transparent',
              color: currentAspect === aspect ? 'white' : 'var(--text-secondary)',
              cursor: loading ? 'wait' : 'pointer',
              transition: 'all 0.15s',
            }}
          >
            {aspect}
          </button>
        ))}
      </div>

      {/* Custom concept input */}
      <form onSubmit={handleConceptSubmit} style={{ display: 'flex', gap: 6 }}>
        <input
          type="text"
          value={concept}
          onChange={(e) => setConcept(e.target.value)}
          placeholder="Custom concept..."
          style={{
            flex: 1, padding: '4px 8px', borderRadius: 6,
            border: '1px solid var(--border)', background: 'var(--bg-tertiary)',
            color: 'var(--text-primary)', fontSize: 12, outline: 'none',
          }}
        />
        <button type="submit" disabled={!concept.trim() || loading} style={{
          padding: '4px 10px', borderRadius: 6, border: 'none',
          background: 'var(--accent)', color: 'white', fontSize: 12,
          cursor: 'pointer', opacity: (!concept.trim() || loading) ? 0.5 : 1,
        }}>
          Go
        </button>
      </form>
    </div>
  );
}
