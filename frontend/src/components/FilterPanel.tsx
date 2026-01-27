import React, { useState } from 'react';

interface FilterPanelProps {
  onFilterChange: (filters: { location?: string; company?: string; title_contains?: string }) => void;
}

export default function FilterPanel({ onFilterChange }: FilterPanelProps) {
  const [location, setLocation] = useState('');
  const [company, setCompany] = useState('');
  const [titleContains, setTitleContains] = useState('');

  const apply = () => {
    onFilterChange({
      location: location || undefined,
      company: company || undefined,
      title_contains: titleContains || undefined,
    });
  };

  const clear = () => {
    setLocation('');
    setCompany('');
    setTitleContains('');
    onFilterChange({});
  };

  return (
    <div style={{ padding: 12 }}>
      <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
        Filters
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <input
          type="text" placeholder="Location..."
          value={location} onChange={(e) => setLocation(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && apply()}
          style={inputStyle}
        />
        <input
          type="text" placeholder="Company..."
          value={company} onChange={(e) => setCompany(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && apply()}
          style={inputStyle}
        />
        <input
          type="text" placeholder="Title contains..."
          value={titleContains} onChange={(e) => setTitleContains(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && apply()}
          style={inputStyle}
        />

        <div style={{ display: 'flex', gap: 6 }}>
          <button onClick={apply} style={{
            flex: 1, padding: '6px', borderRadius: 6, border: 'none',
            background: 'var(--accent)', color: 'white', fontSize: 12, cursor: 'pointer',
          }}>
            Apply
          </button>
          <button onClick={clear} style={{
            padding: '6px 12px', borderRadius: 6, border: '1px solid var(--border)',
            background: 'transparent', color: 'var(--text-secondary)', fontSize: 12, cursor: 'pointer',
          }}>
            Clear
          </button>
        </div>
      </div>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  padding: '6px 8px',
  borderRadius: 6,
  border: '1px solid var(--border)',
  background: 'var(--bg-tertiary)',
  color: 'var(--text-primary)',
  fontSize: 12,
  outline: 'none',
};
