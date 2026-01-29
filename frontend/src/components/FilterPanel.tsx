import React, { useEffect, useState } from 'react';
import { api } from '../api/client';

interface FilterPanelProps {
  onFilterChange: (filters: { location?: string; company?: string; title_contains?: string; language?: string }) => void;
  onKeywordSearch?: (query: string) => void;
  keywordSearchLoading?: boolean;
  jobSource?: string;
  availableLabels?: string[];
  selectedLabels?: string[];
  onLabelFilterChange?: (labels: string[]) => void;
}

export default function FilterPanel({
  onFilterChange,
  onKeywordSearch,
  keywordSearchLoading,
  jobSource,
  availableLabels = [],
  selectedLabels = [],
  onLabelFilterChange,
}: FilterPanelProps) {
  const [location, setLocation] = useState('');
  const [company, setCompany] = useState('');
  const [titleContains, setTitleContains] = useState('');
  const [language, setLanguage] = useState('');
  const [keywordQuery, setKeywordQuery] = useState('');
  const [languages, setLanguages] = useState<{ code: string; count: number }[]>([]);

  useEffect(() => {
    api.getLanguages().then((data) => setLanguages(data.languages)).catch(() => {});
  }, []);

  const apply = () => {
    onFilterChange({
      location: location || undefined,
      company: company || undefined,
      title_contains: titleContains || undefined,
      language: language || undefined,
    });
  };

  const clear = () => {
    setLocation('');
    setCompany('');
    setTitleContains('');
    setLanguage('');
    onFilterChange({});
  };

  const handleKeywordSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (keywordQuery.trim() && onKeywordSearch) {
      onKeywordSearch(keywordQuery.trim());
    }
  };

  const clearKeywordSearch = () => {
    setKeywordQuery('');
    onFilterChange({
      location: location || undefined,
      company: company || undefined,
      title_contains: titleContains || undefined,
      language: language || undefined,
    });
  };

  return (
    <div style={{ padding: 12 }}>
      {/* Keyword Search Section */}
      {onKeywordSearch && (
        <>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Keyword Search
          </div>
          {jobSource === 'agent' && (
            <div style={{ fontSize: 10, color: 'var(--accent)', marginBottom: 6, opacity: 0.8 }}>
              Agent results active — searching will override
            </div>
          )}
          <form onSubmit={handleKeywordSearch} style={{ marginBottom: 12 }}>
            <textarea
              value={keywordQuery}
              onChange={(e) => setKeywordQuery(e.target.value)}
              placeholder={'e.g. ("data engineer" AND python) OR spark'}
              style={{
                ...inputStyle,
                width: '100%',
                height: 64,
                resize: 'vertical',
                fontFamily: 'inherit',
                boxSizing: 'border-box',
              }}
            />
            <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
              <button
                type="submit"
                disabled={!keywordQuery.trim() || keywordSearchLoading}
                style={{
                  flex: 1, padding: '6px', borderRadius: 6, border: 'none',
                  background: 'var(--accent)', color: 'white', fontSize: 12, cursor: 'pointer',
                  opacity: (!keywordQuery.trim() || keywordSearchLoading) ? 0.5 : 1,
                }}
              >
                {keywordSearchLoading ? 'Searching...' : 'Search'}
              </button>
              <button
                type="button"
                onClick={clearKeywordSearch}
                style={{
                  padding: '6px 12px', borderRadius: 6, border: '1px solid var(--border)',
                  background: 'transparent', color: 'var(--text-secondary)', fontSize: 12, cursor: 'pointer',
                }}
              >
                Clear
              </button>
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 4, opacity: 0.7 }}>
              Supports AND, OR, NOT, "quoted phrases", (parentheses)
            </div>
          </form>
          <div style={{ borderTop: '1px solid var(--border)', marginBottom: 12 }} />
        </>
      )}

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
        {languages.length > 0 && (
          <select
            value={language}
            onChange={(e) => { setLanguage(e.target.value); }}
            style={{ ...inputStyle, cursor: 'pointer' }}
          >
            <option value="">All languages</option>
            {languages.map((l) => (
              <option key={l.code} value={l.code}>
                {l.code} ({l.count})
              </option>
            ))}
          </select>
        )}

        {/* Label filter */}
        {availableLabels.length > 0 && (
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4 }}>
              Filter by label:
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {availableLabels.map((label) => {
                const isSelected = selectedLabels.includes(label);
                return (
                  <button
                    key={label}
                    onClick={() => {
                      if (!onLabelFilterChange) return;
                      if (isSelected) {
                        onLabelFilterChange(selectedLabels.filter(l => l !== label));
                      } else {
                        onLabelFilterChange([...selectedLabels, label]);
                      }
                    }}
                    style={{
                      padding: '3px 8px',
                      borderRadius: 10,
                      fontSize: 11,
                      cursor: 'pointer',
                      border: '1px solid',
                      borderColor: isSelected ? 'var(--accent)' : 'var(--border)',
                      background: isSelected ? 'var(--accent)' : 'transparent',
                      color: isSelected ? 'white' : 'var(--text-primary)',
                    }}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
            {selectedLabels.length > 0 && (
              <button
                onClick={() => onLabelFilterChange?.([])}
                style={{
                  marginTop: 4,
                  padding: '2px 6px',
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--text-secondary)',
                  fontSize: 10,
                  cursor: 'pointer',
                  textDecoration: 'underline',
                }}
              >
                Clear label filter
              </button>
            )}
          </div>
        )}

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
