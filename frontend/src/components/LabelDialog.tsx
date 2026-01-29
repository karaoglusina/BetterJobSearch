import { useState, useCallback, useRef, useEffect } from 'react';
import type { JobLabels } from '../hooks/useLabels';

interface LabelDialogProps {
  isOpen: boolean;
  onClose: () => void;
  selectedJobIds: Set<string>;
  existingLabels: string[];
  jobLabels: JobLabels;
  onApplyLabel: (label: string) => void;
  onRemoveLabel: (label: string) => void;
}

export default function LabelDialog({
  isOpen,
  onClose,
  selectedJobIds,
  existingLabels,
  jobLabels,
  onApplyLabel,
  onRemoveLabel,
}: LabelDialogProps) {
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus input when dialog opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  // Get label status for selected jobs
  const getLabelStatus = useCallback((label: string): 'all' | 'some' | 'none' => {
    const jobIds = Array.from(selectedJobIds);
    const count = jobIds.filter(id => (jobLabels[id] || []).includes(label)).length;
    if (count === 0) return 'none';
    if (count === jobIds.length) return 'all';
    return 'some';
  }, [selectedJobIds, jobLabels]);

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = inputValue.trim();
    if (trimmed) {
      onApplyLabel(trimmed);
      setInputValue('');
    }
  }, [inputValue, onApplyLabel]);

  const handleLabelClick = useCallback((label: string) => {
    const status = getLabelStatus(label);
    if (status === 'all') {
      onRemoveLabel(label);
    } else {
      onApplyLabel(label);
    }
  }, [getLabelStatus, onApplyLabel, onRemoveLabel]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    }
  }, [onClose]);

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={onClose}
      onKeyDown={handleKeyDown}
    >
      <div
        style={{
          background: 'var(--bg-primary)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          padding: 20,
          minWidth: 320,
          maxWidth: 400,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <h3 style={{ margin: '0 0 16px', fontSize: 14, fontWeight: 600 }}>
          Label {selectedJobIds.size} job{selectedJobIds.size > 1 ? 's' : ''}
        </h3>

        {/* New label input */}
        <form onSubmit={handleSubmit} style={{ marginBottom: 16 }}>
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Type new label and press Enter..."
            style={{
              width: '100%',
              padding: '8px 12px',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border)',
              borderRadius: 4,
              color: 'var(--text-primary)',
              fontSize: 13,
            }}
          />
        </form>

        {/* Existing labels */}
        {existingLabels.length > 0 && (
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 8 }}>
              Click to toggle label:
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {existingLabels.map((label) => {
                const status = getLabelStatus(label);
                return (
                  <button
                    key={label}
                    onClick={() => handleLabelClick(label)}
                    style={{
                      padding: '4px 10px',
                      borderRadius: 12,
                      fontSize: 12,
                      cursor: 'pointer',
                      border: '1px solid',
                      borderColor: status === 'none' ? 'var(--border)' : 'var(--accent)',
                      background: status === 'all' ? 'var(--accent)' : status === 'some' ? 'rgba(99, 102, 241, 0.3)' : 'transparent',
                      color: status === 'all' ? 'white' : 'var(--text-primary)',
                    }}
                  >
                    {label}
                    {status === 'some' && ' (partial)'}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Current labels on selected jobs */}
        {selectedJobIds.size === 1 && (
          <div style={{ marginTop: 16 }}>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4 }}>
              Labels on this job:
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-primary)' }}>
              {(jobLabels[Array.from(selectedJobIds)[0]] || []).join(', ') || 'None'}
            </div>
          </div>
        )}

        {/* Close button */}
        <div style={{ marginTop: 20, display: 'flex', justifyContent: 'flex-end' }}>
          <button
            onClick={onClose}
            style={{
              padding: '6px 16px',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border)',
              borderRadius: 4,
              color: 'var(--text-primary)',
              cursor: 'pointer',
              fontSize: 12,
            }}
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
