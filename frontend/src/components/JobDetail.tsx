import { useEffect, useState } from 'react';
import { api } from '../api/client';

interface JobDetailProps {
  jobId: string | null;
  onClose: () => void;
}

interface JobFullDetail {
  job_id: string;
  title: string;
  company: string;
  location: string;
  url: string;
  days_old?: number;
  salary?: string;
  full_text: string;
  sections: string[];
  n_chunks: number;
}

export default function JobDetail({ jobId, onClose }: JobDetailProps) {
  const [job, setJob] = useState<JobFullDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!jobId) {
      setJob(null);
      return;
    }
    setLoading(true);
    api.getJob(jobId)
      .then((data) => {
        setJob(data as unknown as JobFullDetail);
      })
      .catch((error) => {
        console.error('Failed to load job detail:', error);
        setJob(null);
      })
      .finally(() => setLoading(false));
  }, [jobId]);

  if (!jobId) return null;

  return (
    <div style={{
      background: 'var(--bg-secondary)', borderTop: '1px solid var(--border)',
      padding: 16, maxHeight: 300, overflowY: 'auto',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <span style={{ fontWeight: 600, fontSize: 14 }}>Job Detail</span>
        <button onClick={onClose} style={{
          background: 'transparent', border: 'none', color: 'var(--text-secondary)',
          cursor: 'pointer', fontSize: 16,
        }}>×</button>
      </div>

      {loading && <div style={{ color: 'var(--text-secondary)' }}>Loading job details...</div>}

      {!loading && !job && (
        <div style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
          Failed to load job details. Check console for errors.
        </div>
      )}

      {job && (
        <div>
          <h3 style={{ fontSize: 16, marginBottom: 4 }}>{job.title}</h3>
          <div style={{ color: 'var(--text-secondary)', fontSize: 13, marginBottom: 8 }}>
            {job.company} — {job.location}
            {job.days_old != null && ` — ${job.days_old}d ago`}
            {job.salary && ` — ${job.salary}`}
          </div>
          {job.url && (
            <a href={job.url} target="_blank" rel="noopener noreferrer" style={{
              color: 'var(--accent)', fontSize: 12, textDecoration: 'none',
            }}>
              Open posting →
            </a>
          )}
          {job.sections.length > 0 && (
            <div style={{ marginTop: 8, fontSize: 12, color: 'var(--text-secondary)' }}>
              Sections: {job.sections.join(', ')}
            </div>
          )}
          <div style={{
            marginTop: 12, fontSize: 13, lineHeight: 1.6,
            whiteSpace: 'pre-wrap', color: 'var(--text-primary)',
            maxHeight: 200, overflowY: 'auto',
          }}>
            {job.full_text}
          </div>
        </div>
      )}
    </div>
  );
}
