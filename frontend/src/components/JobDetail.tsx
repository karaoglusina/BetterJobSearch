import { useEffect, useState } from 'react';
import { api } from '../api/client';

interface JobDetailProps {
  jobId: string | null;
  onClose: () => void;
}

interface ParsedSection {
  name: string;
  raw_name: string | null;
  text: string;
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
  parsed_sections?: ParsedSection[];
  n_chunks: number;
}

// Friendly display names for canonical section types
const SECTION_LABELS: Record<string, string> = {
  intro: 'Overview',
  responsibilities: 'Responsibilities',
  requirements: 'Requirements',
  nice_to_have: 'Nice to Have',
  about_you: 'About You',
  about_company: 'About the Company',
  benefits: 'Benefits',
  culture: 'Culture',
  culture_team: 'Culture & Team',
  skills: 'Skills',
  tools: 'Tools & Tech',
  education: 'Education',
  experience: 'Experience',
  languages: 'Languages',
  how_to_apply: 'How to Apply',
  application_info: 'Application',
  role_description: 'The Role',
  other: 'Other',
};

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

          {/* Render structured sections if available */}
          {job.parsed_sections && job.parsed_sections.length > 0 ? (
            <div style={{ marginTop: 12, maxHeight: 200, overflowY: 'auto' }}>
              {job.parsed_sections.map((section, idx) => (
                <div key={idx} style={{ marginBottom: 12 }}>
                  <div style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: 'var(--accent)',
                    marginBottom: 4,
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                  }}>
                    {SECTION_LABELS[section.name] || section.raw_name || section.name}
                  </div>
                  <div style={{
                    fontSize: 13,
                    lineHeight: 1.6,
                    whiteSpace: 'pre-wrap',
                    color: 'var(--text-primary)',
                    paddingLeft: 8,
                    borderLeft: '2px solid var(--border)',
                  }}>
                    {section.text}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            /* Fallback to full text if no parsed sections */
            <div style={{
              marginTop: 12, fontSize: 13, lineHeight: 1.6,
              whiteSpace: 'pre-wrap', color: 'var(--text-primary)',
              maxHeight: 200, overflowY: 'auto',
            }}>
              {job.full_text}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
