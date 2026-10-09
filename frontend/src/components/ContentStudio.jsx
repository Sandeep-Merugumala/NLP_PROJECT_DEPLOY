import React, { useState } from 'react';
import { Wand2, Baby, Presentation, MessageSquare, Mic, FileText, Image, Copy, Check, AlertCircle, Search } from 'lucide-react';

const FORMATS = [
  { key: 'eli5', label: 'ELI5', icon: <Baby size={16} /> },
  { key: 'slide_outline', label: 'Slides', icon: <Presentation size={16} /> },
  { key: 'tweet_thread', label: 'Tweet thread', icon: <MessageSquare size={16} /> },
  { key: 'podcast_script', label: 'Podcast', icon: <Mic size={16} /> },
  { key: 'blog_post', label: 'Blog post', icon: <FileText size={16} /> },
  { key: 'infographic_brief', label: 'Infographic', icon: <Image size={16} /> },
];

function renderMarkdown(md) {
  if (!md) return null;
  return md.split('\n').map((line, i) => {
    const t = line.trim();
    if (!t) return <div key={i} style={{ height: 8 }} />;
    if (t.startsWith('## ')) return <h3 key={i} style={{ color: 'var(--c-fg-strong)', fontSize: 17, margin: '18px 0 8px', fontFamily: 'var(--font-heading)' }}>{t.slice(3)}</h3>;
    if (t.startsWith('# ')) return <h2 key={i} style={{ color: 'var(--c-fg-strong)', fontSize: 20, margin: '20px 0 10px', fontFamily: 'var(--font-heading)' }}>{t.slice(2)}</h2>;
    if (t.startsWith('- ') || t.startsWith('* ')) return <div key={i} style={{ display: 'flex', gap: 8, margin: '4px 0 4px 8px', color: 'var(--text-secondary)', fontSize: 14, lineHeight: 1.6 }}><span style={{ color: 'var(--accent-color)' }}>•</span><span>{bold(t.slice(2))}</span></div>;
    return <p key={i} style={{ color: 'var(--text-secondary)', fontSize: 14, lineHeight: 1.7, margin: '6px 0' }}>{bold(t)}</p>;
  });
}
function bold(text) {
  return text.split(/\*\*([^*]+)\*\*/g).map((p, i) => i % 2 ? <strong key={i} style={{ color: 'var(--c-fg-strong)' }}>{p}</strong> : p);
}

export default function ContentStudio({ papers = [], geminiApiKey }) {
  const [selected, setSelected] = useState(null);
  const [fmt, setFmt] = useState('eli5');
  const [output, setOutput] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);
  const [q, setQ] = useState('');

  const generate = async (paper, format) => {
    setLoading(true); setError(null); setOutput(null); setSelected(paper); setFmt(format);
    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/content`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paper_id: paper.id, format, api_key: geminiApiKey }),
      });
      const d = await res.json();
      if (!res.ok || d.error) throw new Error(d.error || 'Generation failed');
      setOutput(d);
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  };

  const copy = () => {
    navigator.clipboard?.writeText(output?.content || '');
    setCopied(true); setTimeout(() => setCopied(false), 1500);
  };

  const filtered = papers.filter(p => p.title?.toLowerCase().includes(q.toLowerCase())).slice(0, 30);

  return (
    <div style={{ padding: '24px', maxWidth: '1100px', margin: '0 auto' }}>
      <div style={{ marginBottom: '24px' }}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '12px', margin: '0 0 8px 0', fontSize: '24px', fontFamily: 'var(--font-heading)' }}>
          <Wand2 size={28} style={{ color: 'var(--accent-color)' }} />
          Content Studio
        </h2>
        <p style={{ color: 'var(--text-secondary)', margin: 0 }}>Turn any paper into an ELI5, slide outline, tweet thread, podcast script, blog post or infographic brief.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(260px, 320px) 1fr', gap: '20px', alignItems: 'start' }}>
        {/* Paper list */}
        <div className="glass-panel" style={{ padding: '14px' }}>
          <div style={{ position: 'relative', marginBottom: '10px' }}>
            <Search size={15} style={{ position: 'absolute', left: 10, top: 11, color: 'var(--text-muted)' }} />
            <input className="glass-input" placeholder="Search papers…" value={q} onChange={e => setQ(e.target.value)} style={{ width: '100%', padding: '8px 8px 8px 32px', fontSize: 13 }} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '520px', overflowY: 'auto' }}>
            {filtered.map(p => (
              <button key={p.id} onClick={() => generate(p, fmt)} style={{
                textAlign: 'left', padding: '10px 12px', borderRadius: 8, cursor: 'pointer', fontSize: 13, lineHeight: 1.4,
                background: selected?.id === p.id ? 'rgba(139,92,246,0.18)' : 'rgba(255,255,255,0.03)',
                border: `1px solid ${selected?.id === p.id ? 'var(--accent-color)' : 'rgba(255,255,255,0.06)'}`, color: 'var(--c-fg-strong)',
              }}>{p.title}</button>
            ))}
          </div>
        </div>

        {/* Output */}
        <div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '16px' }}>
            {FORMATS.map(f => (
              <button key={f.key} onClick={() => selected && generate(selected, f.key)} disabled={!selected} style={{
                display: 'flex', alignItems: 'center', gap: 6, padding: '8px 14px', borderRadius: 20, cursor: selected ? 'pointer' : 'not-allowed', fontSize: 13, fontWeight: 600,
                background: fmt === f.key ? 'var(--accent-color)' : 'rgba(255,255,255,0.05)',
                border: `1px solid ${fmt === f.key ? 'var(--accent-color)' : 'rgba(255,255,255,0.1)'}`,
                color: fmt === f.key ? 'var(--c-fg-strong)' : 'var(--text-secondary)', opacity: selected ? 1 : 0.5,
              }}>{f.icon}{f.label}</button>
            ))}
          </div>

          {!selected && <div className="glass-panel" style={{ padding: '48px', textAlign: 'center', color: 'var(--text-muted)' }}>Pick a paper to start generating content.</div>}
          {loading && <div className="loading-spinner"><div className="spinner" /><p style={{ color: 'var(--text-secondary)' }}>Writing your {FORMATS.find(f => f.key === fmt)?.label}…</p></div>}
          {error && <div style={{ padding: 16, background: 'var(--score-negative-bg)', border: '1px solid var(--score-negative-border)', borderRadius: 10, color: 'var(--score-negative-text)', display: 'flex', gap: 10, alignItems: 'center' }}><AlertCircle size={18} />{error}</div>}

          {output && !loading && (
            <div className="glass-panel" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, gap: 10 }}>
                <h3 style={{ margin: 0, color: 'var(--c-fg-strong)', fontSize: 18, fontFamily: 'var(--font-heading)' }}>{output.title}</h3>
                <button className="glass-button-secondary" onClick={copy} style={{ padding: '6px 12px', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
                  {copied ? <><Check size={14} />Copied</> : <><Copy size={14} />Copy</>}
                </button>
              </div>
              {output.metadata && (
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
                  {Object.entries(output.metadata).map(([k, v]) => (
                    <span key={k} style={{ fontSize: 11, color: 'var(--text-muted)', background: 'rgba(255,255,255,0.05)', padding: '3px 10px', borderRadius: 12 }}>{k.replace(/_/g, ' ')}: {String(v)}</span>
                  ))}
                </div>
              )}
              <div>{renderMarkdown(output.content)}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
