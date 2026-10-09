import React, { useState } from 'react';
import { Route, Clock, Target, Milestone, Sparkles, AlertCircle, GraduationCap } from 'lucide-react';

const LEVEL_STYLE = {
  foundational: { bg: 'rgba(16,185,129,0.15)', border: 'rgba(16,185,129,0.35)', text: '#34d399', label: 'Foundational' },
  intermediate: { bg: 'rgba(6,182,212,0.15)', border: 'rgba(6,182,212,0.35)', text: '#22d3ee', label: 'Intermediate' },
  advanced: { bg: 'rgba(139,92,246,0.18)', border: 'rgba(139,92,246,0.4)', text: '#a78bfa', label: 'Advanced' },
};

export default function LearningPath({ geminiApiKey }) {
  const [topic, setTopic] = useState('');
  const [maxPapers, setMaxPapers] = useState(6);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const generate = async () => {
    if (!topic.trim()) return;
    setLoading(true); setError(null); setData(null);
    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/learning-path`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: topic.trim(), max_papers: maxPapers, api_key: geminiApiKey }),
      });
      const d = await res.json();
      if (!res.ok || d.error) throw new Error(d.error || 'Failed to build path');
      setData(d);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const meta = data?.metadata;

  return (
    <div style={{ padding: '24px', maxWidth: '1000px', margin: '0 auto' }}>
      <div style={{ marginBottom: '28px' }}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '12px', margin: '0 0 8px 0', fontSize: '24px', fontFamily: 'var(--font-heading)' }}>
          <Route size={28} style={{ color: 'var(--accent-color)' }} />
          Learning Path
        </h2>
        <p style={{ color: 'var(--text-secondary)', margin: 0 }}>
          Build an ordered reading plan from foundational to advanced papers for any topic.
        </p>
      </div>

      {/* Input */}
      <div className="glass-panel" style={{ padding: '18px', marginBottom: '24px', display: 'flex', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <input
          className="glass-input"
          placeholder="e.g. diffusion models, graph neural networks…"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && generate()}
          style={{ flex: 1, minWidth: '240px', padding: '12px 16px' }}
        />
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '13px' }}>
          <span>Papers</span>
          <input type="range" min="3" max="8" value={maxPapers} onChange={(e) => setMaxPapers(+e.target.value)} />
          <strong style={{ color: 'var(--accent-color)' }}>{maxPapers}</strong>
        </div>
        <button className="glass-button" onClick={generate} disabled={loading || !topic.trim()} style={{ padding: '10px 20px' }}>
          {loading ? 'Building…' : <><Sparkles size={15} style={{ marginRight: 6, display: 'inline', verticalAlign: '-2px' }} />Generate</>}
        </button>
      </div>

      {error && (
        <div style={{ padding: '16px', background: 'var(--score-negative-bg)', border: '1px solid var(--score-negative-border)', borderRadius: '10px', color: 'var(--score-negative-text)', marginBottom: '20px', display: 'flex', gap: '10px', alignItems: 'center' }}>
          <AlertCircle size={18} /> {error}
        </div>
      )}

      {loading && (
        <div className="loading-spinner"><div className="spinner" /><p style={{ color: 'var(--text-secondary)' }}>Classifying difficulty & ordering papers…</p></div>
      )}

      {meta && (
        <>
          {/* Metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '14px', marginBottom: '24px' }}>
            {[
              { icon: <GraduationCap size={18} />, label: 'Papers', value: meta.num_papers },
              { icon: <Clock size={18} />, label: 'Est. reading', value: `${meta.estimated_hours} h` },
              { icon: <Target size={18} />, label: 'Topic coverage', value: `${Math.round((meta.topic_coverage || 0) * 100)}%` },
              { icon: <Milestone size={18} />, label: 'Milestones', value: meta.milestones?.length || 0 },
            ].map((m, i) => (
              <div key={i} className="glass-panel" style={{ padding: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '12px', marginBottom: '6px' }}>
                  <span style={{ color: 'var(--accent-color)' }}>{m.icon}</span>{m.label}
                </div>
                <div style={{ fontSize: '22px', fontWeight: 700, color: 'var(--c-fg-strong)', fontFamily: 'var(--font-heading)' }}>{m.value}</div>
              </div>
            ))}
          </div>

          {/* Difficulty curve */}
          {meta.difficulty_curve?.length > 0 && (
            <div className="glass-panel" style={{ padding: '18px', marginBottom: '24px' }}>
              <h4 style={{ margin: '0 0 14px 0', color: 'var(--text-secondary)', fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Understanding curve</h4>
              <div style={{ display: 'flex', alignItems: 'flex-end', gap: '6px', height: '80px' }}>
                {meta.difficulty_curve.map((v, i) => (
                  <div key={i} title={`Step ${i + 1}: ${Math.round(v * 100)}%`} style={{
                    flex: 1, height: `${Math.max(8, v * 100)}%`, borderRadius: '6px 6px 0 0',
                    background: 'linear-gradient(180deg, var(--accent-color), var(--accent-secondary))', opacity: 0.4 + v * 0.6,
                  }} />
                ))}
              </div>
            </div>
          )}

          {/* Path timeline */}
          <div style={{ position: 'relative', paddingLeft: '28px' }}>
            <div style={{ position: 'absolute', left: '9px', top: '8px', bottom: '8px', width: '2px', background: 'linear-gradient(180deg, var(--accent-color), var(--accent-secondary))', opacity: 0.4 }} />
            {data.path.map((node, idx) => {
              const s = LEVEL_STYLE[node.level] || LEVEL_STYLE.intermediate;
              const ms = meta.milestones?.find(m => m.after_paper === idx);
              return (
                <div key={node.paper_id} style={{ position: 'relative', marginBottom: '16px' }}>
                  <div style={{ position: 'absolute', left: '-28px', top: '18px', width: '20px', height: '20px', borderRadius: '50%', background: s.text, boxShadow: `0 0 0 4px var(--bg-color)`, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#000', fontSize: '11px', fontWeight: 700 }}>{idx + 1}</div>
                  <div className="glass-panel" style={{ padding: '16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', marginBottom: '8px', flexWrap: 'wrap' }}>
                      <span style={{ background: s.bg, border: `1px solid ${s.border}`, color: s.text, padding: '2px 10px', borderRadius: '20px', fontSize: '11px', fontWeight: 600 }}>{s.label}</span>
                      <span style={{ color: 'var(--text-muted)', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px' }}><Clock size={12} /> {node.estimated_read_time} min</span>
                    </div>
                    <h4 style={{ margin: '0 0 6px 0', color: 'var(--c-fg-strong)', fontSize: '15px', lineHeight: 1.4 }}>{node.title}</h4>
                    <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '13px', lineHeight: 1.5 }}>{node.reasoning}</p>
                  </div>
                  {ms && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: '12px 0', color: '#fbbf24', fontSize: '13px', fontWeight: 600 }}>
                      <Milestone size={16} /> {ms.milestone}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
