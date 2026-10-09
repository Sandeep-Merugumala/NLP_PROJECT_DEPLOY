import React, { useState, useEffect } from 'react';
import { NotebookPen, Plus, Search, Trash2, Link2, Tag, X, AlertCircle } from 'lucide-react';

const API = `${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/notes`;

export default function Notebook({ papers = [] }) {
  const [notes, setNotes] = useState([]);
  const [query, setQuery] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({ paper_id: '', title: '', content: '', tags: '' });
  const [paperQ, setPaperQ] = useState('');
  const [related, setRelated] = useState({});
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const paperTitle = (id) => papers.find(p => p.id === id)?.title || id;

  const load = async (q = '') => {
    setLoading(true);
    try {
      const res = await fetch(q ? `${API}?query=${encodeURIComponent(q)}` : API);
      const d = await res.json();
      setNotes(d.notes || []);
    } catch (e) { setError('Could not load notes'); } finally { setLoading(false); }
  };
  useEffect(() => { load(); }, []);

  const save = async () => {
    if (!form.paper_id || !form.content.trim()) { setError('Pick a paper and write something.'); return; }
    setError(null);
    const res = await fetch(API, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'create', ...form, tags: form.tags.split(',').map(t => t.trim()).filter(Boolean) }),
    });
    const d = await res.json();
    if (d.status === 'success') { setShowForm(false); setForm({ paper_id: '', title: '', content: '', tags: '' }); setPaperQ(''); load(); }
    else setError(d.error || 'Could not save note');
  };

  const remove = async (note_id) => {
    await fetch(API, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'delete', note_id }) });
    load(query);
  };

  const findRelated = async (note_id) => {
    if (related[note_id]) { setRelated(r => ({ ...r, [note_id]: null })); return; }
    const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/notes/related/${note_id}`);
    const d = await res.json();
    setRelated(r => ({ ...r, [note_id]: d.papers || [] }));
  };

  const paperMatches = papers.filter(p => p.title?.toLowerCase().includes(paperQ.toLowerCase())).slice(0, 8);

  return (
    <div style={{ padding: '24px', maxWidth: '1000px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, marginBottom: '24px', flexWrap: 'wrap' }}>
        <div>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '12px', margin: '0 0 8px 0', fontSize: '24px', fontFamily: 'var(--font-heading)' }}>
            <NotebookPen size={28} style={{ color: 'var(--accent-color)' }} /> Notebook
          </h2>
          <p style={{ color: 'var(--text-secondary)', margin: 0 }}>Your second brain — notes attached to papers, searchable semantically and auto-linked to related work.</p>
        </div>
        <button className="glass-button" onClick={() => setShowForm(v => !v)} style={{ padding: '10px 18px', display: 'flex', alignItems: 'center', gap: 6 }}>
          <Plus size={16} /> New note
        </button>
      </div>

      {error && <div style={{ padding: 14, background: 'var(--score-negative-bg)', border: '1px solid var(--score-negative-border)', borderRadius: 10, color: 'var(--score-negative-text)', marginBottom: 16, display: 'flex', gap: 10, alignItems: 'center' }}><AlertCircle size={18} />{error}</div>}

      {/* New note form */}
      {showForm && (
        <div className="glass-panel" style={{ padding: '20px', marginBottom: '24px' }}>
          <div style={{ marginBottom: 12 }}>
            {form.paper_id ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
                <span style={{ color: 'var(--text-secondary)' }}>On paper:</span>
                <span style={{ color: 'var(--c-fg-strong)', fontWeight: 600 }}>{paperTitle(form.paper_id).slice(0, 70)}</span>
                <button onClick={() => setForm(f => ({ ...f, paper_id: '' }))} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}><X size={14} /></button>
              </div>
            ) : (
              <>
                <input className="glass-input" placeholder="Search a paper to attach…" value={paperQ} onChange={e => setPaperQ(e.target.value)} style={{ width: '100%', padding: '10px 14px', marginBottom: 6 }} />
                {paperQ && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 180, overflowY: 'auto' }}>
                    {paperMatches.map(p => (
                      <button key={p.id} onClick={() => { setForm(f => ({ ...f, paper_id: p.id })); setPaperQ(''); }} style={{ textAlign: 'left', padding: '8px 10px', borderRadius: 6, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', color: 'var(--c-fg-strong)', cursor: 'pointer', fontSize: 12 }}>{p.title}</button>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
          <input className="glass-input" placeholder="Note title" value={form.title} onChange={e => setForm(f => ({ ...f, title: e.target.value }))} style={{ width: '100%', padding: '10px 14px', marginBottom: 10 }} />
          <textarea className="glass-input" placeholder="Your thoughts, key takeaways, questions… (Markdown ok)" value={form.content} onChange={e => setForm(f => ({ ...f, content: e.target.value }))} rows={4} style={{ width: '100%', padding: '10px 14px', marginBottom: 10, resize: 'vertical', fontFamily: 'inherit' }} />
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <input className="glass-input" placeholder="tags, comma, separated" value={form.tags} onChange={e => setForm(f => ({ ...f, tags: e.target.value }))} style={{ flex: 1, minWidth: 180, padding: '10px 14px' }} />
            <button className="glass-button" onClick={save} style={{ padding: '10px 20px' }}>Save note</button>
          </div>
        </div>
      )}

      {/* Search */}
      <div style={{ position: 'relative', marginBottom: '20px' }}>
        <Search size={16} style={{ position: 'absolute', left: 12, top: 13, color: 'var(--text-muted)' }} />
        <input className="glass-input" placeholder="Semantic search across your notes…" value={query}
          onChange={e => { setQuery(e.target.value); }} onKeyDown={e => e.key === 'Enter' && load(query)}
          style={{ width: '100%', padding: '11px 12px 11px 38px' }} />
      </div>

      {loading ? <div className="loading-spinner"><div className="spinner" /></div> : notes.length === 0 ? (
        <div className="glass-panel" style={{ padding: '48px', textAlign: 'center', color: 'var(--text-muted)' }}>No notes yet. Create your first note to start building your second brain.</div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '16px' }}>
          {notes.map(n => (
            <div key={n.note_id} className="glass-panel" style={{ padding: '18px', display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                <h4 style={{ margin: 0, color: 'var(--c-fg-strong)', fontSize: 15 }}>{n.title || 'Untitled'}</h4>
                <button onClick={() => remove(n.note_id)} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }} title="Delete"><Trash2 size={15} /></button>
              </div>
              <div style={{ fontSize: 11, color: 'var(--accent-secondary)' }}>on “{paperTitle(n.paper_id).slice(0, 48)}”</div>
              <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: 13, lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>{n.content}</p>
              {n.tags?.length > 0 && (
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {n.tags.map(t => <span key={t} style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: 10, color: 'var(--text-muted)', background: 'rgba(255,255,255,0.05)', padding: '2px 8px', borderRadius: 10 }}><Tag size={10} />{t}</span>)}
                </div>
              )}
              {typeof n.search_score === 'number' && <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>match {(n.search_score).toFixed(2)}</div>}
              <button onClick={() => findRelated(n.note_id)} style={{ marginTop: 4, alignSelf: 'flex-start', display: 'flex', alignItems: 'center', gap: 5, background: 'none', border: 'none', color: 'var(--accent-color)', cursor: 'pointer', fontSize: 12, fontWeight: 600 }}>
                <Link2 size={13} /> {related[n.note_id] ? 'Hide related' : 'Related papers'}
              </button>
              {related[n.note_id]?.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4, borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 8 }}>
                  {related[n.note_id].map(p => <div key={p.id} style={{ fontSize: 12, color: 'var(--text-secondary)' }}>• {p.title.slice(0, 60)} <span style={{ color: 'var(--text-muted)' }}>({p.score})</span></div>)}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
