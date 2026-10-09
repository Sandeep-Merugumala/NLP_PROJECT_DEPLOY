import React, { useState, useEffect } from 'react';
import { Bell, Plus, RadioTower, Hash, User, Building2, Check, CheckCheck, AlertCircle, Trash2 } from 'lucide-react';

const API = `${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/watchlists`;
const TYPE_ICON = { keyword: <Hash size={13} />, author: <User size={13} />, venue: <Building2 size={13} />, topic: <RadioTower size={13} /> };

export default function Watchlist() {
  const [watchlists, setWatchlists] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [newName, setNewName] = useState('');
  const [itemDraft, setItemDraft] = useState({});
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      const res = await fetch(API);
      const d = await res.json();
      setWatchlists(d.watchlists || []);
      setAlerts(d.unread_alerts || []);
    } catch (e) { setError('Could not load watchlists'); } finally { setLoading(false); }
  };
  useEffect(() => { load(); }, []);

  const post = async (body) => {
    const res = await fetch(API, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    return res.json();
  };

  const createWatchlist = async () => {
    if (!newName.trim()) return;
    await post({ action: 'create', name: newName.trim() });
    setNewName(''); load();
  };
  const addItem = async (wid) => {
    const draft = itemDraft[wid] || { type: 'keyword', value: '' };
    if (!draft.value.trim()) return;
    await post({ action: 'add_item', watchlist_id: wid, type: draft.type, value: draft.value.trim() });
    setItemDraft(d => ({ ...d, [wid]: { ...draft, value: '' } })); load();
  };
  const removeItem = async (wid, type, value) => { await post({ action: 'remove_item', watchlist_id: wid, type, value }); load(); };
  const scan = async () => { setScanning(true); await post({ action: 'scan' }); await load(); setScanning(false); };
  const markRead = async (alert_id) => { await post({ action: 'mark_read', alert_id }); load(); };
  const markAll = async () => { await post({ action: 'mark_all_read' }); load(); };

  return (
    <div style={{ padding: '24px', maxWidth: '1000px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, marginBottom: '24px', flexWrap: 'wrap' }}>
        <div>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '12px', margin: '0 0 8px 0', fontSize: '24px', fontFamily: 'var(--font-heading)' }}>
            <Bell size={28} style={{ color: 'var(--accent-color)' }} />
            Watchlists & Alerts
            {alerts.length > 0 && <span style={{ fontSize: 13, background: 'var(--score-negative-text)', color: '#000', borderRadius: 20, padding: '2px 10px', fontWeight: 700 }}>{alerts.length}</span>}
          </h2>
          <p style={{ color: 'var(--text-secondary)', margin: 0 }}>Follow keywords, authors and venues. Scan the newest papers for matches.</p>
        </div>
        <button className="glass-button" onClick={scan} disabled={scanning} style={{ padding: '10px 18px', display: 'flex', alignItems: 'center', gap: 6 }}>
          <RadioTower size={16} /> {scanning ? 'Scanning…' : 'Scan new papers'}
        </button>
      </div>

      {error && <div style={{ padding: 14, background: 'var(--score-negative-bg)', border: '1px solid var(--score-negative-border)', borderRadius: 10, color: 'var(--score-negative-text)', marginBottom: 16, display: 'flex', gap: 10, alignItems: 'center' }}><AlertCircle size={18} />{error}</div>}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr minmax(280px, 360px)', gap: '20px', alignItems: 'start' }}>
        {/* Watchlists column */}
        <div>
          <div className="glass-panel" style={{ padding: '14px', marginBottom: '16px', display: 'flex', gap: 10 }}>
            <input className="glass-input" placeholder="New watchlist name…" value={newName} onChange={e => setNewName(e.target.value)} onKeyDown={e => e.key === 'Enter' && createWatchlist()} style={{ flex: 1, padding: '10px 14px' }} />
            <button className="glass-button-secondary" onClick={createWatchlist} style={{ padding: '10px 16px', display: 'flex', alignItems: 'center', gap: 6 }}><Plus size={15} />Create</button>
          </div>

          {loading ? <div className="loading-spinner"><div className="spinner" /></div> : watchlists.length === 0 ? (
            <div className="glass-panel" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>No watchlists yet. Create one and add keywords to follow.</div>
          ) : watchlists.map(w => {
            const draft = itemDraft[w.watchlist_id] || { type: 'keyword', value: '' };
            return (
              <div key={w.watchlist_id} className="glass-panel" style={{ padding: '18px', marginBottom: '16px' }}>
                <h4 style={{ margin: '0 0 12px 0', color: 'var(--c-fg-strong)', fontSize: 16 }}>{w.name}</h4>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
                  {(w.items || []).map((it, i) => (
                    <span key={i} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, color: 'var(--text-secondary)', background: 'rgba(139,92,246,0.12)', border: '1px solid rgba(139,92,246,0.25)', padding: '4px 10px', borderRadius: 16 }}>
                      {TYPE_ICON[it.type]}{it.value}
                      {it.hit_count > 0 && <span style={{ color: 'var(--accent-color)', fontWeight: 700 }}>· {it.hit_count}</span>}
                      <button onClick={() => removeItem(w.watchlist_id, it.type, it.value)} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: 0, display: 'flex' }}><Trash2 size={11} /></button>
                    </span>
                  ))}
                  {(w.items || []).length === 0 && <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>No items yet.</span>}
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <select value={draft.type} onChange={e => setItemDraft(d => ({ ...d, [w.watchlist_id]: { ...draft, type: e.target.value } }))} className="glass-input" style={{ padding: '8px 10px', fontSize: 13 }}>
                    <option value="keyword">Keyword</option><option value="author">Author</option><option value="venue">Venue</option><option value="topic">Topic</option>
                  </select>
                  <input className="glass-input" placeholder="value to follow…" value={draft.value} onChange={e => setItemDraft(d => ({ ...d, [w.watchlist_id]: { ...draft, value: e.target.value } }))} onKeyDown={e => e.key === 'Enter' && addItem(w.watchlist_id)} style={{ flex: 1, padding: '8px 12px', fontSize: 13 }} />
                  <button className="glass-button-secondary" onClick={() => addItem(w.watchlist_id)} style={{ padding: '8px 12px' }}><Plus size={14} /></button>
                </div>
              </div>
            );
          })}
        </div>

        {/* Alerts column */}
        <div className="glass-panel" style={{ padding: '18px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
            <h4 style={{ margin: 0, color: 'var(--c-fg-strong)', fontSize: 15, display: 'flex', alignItems: 'center', gap: 6 }}><Bell size={16} style={{ color: 'var(--accent-color)' }} />Alerts</h4>
            {alerts.length > 0 && <button onClick={markAll} style={{ background: 'none', border: 'none', color: 'var(--accent-color)', cursor: 'pointer', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}><CheckCheck size={14} />Mark all</button>}
          </div>
          {alerts.length === 0 ? (
            <p style={{ color: 'var(--text-muted)', fontSize: 13, margin: 0 }}>No unread alerts. Add items and hit “Scan new papers”.</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, maxHeight: 520, overflowY: 'auto' }}>
              {alerts.map(a => (
                <div key={a.alert_id} style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 10, padding: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, marginBottom: 4 }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: 'var(--accent-secondary)' }}>{TYPE_ICON[a.watch_type]}{a.watch_item}</span>
                    <button onClick={() => markRead(a.alert_id)} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', display: 'flex' }} title="Mark read"><Check size={14} /></button>
                  </div>
                  <div style={{ fontSize: 13, color: 'var(--c-fg-strong)', lineHeight: 1.4 }}>{a.paper_title}</div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>match strength {(a.match_strength).toFixed(2)}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
