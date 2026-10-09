import React, { useState, useEffect, useRef } from 'react';
import { Share2, Users, Building2, RefreshCw, AlertCircle } from 'lucide-react';

export default function Network() {
  const canvasRef = useRef(null);
  const simRef = useRef({ nodes: [], edges: [], raf: null, drag: null });
  const [graph, setGraph] = useState(null);
  const [topAuthors, setTopAuthors] = useState(25);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [hovered, setHovered] = useState(null);

  const fetchGraph = async () => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/network?top_authors=${topAuthors}&min_collaboration=1`);
      const d = await res.json();
      if (!res.ok || d.error) throw new Error(d.error || 'Failed to load network');
      setGraph(d);
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  };
  useEffect(() => { fetchGraph(); }, [topAuthors]);

  // Build & run the force simulation whenever the graph changes
  useEffect(() => {
    if (!graph) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const W = canvas.width = canvas.clientWidth;
    const H = canvas.height = 560;
    const ctx = canvas.getContext('2d');

    const nodes = graph.nodes.map(n => ({
      ...n, x: W / 2 + (Math.random() - 0.5) * 300, y: H / 2 + (Math.random() - 0.5) * 300, vx: 0, vy: 0,
      r: Math.max(6, Math.min(26, n.size)),
    }));
    const index = Object.fromEntries(nodes.map((n, i) => [n.id, i]));
    const edges = graph.edges.filter(e => e.source in index && e.target in index)
      .map(e => ({ s: index[e.source], t: index[e.target], w: e.weight }));
    simRef.current.nodes = nodes; simRef.current.edges = edges;

    let ticks = 0;
    const step = () => {
      const k = 0.02, repel = 1400, spring = 0.015, springLen = 90, center = 0.012, damp = 0.86;
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        a.vx += (W / 2 - a.x) * center * k * 10;
        a.vy += (H / 2 - a.y) * center * k * 10;
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          let dx = a.x - b.x, dy = a.y - b.y; let d2 = dx * dx + dy * dy || 1;
          const f = repel / d2; const d = Math.sqrt(d2);
          const ux = dx / d, uy = dy / d;
          a.vx += ux * f; a.vy += uy * f; b.vx -= ux * f; b.vy -= uy * f;
        }
      }
      for (const e of edges) {
        const a = nodes[e.s], b = nodes[e.t];
        let dx = b.x - a.x, dy = b.y - a.y; const d = Math.sqrt(dx * dx + dy * dy) || 1;
        const f = (d - springLen) * spring; const ux = dx / d, uy = dy / d;
        a.vx += ux * f; a.vy += uy * f; b.vx -= ux * f; b.vy -= uy * f;
      }
      for (const n of nodes) {
        if (simRef.current.drag === n) continue;
        n.vx *= damp; n.vy *= damp; n.x += n.vx; n.y += n.vy;
        n.x = Math.max(n.r, Math.min(W - n.r, n.x)); n.y = Math.max(n.r, Math.min(H - n.r, n.y));
      }

      // draw
      ctx.clearRect(0, 0, W, H);
      ctx.lineWidth = 1;
      for (const e of edges) {
        const a = nodes[e.s], b = nodes[e.t];
        ctx.strokeStyle = `rgba(139,92,246,${Math.min(0.5, 0.15 + e.w * 0.1)})`;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      for (const n of nodes) {
        ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = n.type === 'author' ? '#8b5cf6' : '#06b6d4';
        ctx.globalAlpha = 0.9; ctx.fill(); ctx.globalAlpha = 1;
        if (n.r > 11 || n === simRef.current.hovered) {
          ctx.fillStyle = 'rgba(243,244,246,0.9)'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
          ctx.fillText(n.label.length > 18 ? n.label.slice(0, 17) + '…' : n.label, n.x, n.y - n.r - 5);
        }
      }
      ticks++;
      if (ticks < 600) simRef.current.raf = requestAnimationFrame(step);
    };
    step();
    return () => cancelAnimationFrame(simRef.current.raf);
  }, [graph]);

  // Pointer interaction
  const pos = (e) => {
    const r = canvasRef.current.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };
  const pick = (p) => simRef.current.nodes.find(n => (n.x - p.x) ** 2 + (n.y - p.y) ** 2 <= (n.r + 4) ** 2);
  const onDown = (e) => { const n = pick(pos(e)); if (n) simRef.current.drag = n; };
  const onMove = (e) => {
    const p = pos(e);
    if (simRef.current.drag) { simRef.current.drag.x = p.x; simRef.current.drag.y = p.y; simRef.current.drag.vx = 0; simRef.current.drag.vy = 0; if (!simRef.current.raf) { } }
    const n = pick(p); simRef.current.hovered = n; setHovered(n || null);
    canvasRef.current.style.cursor = n ? 'grab' : 'default';
  };
  const onUp = () => { simRef.current.drag = null; };

  return (
    <div style={{ padding: '24px', maxWidth: '1100px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, marginBottom: '20px', flexWrap: 'wrap' }}>
        <div>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '12px', margin: '0 0 8px 0', fontSize: '24px', fontFamily: 'var(--font-heading)' }}>
            <Share2 size={28} style={{ color: 'var(--accent-color)' }} /> Author & Venue Network
          </h2>
          <p style={{ color: 'var(--text-secondary)', margin: 0 }}>Force-directed graph of co-authors and the venues (arXiv categories) that connect them. Drag nodes to explore.</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Top authors</span>
          <input type="range" min="10" max="50" value={topAuthors} onChange={e => setTopAuthors(+e.target.value)} />
          <strong style={{ color: 'var(--accent-color)' }}>{topAuthors}</strong>
          <button className="glass-button-secondary" onClick={fetchGraph} style={{ padding: '8px 12px' }}><RefreshCw size={14} /></button>
        </div>
      </div>

      {graph?.metadata && (
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 16 }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-secondary)' }}><span style={{ width: 10, height: 10, borderRadius: '50%', background: '#8b5cf6' }} /> {graph.metadata.total_authors} authors</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-secondary)' }}><span style={{ width: 10, height: 10, borderRadius: '50%', background: '#06b6d4' }} /> {graph.metadata.total_venues} venues</span>
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{graph.edges.length} connections shown</span>
        </div>
      )}

      {error && <div style={{ padding: 16, background: 'var(--score-negative-bg)', border: '1px solid var(--score-negative-border)', borderRadius: 10, color: 'var(--score-negative-text)', marginBottom: 16, display: 'flex', gap: 10, alignItems: 'center' }}><AlertCircle size={18} />{error}</div>}
      {loading && <div className="loading-spinner"><div className="spinner" /></div>}

      <div className="glass-panel" style={{ padding: 8, position: 'relative' }}>
        <canvas ref={canvasRef} onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp} style={{ width: '100%', height: 560, borderRadius: 8, display: loading ? 'none' : 'block' }} />
        {hovered && (
          <div style={{ position: 'absolute', top: 16, left: 16, background: 'rgba(8,9,13,0.9)', border: '1px solid var(--panel-border)', borderRadius: 10, padding: '12px 14px', maxWidth: 260, pointerEvents: 'none' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--c-fg-strong)', fontWeight: 600, fontSize: 14 }}>
              {hovered.type === 'author' ? <Users size={14} style={{ color: '#8b5cf6' }} /> : <Building2 size={14} style={{ color: '#06b6d4' }} />}
              {hovered.label}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 6 }}>
              {hovered.type === 'author'
                ? <>Papers: {hovered.metadata?.paper_count} · h-index ≈ {hovered.metadata?.h_index}</>
                : <>Papers: {hovered.metadata?.paper_count}</>}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
