import React, { useState } from 'react';
import { Lightbulb, Search, AlertCircle, Zap, ChevronDown, ChevronUp, ExternalLink, FileText } from 'lucide-react';

export default function IdeaFinder({ geminiApiKey }) {
  const [topicKeyword, setTopicKeyword] = useState('');
  const [ideas, setIdeas] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(null);

  const handleSearchIdeas = async (e) => {
    e.preventDefault();
    if (!topicKeyword.trim()) return;

    try {
      setLoading(true);
      setError(null);
      
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/ideas`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic_keyword: topicKeyword,
          api_key: geminiApiKey
        })
      });
      
      if (!res.ok) throw new Error('Failed to find ideas');
      
      const data = await res.json();
      setIdeas(data);
    } catch (err) {
      console.error('Idea Finder error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '24px', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: '32px' }}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '12px', margin: '0 0 8px 0', fontSize: '24px', fontFamily: 'var(--font-heading)' }}>
          <Lightbulb size={28} style={{ color: '#ffd700' }} />
          Research Idea Finder
        </h2>
        <p style={{ color: 'var(--text-secondary)', margin: 0 }}>Find sparse research areas and discover novel research ideas</p>
      </div>

      {/* Search Form */}
      <form 
        onSubmit={handleSearchIdeas}
        style={{
          background: 'var(--c-sunken)',
          border: '1px solid var(--c-border)',
          borderRadius: '12px',
          padding: '20px',
          marginBottom: '24px',
          display: 'flex',
          gap: '12px'
        }}
      >
        <input
          type="text"
          placeholder="Enter research topic (e.g., 'Large Language Models', 'Computer Vision')..."
          value={topicKeyword}
          onChange={(e) => setTopicKeyword(e.target.value)}
          style={{
            flex: 1,
            padding: '12px 16px',
            background: 'var(--c-surface)',
            border: '1px solid var(--c-border)',
            borderRadius: '8px',
            color: 'var(--c-fg-strong)',
            fontSize: '14px'
          }}
        />
        <button
          type="submit"
          disabled={loading || !topicKeyword.trim()}
          style={{
            padding: '12px 24px',
            background: 'var(--accent-color)',
            color: '#000',
            border: 'none',
            borderRadius: '8px',
            fontWeight: '600',
            cursor: loading ? 'wait' : 'pointer',
            fontSize: '14px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            opacity: loading || !topicKeyword.trim() ? 0.5 : 1
          }}
        >
          <Search size={18} />
          Find Ideas
        </button>
      </form>

      {/* Loading State */}
      {loading && (
        <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px' }}>
            <div style={{ width: '20px', height: '20px', border: '2px solid rgba(255,255,255,0.2)', borderTopColor: 'var(--accent-color)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
            Analyzing research gaps and generating ideas...
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div style={{ padding: '16px', background: 'rgba(255, 100, 100, 0.1)', border: '1px solid rgba(255, 100, 100, 0.2)', borderRadius: '8px', color: '#ff6b6b', marginBottom: '24px', display: 'flex', gap: '12px', alignItems: 'center' }}>
          <AlertCircle size={20} />
          {error}
        </div>
      )}

      {/* Results */}
      {ideas && !loading && (
        <div>
          {ideas.error ? (
            <div style={{ padding: '20px', background: 'rgba(255, 200, 100, 0.1)', border: '1px solid rgba(255, 200, 100, 0.2)', borderRadius: '8px', color: 'var(--text-secondary)' }}>
              <p style={{ margin: 0 }}>{ideas.error}</p>
            </div>
          ) : (
            <>
              {/* Summary */}
              <div style={{
                background: 'rgba(100, 200, 255, 0.1)',
                border: '1px solid rgba(100, 200, 255, 0.2)',
                borderRadius: '12px',
                padding: '16px',
                marginBottom: '24px'
              }}>
                <h3 style={{ margin: '0 0 8px 0', color: 'var(--accent-color)' }}>Research Area: {ideas.topic}</h3>
                <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '14px' }}>
                  {ideas.sparse_regions?.length || 0} sparse region(s) identified
                </p>
              </div>

              {/* Ideas Grid */}
              {ideas.ideas && ideas.ideas.length > 0 ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px', marginBottom: '24px' }}>
                  {ideas.ideas.map((idea, idx) => (
                    <div
                      key={idx}
                      style={{
                        background: 'var(--c-sunken)',
                        border: '1px solid var(--c-border)',
                        borderRadius: '12px',
                        padding: '20px',
                        transition: 'all 0.2s'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = 'var(--c-sunken)';
                        e.currentTarget.style.borderColor = 'var(--accent-color)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = 'var(--c-sunken)';
                        e.currentTarget.style.borderColor = 'var(--c-border)';
                      }}
                    >
                      <div style={{ display: 'flex', gap: '12px', marginBottom: '12px', alignItems: 'start' }}>
                        <Zap size={20} style={{ color: '#ffd700', marginTop: '2px', flexShrink: 0 }} />
                        <h4 style={{ margin: 0, color: 'var(--c-fg-strong)', fontSize: '16px', fontFamily: 'var(--font-heading)' }}>
                          {idea.title || `Idea ${idx + 1}`}
                        </h4>
                      </div>
                      
                      <p style={{ margin: '0 0 12px 0', color: 'var(--text-secondary)', fontSize: '14px', lineHeight: '1.5' }}>
                        {idea.description || 'Novel research direction combining multiple perspectives'}
                      </p>

                      {/* Novelty Score */}
                      {ideas.novelty_scores && ideas.novelty_scores[idx] !== undefined && (
                        <div style={{ marginBottom: '12px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontSize: '12px' }}>
                            <span style={{ color: 'var(--text-secondary)' }}>Novelty Score</span>
                            <span style={{ color: 'var(--accent-color)' }}>
                              {(ideas.novelty_scores[idx] * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div style={{
                            height: '6px',
                            background: 'var(--c-border)',
                            borderRadius: '3px',
                            overflow: 'hidden'
                          }}>
                            <div
                              style={{
                                height: '100%',
                                width: `${ideas.novelty_scores[idx] * 100}%`,
                                background: 'linear-gradient(90deg, var(--accent-color), #ffd700)',
                                transition: 'width 0.3s'
                              }}
                            />
                          </div>
                        </div>
                      )}

                      <button
                        onClick={() => setExpanded(expanded === idx ? null : idx)}
                        style={{
                          width: '100%',
                          padding: '10px',
                          background: expanded === idx ? 'rgba(100, 200, 255, 0.3)' : 'rgba(100, 200, 255, 0.2)',
                          border: '1px solid rgba(100, 200, 255, 0.3)',
                          borderRadius: '6px',
                          color: 'var(--accent-color)',
                          fontWeight: '600',
                          cursor: 'pointer',
                          fontSize: '12px',
                          transition: 'all 0.2s',
                          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px'
                        }}
                      >
                        {expanded === idx ? <>Hide details <ChevronUp size={14} /></> : <>Explore Further <ChevronDown size={14} /></>}
                      </button>

                      {expanded === idx && (
                        <div style={{ marginTop: '14px', paddingTop: '14px', borderTop: '1px solid rgba(255,255,255,0.08)' }}>
                          <p style={{ margin: '0 0 12px 0', color: 'var(--text-secondary)', fontSize: '13px', lineHeight: 1.6 }}>
                            {idea.rationale || idea.description || 'This direction combines under-explored regions of the embedding space. The supporting papers below sit on the boundary of the gap and are good starting points.'}
                          </p>
                          <div style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 700, marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <FileText size={14} style={{ color: 'var(--accent-color)' }} /> Papers near this gap
                          </div>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            {(ideas.supporting_papers || []).slice(0, 4).map((p, pi) => (
                              <a key={pi} href={p.arxiv_url || `https://arxiv.org/abs/${p.id}`} target="_blank" rel="noreferrer"
                                 style={{ display: 'block', textDecoration: 'none', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '8px', padding: '10px 12px' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'start' }}>
                                  <span style={{ color: 'var(--c-fg-strong)', fontSize: '13px', fontWeight: 600, lineHeight: 1.4 }}>{p.title}</span>
                                  <ExternalLink size={13} style={{ color: 'var(--accent-color)', flexShrink: 0, marginTop: 2 }} />
                                </div>
                                <span style={{ fontSize: '11px', color: 'var(--accent-secondary)' }}>{p.category_name} · {String(p.published).slice(0, 10)}</span>
                              </a>
                            ))}
                            {(!ideas.supporting_papers || ideas.supporting_papers.length === 0) && (
                              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>No supporting papers returned for this gap.</span>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                  No ideas generated. Try a different topic.
                </div>
              )}

              {/* Supporting Papers */}
              {ideas.supporting_papers && ideas.supporting_papers.length > 0 && (
                <div style={{
                  background: 'var(--c-sunken)',
                  border: '1px solid var(--c-border)',
                  borderRadius: '12px',
                  padding: '16px'
                }}>
                  <h4 style={{ margin: '0 0 12px 0', color: 'var(--text-secondary)', fontSize: '14px' }}>Supporting Papers</h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {ideas.supporting_papers.map((paper, idx) => (
                      <div key={idx} style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                        <strong style={{ color: 'var(--c-fg-strong)' }}>{paper.title}</strong>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Empty State */}
      {!ideas && !loading && !error && (
        <div style={{
          padding: '60px 20px',
          textAlign: 'center',
          color: 'var(--text-secondary)'
        }}>
          <Lightbulb size={48} style={{ marginBottom: '16px', opacity: 0.3 }} />
          <p>Enter a research topic to find sparse areas and discover novel research ideas</p>
        </div>
      )}
    </div>
  );
}
