import React, { useState, useEffect, useRef } from 'react';
import { 
  ThumbsUp, ThumbsDown, Bookmark, FileText, Compass, 
  X, Calendar, ChevronRight, Award, Sparkles, BookOpen,
  Info, Layers, TrendingUp, Tag
} from 'lucide-react';

export default function DailyDigest({ 
  papers, 
  ratings, 
  onRate, 
  onBookmark, 
  bookmarkedIds, 
  allPapers,
  geminiApiKey,
  onViewPdf,
  onRefreshDigest
}) {
  const [startDate, setStartDate] = useState('2024-01-01');
  const [endDate, setEndDate] = useState(() => new Date().toISOString().slice(0, 10));

  // Feature C: MMR Diversity Slider
  const diversityLam = 1.0; // MMR λ fixed to pure-relevance (diversity slider removed)
  const [digestPapers, setDigestPapers] = useState(papers || []);

  // Feature D: Interest Profile Chips
  const [interests, setInterests] = useState([]);
  const [activeInterest, setActiveInterest] = useState(null);

  // Browse mode: ranked "Top picks" (digest) vs the full catalogue, + search
  const [showAll, setShowAll] = useState(false);
  const [query, setQuery] = useState('');
  const ALL_RENDER_CAP = 150;

  // Recommender mode: exploit (relevance) vs explore (LinUCB bandit)
  const [explore, setExplore] = useState(false);
  const [ranker, setRanker] = useState('');

  // Modals / Drawers State
  const [activeBibtex, setActiveBibtex] = useState(null);
  const [activeSimilar, setActiveSimilar] = useState(null);
  const [similarPapers, setSimilarPapers] = useState([]);
  const [loadingSimilar, setLoadingSimilar] = useState(false);

  // Feature E: Explanation Popovers
  const [aiExplanations, setAiExplanations] = useState({});
  const [loadingAi, setLoadingAi] = useState({});
  const [openPopover, setOpenPopover] = useState(null);

  // Sync incoming papers on prop change
  useEffect(() => {
    setDigestPapers(papers || []);
  }, [papers]);

  // Fetch digest with MMR lambda whenever slider or dates change
  const fetchDigest = async (lam, start, end, exploreMode = explore) => {
    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/digest?start_date=${start}&end_date=${end}&diversity=${lam}&explore=${exploreMode ? 1 : 0}`);
      const data = await res.json();
      setDigestPapers(data.papers || []);
      setInterests(data.interests || []);
      setRanker(data.ranker || '');
    } catch (err) {
      console.warn('Could not fetch digest with diversity param');
    }
  };

  // Re-fetch when dates or explore mode change
  useEffect(() => {
    fetchDigest(diversityLam, startDate, endDate, explore);
  }, [startDate, endDate, explore]);

  // Initial interest fetch
  useEffect(() => {
    fetchInterests();
  }, []);

  const fetchInterests = async () => {
    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/interests`);
      const data = await res.json();
      setInterests(data.interests || []);
    } catch (err) { /* silent */ }
  };

  // Feature E: Fetch AI explanation and show popover
  const handleAskAi = async (paperId) => {
    if (openPopover === paperId) { setOpenPopover(null); return; }
    setOpenPopover(paperId);
    if (aiExplanations[paperId]) return; // cached
    setLoadingAi(prev => ({ ...prev, [paperId]: true }));
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paper_id: paperId, api_key: geminiApiKey })
      });
      const data = await response.json();
      setAiExplanations(prev => ({ ...prev, [paperId]: data }));
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingAi(prev => ({ ...prev, [paperId]: false }));
    }
  };

  // Feature B: Fetch similar papers from backend /api/similar
  const fetchSimilar = async (paper) => {
    setActiveSimilar(paper);
    setSimilarPapers([]);
    setLoadingSimilar(true);
    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/similar/${encodeURIComponent(paper.id)}`);
      const data = await res.json();
      setSimilarPapers(data.papers || []);
    } catch (err) {
      // Fallback: local cosine on embeddings
      if (paper.embedding) {
        const candidates = allPapers.filter(p => p.id !== paper.id);
        const sims = candidates.map(p => {
          let dot = 0;
          for (let i = 0; i < paper.embedding.length; i++) dot += paper.embedding[i] * (p.embedding?.[i] || 0);
          return { ...p, similarity: dot };
        });
        setSimilarPapers(sims.sort((a, b) => b.similarity - a.similarity).slice(0, 5));
      }
    } finally {
      setLoadingSimilar(false);
    }
  };

  // Choose the source list: the ranked digest, or the full catalogue (sorted by relevance).
  const basePapers = showAll
    ? [...(allPapers || [])].sort((a, b) => (b.relevance_score || 0) - (a.relevance_score || 0))
    : digestPapers;

  // Filter by active interest + free-text search over title/abstract/authors.
  const q = query.trim().toLowerCase();
  const matchedPapers = basePapers.filter(p => {
    if (activeInterest) {
      const interest = interests.find(i => i.cluster === activeInterest);
      if (interest) {
        const content = (p.title + ' ' + p.abstract).toLowerCase();
        if (!interest.top_terms.some(t => content.includes(t.toLowerCase()))) return false;
      }
    }
    if (q) {
      const hay = (p.title + ' ' + p.abstract + ' ' + (p.authors || []).join(' ')).toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  // Cap the number of heavy cards rendered in "All papers" mode for performance.
  const totalMatched = matchedPapers.length;
  const filteredPapers = (showAll && !q) ? matchedPapers.slice(0, ALL_RENDER_CAP) : matchedPapers.slice(0, 400);
  const cappedNote = totalMatched > filteredPapers.length
    ? `Showing ${filteredPapers.length} of ${totalMatched}. Use search to find a specific paper.`
    : null;

  // Interest chip colour palette
  const chipColors = [
    { bg: 'rgba(139,92,246,0.18)', border: 'rgba(139,92,246,0.5)', text: '#c084fc' },
    { bg: 'rgba(6,182,212,0.15)', border: 'rgba(6,182,212,0.5)', text: '#22d3ee' },
    { bg: 'rgba(251,146,60,0.15)', border: 'rgba(251,146,60,0.5)', text: '#fb923c' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', width: '100%' }}>
      {/* Upper Control Bar */}
      <div className="digest-header glass-panel" style={{ padding: '16px 24px' }}>
        <div className="digest-title-group" style={{ textAlign: 'left' }}>
          <h1>Daily Digest</h1>
          <p>Ranked updates from open-access preprint servers based on your preferences</p>
        </div>
        
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
          {/* Date Range Pickers */}
          <div className="date-picker-bar glass-panel" style={{ border: '1px solid rgba(255,255,255,0.05)' }}>
            <Calendar size={16} style={{ color: 'var(--accent-color)' }} />
            <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
            <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>to</span>
            <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
          </div>

          {/* Recommender mode: For you (exploit) vs Explore (LinUCB bandit) */}
          <div style={{ display: 'flex', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: '8px', padding: '3px' }} title="For you = rank by predicted relevance. Explore = LinUCB bandit surfaces promising, uncertain papers (reinforcement learning).">
            {[{ k: false, label: '🎯 For you' }, { k: true, label: '🧭 Explore' }].map(opt => (
              <button
                key={String(opt.k)}
                onClick={() => setExplore(opt.k)}
                style={{
                  padding: '6px 12px', borderRadius: '6px', border: 'none', cursor: 'pointer',
                  fontSize: '12px', fontWeight: 600,
                  background: explore === opt.k ? 'var(--accent-secondary)' : 'transparent',
                  color: explore === opt.k ? '#001014' : 'var(--text-secondary)',
                  transition: 'all 0.2s'
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {ranker && (
            <span className="tag" style={{ fontSize: '10px', color: 'var(--accent-secondary)', border: '1px solid rgba(6,182,212,0.3)', background: 'rgba(6,182,212,0.08)' }} title="Active ranking model">
              {ranker}
            </span>
          )}

          {/* Browse mode toggle: Top picks vs All papers */}
          <div style={{ display: 'flex', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: '8px', padding: '3px' }}>
            {[{ k: false, label: 'Top picks' }, { k: true, label: 'All papers' }].map(opt => (
              <button
                key={String(opt.k)}
                onClick={() => setShowAll(opt.k)}
                style={{
                  padding: '6px 12px', borderRadius: '6px', border: 'none', cursor: 'pointer',
                  fontSize: '12px', fontWeight: 600,
                  background: showAll === opt.k ? 'var(--accent-color)' : 'transparent',
                  color: showAll === opt.k ? 'var(--c-fg-strong)' : 'var(--text-secondary)',
                  transition: 'all 0.2s'
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {/* Search over the shown papers */}
          <input
            className="glass-input"
            placeholder={showAll ? `Search all ${(allPapers || []).length} papers…` : 'Search these papers…'}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{ fontSize: '12px', padding: '7px 12px', minWidth: '200px' }}
          />

          <span className="score-badge neutral" style={{ fontSize: '12px', whiteSpace: 'nowrap' }}>
            {totalMatched} papers
          </span>
        </div>
      </div>

      {cappedNote && (
        <div style={{ fontSize: '12px', color: 'var(--text-muted)', textAlign: 'left' }}>{cappedNote}</div>
      )}

      {/* Feature D: Interest Profile Chips */}
      {interests.length > 0 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <Layers size={14} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Your interests:</span>
          <button
            onClick={() => setActiveInterest(null)}
            style={{
              padding: '4px 10px',
              borderRadius: '20px',
              border: `1px solid ${activeInterest === null ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.1)'}`,
              background: activeInterest === null ? 'rgba(255,255,255,0.08)' : 'transparent',
              color: activeInterest === null ? 'var(--c-fg-strong)' : 'var(--text-secondary)',
              fontSize: '11px',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            All
          </button>
          {interests.map((interest, i) => {
            const col = chipColors[i % chipColors.length];
            const isActive = activeInterest === interest.cluster;
            return (
              <button
                key={interest.cluster}
                onClick={() => setActiveInterest(isActive ? null : interest.cluster)}
                title={`Representative: "${interest.representative}"`}
                style={{
                  padding: '4px 12px',
                  borderRadius: '20px',
                  border: `1px solid ${isActive ? col.border : 'rgba(255,255,255,0.1)'}`,
                  background: isActive ? col.bg : 'transparent',
                  color: isActive ? col.text : 'var(--text-secondary)',
                  fontSize: '11px',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px'
                }}
              >
                <Tag size={10} />
                {interest.label}
                <span style={{ opacity: 0.6 }}>({interest.count})</span>
              </button>
            );
          })}
        </div>
      )}

      {/* Recommended Papers List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {filteredPapers.length === 0 ? (
          <div className="glass-panel" style={{ padding: '60px', textAlign: 'center' }}>
            <Award size={48} style={{ color: 'var(--text-muted)', marginBottom: '16px' }} />
            <h3 style={{ margin: '0 0 8px 0', fontFamily: 'var(--font-heading)' }}>No Recommendations Yet</h3>
            <p style={{ color: 'var(--text-secondary)', maxWidth: '400px', margin: '0 auto' }}>
              Try rating some papers in the <strong>Scholar Map</strong> or adding your research interests to train the recommender!
            </p>
          </div>
        ) : (
          filteredPapers.map((paper) => {
            const userRating = ratings[paper.id] || 0;
            const isBookmarked = bookmarkedIds.includes(paper.id);
            const score = paper.relevance_score;
            
            let scoreClass = 'neutral';
            if (score > 15.0) scoreClass = 'positive';
            else if (score < -15.0) scoreClass = 'negative';
            
            const explanation = aiExplanations[paper.id];
            const isPopoverOpen = openPopover === paper.id;

            return (
              <div key={paper.id} className="paper-card glass-panel" style={{ position: 'relative' }}>
                <div className="paper-card-header">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', flex: 1 }}>
                    <h2 className="paper-title">{paper.title}</h2>
                    <div className="paper-authors">
                      {paper.authors.slice(0, 5).join(', ')}{paper.authors.length > 5 ? ' et al.' : ''}
                    </div>
                    <div className="paper-meta">
                      <span className={`score-badge ${scoreClass}`}>
                        {score > 0 ? `+${score}` : score} Relevance
                      </span>
                      <span className="tag category">{paper.category_name}</span>
                      <span className="tag" style={{ color: 'var(--text-secondary)' }}>{paper.published}</span>
                    </div>
                  </div>
                  
                  {/* Paper Card Actions */}
                  <div className="paper-actions">
                    <button 
                      className={`action-btn upvote ${userRating === 1 ? 'upvoted' : ''}`}
                      onClick={() => onRate(paper.id, userRating === 1 ? 0 : 1)}
                      title="Upvote (This is relevant)"
                    >
                      <ThumbsUp size={16} />
                    </button>
                    <button 
                      className={`action-btn downvote ${userRating === -1 ? 'downvoted' : ''}`}
                      onClick={() => onRate(paper.id, userRating === -1 ? 0 : -1)}
                      title="Downvote (This is not relevant)"
                    >
                      <ThumbsDown size={16} />
                    </button>
                    <button 
                      className={`action-btn ${isBookmarked ? 'bookmarked' : ''}`}
                      onClick={() => onBookmark(paper.id)}
                      title="Bookmark to Collection"
                    >
                      <Bookmark size={16} />
                    </button>
                    {/* Feature E: Why Recommended info icon */}
                    <button
                      className={`action-btn ${isPopoverOpen ? 'bookmarked' : ''}`}
                      onClick={() => handleAskAi(paper.id)}
                      title="Why Recommended?"
                    >
                      <Info size={16} />
                    </button>
                  </div>
                </div>
                
                {/* Highlighted Abstract */}
                <div className="paper-abstract">
                  {renderAbstract(paper.abstract, paper.highlight)}
                </div>
                
                {/* Paper Footer with Drawers Links */}
                <div className="paper-card-footer">
                  <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                    Source: arXiv preprint
                  </span>
                  
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <button 
                      className="glass-button-secondary" 
                      style={{ fontSize: '12px', padding: '6px 12px' }}
                      onClick={() => setActiveBibtex(paper)}
                    >
                      <FileText size={14} /> BibTeX
                    </button>
                    <button 
                      className="glass-button-secondary" 
                      style={{ fontSize: '12px', padding: '6px 12px' }}
                      onClick={() => onViewPdf(paper)}
                    >
                      <BookOpen size={14} /> View PDF
                    </button>
                    <button 
                      className="glass-button-secondary" 
                      style={{ fontSize: '12px', padding: '6px 12px' }}
                      onClick={() => fetchSimilar(paper)}
                    >
                      <Compass size={14} /> More Like This
                    </button>
                    <button 
                      className="glass-button-secondary" 
                      style={{ fontSize: '12px', padding: '6px 12px', borderColor: 'rgba(139,92,246,0.3)' }}
                      onClick={() => handleAskAi(paper.id)}
                      disabled={loadingAi[paper.id]}
                    >
                      <Sparkles size={14} style={{ color: 'var(--accent-color)' }} /> 
                      {loadingAi[paper.id] ? 'Analyzing...' : 'Why Recommended'}
                    </button>
                  </div>
                </div>

                {/* Feature E: Explanation Popover */}
                {isPopoverOpen && (
                  <div 
                    className="glass-panel" 
                    style={{
                      marginTop: '16px',
                      padding: '16px',
                      background: 'rgba(139, 92, 246, 0.05)',
                      borderLeft: '4px solid var(--accent-color)',
                      fontSize: '13px',
                      textAlign: 'left',
                      animation: 'fadeIn 0.2s ease'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px', alignItems: 'center' }}>
                      <span style={{ fontWeight: '700', color: 'var(--accent-secondary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <TrendingUp size={14} /> Why Recommended
                      </span>
                      {explanation && (
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                          powered by {explanation.source}
                        </span>
                      )}
                      <button className="action-btn" onClick={() => setOpenPopover(null)} style={{ width: '24px', height: '24px' }}>
                        <X size={12} />
                      </button>
                    </div>

                    {/* Relevance Score Bar */}
                    <div style={{ marginBottom: '10px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', fontSize: '11px', color: 'var(--text-muted)' }}>
                        <span>Relevance Score</span>
                        <span style={{ color: score > 0 ? '#4ade80' : '#f87171' }}>{score > 0 ? `+${score}` : score}</span>
                      </div>
                      <div style={{ height: '4px', background: 'rgba(255,255,255,0.08)', borderRadius: '2px', overflow: 'hidden' }}>
                        <div style={{
                          height: '100%',
                          width: `${Math.min(100, Math.abs(score))}%`,
                          background: score > 0 ? 'linear-gradient(90deg, #4ade80, #22d3ee)' : 'linear-gradient(90deg, #f87171, #fb923c)',
                          borderRadius: '2px',
                          transition: 'width 0.5s ease'
                        }} />
                      </div>
                    </div>

                    {loadingAi[paper.id] ? (
                      <div style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div style={{ width: '12px', height: '12px', border: '2px solid rgba(255,255,255,0.1)', borderTopColor: 'var(--accent-color)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
                        Analyzing match with your profile...
                      </div>
                    ) : explanation ? (
                      <>
                        <p style={{ color: 'var(--c-fg-strong)', marginBottom: '8px', lineHeight: '1.5', margin: '0 0 8px 0' }}>
                          <strong style={{ color: 'var(--text-secondary)' }}>Relevance: </strong>{explanation.explanation}
                        </p>
                        {explanation.tailored_summary && (
                          <p style={{ color: 'var(--text-secondary)', margin: 0, lineHeight: '1.5' }}>
                            <strong>Summary: </strong>{explanation.tailored_summary}
                          </p>
                        )}
                      </>
                    ) : (
                      <p style={{ color: 'var(--text-muted)', margin: 0 }}>
                        Relevance score: <strong>{score > 0 ? `+${score}` : score}</strong>. 
                        Add a Gemini API key for detailed AI explanations.
                      </p>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* BibTeX Modal */}
      {activeBibtex && (
        <div className="modal-overlay" onClick={() => setActiveBibtex(null)}>
          <div className="modal-content glass-panel" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setActiveBibtex(null)}><X size={20} /></button>
            <h3 style={{ margin: '0 0 16px 0', fontFamily: 'var(--font-heading)' }}>Export BibTeX Reference</h3>
            <pre style={{
              background: '#0a0c14',
              padding: '16px',
              borderRadius: '8px',
              fontFamily: 'var(--font-mono)',
              fontSize: '13px',
              color: 'var(--accent-secondary)',
              textAlign: 'left',
              overflowX: 'auto',
              border: '1px solid rgba(255,255,255,0.05)'
            }}>
              {activeBibtex.bibtex}
            </pre>
            <button 
              className="glass-button" 
              style={{ marginTop: '16px', width: '100%', justifyContent: 'center' }}
              onClick={() => {
                navigator.clipboard.writeText(activeBibtex.bibtex);
                alert("BibTeX copied to clipboard!");
              }}
            >
              Copy Reference
            </button>
          </div>
        </div>
      )}

      {/* Feature B: Similar Papers Drawer */}
      <div className={`similar-drawer glass-panel ${activeSimilar ? 'open' : ''}`}>
        <div className="similar-drawer-header">
          <h3 style={{ margin: 0, fontFamily: 'var(--font-heading)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Compass size={16} style={{ color: 'var(--accent-color)' }} /> More Like This
          </h3>
          <button className="action-btn" onClick={() => { setActiveSimilar(null); setSimilarPapers([]); }}>
            <X size={16} />
          </button>
        </div>
        
        {activeSimilar && (
          <div style={{ textAlign: 'left', fontSize: '13px', marginBottom: '12px' }}>
            <p style={{ color: 'var(--text-secondary)', margin: 0 }}>
              Papers similar to <strong>"{activeSimilar.title.slice(0, 60)}..."</strong>
            </p>
          </div>
        )}
        
        <div className="similar-drawer-list">
          {loadingSimilar ? (
            <div style={{ textAlign: 'center', padding: '20px', color: 'var(--text-muted)', fontSize: '13px' }}>
              <div style={{ width: '20px', height: '20px', border: '2px solid rgba(255,255,255,0.1)', borderTopColor: 'var(--accent-color)', borderRadius: '50%', animation: 'spin 0.8s linear infinite', margin: '0 auto 8px' }} />
              Computing similarities...
            </div>
          ) : similarPapers.map((paper) => (
            <div 
              key={paper.id} 
              className="glass-panel" 
              style={{ padding: '14px', background: 'rgba(255,255,255,0.01)' }}
            >
              <h4 style={{ margin: '0 0 6px 0', fontSize: '13px', color: 'var(--c-fg-strong)', textAlign: 'left', lineHeight: '1.4' }}>
                {paper.title}
              </h4>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', color: 'var(--text-secondary)', marginTop: '8px' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <TrendingUp size={10} style={{ color: 'var(--accent-color)' }} />
                  Cosine Sim: {paper.similarity != null ? (paper.similarity * 100).toFixed(1) : '—'}%
                </span>
                <button 
                  className="glass-button-secondary"
                  style={{ fontSize: '10px', padding: '2px 8px' }}
                  onClick={() => onViewPdf(paper)}
                >
                  <BookOpen size={10} /> View PDF
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Highlight helper: puts a highlighted CSS span around the key sentence in the abstract
function renderAbstract(abstract, highlight) {
  if (!highlight || !abstract) return abstract;
  const index = abstract.indexOf(highlight);
  if (index === -1) return abstract;
  return (
    <>
      {abstract.slice(0, index)}
      <span className="highlighted-sentence">{highlight}</span>
      {abstract.slice(index + highlight.length)}
    </>
  );
}
