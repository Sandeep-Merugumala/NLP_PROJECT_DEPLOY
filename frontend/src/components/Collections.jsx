import React, { useState, useEffect, useRef } from 'react';
import { 
  Search, Folder, Plus, Trash2, FolderPlus, ThumbsUp, ThumbsDown, 
  Bookmark, BookOpen, Clock, ThumbsDown as Down, ChevronRight,
  History, Zap, X
} from 'lucide-react';

export default function Collections({ 
  ratings, 
  onRate, 
  allPapers, 
  collections, 
  onManageCollection,
  onViewPdf
}) {
  const [activeCollection, setActiveCollection] = useState('My Library');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [newCollectionName, setNewCollectionName] = useState('');
  const [collectionRecs, setCollectionRecs] = useState([]);

  // Feature G: Search autocomplete
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [expandedQuery, setExpandedQuery] = useState('');
  const [queryExpanded, setQueryExpanded] = useState(false);
  const suggestionsRef = useRef(null);

  // Feature F: Reading history
  const [historyTab, setHistoryTab] = useState('library'); // 'library' | 'history'
  const [historyEvents, setHistoryEvents] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);

  // Compute TF-IDF-based autocomplete suggestions from allPapers titles
  const computeSuggestions = (query) => {
    if (!query || query.length < 2) {
      setSuggestions([]);
      return;
    }
    const q = query.toLowerCase();
    const titleWords = new Set();
    for (const paper of allPapers) {
      const words = paper.title.toLowerCase().split(/\s+/);
      for (const word of words) {
        if (word.startsWith(q) && word.length > q.length && word.length > 4) {
          titleWords.add(word);
        }
      }
      if (titleWords.size >= 8) break;
    }
    setSuggestions([...titleWords].slice(0, 6));
  };

  const handleQueryChange = (val) => {
    setSearchQuery(val);
    computeSuggestions(val);
    setShowSuggestions(true);
    setQueryExpanded(false);
    setExpandedQuery('');
  };

  const applySuggestion = (word) => {
    setSearchQuery(word);
    setSuggestions([]);
    setShowSuggestions(false);
  };

  // Perform Search with optional expansion
  const handleSearch = async (e, useExpand = false) => {
    if (e) e.preventDefault();
    setShowSuggestions(false);
    
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery, expand: useExpand })
      });
      const data = await response.json();
      setSearchResults(data.results || []);
      if (data.expanded_query && data.expanded_query !== searchQuery) {
        setExpandedQuery(data.expanded_query);
        setQueryExpanded(useExpand);
      }
    } catch (err) {
      // Local fallback
      const queryWords = searchQuery.toLowerCase().split(' ').filter(Boolean);
      const localResults = allPapers.filter(p => {
        const text = (p.title + " " + p.abstract).toLowerCase();
        return queryWords.every(word => {
          const escapedWord = word.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
          const regex = new RegExp('\\b' + escapedWord);
          return regex.test(text);
        });
      });
      setSearchResults(localResults);
    }
  };

  // Feature F: Load reading history
  const fetchHistory = async () => {
    setLoadingHistory(true);
    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/history?limit=30`);
      const data = await res.json();
      setHistoryEvents(data.events || []);
    } catch (err) {
      setHistoryEvents([]);
    } finally {
      setLoadingHistory(false);
    }
  };

  useEffect(() => {
    if (historyTab === 'history') {
      fetchHistory();
    }
  }, [historyTab]);

  // Create Collection
  const handleCreateCollection = () => {
    if (!newCollectionName.trim()) return;
    onManageCollection('create', newCollectionName);
    setActiveCollection(newCollectionName);
    setNewCollectionName('');
  };

  // Fetch proposals to expand collection
  const fetchCollectionRecs = async (colName) => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/collections/${encodeURIComponent(colName)}/recommendations`);
      const data = await response.json();
      setCollectionRecs(data.papers || []);
    } catch (err) {
      setCollectionRecs([]);
    }
  };

  useEffect(() => {
    if (activeCollection) {
      fetchCollectionRecs(activeCollection);
    }
  }, [activeCollection, collections]);

  // Close suggestions on outside click
  useEffect(() => {
    const handler = (e) => {
      if (suggestionsRef.current && !suggestionsRef.current.contains(e.target)) {
        setShowSuggestions(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const getCollectionPapers = () => {
    const paperIds = collections[activeCollection] || [];
    return allPapers.filter(p => paperIds.includes(p.id));
  };

  const activePapers = getCollectionPapers();

  // Event type icon + colour mapping
  const eventStyle = {
    upvote: { icon: <ThumbsUp size={12} />, color: '#4ade80', label: 'Upvoted' },
    downvote: { icon: <ThumbsDown size={12} />, color: '#f87171', label: 'Downvoted' },
    clear: { icon: <X size={12} />, color: 'var(--text-muted)', label: 'Cleared' },
    bookmark: { icon: <Bookmark size={12} />, color: '#fb923c', label: 'Bookmarked' },
    view_pdf: { icon: <BookOpen size={12} />, color: '#22d3ee', label: 'Viewed PDF' },
  };

  return (
    <div className="search-collections-layout">
      {/* Sidebar folders */}
      <div className="collections-sidebar glass-panel">
        <h3 style={{ margin: '0 0 16px 0', fontFamily: 'var(--font-heading)', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Folder size={18} style={{ color: 'var(--accent-color)' }} /> Libraries
        </h3>
        
        {/* Create Folder Input */}
        <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
          <input 
            type="text" 
            className="glass-input" 
            placeholder="New folder..." 
            value={newCollectionName}
            onChange={(e) => setNewCollectionName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCreateCollection()}
            style={{ padding: '6px 12px', fontSize: '13px', width: '100%' }}
          />
          <button 
            className="action-btn" 
            onClick={handleCreateCollection}
            style={{ width: '32px', height: '32px' }}
          >
            <Plus size={14} />
          </button>
        </div>

        {/* Folder List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '20px' }}>
          {Object.keys(collections).map(name => {
            const count = collections[name]?.length || 0;
            return (
              <div 
                key={name}
                className={`collection-item ${activeCollection === name && historyTab === 'library' ? 'active' : ''}`}
                onClick={() => { setActiveCollection(name); setHistoryTab('library'); }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <FolderPlus size={14} style={{ color: activeCollection === name && historyTab === 'library' ? 'var(--c-fg-strong)' : 'var(--text-secondary)' }} />
                  <span style={{ fontSize: '14px', fontWeight: '500' }}>{name}</span>
                </div>
                <span className="tag" style={{ fontSize: '11px', padding: '2px 8px' }}>{count}</span>
              </div>
            );
          })}
        </div>

        {/* Feature F: History Tab Button */}
        <div
          className={`collection-item ${historyTab === 'history' ? 'active' : ''}`}
          onClick={() => setHistoryTab('history')}
          style={{ borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '12px', marginTop: '4px' }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <History size={14} style={{ color: historyTab === 'history' ? 'var(--c-fg-strong)' : 'var(--accent-color)' }} />
            <span style={{ fontSize: '14px', fontWeight: '500' }}>Reading History</span>
          </div>
          <span className="tag" style={{ fontSize: '11px', padding: '2px 8px' }}>{historyEvents.length || '—'}</span>
        </div>
      </div>

      {/* Main Panel */}
      <div className="search-results-pane">
        {/* Feature G: Search Bar with Autocomplete */}
        <div ref={suggestionsRef} style={{ position: 'relative' }}>
          <form onSubmit={(e) => handleSearch(e, false)} className="search-bar-container glass-panel" style={{ padding: '12px', gap: '8px' }}>
            <input 
              type="text" 
              className="glass-input" 
              placeholder="Search papers semantically or lexically (e.g. 'Language Agents for coding')..." 
              value={searchQuery}
              onChange={(e) => handleQueryChange(e.target.value)}
              onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
              style={{ background: 'transparent', border: 'none', padding: '6px 12px', flex: 1 }}
            />
            <button className="glass-button-secondary" type="button" onClick={(e) => handleSearch(e, true)} title="Search with interest-based query expansion">
              <Zap size={14} style={{ color: 'var(--accent-color)' }} /> Smart Search
            </button>
            <button className="glass-button" type="submit">
              <Search size={16} /> Search
            </button>
          </form>

          {/* Autocomplete Dropdown */}
          {showSuggestions && suggestions.length > 0 && (
            <div style={{
              position: 'absolute',
              top: '100%',
              left: 0,
              right: 0,
              zIndex: 100,
              background: '#12141f',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '0 0 10px 10px',
              overflow: 'hidden',
              boxShadow: '0 12px 40px rgba(0,0,0,0.5)'
            }}>
              {suggestions.map((s, i) => (
                <div
                  key={i}
                  onClick={() => applySuggestion(s)}
                  style={{
                    padding: '8px 16px',
                    fontSize: '13px',
                    color: 'var(--text-secondary)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    borderBottom: '1px solid rgba(255,255,255,0.04)',
                    transition: 'background 0.15s'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(139,92,246,0.1)'}
                  onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                >
                  <Search size={12} style={{ color: 'var(--accent-color)', flexShrink: 0 }} />
                  {s}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Query expansion notice */}
        {queryExpanded && expandedQuery && (
          <div style={{ 
            fontSize: '12px', 
            color: 'var(--accent-secondary)', 
            padding: '6px 12px', 
            background: 'rgba(139,92,246,0.07)', 
            borderRadius: '6px',
            display: 'flex', 
            alignItems: 'center', 
            gap: '6px',
            border: '1px solid rgba(139,92,246,0.2)'
          }}>
            <Zap size={12} style={{ color: 'var(--accent-color)' }} />
            <span>Query expanded using your interest profile: <em>"{expandedQuery.slice(0, 80)}..."</em></span>
          </div>
        )}

        {/* Feature F: Reading History Timeline */}
        {historyTab === 'history' ? (
          <div className="glass-panel" style={{ padding: '24px', textAlign: 'left' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h3 style={{ margin: 0, fontFamily: 'var(--font-heading)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <History size={18} style={{ color: 'var(--accent-color)' }} /> Reading History
              </h3>
              <button className="glass-button-secondary" style={{ fontSize: '11px', padding: '4px 10px' }} onClick={fetchHistory}>
                Refresh
              </button>
            </div>

            {loadingHistory ? (
              <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                <div style={{ width: '24px', height: '24px', border: '2px solid rgba(255,255,255,0.1)', borderTopColor: 'var(--accent-color)', borderRadius: '50%', animation: 'spin 0.8s linear infinite', margin: '0 auto 12px' }} />
                Loading history...
              </div>
            ) : historyEvents.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                <Clock size={40} style={{ marginBottom: '12px', opacity: 0.4 }} />
                <p>No activity yet. Start rating and bookmarking papers!</p>
              </div>
            ) : (
              <div style={{ position: 'relative', paddingLeft: '24px' }}>
                {/* Timeline vertical line */}
                <div style={{
                  position: 'absolute',
                  left: '7px',
                  top: 0,
                  bottom: 0,
                  width: '2px',
                  background: 'linear-gradient(180deg, var(--accent-color), transparent)',
                  borderRadius: '1px'
                }} />

                {historyEvents.map((event, idx) => {
                  const style = eventStyle[event.event_type] || { icon: <Clock size={12} />, color: 'var(--text-muted)', label: event.event_type };
                  const ts = event.timestamp ? new Date(event.timestamp) : null;
                  return (
                    <div key={idx} style={{ 
                      display: 'flex', 
                      gap: '12px', 
                      marginBottom: '16px',
                      position: 'relative',
                      alignItems: 'flex-start'
                    }}>
                      {/* Timeline dot */}
                      <div style={{
                        position: 'absolute',
                        left: '-24px',
                        top: '2px',
                        width: '16px',
                        height: '16px',
                        borderRadius: '50%',
                        background: style.color,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        border: '2px solid #0f111a',
                        color: '#000',
                        flexShrink: 0
                      }}>
                        {style.icon}
                      </div>

                      <div style={{ flex: 1, background: 'rgba(255,255,255,0.02)', borderRadius: '8px', padding: '10px 14px', border: '1px solid rgba(255,255,255,0.05)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                          <span style={{ fontSize: '11px', fontWeight: '600', color: style.color }}>
                            {style.label}
                          </span>
                          <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                            {ts ? ts.toLocaleString() : '—'}
                          </span>
                        </div>
                        <p style={{ margin: 0, fontSize: '13px', color: 'var(--c-fg-strong)', lineHeight: '1.4' }}>
                          {event.paper_title || event.paper_id}
                        </p>
                        {event.paper_authors?.length > 0 && (
                          <p style={{ margin: '4px 0 0', fontSize: '11px', color: 'var(--text-muted)' }}>
                            {event.paper_authors.join(', ')} · {event.category_name}
                          </p>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ) : searchQuery ? (
          /* Search Results */
          <div className="glass-panel" style={{ padding: '24px', textAlign: 'left' }}>
            <h3 style={{ margin: '0 0 16px 0', fontFamily: 'var(--font-heading)' }}>
              Search Results ({searchResults.length})
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {searchResults.length === 0 ? (
                <p style={{ color: 'var(--text-secondary)' }}>No matching papers found.</p>
              ) : (
                searchResults.map(paper => {
                  const rating = ratings[paper.id] || 0;
                  const isBookmarked = collections[activeCollection]?.includes(paper.id);
                  
                  return (
                    <div 
                      key={paper.id} 
                      className="glass-panel" 
                      style={{ padding: '16px', background: 'rgba(255,255,255,0.01)', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '16px' }}
                    >
                      <div style={{ flex: 1 }}>
                        <h4 style={{ margin: '0 0 6px 0', color: 'var(--c-fg-strong)', fontSize: '16px' }}>{paper.title}</h4>
                        <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                          {paper.authors.slice(0, 3).join(', ')}
                        </p>
                        <p style={{ fontSize: '12px', color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' }}>
                          {paper.abstract}
                        </p>
                      </div>
                      
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', alignItems: 'flex-end' }}>
                        <span className="tag category" style={{ fontSize: '11px' }}>{paper.category_name}</span>
                        <button 
                          className="glass-button-secondary"
                          style={{ fontSize: '11px', padding: '4px 10px' }}
                          onClick={() => onViewPdf(paper)}
                        >
                          <BookOpen size={12} /> View PDF
                        </button>
                        <button 
                          className={`glass-button-secondary ${isBookmarked ? 'active' : ''}`}
                          style={{ fontSize: '11px', padding: '4px 10px' }}
                          onClick={() => onManageCollection(isBookmarked ? 'remove' : 'add', activeCollection, paper.id)}
                        >
                          <Bookmark size={12} /> {isBookmarked ? 'Bookmarked' : 'Add to Collection'}
                        </button>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        ) : (
          /* Collection Contents */
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div className="glass-panel" style={{ padding: '24px', textAlign: 'left' }}>
              <h3 style={{ margin: '0 0 16px 0', fontFamily: 'var(--font-heading)' }}>
                {activeCollection} ({activePapers.length} papers)
              </h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {activePapers.length === 0 ? (
                  <p style={{ color: 'var(--text-secondary)' }}>This library is empty. Use Search above or add papers from the Map!</p>
                ) : (
                  activePapers.map(paper => (
                    <div 
                      key={paper.id} 
                      className="glass-panel" 
                      style={{ padding: '16px', background: 'rgba(255,255,255,0.01)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                    >
                      <div>
                        <h4 style={{ margin: '0 0 4px 0', color: 'var(--c-fg-strong)', fontSize: '15px' }}>{paper.title}</h4>
                        <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                          {paper.authors.slice(0, 2).join(', ')} | {paper.category_name}
                        </span>
                      </div>
                      <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                        <button 
                          className="glass-button-secondary"
                          style={{ fontSize: '11px', padding: '4px 10px' }}
                          onClick={() => onViewPdf(paper)}
                        >
                          <BookOpen size={12} /> View PDF
                        </button>
                        <button 
                          className="action-btn"
                          onClick={() => onManageCollection('remove', activeCollection, paper.id)}
                          title="Remove from library"
                        >
                          <Trash2 size={14} style={{ color: 'var(--score-negative-text)' }} />
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Recommendations to Expand Collection */}
            {activePapers.length > 0 && (
              <div className="glass-panel" style={{ padding: '24px', textAlign: 'left' }}>
                <h3 style={{ margin: '0 0 6px 0', fontFamily: 'var(--font-heading)', color: 'var(--accent-secondary)' }}>
                  Proposed Additions
                </h3>
                <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
                  Recommendations based on the papers inside <strong>"{activeCollection}"</strong>:
                </p>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {collectionRecs.map(paper => (
                    <div 
                      key={paper.id} 
                      className="glass-panel" 
                      style={{ padding: '14px', background: 'rgba(255,255,255,0.01)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                    >
                      <div style={{ flex: 1, paddingRight: '16px' }}>
                        <h4 style={{ margin: '0 0 4px 0', color: 'var(--c-fg-strong)', fontSize: '14px' }}>{paper.title}</h4>
                        <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>{paper.category_name}</span>
                      </div>
                      <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                        <button 
                          className="glass-button-secondary"
                          style={{ fontSize: '11px', padding: '4px 10px' }}
                          onClick={() => onViewPdf(paper)}
                        >
                          <BookOpen size={12} /> View PDF
                        </button>
                        <button 
                          className="glass-button" 
                          style={{ padding: '4px 10px', fontSize: '11px' }}
                          onClick={() => onManageCollection('add', activeCollection, paper.id)}
                        >
                          <Plus size={12} /> Add
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
