import React, { useState, useEffect } from 'react';
import { Mail, Compass, Folder, Calendar, Sparkles, AlertTriangle, Key, Brain, X, ChevronRight, BookOpen, BarChart3, Columns3, BookMarked, Route, NotebookPen, Wand2, Share2, Bell, RefreshCw, FileText, ThumbsUp, ThumbsDown, Bookmark, Trophy, Sun, Moon, ChevronLeft } from 'lucide-react';

// Grouped navigation for the sidebar
// `isNew: true` marks the 10 newer features — shown in cyan in the sidebar.
const NAV = [
  { group: 'Discover', items: [
    { id: 'digest', label: 'Daily Digest', Icon: Mail },
    { id: 'map', label: 'Scholar Maps', Icon: Compass },
    { id: 'trends', label: 'Trends', Icon: BarChart3, isNew: true },
    { id: 'ideas', label: 'Ideas', Icon: Sparkles, isNew: true },
    { id: 'network', label: 'Network', Icon: Share2, isNew: true },
  ]},
  { group: 'Read & Study', items: [
    { id: 'collections', label: 'Collections', Icon: Folder },
    { id: 'planner', label: 'Conferences', Icon: Calendar },
    { id: 'study', label: 'Study', Icon: BookMarked, isNew: true },
    { id: 'learn', label: 'Learning Path', Icon: Route, isNew: true },
  ]},
  { group: 'Create', items: [
    { id: 'workspace', label: 'AI Workspace', Icon: Brain },
    { id: 'compare', label: 'Compare', Icon: Columns3, isNew: true },
    { id: 'cite', label: 'Write & Cite', Icon: BookOpen, isNew: true },
    { id: 'content', label: 'Content Studio', Icon: Wand2, isNew: true },
    { id: 'notes', label: 'Notebook', Icon: NotebookPen, isNew: true },
  ]},
  { group: 'Track', items: [
    { id: 'watch', label: 'Watchlists', Icon: Bell, isNew: true },
  ]},
];
import DailyDigest from './components/DailyDigest';
import ScholarMap from './components/ScholarMap';
import Collections from './components/Collections';
import ConferencePlanner from './components/ConferencePlanner';
import ResearchWorkspace from './components/ResearchWorkspace';
import Trends from './components/Trends';
import IdeaFinder from './components/IdeaFinder';
import CiteFinder from './components/CiteFinder';
import Compare from './components/Compare';
import Study from './components/Study';
import LearningPath from './components/LearningPath';
import Notebook from './components/Notebook';
import ContentStudio from './components/ContentStudio';
import Network from './components/Network';
import Watchlist from './components/Watchlist';

export default function App() {
  const [activeTab, setActiveTab] = useState('digest'); // digest, map, collections, planner, trends, ideas, cite, compare, study
  const [papers, setPapers] = useState([]);
  const [digestPapers, setDigestPapers] = useState([]);
  const [ratings, setRatings] = useState({});
  const [collections, setCollections] = useState({ "My Library": [] });
  const [activeLearningPapers, setActiveLearningPapers] = useState([]);
  
  const [loading, setLoading] = useState(true);
  const [backendError, setBackendError] = useState(false);

  // Feature A: Live Stats
  const [stats, setStats] = useState(null);

  // Corpus freshness + refresh-from-arXiv
  const [freshness, setFreshness] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchFreshness = async () => {
    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/data-freshness`);
      if (res.ok) setFreshness(await res.json());
    } catch { /* silent */ }
  };

  const handleRefreshPapers = async () => {
    if (refreshing) return;
    if (!window.confirm('Re-crawl the newest papers from arXiv and rebuild the map? This can take ~1 minute.')) return;
    setRefreshing(true);
    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/refresh-papers`, { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        await loadAllData();
        await fetchFreshness();
      } else {
        alert('Refresh failed: ' + (data.error || 'unknown error'));
      }
    } catch (e) {
      alert('Refresh failed: could not reach the backend.');
    } finally {
      setRefreshing(false);
    }
  };

  // Gemini API Key State
  const [geminiApiKey, setGeminiApiKey] = useState(() => localStorage.getItem('gemini_api_key') || '');
  const [showKeyInput, setShowKeyInput] = useState(false);

  // Theme (dark/light) + sidebar collapse
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('theme') || 'dark'; } catch { return 'dark'; }
  });
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('theme', theme); } catch { /* ignore */ }
  }, [theme]);
  const pageTitle = NAV.flatMap(s => s.items).find(i => i.id === activeTab)?.label || 'Scholar Inbox';

  const handleKeyChange = (val) => {
    setGeminiApiKey(val);
    localStorage.setItem('gemini_api_key', val);
  };

  // Global PDF Viewer & Chat States
  const [activePdfPaper, setActivePdfPaper] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);

  useEffect(() => {
    if (activePdfPaper) {
      setChatHistory([
        {
          sender: 'ai',
          text: `Hi! I am your AI assistant for this paper. Ask me anything about the methodology, benchmarks, or results in "${activePdfPaper.title}".`
        }
      ]);
    } else {
      setChatHistory([]);
    }
    setChatInput('');
    setChatLoading(false);
  }, [activePdfPaper]);

  const handleSendChatMessage = async (e) => {
    if (e) e.preventDefault();
    if (!chatInput.trim() || chatLoading) return;

    const userMsg = chatInput.trim();
    setChatHistory(prev => [...prev, { sender: 'user', text: userMsg }]);
    setChatInput('');
    setChatLoading(true);

    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/chat-pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          paper_id: activePdfPaper.id,
          question: userMsg,
          api_key: geminiApiKey
        })
      });
      const data = await response.json();
      if (data.answer) {
        setChatHistory(prev => [...prev, { sender: 'ai', text: data.answer }]);
      } else {
        setChatHistory(prev => [...prev, { sender: 'ai', text: `Error: ${data.error || 'Failed to process question'}` }]);
      }
    } catch (err) {
      console.error(err);
      setChatHistory(prev => [...prev, { sender: 'ai', text: 'Error: Failed to connect to server.' }]);
    } finally {
      setChatLoading(false);
    }
  };

  // 1. Fetch main papers and ratings
  const fetchPapersAndRatings = async () => {
    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/papers`);
      if (!res.ok) throw new Error();
      const data = await res.json();
      setPapers(data.papers || []);
      setRatings(data.ratings || {});
      setBackendError(false);
      return data;
    } catch (err) {
      console.warn("Failed to connect to backend server. Re-attempting...");
      setBackendError(true);
      return null;
    }
  };

  // Feature A: Fetch live stats
  const fetchStats = async () => {
    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/stats`);
      if (!res.ok) return;
      const data = await res.json();
      setStats(data);
    } catch (err) {
      // silent
    }
  };

  // 2. Fetch daily/weekly digest recommendations
  const fetchDigest = async () => {
    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/digest?range=day`);
      if (!res.ok) throw new Error();
      const data = await res.json();
      setDigestPapers(data.papers || []);
    } catch (err) {
      console.warn("Could not fetch digest from backend.");
    }
  };

  // 3. Fetch active learning query (papers near decision boundary)
  const fetchActiveLearning = async () => {
    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/active-learning`);
      if (!res.ok) throw new Error();
      const data = await res.json();
      setActiveLearningPapers(data.papers || []);
    } catch (err) {
      console.warn("Could not fetch active learning papers.");
    }
  };

  // 4. Fetch user paper collections
  const fetchCollections = async () => {
    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/collections`);
      if (!res.ok) throw new Error();
      const data = await res.json();
      setCollections(data.collections || { "My Library": [] });
    } catch (err) {
      console.warn("Could not fetch collections.");
    }
  };

  // Coordinated Initial Load
  const loadAllData = async () => {
    setLoading(true);
    const mainData = await fetchPapersAndRatings();
    if (mainData) {
      await Promise.all([
        fetchDigest(),
        fetchActiveLearning(),
        fetchCollections(),
        fetchStats(),
        fetchFreshness()
      ]);
    }
    setLoading(false);
  };

  useEffect(() => {
    loadAllData();
  }, []);

  // Submit Paper Rating (Thumbs Up/Down)
  const handleRatePaper = async (paperId, rating) => {
    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/rate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paper_id: paperId, rating })
      });
      const data = await res.json();
      
      if (data.status === 'success') {
        // Update states
        setPapers(data.papers);
        setRatings(data.ratings);
        
        // Refresh digest, active learning, and stats
        await Promise.all([
          fetchDigest(),
          fetchActiveLearning(),
          fetchStats()
        ]);
      }
    } catch (err) {
      console.error("Failed to submit rating:", err);
    }
  };

  // Collection Folder additions / removals
  const handleManageCollection = async (action, name, paperId = null) => {
    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/collections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, name, paper_id: paperId })
      });
      const data = await res.json();
      
      if (data.status === 'success') {
        setCollections(data.collections);
      }
    } catch (err) {
      console.error("Failed to manage collection:", err);
    }
  };

  return (
    <div className={`app-shell ${sidebarCollapsed ? 'nav-collapsed' : ''}`}>
      {/* Left sidebar navigation */}
      <aside className="sidebar">
        <div className="side-brand">
          <div className="logo-icon"><Sparkles size={20} fill="#fff" /></div>
          {!sidebarCollapsed && <span className="logo-text">Scholar Inbox</span>}
        </div>

        <nav className="side-nav">
          {NAV.map(sec => (
            <div className="side-group" key={sec.group}>
              {!sidebarCollapsed && <div className="side-group-label">{sec.group}</div>}
              {sec.items.map(({ id, label, Icon, isNew }) => (
                <button
                  key={id}
                  className={`side-link ${isNew ? 'side-link-new' : ''} ${activeTab === id ? 'active' : ''}`}
                  onClick={() => setActiveTab(id)}
                  title={label}
                >
                  <Icon size={18} className="side-ico" />
                  {!sidebarCollapsed && <span>{label}</span>}
                </button>
              ))}
            </div>
          ))}
        </nav>

        <button className="side-collapse" onClick={() => setSidebarCollapsed(v => !v)} title="Collapse sidebar">
          <ChevronLeft size={16} style={{ transform: sidebarCollapsed ? 'rotate(180deg)' : 'none', transition: 'transform .2s' }} />
          {!sidebarCollapsed && <span>Collapse</span>}
        </button>
      </aside>

      {/* Content column: top bar + main viewport */}
      <div className="content">
        <header className="topbar">
          <h1 className="topbar-title">{pageTitle}</h1>

          <div className="topbar-right">
            {stats && (
              <div className="stat-chips">
                <span className="stat-chip" title="Total papers loaded">
                  <FileText size={13} className="stat-ico" style={{ color: 'var(--accent-color)' }} />
                  <strong>{stats.total_papers}</strong>
                  <span className="stat-lbl">papers</span>
                </span>
                <span className="stat-chip chip-like" title="Papers you liked">
                  <ThumbsUp size={13} className="stat-ico" />
                  <strong>{stats.upvoted}</strong>
                  <span className="stat-lbl">liked</span>
                </span>
                <span className="stat-chip chip-dislike" title="Papers you disliked">
                  <ThumbsDown size={13} className="stat-ico" />
                  <strong>{stats.downvoted}</strong>
                  <span className="stat-lbl">disliked</span>
                </span>
                <span className="stat-chip chip-save" title="Papers you saved">
                  <Bookmark size={13} className="stat-ico" />
                  <strong>{stats.bookmarked}</strong>
                  <span className="stat-lbl">saved</span>
                </span>
              </div>
            )}

            {freshness?.newest && (
              <span className="freshness-pill" title={`Corpus: ${freshness.count} papers, ${freshness.oldest} → ${freshness.newest}`}>
                through <strong>{freshness.newest}</strong>
              </span>
            )}

            <button className="icon-btn" onClick={handleRefreshPapers} title="Re-crawl the newest papers from arXiv" disabled={refreshing}>
              <RefreshCw size={16} style={refreshing ? { animation: 'spin 0.9s linear infinite' } : undefined} />
            </button>

            <button className="icon-btn" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} title="Toggle light / dark">
              {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
            </button>

            <button className={`icon-btn ${geminiApiKey ? 'active' : ''}`} onClick={() => setShowKeyInput(!showKeyInput)} title="Configure Gemini API Key">
              <Key size={16} />
            </button>
            {showKeyInput && (
              <input
                type="password"
                className="glass-input"
                placeholder="Enter Gemini API Key..."
                value={geminiApiKey}
                onChange={(e) => handleKeyChange(e.target.value)}
                style={{ padding: '7px 12px', fontSize: '12px', width: '200px' }}
              />
            )}
          </div>
        </header>

        {/* Main Panel Viewport */}
        <main className="main-viewport">
        {/* Backend offline warning banner */}
        {backendError && (
          <div 
            className="glass-panel" 
            style={{
              padding: '16px',
              marginBottom: '24px',
              borderColor: 'rgba(245, 158, 11, 0.4)',
              background: 'rgba(245, 158, 11, 0.1)',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              textAlign: 'left'
            }}
          >
            <AlertTriangle size={24} style={{ color: '#f59e0b', flexShrink: 0 }} />
            <div>
              <h4 style={{ margin: '0 0 4px 0', color: '#fff', fontSize: '15px' }}>Backend Offline / Connecting...</h4>
              <p style={{ margin: 0, fontSize: '13px', color: 'var(--text-secondary)' }}>
                  Please run the backend server in your terminal: 
                  <code style={{ marginLeft: '8px', fontSize: '12px', background: 'rgba(0,0,0,0.3)', padding: '2px 6px', color: '#c084fc' }}>
                    python backend/app.py
                  </code>
              </p>
            </div>
            <button 
              className="glass-button" 
              style={{ marginLeft: 'auto', padding: '6px 12px', fontSize: '12px' }}
              onClick={loadAllData}
            >
              Retry Connection
            </button>
          </div>
        )}

        {loading ? (
          <div className="loading-spinner">
            <div className="spinner"></div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Loading papers and clusters database...</p>
          </div>
        ) : (
          /* Render Active View Component */
          <>
            {activeTab === 'digest' && (
              <DailyDigest 
                papers={digestPapers}
                ratings={ratings}
                onRate={handleRatePaper}
                onBookmark={(pid) => handleManageCollection(collections['My Library']?.includes(pid) ? 'remove' : 'add', 'My Library', pid)}
                bookmarkedIds={collections['My Library'] || []}
                allPapers={papers}
                geminiApiKey={geminiApiKey}
                onViewPdf={setActivePdfPaper}
              />
            )}
            
            {activeTab === 'map' && (
              <ScholarMap 
                papers={papers}
                ratings={ratings}
                onRate={handleRatePaper}
                activeLearningPapers={activeLearningPapers}
                onViewPdf={setActivePdfPaper}
              />
            )}
            
            {activeTab === 'collections' && (
              <Collections 
                ratings={ratings}
                onRate={handleRatePaper}
                allPapers={papers}
                collections={collections}
                onManageCollection={handleManageCollection}
                onViewPdf={setActivePdfPaper}
              />
            )}
            
            {activeTab === 'planner' && (
              <ConferencePlanner 
                ratings={ratings}
                allPapers={papers}
                onViewPdf={setActivePdfPaper}
              />
            )}
            
            {activeTab === 'workspace' && (
              <ResearchWorkspace 
                apiKey={geminiApiKey}
                onViewPdf={setActivePdfPaper}
                ratings={ratings}
                onRate={handleRatePaper}
                onBookmark={(pid) => handleManageCollection(collections['My Library']?.includes(pid) ? 'remove' : 'add', 'My Library', pid)}
                bookmarkedIds={collections['My Library'] || []}
              />
            )}

            {activeTab === 'trends' && (
              <Trends geminiApiKey={geminiApiKey} />
            )}
            
            {activeTab === 'ideas' && (
              <IdeaFinder geminiApiKey={geminiApiKey} />
            )}
            
            {activeTab === 'cite' && (
              <CiteFinder geminiApiKey={geminiApiKey} />
            )}
            
            {activeTab === 'compare' && (
              <Compare papers={papers} geminiApiKey={geminiApiKey} />
            )}
            
            {activeTab === 'study' && (
              <Study papers={papers} geminiApiKey={geminiApiKey} />
            )}

            {activeTab === 'learn' && (
              <LearningPath geminiApiKey={geminiApiKey} />
            )}

            {activeTab === 'content' && (
              <ContentStudio papers={papers} geminiApiKey={geminiApiKey} />
            )}

            {activeTab === 'notes' && (
              <Notebook papers={papers} />
            )}

            {activeTab === 'network' && (
              <Network />
            )}

            {activeTab === 'watch' && (
              <Watchlist />
            )}

          </>
        )}
        </main>
      </div>

      {/* Global PDF Document & AI Chat Assistant Modal */}
      {activePdfPaper && (
        <div className="modal-overlay" onClick={() => setActivePdfPaper(null)}>
          <div className="modal-content glass-panel" style={{ maxWidth: '1200px', width: '95%', height: '90vh', display: 'flex', flexDirection: 'column' }} onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setActivePdfPaper(null)}><X size={20} /></button>
            <h3 style={{ margin: '0 0 4px 0', fontFamily: 'var(--font-heading)' }}>Official PDF Document & AI Chat Assistant</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '13px', marginBottom: '4px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              Document: <strong>"{activePdfPaper.title}"</strong>
            </p>
            
            <div style={{ display: 'flex', flex: 1, gap: '16px', minHeight: 0, marginTop: '12px' }}>
              {/* Left pane: PDF Iframe */}
              <div style={{ flex: 1, background: '#0f111a', borderRadius: '8px', overflow: 'hidden', height: '100%' }}>
                <iframe 
                  src={`https://arxiv.org/pdf/${activePdfPaper.id}.pdf`}
                  style={{ width: '100%', height: '100%', border: 'none' }}
                  title={activePdfPaper.title}
                />
              </div>
              
              {/* Right pane: AI RAG Chat Sidebar */}
              <div style={{ 
                width: '340px', 
                display: 'flex', 
                flexDirection: 'column', 
                background: 'rgba(0, 0, 0, 0.2)', 
                border: '1px solid rgba(255,255,255,0.05)', 
                borderRadius: '8px', 
                padding: '16px', 
                minHeight: 0,
                height: '100%'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '12px', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '8px' }}>
                  <Sparkles size={16} style={{ color: 'var(--accent-color)' }} />
                  <span style={{ fontWeight: '700', fontSize: '14px', color: '#fff' }}>Paper Assistant (RAG)</span>
                </div>
                
                {/* Chat Messages Log */}
                <div style={{ 
                  flex: 1, 
                  overflowY: 'auto', 
                  marginBottom: '12px', 
                  display: 'flex', 
                  flexDirection: 'column', 
                  gap: '10px',
                  paddingRight: '4px'
                }}>
                  {chatHistory.map((msg, idx) => (
                    <div 
                      key={idx} 
                      style={{ 
                        alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                        maxWidth: '90%',
                        padding: '10px 12px',
                        borderRadius: '8px',
                        fontSize: '13px',
                        lineHeight: '1.4',
                        textAlign: 'left',
                        whiteSpace: msg.sender === 'user' ? 'pre-wrap' : 'normal',
                        background: msg.sender === 'user' ? 'var(--accent-color)' : 'rgba(255,255,255,0.05)',
                        color: msg.sender === 'user' ? '#fff' : 'var(--text-secondary)',
                        border: msg.sender === 'user' ? 'none' : '1px solid rgba(255,255,255,0.02)'
                      }}
                    >
                      {msg.sender === 'user' ? msg.text : renderChatText(msg.text)}
                    </div>
                  ))}
                  {chatLoading && (
                    <div 
                      style={{ 
                        alignSelf: 'flex-start',
                        background: 'rgba(255,255,255,0.05)',
                        padding: '10px 12px',
                        borderRadius: '8px',
                        fontSize: '13px',
                        color: 'var(--text-muted)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px'
                      }}
                    >
                      <div style={{ width: '12px', height: '12px', border: '2px solid rgba(255,255,255,0.2)', borderTopColor: 'var(--accent-color)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }}></div>
                      Reading PDF & thinking...
                    </div>
                  )}
                </div>
                
                {/* Chat Input Field */}
                <form 
                  onSubmit={handleSendChatMessage}
                  style={{ display: 'flex', gap: '8px', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '12px' }}
                >
                  <input 
                    type="text" 
                    className="glass-input"
                    placeholder="Ask about methodology, results..."
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    disabled={chatLoading}
                    style={{ flex: 1, fontSize: '12px', padding: '8px 12px' }}
                  />
                  <button 
                    type="submit" 
                    className="glass-button" 
                    disabled={chatLoading || !chatInput.trim()}
                    style={{ padding: '8px' }}
                  >
                    <ChevronRight size={18} />
                  </button>
                </form>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Simple bold parser helper
function parseBoldText(text) {
  if (!text) return "";
  const parts = text.split(/\*\*([^*]+)\*\*/g);
  return parts.map((part, index) => {
    return index % 2 === 1 ? <strong style={{ color: '#fff', fontWeight: '600' }} key={index}>{part}</strong> : part;
  });
}

// Simple chat message line splitter helper
function renderChatText(text) {
  if (!text) return null;
  const lines = text.split('\n');
  return lines.map((line, lIdx) => {
    let content = line.trim();
    if (!content) return <div key={lIdx} style={{ height: '4px' }} />;
    
    if (content.startsWith('-') || content.startsWith('*')) {
      const bulletContent = content.substring(1).trim();
      return (
        <div key={lIdx} style={{ marginLeft: '8px', display: 'flex', gap: '6px', marginBottom: '4px', fontSize: '13px', lineHeight: '1.4' }}>
          <span style={{ color: 'var(--accent-color)' }}>•</span>
          <span>{parseBoldText(bulletContent)}</span>
        </div>
      );
    }
    return (
      <p key={lIdx} style={{ margin: '0 0 6px 0', fontSize: '13px', lineHeight: '1.4' }}>
        {parseBoldText(content)}
      </p>
    );
  });
}
