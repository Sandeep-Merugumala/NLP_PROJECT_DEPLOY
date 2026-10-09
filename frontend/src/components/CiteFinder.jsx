import React, { useState } from 'react';
import { BookOpen, Send, Copy, Download, AlertCircle } from 'lucide-react';

export default function CiteFinder({ geminiApiKey }) {
  const [userText, setUserText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('suggestions'); // 'suggestions' or 'bibtex'
  const [copied, setCopied] = useState(false);

  const handleProcessText = async (e) => {
    e.preventDefault();
    if (!userText.trim()) return;

    try {
      setLoading(true);
      setError(null);
      
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/cite-suggest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: userText,
          api_key: geminiApiKey
        })
      });
      
      if (!res.ok) throw new Error('Failed to process citations');
      
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error('CiteFinder error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadBibTeX = () => {
    if (!result?.bibtex) return;
    
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(result.bibtex));
    element.setAttribute('download', 'references.bib');
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div style={{ padding: '24px', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: '32px' }}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '12px', margin: '0 0 8px 0', fontSize: '24px', fontFamily: 'var(--font-heading)' }}>
          <BookOpen size={28} style={{ color: 'var(--accent-color)' }} />
          Citation Finder & Related Work Writer
        </h2>
        <p style={{ color: 'var(--text-secondary)', margin: 0 }}>Paste your draft text to find related papers and auto-generate citations</p>
      </div>

      {/* Input Section */}
      <div style={{
        background: 'var(--c-sunken)',
        border: '1px solid var(--c-border)',
        borderRadius: '12px',
        padding: '20px',
        marginBottom: '24px'
      }}>
        <form onSubmit={handleProcessText}>
          <textarea
            placeholder="Paste your paragraph or draft text here..."
            value={userText}
            onChange={(e) => setUserText(e.target.value)}
            style={{
              width: '100%',
              minHeight: '140px',
              padding: '16px',
              background: 'var(--c-surface)',
              border: '1px solid var(--c-border)',
              borderRadius: '8px',
              color: 'var(--c-fg-strong)',
              fontFamily: 'monospace',
              fontSize: '13px',
              resize: 'vertical',
              marginBottom: '16px'
            }}
          />
          
          <button
            type="submit"
            disabled={loading || !userText.trim()}
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
              opacity: loading || !userText.trim() ? 0.5 : 1
            }}
          >
            <Send size={16} />
            Find Citations
          </button>
        </form>
      </div>

      {/* Loading */}
      {loading && (
        <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px' }}>
            <div style={{ width: '20px', height: '20px', border: '2px solid rgba(255,255,255,0.2)', borderTopColor: 'var(--accent-color)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
            Analyzing text and finding related papers...
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ padding: '16px', background: 'rgba(255, 100, 100, 0.1)', border: '1px solid rgba(255, 100, 100, 0.2)', borderRadius: '8px', color: '#ff6b6b', marginBottom: '24px', display: 'flex', gap: '12px', alignItems: 'center' }}>
          <AlertCircle size={20} />
          {error}
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <div>
          {/* Stats */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
            gap: '16px',
            marginBottom: '24px'
          }}>
            <div style={{
              background: 'rgba(100, 200, 255, 0.1)',
              border: '1px solid rgba(100, 200, 255, 0.2)',
              borderRadius: '8px',
              padding: '16px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '24px', fontWeight: '700', color: 'var(--accent-color)', marginBottom: '4px' }}>
                {result.statistics?.papers_found || 0}
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Papers Found</div>
            </div>
            
            <div style={{
              background: 'rgba(100, 200, 255, 0.1)',
              border: '1px solid rgba(100, 200, 255, 0.2)',
              borderRadius: '8px',
              padding: '16px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '24px', fontWeight: '700', color: 'var(--accent-color)', marginBottom: '4px' }}>
                {result.statistics?.citations_suggested || 0}
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Suggestions</div>
            </div>

            <div style={{
              background: 'rgba(100, 200, 255, 0.1)',
              border: '1px solid rgba(100, 200, 255, 0.2)',
              borderRadius: '8px',
              padding: '16px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '24px', fontWeight: '700', color: 'var(--accent-color)', marginBottom: '4px' }}>
                {result.statistics?.sentences_analyzed || 0}
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Sentences</div>
            </div>
          </div>

          {/* Tabs */}
          <div style={{ marginBottom: '24px', borderBottom: '1px solid var(--c-border)' }}>
            <div style={{ display: 'flex', gap: '4px' }}>
              {[
                { id: 'suggestions', label: 'Citation Suggestions' },
                { id: 'rewritten', label: 'Rewritten with Citations' },
                { id: 'bibtex', label: 'BibTeX Export' }
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  style={{
                    padding: '12px 16px',
                    background: activeTab === tab.id ? 'rgba(100, 200, 255, 0.2)' : 'transparent',
                    border: 'none',
                    borderBottom: activeTab === tab.id ? '2px solid var(--accent-color)' : 'none',
                    color: activeTab === tab.id ? 'var(--c-fg-strong)' : 'var(--text-secondary)',
                    cursor: 'pointer',
                    fontSize: '13px',
                    fontWeight: activeTab === tab.id ? '600' : '400'
                  }}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          {/* Suggestions Tab */}
          {activeTab === 'suggestions' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {result.citations?.map((sentence, idx) => (
                <div key={idx} style={{
                  background: 'var(--c-sunken)',
                  border: '1px solid var(--c-border)',
                  borderRadius: '8px',
                  padding: '16px'
                }}>
                  <p style={{ margin: '0 0 12px 0', color: 'var(--c-fg-strong)', fontSize: '14px', fontStyle: 'italic' }}>
                    "{sentence.text}"
                  </p>
                  
                  {sentence.suggested_papers?.length > 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {sentence.suggested_papers.map((paper, pidx) => (
                        <div key={pidx} style={{
                          background: 'rgba(100, 200, 255, 0.1)',
                          border: '1px solid rgba(100, 200, 255, 0.2)',
                          borderRadius: '6px',
                          padding: '10px 12px',
                          fontSize: '12px'
                        }}>
                          <div style={{ color: 'var(--accent-color)', fontWeight: '600', marginBottom: '4px' }}>
                            [{paper.id}] {paper.title.substring(0, 60)}...
                          </div>
                          <div style={{ color: 'var(--text-secondary)', fontSize: '11px' }}>
                            Similarity: {(paper.similarity_score * 100).toFixed(0)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p style={{ color: 'var(--text-secondary)', fontSize: '12px', margin: 0 }}>No related papers found</p>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Rewritten Tab */}
          {activeTab === 'rewritten' && (
            <div style={{
              background: 'var(--c-sunken)',
              border: '1px solid var(--c-border)',
              borderRadius: '8px',
              padding: '16px',
              position: 'relative'
            }}>
              <button
                onClick={() => handleCopy(result.rewritten_paragraph)}
                style={{
                  position: 'absolute',
                  top: '12px',
                  right: '12px',
                  padding: '6px 12px',
                  background: 'rgba(100, 200, 255, 0.2)',
                  border: '1px solid rgba(100, 200, 255, 0.3)',
                  borderRadius: '4px',
                  color: 'var(--accent-color)',
                  cursor: 'pointer',
                  fontSize: '11px',
                  fontWeight: '500',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px'
                }}
              >
                <Copy size={14} />
                {copied ? 'Copied!' : 'Copy'}
              </button>

              <p style={{
                color: 'var(--c-fg-strong)',
                fontSize: '14px',
                lineHeight: '1.6',
                margin: 0,
                paddingRight: '100px',
                whiteSpace: 'pre-wrap'
              }}>
                {result.rewritten_paragraph}
              </p>
            </div>
          )}

          {/* BibTeX Tab */}
          {activeTab === 'bibtex' && (
            <div style={{
              background: 'var(--c-sunken)',
              border: '1px solid var(--c-border)',
              borderRadius: '8px',
              padding: '16px',
              position: 'relative'
            }}>
              <button
                onClick={handleDownloadBibTeX}
                style={{
                  position: 'absolute',
                  top: '12px',
                  right: '12px',
                  padding: '8px 16px',
                  background: 'var(--accent-color)',
                  color: '#000',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '12px',
                  fontWeight: '600',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}
              >
                <Download size={14} />
                Download .bib
              </button>

              <pre style={{
                color: 'var(--text-secondary)',
                fontSize: '11px',
                overflow: 'auto',
                maxHeight: '400px',
                margin: 0,
                paddingRight: '150px',
                fontFamily: 'monospace',
                lineHeight: '1.4'
              }}>
                {result.bibtex}
              </pre>
            </div>
          )}
        </div>
      )}

      {/* Empty State */}
      {!result && !loading && !error && (
        <div style={{
          padding: '60px 20px',
          textAlign: 'center',
          color: 'var(--text-secondary)'
        }}>
          <BookOpen size={48} style={{ marginBottom: '16px', opacity: 0.3 }} />
          <p>Paste your research text to find related papers and auto-generate citations</p>
        </div>
      )}
    </div>
  );
}
