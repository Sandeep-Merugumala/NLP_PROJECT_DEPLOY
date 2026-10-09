import React, { useState } from 'react';
import { Columns3, AlertCircle, Plus, X } from 'lucide-react';

export default function Compare({ papers = [], geminiApiKey }) {
  const [selectedPapers, setSelectedPapers] = useState([]);
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showPaperList, setShowPaperList] = useState(false);

  const handleSelectPaper = (paper) => {
    if (selectedPapers.length < 3 && !selectedPapers.find(p => p.id === paper.id)) {
      setSelectedPapers([...selectedPapers, paper]);
    }
  };

  const handleRemovePaper = (paperId) => {
    setSelectedPapers(selectedPapers.filter(p => p.id !== paperId));
  };

  const handleCompare = async () => {
    if (selectedPapers.length < 2) {
      setError('Please select at least 2 papers to compare');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          paper_ids: selectedPapers.map(p => p.id),
          api_key: geminiApiKey
        })
      });
      
      if (!res.ok) throw new Error('Failed to compare papers');
      
      const data = await res.json();
      setComparison(data);
    } catch (err) {
      console.error('Compare error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: '32px' }}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '12px', margin: '0 0 8px 0', fontSize: '24px', fontFamily: 'var(--font-heading)' }}>
          <Columns3 size={28} style={{ color: 'var(--accent-color)' }} />
          Paper Comparison Studio
        </h2>
        <p style={{ color: 'var(--text-secondary)', margin: 0 }}>Select 2-3 papers to compare side-by-side</p>
      </div>

      {/* Error */}
      {error && (
        <div style={{ padding: '16px', background: 'rgba(255, 100, 100, 0.1)', border: '1px solid rgba(255, 100, 100, 0.2)', borderRadius: '8px', color: '#ff6b6b', marginBottom: '24px', display: 'flex', gap: '12px', alignItems: 'center' }}>
          <AlertCircle size={20} />
          {error}
        </div>
      )}

      {/* Selection Section */}
      <div style={{
        background: 'var(--c-sunken)',
        border: '1px solid var(--c-border)',
        borderRadius: '12px',
        padding: '20px',
        marginBottom: '24px'
      }}>
        <h3 style={{ margin: '0 0 16px 0', color: 'var(--c-fg-strong)', fontSize: '16px' }}>
          Selected Papers ({selectedPapers.length}/3)
        </h3>

        {selectedPapers.length > 0 ? (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', marginBottom: '16px' }}>
            {selectedPapers.map(paper => (
              <div
                key={paper.id}
                style={{
                  background: 'rgba(100, 200, 255, 0.1)',
                  border: '1px solid rgba(100, 200, 255, 0.3)',
                  borderRadius: '8px',
                  padding: '12px 16px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  maxWidth: '300px'
                }}
              >
                <span style={{ flex: 1, color: 'var(--accent-color)', fontSize: '13px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {paper.title?.substring(0, 40)}...
                </span>
                <button
                  onClick={() => handleRemovePaper(paper.id)}
                  style={{
                    background: 'rgba(255, 100, 100, 0.2)',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#ff6b6b',
                    padding: '4px 6px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                >
                  <X size={16} />
                </button>
              </div>
            ))}
          </div>
        ) : (
          <p style={{ color: 'var(--text-secondary)', margin: '0 0 16px 0', fontSize: '13px' }}>
            No papers selected yet. Add papers from your library to compare.
          </p>
        )}

        <div style={{ display: 'flex', gap: '12px' }}>
          <button
            onClick={() => setShowPaperList(!showPaperList)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '10px 16px',
              background: 'rgba(100, 200, 255, 0.2)',
              border: '1px solid rgba(100, 200, 255, 0.3)',
              borderRadius: '8px',
              color: 'var(--accent-color)',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: '500'
            }}
          >
            <Plus size={16} />
            Add Paper
          </button>

          {selectedPapers.length >= 2 && (
            <button
              onClick={handleCompare}
              disabled={loading}
              style={{
                padding: '10px 20px',
                background: 'var(--accent-color)',
                color: '#000',
                border: 'none',
                borderRadius: '8px',
                fontWeight: '600',
                cursor: loading ? 'wait' : 'pointer',
                fontSize: '13px',
                opacity: loading ? 0.5 : 1
              }}
            >
              {loading ? 'Comparing...' : 'Compare Selected'}
            </button>
          )}
        </div>
      </div>

      {/* Paper List */}
      {showPaperList && papers.length > 0 && (
        <div style={{
          background: 'var(--c-sunken)',
          border: '1px solid var(--c-border)',
          borderRadius: '12px',
          padding: '16px',
          marginBottom: '24px',
          maxHeight: '300px',
          overflowY: 'auto'
        }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {papers.slice(0, 20).map(paper => (
              <button
                key={paper.id}
                onClick={() => {
                  handleSelectPaper(paper);
                  if (selectedPapers.length >= 2) setShowPaperList(false);
                }}
                disabled={selectedPapers.find(p => p.id === paper.id)}
                style={{
                  padding: '12px',
                  background: selectedPapers.find(p => p.id === paper.id) ? 'rgba(100, 200, 255, 0.2)' : 'var(--c-surface)',
                  border: '1px solid var(--c-border)',
                  borderRadius: '8px',
                  color: 'var(--c-fg-strong)',
                  textAlign: 'left',
                  cursor: selectedPapers.find(p => p.id === paper.id) ? 'default' : 'pointer',
                  fontSize: '13px',
                  opacity: selectedPapers.find(p => p.id === paper.id) ? 0.6 : 1,
                  transition: 'all 0.2s'
                }}
              >
                <strong>{paper.title?.substring(0, 60)}...</strong>
                <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                  {paper.published || 'Unknown'} • {paper.id}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Comparison Results */}
      {comparison && !loading && (
        <div>
          {/* Similarity Score */}
          <div style={{
            background: 'rgba(100, 200, 255, 0.1)',
            border: '1px solid rgba(100, 200, 255, 0.2)',
            borderRadius: '12px',
            padding: '16px',
            marginBottom: '24px'
          }}>
            <h4 style={{ margin: '0 0 12px 0', color: 'var(--accent-color)' }}>Comparison Summary</h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '12px' }}>
              <div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>Similarity Score</div>
                <div style={{ fontSize: '20px', fontWeight: '700', color: 'var(--accent-color)' }}>
                  {comparison.similarity_score ? (comparison.similarity_score * 100).toFixed(0) : 'N/A'}%
                </div>
              </div>
              <div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>Papers Compared</div>
                <div style={{ fontSize: '20px', fontWeight: '700', color: 'var(--accent-color)' }}>
                  {comparison.count || selectedPapers.length}
                </div>
              </div>
            </div>

            {comparison.when_to_use && (
              <>
                <h4 style={{ margin: '16px 0 8px 0', color: 'var(--accent-color)', fontSize: '13px' }}>When to Use Each</h4>
                <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '13px', lineHeight: '1.5' }}>
                  {comparison.when_to_use}
                </p>
              </>
            )}
          </div>

          {/* Comparison Table */}
          {comparison.comparison_table && (
            <div style={{
              background: 'var(--c-sunken)',
              border: '1px solid var(--c-border)',
              borderRadius: '12px',
              overflow: 'hidden'
            }}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  fontSize: '12px'
                }}>
                  <tbody>
                    {['Paper', 'Year', 'Problem', 'Methodology', 'Dataset', 'Results', 'Limitations'].map((field, fidx) => (
                      <tr key={fidx} style={{ borderBottom: '1px solid var(--c-surface)' }}>
                        <td style={{
                          padding: '12px',
                          background: 'rgba(100, 200, 255, 0.05)',
                          color: 'var(--accent-color)',
                          fontWeight: '600',
                          minWidth: '100px',
                          borderRight: '1px solid var(--c-surface)'
                        }}>
                          {field}
                        </td>
                        
                        {comparison.comparison_table[field]?.map((value, cidx) => (
                          <td key={cidx} style={{
                            padding: '12px',
                            background: cidx % 2 === 0 ? 'var(--c-surface)' : 'var(--c-surface)',
                            color: 'var(--text-secondary)',
                            minWidth: '150px',
                            borderRight: '1px solid var(--c-surface)',
                            verticalAlign: 'top'
                          }}>
                            <div style={{ maxHeight: '80px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                              {String(value).substring(0, 150)}
                              {String(value).length > 150 ? '...' : ''}
                            </div>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Empty State */}
      {!comparison && !loading && (
        <div style={{
          padding: '60px 20px',
          textAlign: 'center',
          color: 'var(--text-secondary)',
          background: 'var(--c-sunken)',
          borderRadius: '12px',
          border: '1px dashed var(--c-border)'
        }}>
          <Columns3 size={48} style={{ marginBottom: '16px', opacity: 0.3 }} />
          <p>Select 2-3 papers to view a detailed side-by-side comparison</p>
        </div>
      )}
    </div>
  );
}
