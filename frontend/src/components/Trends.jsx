import React, { useState, useEffect } from 'react';
import { TrendingUp, Calendar, AlertCircle } from 'lucide-react';

export default function Trends({ geminiApiKey }) {
  const [trendsData, setTrendsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedTopic, setSelectedTopic] = useState(null);

  useEffect(() => {
    fetchTrends();
  }, []);

  const fetchTrends = async () => {
    try {
      setLoading(true);
      const url = geminiApiKey
        ? `${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/trends?api_key=${encodeURIComponent(geminiApiKey)}`
        : `${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/trends`;
      const res = await fetch(url, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!res.ok) throw new Error('Failed to fetch trends');
      
      const data = await res.json();
      setTrendsData(data);
      setError(null);
    } catch (err) {
      console.error('Trends error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px', marginBottom: '20px' }}>
          <div style={{ width: '20px', height: '20px', border: '2px solid rgba(255,255,255,0.2)', borderTopColor: 'var(--accent-color)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
          Loading trends...
        </div>
      </div>
    );
  }

  if (error || !trendsData) {
    return (
      <div style={{ padding: '40px', color: 'var(--error-color)' }}>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <AlertCircle size={20} />
          <span>Error loading trends: {error || 'Unknown error'}</span>
        </div>
      </div>
    );
  }

  const risingTopics = trendsData.rising_topics || [];
  const topTopics = trendsData.top_topics || [];
  const newThisWeek = trendsData.new_this_week || [];
  const summary = trendsData.trend_summary || '';

  return (
    <div style={{ padding: '24px', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: '32px' }}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '12px', margin: '0 0 8px 0', fontSize: '24px', fontFamily: 'var(--font-heading)' }}>
          <TrendingUp size={28} style={{ color: 'var(--accent-color)' }} />
          Research Trends
        </h2>
        <p style={{ color: 'var(--text-secondary)', margin: 0 }}>Real-time analysis of emerging topics and research activity</p>
      </div>

      {/* Summary */}
      {summary && (
        <div style={{ 
          background: 'rgba(100, 200, 255, 0.1)', 
          border: '1px solid rgba(100, 200, 255, 0.2)',
          borderRadius: '12px',
          padding: '16px',
          marginBottom: '24px',
          lineHeight: '1.6'
        }}>
          <h4 style={{ margin: '0 0 8px 0', color: 'var(--accent-color)' }}>Trend Summary</h4>
          <p style={{ margin: 0, color: 'var(--text-secondary)' }}>{summary}</p>
        </div>
      )}

      {/* Grid Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
        
        {/* Rising Topics Card */}
        <div style={{
          background: 'var(--c-sunken)',
          border: '1px solid var(--c-border)',
          borderRadius: '12px',
          padding: '20px'
        }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: '0 0 16px 0', fontSize: '18px', fontFamily: 'var(--font-heading)' }}>
            <TrendingUp size={20} style={{ color: '#ff6b6b' }} />
            Rising Topics
          </h3>
          
          {risingTopics.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {risingTopics.slice(0, 5).map((topic, idx) => (
                <div 
                  key={idx}
                  onClick={() => setSelectedTopic(topic)}
                  style={{
                    background: 'var(--c-surface)',
                    border: '1px solid var(--c-border)',
                    borderRadius: '8px',
                    padding: '12px',
                    cursor: 'pointer',
                    transition: 'all 0.2s'
                  }}
                  onMouseEnter={(e) => e.target.style.background = 'var(--c-border)'}
                  onMouseLeave={(e) => e.target.style.background = 'var(--c-surface)'}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '4px' }}>
                    <span style={{ fontWeight: '600', color: 'var(--c-fg-strong)' }}>{topic.topic}</span>
                    <span style={{ background: 'rgba(255, 100, 100, 0.3)', color: '#ff6b6b', padding: '4px 8px', borderRadius: '4px', fontSize: '12px', fontWeight: '600' }}>
                      +{(topic.growth_rate * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                    {topic.this_week} papers on the latest day vs {topic.prev_week} avg earlier
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p style={{ color: 'var(--text-secondary)', margin: 0 }}>No rising topics yet</p>
          )}
        </div>

        {/* Top Topics Card */}
        <div style={{
          background: 'var(--c-sunken)',
          border: '1px solid var(--c-border)',
          borderRadius: '12px',
          padding: '20px'
        }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: '0 0 16px 0', fontSize: '18px', fontFamily: 'var(--font-heading)' }}>
            <Calendar size={20} style={{ color: '#64b5f6' }} />
            Top Topics
          </h3>
          
          {topTopics.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {topTopics.map((topic, idx) => (
                <div 
                  key={idx}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    padding: '12px',
                    background: 'rgba(100, 181, 246, 0.1)',
                    borderRadius: '8px'
                  }}
                >
                  <span style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '24px',
                    height: '24px',
                    borderRadius: '50%',
                    background: 'var(--accent-color)',
                    color: '#000',
                    fontWeight: 'bold',
                    fontSize: '12px'
                  }}>
                    {idx + 1}
                  </span>
                  <span style={{ flex: 1, color: 'var(--c-fg-strong)' }}>{topic}</span>
                </div>
              ))}
            </div>
          ) : (
            <p style={{ color: 'var(--text-secondary)', margin: 0 }}>No topics available</p>
          )}
        </div>

        {/* New This Week */}
        <div style={{
          background: 'var(--c-sunken)',
          border: '1px solid var(--c-border)',
          borderRadius: '12px',
          padding: '20px'
        }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: '0 0 16px 0', fontSize: '18px', fontFamily: 'var(--font-heading)' }}>
            ✨ New This Week
          </h3>
          
          {newThisWeek.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {newThisWeek.map((topic, idx) => (
                <div 
                  key={idx}
                  style={{
                    background: 'rgba(255, 200, 100, 0.1)',
                    border: '1px solid rgba(255, 200, 100, 0.3)',
                    borderRadius: '8px',
                    padding: '10px 12px',
                    color: 'var(--text-secondary)',
                    fontSize: '14px'
                  }}
                >
                  {topic}
                </div>
              ))}
            </div>
          ) : (
            <p style={{ color: 'var(--text-secondary)', margin: 0 }}>No new topics this week</p>
          )}
        </div>
      </div>

      {/* Refresh Button */}
      <div style={{ marginTop: '24px', textAlign: 'center' }}>
        <button
          onClick={fetchTrends}
          style={{
            padding: '10px 20px',
            background: 'var(--accent-color)',
            color: '#000',
            border: 'none',
            borderRadius: '8px',
            fontWeight: '600',
            cursor: 'pointer',
            fontSize: '14px'
          }}
          onMouseEnter={(e) => e.target.style.opacity = '0.9'}
          onMouseLeave={(e) => e.target.style.opacity = '1'}
        >
          Refresh Trends
        </button>
      </div>
    </div>
  );
}
