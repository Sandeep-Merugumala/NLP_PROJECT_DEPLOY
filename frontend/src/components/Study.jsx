import React, { useState, useEffect } from 'react';
import { BookMarked, RotateCcw, CheckCircle, XCircle, AlertCircle, Search, Brain, Sparkles, Send } from 'lucide-react';

export default function Study({ papers = [], geminiApiKey }) {
  const [paperSearch, setPaperSearch] = useState('');
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [cards, setCards] = useState([]);
  const [currentCard, setCurrentCard] = useState(null);   // RL-selected card being shown
  const [sessionLen, setSessionLen] = useState(0);        // how many reviews make a session
  const [isFlipped, setIsFlipped] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({ correct: 0, incorrect: 0, reviewed: 0 });
  const [showMCQOptions, setShowMCQOptions] = useState(false);

  // ---- NLP STS Grading State ----
  const [typedAnswer, setTypedAnswer] = useState('');
  const [stsResult, setStsResult] = useState(null);       // {similarity, quality, correct, feedback, method}
  const [grading, setGrading] = useState(false);           // loading state for grading

  const handleGenerateCards = async (paper) => {
    try {
      setLoading(true);
      setError(null);
      setSelectedPaper(paper);
      
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/study/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          paper_id: paper.id,
          api_key: geminiApiKey
        })
      });
      
      if (!res.ok) throw new Error('Failed to generate study cards');
      
      const data = await res.json();
      const deck = data.cards || [];
      setCards(deck);
      setCurrentCard(deck[0] || null);
      setSessionLen(deck.length);
      setIsFlipped(false);
      setStats({ correct: 0, incorrect: 0, reviewed: 0 });
      setShowMCQOptions(false);
      setTypedAnswer('');
      setStsResult(null);
    } catch (err) {
      console.error('Study error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ---- NLP Feature: Submit typed answer for STS grading ----
  const handleTypedAnswerSubmit = async () => {
    if (!currentCard || !typedAnswer.trim()) return;
    setGrading(true);
    setStsResult(null);

    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/study/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          card_id: currentCard.id,
          typed_answer: typedAnswer.trim(),
          paper_id: selectedPaper.id,
          api_key: geminiApiKey
        })
      });
      const data = await res.json().catch(() => ({}));

      const sts = data.sts_grading || {};
      setStsResult(sts);

      const correct = sts.correct || false;
      const reviewedNow = stats.reviewed + 1;
      setStats(prev => ({
        ...prev,
        correct: prev.correct + (correct ? 1 : 0),
        incorrect: prev.incorrect + (correct ? 0 : 1),
        reviewed: reviewedNow
      }));

      // Store the next card info for later advancement
      setCurrentCard(prev => ({
        ...prev,
        _nextCard: data.next_card,
        _reviewedNow: reviewedNow
      }));
    } catch (err) {
      console.error('STS grading error:', err);
      setStsResult({ similarity: 0, quality: 0, correct: false, feedback: '❌ Error grading answer.', method: 'error' });
    } finally {
      setGrading(false);
    }
  };

  // ---- Advance to next card after viewing STS result ----
  const handleAdvanceAfterSTS = () => {
    const reviewedNow = currentCard?._reviewedNow || stats.reviewed;
    const next = currentCard?._nextCard;

    if (reviewedNow >= sessionLen || !next) {
      setStats(prev => ({ ...prev, complete: true }));
    } else {
      setCurrentCard(next);
      setIsFlipped(false);
      setShowMCQOptions(false);
      setTypedAnswer('');
      setStsResult(null);
    }
  };

  const handleCardResponse = async (correct, quality = null) => {
    if (!currentCard) return;

    try {
      // Log the review — the backend updates SM-2 + the Thompson-Sampling bandit
      // and returns the RL-selected next card (based on your correct/incorrect history).
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001'}/api/study/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          card_id: currentCard.id,
          correct,
          quality: quality || (correct ? 4 : 1),
          paper_id: selectedPaper.id,
          api_key: geminiApiKey
        })
      });
      const data = await res.json().catch(() => ({}));

      const reviewedNow = stats.reviewed + 1;
      setStats(prev => ({
        ...prev,
        correct: prev.correct + (correct ? 1 : 0),
        incorrect: prev.incorrect + (correct ? 0 : 1),
        reviewed: reviewedNow
      }));

      // Advance to the RL-chosen next card; end the session after `sessionLen` reviews.
      const next = data.next_card;
      if (reviewedNow >= sessionLen || !next) {
        setStats(prev => ({ ...prev, complete: true }));
      } else {
        setCurrentCard(next);
        setIsFlipped(false);
        setShowMCQOptions(false);
        setTypedAnswer('');
        setStsResult(null);
      }
    } catch (err) {
      console.error('Error logging review:', err);
    }
  };

  const isComplete = stats.complete;

  // Similarity bar color
  const getSimilarityColor = (sim) => {
    if (sim >= 0.75) return '#4ade80';
    if (sim >= 0.55) return '#a3e635';
    if (sim >= 0.40) return '#facc15';
    if (sim >= 0.25) return '#fb923c';
    return '#ff6b6b';
  };

  return (
    <div style={{ padding: '24px', maxWidth: '1000px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: '32px' }}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '12px', margin: '0 0 8px 0', fontSize: '24px', fontFamily: 'var(--font-heading)' }}>
          <BookMarked size={28} style={{ color: 'var(--accent-color)' }} />
          Study Mode
          <span style={{
            display: 'inline-flex', alignItems: 'center', gap: '4px',
            fontSize: '11px', fontWeight: '600',
            background: 'linear-gradient(135deg, rgba(147, 51, 234, 0.2), rgba(59, 130, 246, 0.2))',
            border: '1px solid rgba(147, 51, 234, 0.3)',
            borderRadius: '20px', padding: '3px 10px',
            color: '#a78bfa'
          }}>
            <Brain size={12} />
            NLP Pipeline (KeyBERT + T5 + SBERT)
          </span>
        </h2>
        <p style={{ color: 'var(--text-secondary)', margin: 0 }}>
          AI-generated flashcards with <strong>semantic auto-grading</strong> — type your answer and get instant NLP-powered feedback
        </p>
      </div>

      {/* Error */}
      {error && (
        <div style={{ padding: '16px', background: 'rgba(255, 100, 100, 0.1)', border: '1px solid rgba(255, 100, 100, 0.2)', borderRadius: '8px', color: '#ff6b6b', marginBottom: '24px', display: 'flex', gap: '12px', alignItems: 'center' }}>
          <AlertCircle size={20} />
          {error}
        </div>
      )}

      {/* Study Session Active */}
      {selectedPaper && cards.length > 0 ? (
        <div>
          {/* Progress Bar */}
          <div style={{
            background: 'var(--c-sunken)',
            borderRadius: '12px',
            padding: '16px',
            marginBottom: '24px'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <h3 style={{ margin: 0, color: 'var(--c-fg-strong)', fontSize: '16px' }}>
                {selectedPaper.title?.substring(0, 60)}...
              </h3>
              <button
                onClick={() => {
                  setSelectedPaper(null);
                  setCards([]);
                  setCurrentCard(null);
                  setSessionLen(0);
                  setStats({ correct: 0, incorrect: 0, reviewed: 0 });
                  setTypedAnswer('');
                  setStsResult(null);
                }}
                style={{
                  padding: '6px 12px',
                  background: 'rgba(255, 100, 100, 0.2)',
                  border: '1px solid rgba(255, 100, 100, 0.3)',
                  borderRadius: '6px',
                  color: '#ff6b6b',
                  cursor: 'pointer',
                  fontSize: '12px',
                  fontWeight: '500'
                }}
              >
                Exit
              </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginBottom: '12px' }}>
              <div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>Progress</div>
                <div style={{ fontSize: '18px', fontWeight: '700', color: 'var(--accent-color)' }}>
                  {Math.min(stats.reviewed + 1, sessionLen)} / {sessionLen}
                </div>
              </div>
              <div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>Correct</div>
                <div style={{ fontSize: '18px', fontWeight: '700', color: '#4ade80' }}>
                  {stats.correct}
                </div>
              </div>
              <div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>Incorrect</div>
                <div style={{ fontSize: '18px', fontWeight: '700', color: '#ff6b6b' }}>
                  {stats.incorrect}
                </div>
              </div>
            </div>

            {/* Progress Bar */}
            <div style={{
              height: '6px',
              background: 'var(--c-border)',
              borderRadius: '3px',
              overflow: 'hidden'
            }}>
              <div
                style={{
                  height: '100%',
                  width: `${(stats.reviewed / Math.max(1, sessionLen)) * 100}%`,
                  background: 'linear-gradient(90deg, var(--accent-color), #ffd700)',
                  transition: 'width 0.3s'
                }}
              />
            </div>
          </div>

          {/* Card Display */}
          {!isComplete ? (
            <div>
              <div
                style={{
                  background: 'linear-gradient(135deg, rgba(100, 200, 255, 0.15), rgba(200, 100, 255, 0.15))',
                  border: '1px solid var(--c-border)',
                  borderRadius: '12px',
                  padding: '40px 24px',
                  minHeight: '280px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  textAlign: 'center',
                  marginBottom: '24px',
                  position: 'relative',
                  transition: 'all 0.2s'
                }}
              >
                {/* Card type badge */}
                <div style={{ position: 'absolute', top: '16px', left: '16px', display: 'flex', gap: '8px' }}>
                  <span style={{
                    fontSize: '11px', fontWeight: '600',
                    background: currentCard?.card_type === 'mcq'
                      ? 'rgba(59, 130, 246, 0.2)' : 'rgba(147, 51, 234, 0.2)',
                    border: `1px solid ${currentCard?.card_type === 'mcq'
                      ? 'rgba(59, 130, 246, 0.3)' : 'rgba(147, 51, 234, 0.3)'}`,
                    borderRadius: '4px', padding: '2px 8px',
                    color: currentCard?.card_type === 'mcq' ? '#60a5fa' : '#a78bfa'
                  }}>
                    {currentCard?.card_type === 'mcq' ? 'MCQ' : 'Flashcard'}
                  </span>
                  {currentCard?.card_type === 'flashcard' && (
                    <span style={{
                      fontSize: '11px', fontWeight: '600',
                      background: 'rgba(16, 185, 129, 0.2)',
                      border: '1px solid rgba(16, 185, 129, 0.3)',
                      borderRadius: '4px', padding: '2px 8px',
                      color: '#34d399',
                      display: 'flex', alignItems: 'center', gap: '3px'
                    }}>
                      <Sparkles size={10} />
                      STS Auto-Grading
                    </span>
                  )}
                </div>

                {currentCard?.card_type === 'mcq' ? (
                  /* ========== MCQ Card ========== */
                  <div style={{ width: '100%' }}>
                    <h4 style={{ margin: '0 0 24px 0', color: 'var(--c-fg-strong)', fontSize: '16px', fontWeight: '600' }}>
                      {currentCard.question}
                    </h4>
                    {!showMCQOptions ? (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setShowMCQOptions(true);
                        }}
                        style={{
                          padding: '12px 24px',
                          background: 'rgba(100, 200, 255, 0.2)',
                          border: '1px solid rgba(100, 200, 255, 0.3)',
                          borderRadius: '8px',
                          color: 'var(--accent-color)',
                          cursor: 'pointer',
                          fontSize: '14px',
                          fontWeight: '600'
                        }}
                      >
                        Show Options
                      </button>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {['A', 'B', 'C', 'D'].map((opt, idx) => (
                          <button
                            key={opt}
                            onClick={(e) => {
                              e.stopPropagation();
                              const correct = opt === currentCard.answer;
                              handleCardResponse(correct, correct ? 4 : 1);
                            }}
                            style={{
                              padding: '12px',
                              background: 'rgba(100, 200, 255, 0.1)',
                              border: '1px solid rgba(100, 200, 255, 0.2)',
                              borderRadius: '8px',
                              color: 'var(--accent-color)',
                              cursor: 'pointer',
                              fontSize: '14px',
                              fontWeight: '500',
                              transition: 'all 0.2s',
                              textAlign: 'left'
                            }}
                            onMouseEnter={(e) => {
                              e.target.style.background = 'rgba(100, 200, 255, 0.2)';
                            }}
                            onMouseLeave={(e) => {
                              e.target.style.background = 'rgba(100, 200, 255, 0.1)';
                            }}
                          >
                            {currentCard.options && currentCard.options[idx] 
                              ? (currentCard.options[idx].startsWith(opt + ')') ? currentCard.options[idx] : `${opt}) ${currentCard.options[idx]}`)
                              : `${opt})`}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  /* ========== Flashcard with STS Grading ========== */
                  <div style={{ width: '100%' }}>
                    <h4 style={{ margin: '0 0 24px 0', color: 'var(--c-fg-strong)', fontSize: '18px', fontWeight: '600', lineHeight: '1.6' }}>
                      {currentCard.question}
                    </h4>

                    {/* STS Result Display */}
                    {stsResult ? (
                      <div style={{ width: '100%' }}>
                        {/* Similarity Score Bar */}
                        <div style={{
                          background: 'rgba(0,0,0,0.2)',
                          borderRadius: '12px',
                          padding: '20px',
                          marginBottom: '16px',
                          border: `1px solid ${getSimilarityColor(stsResult.similarity)}40`
                        }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: '500' }}>
                              Semantic Similarity (SBERT Cosine)
                            </span>
                            <span style={{
                              fontSize: '22px', fontWeight: '800',
                              color: getSimilarityColor(stsResult.similarity)
                            }}>
                              {(stsResult.similarity * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div style={{
                            height: '8px',
                            background: 'rgba(255,255,255,0.1)',
                            borderRadius: '4px',
                            overflow: 'hidden'
                          }}>
                            <div style={{
                              height: '100%',
                              width: `${stsResult.similarity * 100}%`,
                              background: `linear-gradient(90deg, ${getSimilarityColor(stsResult.similarity)}, ${getSimilarityColor(stsResult.similarity)}aa)`,
                              borderRadius: '4px',
                              transition: 'width 0.8s ease-out'
                            }} />
                          </div>

                          {/* Quality badge */}
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px' }}>
                            <span style={{ fontSize: '14px', fontWeight: '600', color: getSimilarityColor(stsResult.similarity) }}>
                              {stsResult.feedback}
                            </span>
                            <span style={{
                              fontSize: '11px', fontWeight: '600',
                              background: 'rgba(147, 51, 234, 0.2)',
                              border: '1px solid rgba(147, 51, 234, 0.3)',
                              borderRadius: '4px', padding: '2px 8px',
                              color: '#a78bfa'
                            }}>
                              SM-2 Quality: {stsResult.quality}/4
                            </span>
                          </div>
                        </div>

                        {/* Your answer vs Reference */}
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '16px' }}>
                          <div style={{
                            background: 'rgba(255,255,255,0.05)',
                            borderRadius: '8px', padding: '12px',
                            border: `1px solid ${stsResult.correct ? 'rgba(74, 222, 128, 0.3)' : 'rgba(255, 107, 107, 0.3)'}`
                          }}>
                            <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: '600' }}>
                              YOUR ANSWER
                            </div>
                            <div style={{ fontSize: '13px', color: 'var(--c-fg-strong)', lineHeight: '1.5' }}>
                              {typedAnswer}
                            </div>
                          </div>
                          <div style={{
                            background: 'rgba(74, 222, 128, 0.05)',
                            borderRadius: '8px', padding: '12px',
                            border: '1px solid rgba(74, 222, 128, 0.2)'
                          }}>
                            <div style={{ fontSize: '11px', color: '#4ade80', marginBottom: '6px', fontWeight: '600' }}>
                              REFERENCE ANSWER
                            </div>
                            <div style={{ fontSize: '13px', color: 'var(--c-fg-strong)', lineHeight: '1.5' }}>
                              {currentCard.answer}
                            </div>
                          </div>
                        </div>

                        {/* Next button */}
                        <button
                          onClick={handleAdvanceAfterSTS}
                          style={{
                            padding: '12px 32px',
                            background: 'var(--accent-color)',
                            color: '#000',
                            border: 'none',
                            borderRadius: '8px',
                            fontWeight: '700',
                            cursor: 'pointer',
                            fontSize: '14px',
                            transition: 'all 0.2s'
                          }}
                        >
                          Next Card →
                        </button>
                      </div>
                    ) : (
                      /* ---- Type Answer Input ---- */
                      <div style={{ width: '100%' }}>
                        <textarea
                          value={typedAnswer}
                          onChange={(e) => setTypedAnswer(e.target.value)}
                          placeholder="Type your answer here... (SBERT will grade it semantically)"
                          style={{
                            width: '100%',
                            minHeight: '100px',
                            padding: '14px',
                            background: 'rgba(255,255,255,0.05)',
                            border: '1px solid var(--c-border)',
                            borderRadius: '8px',
                            color: 'var(--c-fg-strong)',
                            fontSize: '14px',
                            lineHeight: '1.6',
                            resize: 'vertical',
                            fontFamily: 'inherit',
                            marginBottom: '12px',
                            outline: 'none',
                            boxSizing: 'border-box'
                          }}
                          onFocus={(e) => { e.target.style.borderColor = 'var(--accent-color)'; }}
                          onBlur={(e) => { e.target.style.borderColor = 'var(--c-border)'; }}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' && e.ctrlKey) handleTypedAnswerSubmit();
                          }}
                        />
                        <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
                          <button
                            onClick={handleTypedAnswerSubmit}
                            disabled={!typedAnswer.trim() || grading}
                            style={{
                              display: 'flex', alignItems: 'center', gap: '8px',
                              padding: '12px 24px',
                              background: typedAnswer.trim() ? 'linear-gradient(135deg, #8b5cf6, #3b82f6)' : 'rgba(100,100,100,0.2)',
                              border: 'none',
                              borderRadius: '8px',
                              color: typedAnswer.trim() ? '#fff' : 'var(--text-secondary)',
                              cursor: typedAnswer.trim() ? 'pointer' : 'not-allowed',
                              fontSize: '14px',
                              fontWeight: '700',
                              transition: 'all 0.2s'
                            }}
                          >
                            {grading ? (
                              <>
                                <div style={{ width: '16px', height: '16px', border: '2px solid rgba(255,255,255,0.3)', borderTopColor: '#fff', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
                                Grading with SBERT...
                              </>
                            ) : (
                              <>
                                <Send size={16} />
                                Submit Answer (Ctrl+Enter)
                              </>
                            )}
                          </button>
                          <button
                            onClick={() => handleCardResponse(false, 0)}
                            style={{
                              display: 'flex', alignItems: 'center', gap: '8px',
                              padding: '12px 16px',
                              background: 'rgba(255, 107, 107, 0.1)',
                              border: '1px solid rgba(255, 107, 107, 0.3)',
                              borderRadius: '8px',
                              color: '#ff6b6b',
                              cursor: 'pointer',
                              fontSize: '13px',
                              fontWeight: '500',
                              transition: 'all 0.2s'
                            }}
                          >
                            <XCircle size={16} />
                            Skip
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* Session Complete */
            <div style={{
              background: 'rgba(100, 200, 255, 0.1)',
              border: '1px solid rgba(100, 200, 255, 0.2)',
              borderRadius: '12px',
              padding: '40px 24px',
              textAlign: 'center'
            }}>
              <h3 style={{ margin: '0 0 16px 0', color: '#4ade80', fontSize: '24px' }}>Session Complete! 🎉</h3>
              <div style={{ margin: '24px 0', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div>
                  <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '8px' }}>Correct</div>
                  <div style={{ fontSize: '32px', fontWeight: '700', color: '#4ade80' }}>{stats.correct}</div>
                </div>
                <div>
                  <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '8px' }}>Incorrect</div>
                  <div style={{ fontSize: '32px', fontWeight: '700', color: '#ff6b6b' }}>{stats.incorrect}</div>
                </div>
              </div>
              <div style={{ fontSize: '18px', fontWeight: '600', color: 'var(--accent-color)', marginBottom: '8px' }}>
                Score: {Math.round((stats.correct / (stats.correct + stats.incorrect || 1)) * 100)}%
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '20px' }}>
                Graded by NLP Semantic Textual Similarity (SBERT)
              </div>
              <button
                onClick={() => {
                  setSelectedPaper(null);
                  setCards([]);
                  setCurrentCard(null);
                  setSessionLen(0);
                  setStats({ correct: 0, incorrect: 0, reviewed: 0 });
                  setTypedAnswer('');
                  setStsResult(null);
                }}
                style={{
                  padding: '12px 24px',
                  background: 'var(--accent-color)',
                  color: '#000',
                  border: 'none',
                  borderRadius: '8px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  fontSize: '14px'
                }}
              >
                <RotateCcw size={16} style={{ marginRight: '8px', display: 'inline' }} />
                New Session
              </button>
            </div>
          )}
        </div>
      ) : (
        /* Paper Selection */
        <div>
          {loading ? (
            <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px' }}>
                <div style={{ width: '20px', height: '20px', border: '2px solid rgba(255,255,255,0.2)', borderTopColor: 'var(--accent-color)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
                Generating study cards (KeyBERT + T5 NLP pipeline)...
              </div>
            </div>
          ) : (
            <>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '12px', fontSize: '14px' }}>
                Select any paper to generate 10 study cards (7 MCQs + 3 flashcards) using <strong>KeyBERT + T5</strong>:
              </p>
              <div style={{ position: 'relative', marginBottom: '16px', maxWidth: '480px' }}>
                <Search size={16} style={{ position: 'absolute', left: 12, top: 13, color: 'var(--text-muted)' }} />
                <input
                  className="glass-input"
                  placeholder={`Search ${papers.length} papers by title…`}
                  value={paperSearch}
                  onChange={(e) => setPaperSearch(e.target.value)}
                  style={{ width: '100%', padding: '11px 12px 11px 38px' }}
                />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '16px', maxHeight: '60vh', overflowY: 'auto', paddingRight: '4px' }}>
                {papers
                  .filter(p => p.title?.toLowerCase().includes(paperSearch.toLowerCase()))
                  .slice(0, 60)
                  .map(paper => (
                  <button
                    key={paper.id}
                    onClick={() => handleGenerateCards(paper)}
                    style={{
                      padding: '16px',
                      background: 'var(--c-sunken)',
                      border: '1px solid var(--c-border)',
                      borderRadius: '8px',
                      color: 'var(--c-fg-strong)',
                      cursor: 'pointer',
                      textAlign: 'left',
                      transition: 'all 0.2s',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '8px'
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
                    <strong>{paper.title?.substring(0, 60)}{paper.title?.length > 60 ? '…' : ''}</strong>
                    <p style={{ margin: 0, fontSize: '12px', color: 'var(--accent-secondary)' }}>
                      {paper.category_name || 'Machine Learning'} · {(paper.published || '').slice(0, 10) || '—'}
                    </p>
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
