import React, { useState } from 'react';
import { analyzeNews, submitFeedback } from '../services/api';
import ResultDisplay from './ResultDisplay';
import HistoryList from './HistoryList';

const AnalysisForm = () => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [history, setHistory] = useState(() => {
    const saved = localStorage.getItem('analysisHistory');
    return saved ? JSON.parse(saved) : [];
  });
  const [showHistory, setShowHistory] = useState(false);

  const handleAnalyze = async (e) => {
    e.preventDefault();
    
    if (text.length < 10) {
      setError('Please enter at least 10 characters');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      const response = await analyzeNews(text);
      setResult(response);
      
      // Save to history
      const newEntry = {
        id: Date.now(),
        text: text.substring(0, 100) + (text.length > 100 ? '...' : ''),
        result: response,
        timestamp: new Date().toISOString()
      };
      
      const updatedHistory = [newEntry, ...history].slice(0, 10); // Keep last 10
      setHistory(updatedHistory);
      localStorage.setItem('analysisHistory', JSON.stringify(updatedHistory));
      
    } catch (err) {
      setError(err.response?.data?.detail || 'Error analyzing text. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setText('');
    setError('');
  };

  const handleFeedback = async () => {
    const correction = result.label === 'Real' ? 'Fake' : 'Real';
    
    try {
      await submitFeedback({
        text,
        predicted_label: result.label,
        confidence_score: result.confidence_score,
        user_correction: correction
      });
      
      alert('Thank you for your feedback! This helps improve our model.');
      handleReset();
    } catch (err) {
      alert('Error submitting feedback. Please try again.');
    }
  };

  if (result) {
    return (
      <ResultDisplay 
        result={result} 
        onReset={handleReset}
        onFeedback={handleFeedback}
      />
    );
  }

  return (
    <div className="space-y-6">
      <div className="card">
        <form onSubmit={handleAnalyze} className="space-y-4">
          <div>
            <label htmlFor="newsText" className="block text-sm font-semibold text-gray-700 mb-2">
              Paste News Article or Text
            </label>
            <textarea
              id="newsText"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter or paste the news article you want to verify..."
              className="w-full h-48 px-4 py-3 rounded-lg border-2 border-gray-200 focus:border-blue-500 focus:outline-none resize-none"
              disabled={loading}
            />
            <div className="flex justify-between items-center mt-2">
              <span className="text-sm text-gray-700 font-medium">
                {text.length} characters
              </span>
              {error && (
                <span className="text-sm text-red-600 font-semibold bg-red-50 px-2 py-1 rounded">
                  {error}
                </span>
              )}
            </div>
          </div>
          
          <button
            type="submit"
            disabled={loading || text.length < 10}
            className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Analyzing...
              </span>
            ) : (
              'Analyze Text'
            )}
          </button>
        </form>
      </div>
      
      {/* History Toggle */}
      {history.length > 0 && (
        <div>
          <button
            onClick={() => setShowHistory(!showHistory)}
            className="w-full py-3 px-4 bg-white/20 backdrop-blur-sm text-white font-semibold rounded-lg border border-white/30 hover:bg-white/30 transition-all duration-200"
          >
            {showHistory ? 'Hide' : 'Show'} History ({history.length})
          </button>
          
          {showHistory && (
            <HistoryList 
              history={history} 
              onSelect={(item) => {
                setText(item.text);
                setShowHistory(false);
              }}
              onClear={() => {
                setHistory([]);
                localStorage.removeItem('analysisHistory');
              }}
            />
          )}
        </div>
      )}
    </div>
  );
};

export default AnalysisForm;
