import React from 'react';
import AnalysisForm from './components/AnalysisForm';
import './styles/index.css';

function App() {
  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-block p-4 bg-white/10 backdrop-blur-lg rounded-full mb-6">
            <svg className="w-16 h-16 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          
          <h1 className="text-5xl font-bold text-white mb-4 drop-shadow-lg">
            AI Fake News Detector
          </h1>
          
          <p className="text-xl text-white/90 mb-2">
            Powered by Advanced Machine Learning
          </p>
          
          <p className="text-white/70 max-w-2xl mx-auto">
            Analyze news articles and text content using state-of-the-art AI to determine authenticity.
            Our model is trained on thousands of verified articles to help you identify misinformation.
          </p>
        </div>
        
        {/* Main Content */}
        <AnalysisForm />
        
        {/* Footer */}
        <div className="mt-12 text-center text-white/60 text-sm">
          <p>Built with React, FastAPI, and PyTorch • Powered by Transformers (RoBERTa)</p>
          <p className="mt-2">
            This tool provides AI-based predictions. Always verify information from multiple sources.
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;
