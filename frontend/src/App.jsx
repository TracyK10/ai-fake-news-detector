import React from 'react';
import AnalysisForm from './components/AnalysisForm';
import './styles/index.css';

function App() {
  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-block p-4 bg-white/20 backdrop-blur-lg rounded-full mb-6 border-2 border-white/30">
            <svg className="w-16 h-16 text-gray-800" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          
          <h1 className="text-5xl font-bold text-gray-900 mb-4 drop-shadow-lg">
            AI Fake News Detector
          </h1>
          
          <p className="text-xl text-gray-800 mb-2 font-semibold">
            Powered by Advanced Machine Learning
          </p>
          
          <p className="text-gray-700 max-w-2xl mx-auto">
            Analyze news articles and text content using state-of-the-art AI to determine authenticity.
            This model is trained on thousands of verified articles to help you identify misinformation.
          </p>
        </div>
        
        {/* Main Content */}
        <AnalysisForm />
        
        {/* Footer */}
        <footer className="mt-16 py-8 border-t border-gray-700/20">
          <div className="text-center space-y-4">
            {/* Credits */}
            <p className="text-gray-800 text-sm font-medium">
              Made with ❤️ by{' '}
              <a 
                href="https://github.com/TracyK10" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-900 font-bold hover:text-orange-700 transition-colors duration-200 underline decoration-gray-700/40 hover:decoration-orange-700"
              >
                Tracy Karanja
              </a>
            </p>
            
            {/* Repository Link */}
            <div className="flex justify-center items-center gap-2">
              <a
                href="https://github.com/TracyK10/ai-fake-news-detector"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 bg-white/30 backdrop-blur-sm text-gray-900 font-semibold rounded-lg border border-gray-700/20 hover:bg-white/40 hover:border-gray-700/30 transition-all duration-200"
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                </svg>
                View Project Source
              </a>
            </div>
            
            {/* Tech Stack Note */}
            <p className="text-gray-700 text-xs">
              Built with React, FastAPI, and PyTorch • Powered by Transformers (RoBERTa)
            </p>
            
            <p className="text-gray-600 text-xs mt-2">
              This tool provides AI-based predictions. Always verify information from multiple sources.
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
