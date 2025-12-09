import React from 'react';

const ResultDisplay = ({ result, onReset, onFeedback }) => {
  const isFake = result.label === 'Fake';
  const confidence = (result.confidence_score * 100).toFixed(1);
  
  return (
    <div className="card animate-fade-in">
      <div className="text-center mb-6">
        <div className={`inline-flex items-center justify-center w-20 h-20 rounded-full mb-4 ${
          isFake ? 'bg-red-100' : 'bg-green-100'
        }`}>
          {isFake ? (
            <svg className="w-10 h-10 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          ) : (
            <svg className="w-10 h-10 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          )}
        </div>
        
        <h2 className={`text-3xl font-bold mb-2 ${isFake ? 'text-red-600' : 'text-green-600'}`}>
          {result.label} News
        </h2>
        
        <p className="text-gray-600 mb-4">
          Confidence: <span className="font-semibold">{confidence}%</span>
        </p>
      </div>
      
      {/* Confidence Meter */}
      <div className="mb-6">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-green-600 font-semibold">Real</span>
          <span className="text-red-600 font-semibold">Fake</span>
        </div>
        <div className="relative h-4 bg-gray-200 rounded-full overflow-hidden">
          <div 
            className={`absolute left-0 top-0 h-full transition-all duration-500 ${
              isFake ? 'bg-gradient-to-r from-yellow-400 to-red-600' : 'bg-gradient-to-r from-green-400 to-blue-500'
            }`}
            style={{ width: `${confidence}%` }}
          />
        </div>
        <div className="flex justify-between text-xs mt-1 text-gray-500">
          <span>{(result.probabilities.real * 100).toFixed(1)}%</span>
          <span>{(result.probabilities.fake * 100).toFixed(1)}%</span>
        </div>
      </div>
      
      {/* Action Buttons */}
      <div className="flex gap-3">
        <button
          onClick={onReset}
          className="btn-primary flex-1"
        >
          Analyze Another
        </button>
        <button
          onClick={onFeedback}
          className="btn-secondary flex-1"
        >
          Report Incorrect
        </button>
      </div>
    </div>
  );
};

export default ResultDisplay;
