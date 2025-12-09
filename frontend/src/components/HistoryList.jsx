import React from 'react';

const HistoryList = ({ history, onSelect, onClear }) => {
  return (
    <div className="card mt-4">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-bold text-gray-800">Recent Analyses</h3>
        <button
          onClick={onClear}
          className="text-sm text-red-600 hover:text-red-800 font-semibold"
        >
          Clear All
        </button>
      </div>
      
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {history.map((item) => (
          <div
            key={item.id}
            onClick={() => onSelect(item)}
            className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 cursor-pointer transition-all duration-200 border border-gray-200"
          >
            <div className="flex justify-between items-start mb-1">
              <span className={`text-sm font-semibold ${
                item.result.label === 'Fake' ? 'text-red-600' : 'text-green-600'
              }`}>
                {item.result.label}
              </span>
              <span className="text-xs text-gray-500">
                {new Date(item.timestamp).toLocaleDateString()}
              </span>
            </div>
            <p className="text-sm text-gray-700 line-clamp-2">{item.text}</p>
            <div className="mt-1 text-xs text-gray-500">
              Confidence: {(item.result.confidence_score * 100).toFixed(1)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default HistoryList;
