import React from 'react';

function ProgressBar({ progress, stage }) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-2">
        <span className="text-gray-400">{stage}</span>
        <span className="font-mono">{progress}%</span>
      </div>

      <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-white rounded-full transition-all duration-500 progress-animate"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}

export default ProgressBar;
