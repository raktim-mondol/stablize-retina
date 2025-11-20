import React from 'react';

function ModelSelector({ models, selected, setSelected }) {
  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4">Select Model</h2>

      <div className="space-y-3">
        {models.map((model) => (
          <label
            key={model.id}
            className={`
              block p-4 rounded-lg border cursor-pointer
              transition-colors duration-200
              ${selected === model.id
                ? 'border-white bg-dark-700'
                : 'border-dark-600 hover:border-gray-400'
              }
            `}
          >
            <div className="flex items-start">
              <input
                type="radio"
                name="model"
                value={model.id}
                checked={selected === model.id}
                onChange={(e) => setSelected(e.target.value)}
                className="mt-1 mr-3"
              />
              <div className="flex-1">
                <div className="flex items-center">
                  <span className="font-medium">{model.name}</span>
                  {model.recommended && (
                    <span className="ml-2 text-xs bg-white text-black px-2 py-0.5 rounded">
                      Recommended
                    </span>
                  )}
                </div>
                <p className="text-sm text-gray-400 mt-1">
                  {model.description}
                </p>
              </div>
            </div>
          </label>
        ))}
      </div>
    </div>
  );
}

export default ModelSelector;
