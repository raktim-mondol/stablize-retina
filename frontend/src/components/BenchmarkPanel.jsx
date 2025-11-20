import React from 'react';

function BenchmarkPanel({ benchmark }) {
  if (!benchmark) return null;

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4">Benchmark Scores</h2>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 gap-3 mb-6">
        <div className="bg-dark-700 rounded-lg p-4">
          <div className="metric-label mb-1">Total Time</div>
          <div className="metric-value">
            {benchmark.total_time.toFixed(1)}
            <span className="text-sm text-gray-400 ml-1">s</span>
          </div>
        </div>
        <div className="bg-dark-700 rounded-lg p-4">
          <div className="metric-label mb-1">Average FPS</div>
          <div className="metric-value">
            {benchmark.avg_fps.toFixed(1)}
          </div>
        </div>
        <div className="bg-dark-700 rounded-lg p-4">
          <div className="metric-label mb-1">Frames Processed</div>
          <div className="metric-value">
            {benchmark.frames_processed}
          </div>
        </div>
        <div className="bg-dark-700 rounded-lg p-4">
          <div className="metric-label mb-1">Device</div>
          <div className="metric-value text-lg">
            {benchmark.device.toUpperCase()}
          </div>
        </div>
      </div>

      {/* Model Info */}
      <div className="bg-dark-700 rounded-lg p-4 mb-6">
        <div className="metric-label mb-1">Model Used</div>
        <div className="text-lg font-medium capitalize">
          {benchmark.model_used === 'neural'
            ? 'Neural (RAFT + U-Net)'
            : 'Classical (Frangi)'
          }
        </div>
      </div>

      {/* Stage Breakdown (if available) */}
      {benchmark.preprocessing_time !== undefined && (
        <div>
          <h3 className="text-sm text-gray-400 uppercase tracking-wider mb-3">
            Stage Breakdown
          </h3>
          <div className="space-y-2">
            <StageBar
              label="Preprocessing"
              time={benchmark.preprocessing_time}
              total={benchmark.total_time}
            />
            <StageBar
              label="Motion Estimation"
              time={benchmark.motion_estimation_time}
              total={benchmark.total_time}
            />
            <StageBar
              label="Smoothing"
              time={benchmark.smoothing_time}
              total={benchmark.total_time}
            />
            <StageBar
              label="Warping"
              time={benchmark.warping_time}
              total={benchmark.total_time}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function StageBar({ label, time, total }) {
  const percentage = total > 0 ? (time / total) * 100 : 0;

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-400">{label}</span>
        <span className="font-mono">{time.toFixed(2)}s</span>
      </div>
      <div className="h-1.5 bg-dark-600 rounded-full overflow-hidden">
        <div
          className="h-full bg-gray-400 rounded-full"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

export default BenchmarkPanel;
