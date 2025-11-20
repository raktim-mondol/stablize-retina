import React from 'react';

function MetricsPanel({ metrics }) {
  if (!metrics) return null;

  const MetricItem = ({ label, value, unit = '', good = true }) => (
    <div className="bg-dark-700 rounded-lg p-4">
      <div className="metric-label mb-1">{label}</div>
      <div className={`metric-value ${good ? 'text-white' : 'text-gray-400'}`}>
        {value}
        {unit && <span className="text-sm text-gray-400 ml-1">{unit}</span>}
      </div>
    </div>
  );

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4">Performance Metrics</h2>

      {/* Stability Metrics */}
      <div className="mb-6">
        <h3 className="text-sm text-gray-400 uppercase tracking-wider mb-3">
          Stability
        </h3>
        <div className="grid grid-cols-2 gap-3">
          <MetricItem
            label="ITF Score"
            value={metrics.stability.itf.toFixed(3)}
          />
          <MetricItem
            label="Residual Motion"
            value={metrics.stability.residual_motion.toFixed(2)}
            unit="px"
          />
          <MetricItem
            label="Jitter Reduction X"
            value={metrics.stability.jitter_reduction_x.toFixed(1)}
            unit="%"
          />
          <MetricItem
            label="Jitter Reduction Y"
            value={metrics.stability.jitter_reduction_y.toFixed(1)}
            unit="%"
          />
        </div>
      </div>

      {/* Quality Metrics */}
      <div className="mb-6">
        <h3 className="text-sm text-gray-400 uppercase tracking-wider mb-3">
          Quality
        </h3>
        <div className="grid grid-cols-3 gap-3">
          <MetricItem
            label="PSNR"
            value={metrics.quality.psnr.toFixed(1)}
            unit="dB"
          />
          <MetricItem
            label="SSIM"
            value={metrics.quality.ssim.toFixed(3)}
          />
          <MetricItem
            label="Sharpness"
            value={metrics.quality.sharpness_retention.toFixed(1)}
            unit="%"
          />
        </div>
      </div>

      {/* Efficiency Metrics */}
      <div className="mb-6">
        <h3 className="text-sm text-gray-400 uppercase tracking-wider mb-3">
          Efficiency
        </h3>
        <div className="grid grid-cols-3 gap-3">
          <MetricItem
            label="FOV Retention"
            value={metrics.efficiency.fov_retention.toFixed(1)}
            unit="%"
          />
          <MetricItem
            label="Processing"
            value={metrics.efficiency.processing_fps.toFixed(1)}
            unit="FPS"
          />
          <MetricItem
            label="Realtime Factor"
            value={metrics.efficiency.realtime_factor.toFixed(2)}
            unit="x"
          />
        </div>
      </div>

      {/* Temporal Metrics */}
      <div>
        <h3 className="text-sm text-gray-400 uppercase tracking-wider mb-3">
          Temporal
        </h3>
        <div className="grid grid-cols-3 gap-3">
          <MetricItem
            label="Smoothness"
            value={metrics.temporal.smoothness.toFixed(3)}
          />
          <MetricItem
            label="Velocity Consistency"
            value={metrics.temporal.velocity_consistency.toFixed(3)}
          />
          <MetricItem
            label="Tremor Reduction"
            value={metrics.temporal.tremor_reduction.toFixed(1)}
            unit="%"
          />
        </div>
      </div>
    </div>
  );
}

export default MetricsPanel;
