"""
Comprehensive evaluation metrics for retinal video stabilization.

This module provides various metrics to evaluate stabilization quality:
- Stability metrics (ITF, residual motion, jitter reduction)
- Quality metrics (PSNR, SSIM, sharpness)
- Efficiency metrics (FOV retention, cropping ratio)
- Temporal consistency metrics
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import time
import json
from dataclasses import dataclass, asdict
from scipy import signal
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


@dataclass
class StabilityMetrics:
    """Stability-related metrics."""
    itf: float  # Inter-frame Transformation Fidelity
    residual_motion: float  # Median residual motion in pixels
    motion_reduction: float  # Percentage reduction in motion
    jitter_x: float  # Jitter in x direction
    jitter_y: float  # Jitter in y direction
    jitter_rotation: float  # Rotational jitter
    jitter_scale: float  # Scale jitter


@dataclass
class QualityMetrics:
    """Image quality metrics."""
    avg_psnr: float  # Average PSNR compared to original
    avg_ssim: float  # Average SSIM compared to original
    sharpness_original: float  # Average sharpness of original
    sharpness_stabilized: float  # Average sharpness of stabilized
    sharpness_retention: float  # Percentage of sharpness retained


@dataclass
class EfficiencyMetrics:
    """Efficiency and coverage metrics."""
    fov_retention: float  # Field of view retention percentage
    crop_ratio: float  # Cropping ratio
    processing_time: float  # Total processing time in seconds
    fps_processed: float  # Frames processed per second
    realtime_factor: float  # Speed relative to real-time


@dataclass
class TemporalMetrics:
    """Temporal consistency metrics."""
    temporal_smoothness: float  # Smoothness of frame transitions
    velocity_consistency: float  # Consistency of velocity profile
    acceleration_consistency: float  # Consistency of acceleration
    tremor_reduction: float  # High-frequency tremor reduction


@dataclass
class EvaluationResult:
    """Complete evaluation results for a video."""
    video_name: str
    n_frames: int
    duration: float
    stability: StabilityMetrics
    quality: QualityMetrics
    efficiency: EfficiencyMetrics
    temporal: TemporalMetrics

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'video_name': self.video_name,
            'n_frames': self.n_frames,
            'duration': self.duration,
            'stability': asdict(self.stability),
            'quality': asdict(self.quality),
            'efficiency': asdict(self.efficiency),
            'temporal': asdict(self.temporal)
        }


class VideoEvaluator:
    """Evaluates stabilization performance with comprehensive metrics."""

    def __init__(self):
        self.results: List[EvaluationResult] = []

    def evaluate(
        self,
        original_frames: List[np.ndarray],
        stabilized_frames: List[np.ndarray],
        pipeline_metrics: Dict[str, Any],
        video_name: str = "video",
        fps: float = 30.0
    ) -> EvaluationResult:
        """
        Evaluate stabilization performance.

        Args:
            original_frames: List of original video frames
            stabilized_frames: List of stabilized video frames
            pipeline_metrics: Metrics from the stabilization pipeline
            video_name: Name identifier for the video
            fps: Frame rate of the video

        Returns:
            EvaluationResult with all computed metrics
        """
        n_frames = len(stabilized_frames)
        duration = n_frames / fps

        # Compute all metric categories
        stability = self._compute_stability_metrics(
            original_frames, stabilized_frames, pipeline_metrics
        )
        quality = self._compute_quality_metrics(
            original_frames, stabilized_frames
        )
        efficiency = self._compute_efficiency_metrics(
            original_frames, stabilized_frames, pipeline_metrics, fps
        )
        temporal = self._compute_temporal_metrics(
            original_frames, stabilized_frames, fps
        )

        result = EvaluationResult(
            video_name=video_name,
            n_frames=n_frames,
            duration=duration,
            stability=stability,
            quality=quality,
            efficiency=efficiency,
            temporal=temporal
        )

        self.results.append(result)
        return result

    def _compute_stability_metrics(
        self,
        original_frames: List[np.ndarray],
        stabilized_frames: List[np.ndarray],
        pipeline_metrics: Dict[str, Any]
    ) -> StabilityMetrics:
        """Compute stability-related metrics."""
        # ITF - Inter-frame Transformation Fidelity
        itf_original = self._compute_itf(original_frames)
        itf_stabilized = self._compute_itf(stabilized_frames)
        itf = itf_stabilized / max(itf_original, 1e-6)

        # Residual motion
        residual_motion = self._compute_residual_motion(stabilized_frames)
        original_motion = self._compute_residual_motion(original_frames)
        motion_reduction = 1.0 - (residual_motion / max(original_motion, 1e-6))
        motion_reduction = max(0, min(1, motion_reduction)) * 100

        # Extract jitter from pipeline metrics
        smoothing = pipeline_metrics.get('smoothing', {})
        jitter_reduction = smoothing.get('jitter_reduction', [0, 0, 0, 0])

        return StabilityMetrics(
            itf=itf,
            residual_motion=residual_motion,
            motion_reduction=motion_reduction,
            jitter_x=jitter_reduction[0] if len(jitter_reduction) > 0 else 0,
            jitter_y=jitter_reduction[1] if len(jitter_reduction) > 1 else 0,
            jitter_rotation=jitter_reduction[2] if len(jitter_reduction) > 2 else 0,
            jitter_scale=jitter_reduction[3] if len(jitter_reduction) > 3 else 0
        )

    def _compute_itf(self, frames: List[np.ndarray]) -> float:
        """
        Compute Inter-frame Transformation Fidelity.
        Lower values indicate more stable video.
        """
        if len(frames) < 2:
            return 0.0

        total_motion = 0.0
        valid_pairs = 0

        for i in range(len(frames) - 1):
            # Convert to grayscale if needed
            if len(frames[i].shape) == 3:
                gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = frames[i], frames[i + 1]

            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # Compute flow magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            total_motion += np.mean(magnitude)
            valid_pairs += 1

        return total_motion / max(valid_pairs, 1)

    def _compute_residual_motion(
        self,
        frames: List[np.ndarray],
        window_size: int = 5
    ) -> float:
        """Compute median residual motion in stabilized video."""
        if len(frames) < 2:
            return 0.0

        magnitudes = []

        for i in range(len(frames) - 1):
            if len(frames[i].shape) == 3:
                gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = frames[i], frames[i + 1]

            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            magnitudes.append(np.median(magnitude))

        return np.mean(magnitudes)

    def _compute_quality_metrics(
        self,
        original_frames: List[np.ndarray],
        stabilized_frames: List[np.ndarray]
    ) -> QualityMetrics:
        """Compute image quality metrics."""
        # Handle different frame counts (due to cropping)
        n_compare = min(len(original_frames), len(stabilized_frames))

        psnr_values = []
        ssim_values = []
        sharpness_original = []
        sharpness_stabilized = []

        for i in range(n_compare):
            orig = original_frames[i]
            stab = stabilized_frames[i]

            # Resize if dimensions differ
            if orig.shape[:2] != stab.shape[:2]:
                stab_resized = cv2.resize(stab, (orig.shape[1], orig.shape[0]))
            else:
                stab_resized = stab

            # Convert to grayscale for metrics
            if len(orig.shape) == 3:
                orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
                stab_gray = cv2.cvtColor(stab_resized, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = orig
                stab_gray = stab_resized

            # PSNR
            try:
                p = psnr(orig_gray, stab_gray)
                if not np.isinf(p):
                    psnr_values.append(p)
            except Exception:
                pass

            # SSIM
            try:
                s = ssim(orig_gray, stab_gray)
                ssim_values.append(s)
            except Exception:
                pass

            # Sharpness (Laplacian variance)
            sharp_orig = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
            sharp_stab = cv2.Laplacian(stab_gray, cv2.CV_64F).var()
            sharpness_original.append(sharp_orig)
            sharpness_stabilized.append(sharp_stab)

        avg_sharp_orig = np.mean(sharpness_original) if sharpness_original else 0
        avg_sharp_stab = np.mean(sharpness_stabilized) if sharpness_stabilized else 0
        sharp_retention = (avg_sharp_stab / max(avg_sharp_orig, 1e-6)) * 100

        return QualityMetrics(
            avg_psnr=np.mean(psnr_values) if psnr_values else 0,
            avg_ssim=np.mean(ssim_values) if ssim_values else 0,
            sharpness_original=avg_sharp_orig,
            sharpness_stabilized=avg_sharp_stab,
            sharpness_retention=min(sharp_retention, 100)
        )

    def _compute_efficiency_metrics(
        self,
        original_frames: List[np.ndarray],
        stabilized_frames: List[np.ndarray],
        pipeline_metrics: Dict[str, Any],
        fps: float
    ) -> EfficiencyMetrics:
        """Compute efficiency and coverage metrics."""
        # FOV retention
        orig_area = original_frames[0].shape[0] * original_frames[0].shape[1]
        stab_area = stabilized_frames[0].shape[0] * stabilized_frames[0].shape[1]
        fov_retention = pipeline_metrics.get('fov_retention', stab_area / orig_area)

        # Crop ratio
        crop_ratio = stab_area / orig_area

        # Timing
        processing_time = pipeline_metrics.get('processing_time', 0)
        n_frames = len(stabilized_frames)
        fps_processed = n_frames / max(processing_time, 1e-6)

        # Realtime factor
        video_duration = n_frames / fps
        realtime_factor = video_duration / max(processing_time, 1e-6)

        return EfficiencyMetrics(
            fov_retention=fov_retention,
            crop_ratio=crop_ratio,
            processing_time=processing_time,
            fps_processed=fps_processed,
            realtime_factor=realtime_factor
        )

    def _compute_temporal_metrics(
        self,
        original_frames: List[np.ndarray],
        stabilized_frames: List[np.ndarray],
        fps: float
    ) -> TemporalMetrics:
        """Compute temporal consistency metrics."""
        # Compute motion trajectories
        orig_motion = self._extract_motion_trajectory(original_frames)
        stab_motion = self._extract_motion_trajectory(stabilized_frames)

        # Temporal smoothness (inverse of velocity variance)
        if len(stab_motion) > 1:
            velocities = np.diff(stab_motion, axis=0)
            temporal_smoothness = 1.0 / (np.var(velocities) + 1e-6)
            temporal_smoothness = min(temporal_smoothness, 100)  # Cap for readability
        else:
            temporal_smoothness = 100.0

        # Velocity consistency
        if len(stab_motion) > 2:
            velocities = np.diff(stab_motion, axis=0)
            velocity_std = np.std(velocities)
            velocity_consistency = 1.0 / (velocity_std + 1e-6)
            velocity_consistency = min(velocity_consistency, 100)
        else:
            velocity_consistency = 100.0

        # Acceleration consistency
        if len(stab_motion) > 3:
            velocities = np.diff(stab_motion, axis=0)
            accelerations = np.diff(velocities, axis=0)
            accel_std = np.std(accelerations)
            acceleration_consistency = 1.0 / (accel_std + 1e-6)
            acceleration_consistency = min(acceleration_consistency, 100)
        else:
            acceleration_consistency = 100.0

        # Tremor reduction (high-frequency content)
        tremor_reduction = self._compute_tremor_reduction(orig_motion, stab_motion, fps)

        return TemporalMetrics(
            temporal_smoothness=temporal_smoothness,
            velocity_consistency=velocity_consistency,
            acceleration_consistency=acceleration_consistency,
            tremor_reduction=tremor_reduction
        )

    def _extract_motion_trajectory(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract motion trajectory from frames."""
        if len(frames) < 2:
            return np.array([[0, 0]])

        trajectory = [[0, 0]]  # Start at origin
        cumulative_x, cumulative_y = 0, 0

        for i in range(len(frames) - 1):
            if len(frames[i].shape) == 3:
                gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = frames[i], frames[i + 1]

            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # Get mean displacement
            dx = np.mean(flow[..., 0])
            dy = np.mean(flow[..., 1])

            cumulative_x += dx
            cumulative_y += dy
            trajectory.append([cumulative_x, cumulative_y])

        return np.array(trajectory)

    def _compute_tremor_reduction(
        self,
        orig_motion: np.ndarray,
        stab_motion: np.ndarray,
        fps: float
    ) -> float:
        """Compute reduction in high-frequency tremor (8-12 Hz range)."""
        if len(orig_motion) < 10 or len(stab_motion) < 10:
            return 0.0

        # Compute power spectrum for original
        orig_x = orig_motion[:, 0]
        orig_freq = np.fft.fftfreq(len(orig_x), 1/fps)
        orig_fft = np.abs(np.fft.fft(orig_x))

        # Compute power spectrum for stabilized
        stab_x = stab_motion[:, 0]
        # Pad or truncate to match length
        if len(stab_x) != len(orig_x):
            min_len = min(len(stab_x), len(orig_x))
            stab_x = stab_x[:min_len]
            orig_x = orig_x[:min_len]
            orig_freq = np.fft.fftfreq(min_len, 1/fps)
            orig_fft = np.abs(np.fft.fft(orig_x))

        stab_fft = np.abs(np.fft.fft(stab_x))

        # Focus on tremor frequency range (8-12 Hz)
        tremor_mask = (np.abs(orig_freq) >= 8) & (np.abs(orig_freq) <= 12)

        orig_tremor_power = np.sum(orig_fft[tremor_mask])
        stab_tremor_power = np.sum(stab_fft[tremor_mask])

        if orig_tremor_power > 1e-6:
            reduction = (1.0 - stab_tremor_power / orig_tremor_power) * 100
            return max(0, min(100, reduction))

        return 0.0

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate a comprehensive evaluation report."""
        if not self.results:
            return "No evaluation results available."

        report_lines = [
            "=" * 80,
            "RETINAL VIDEO STABILIZATION EVALUATION REPORT",
            "=" * 80,
            ""
        ]

        for result in self.results:
            report_lines.extend([
                f"\nVideo: {result.video_name}",
                f"Frames: {result.n_frames} | Duration: {result.duration:.2f}s",
                "-" * 40,
                "",
                "STABILITY METRICS:",
                f"  ITF Score: {result.stability.itf:.4f}",
                f"  Residual Motion: {result.stability.residual_motion:.3f} px",
                f"  Motion Reduction: {result.stability.motion_reduction:.1f}%",
                f"  Jitter Reduction (X/Y/Rot/Scale): "
                f"{result.stability.jitter_x:.1f}% / {result.stability.jitter_y:.1f}% / "
                f"{result.stability.jitter_rotation:.1f}% / {result.stability.jitter_scale:.1f}%",
                "",
                "QUALITY METRICS:",
                f"  Average PSNR: {result.quality.avg_psnr:.2f} dB",
                f"  Average SSIM: {result.quality.avg_ssim:.4f}",
                f"  Sharpness Retention: {result.quality.sharpness_retention:.1f}%",
                "",
                "EFFICIENCY METRICS:",
                f"  FOV Retention: {result.efficiency.fov_retention * 100:.1f}%",
                f"  Processing Speed: {result.efficiency.fps_processed:.1f} FPS",
                f"  Real-time Factor: {result.efficiency.realtime_factor:.2f}x",
                "",
                "TEMPORAL METRICS:",
                f"  Temporal Smoothness: {result.temporal.temporal_smoothness:.2f}",
                f"  Velocity Consistency: {result.temporal.velocity_consistency:.2f}",
                f"  Tremor Reduction: {result.temporal.tremor_reduction:.1f}%",
                ""
            ])

        # Compute aggregate statistics if multiple videos
        if len(self.results) > 1:
            report_lines.extend([
                "",
                "=" * 80,
                "AGGREGATE STATISTICS",
                "=" * 80,
                ""
            ])

            # Compute means
            metrics = {
                'ITF': [r.stability.itf for r in self.results],
                'Residual Motion (px)': [r.stability.residual_motion for r in self.results],
                'Motion Reduction (%)': [r.stability.motion_reduction for r in self.results],
                'PSNR (dB)': [r.quality.avg_psnr for r in self.results],
                'SSIM': [r.quality.avg_ssim for r in self.results],
                'FOV Retention (%)': [r.efficiency.fov_retention * 100 for r in self.results],
                'Processing Speed (FPS)': [r.efficiency.fps_processed for r in self.results],
                'Tremor Reduction (%)': [r.temporal.tremor_reduction for r in self.results],
            }

            report_lines.append(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
            report_lines.append("-" * 65)

            for name, values in metrics.items():
                mean = np.mean(values)
                std = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                report_lines.append(
                    f"{name:<25} {mean:>10.2f} {std:>10.2f} {min_val:>10.2f} {max_val:>10.2f}"
                )

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report

    def save_results_json(self, output_path: str):
        """Save all results to JSON file."""
        data = {
            'results': [r.to_dict() for r in self.results],
            'summary': self._compute_summary() if self.results else {}
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _compute_summary(self) -> Dict:
        """Compute summary statistics."""
        if not self.results:
            return {}

        return {
            'n_videos': len(self.results),
            'total_frames': sum(r.n_frames for r in self.results),
            'total_duration': sum(r.duration for r in self.results),
            'mean_motion_reduction': np.mean([r.stability.motion_reduction for r in self.results]),
            'mean_psnr': np.mean([r.quality.avg_psnr for r in self.results]),
            'mean_ssim': np.mean([r.quality.avg_ssim for r in self.results]),
            'mean_fov_retention': np.mean([r.efficiency.fov_retention for r in self.results]),
            'mean_realtime_factor': np.mean([r.efficiency.realtime_factor for r in self.results]),
        }

    def clear_results(self):
        """Clear all stored results."""
        self.results = []


def evaluate_stabilization(
    original_frames: List[np.ndarray],
    stabilized_frames: List[np.ndarray],
    pipeline_metrics: Dict[str, Any],
    video_name: str = "video",
    fps: float = 30.0
) -> EvaluationResult:
    """
    Convenience function for single video evaluation.

    Args:
        original_frames: Original video frames
        stabilized_frames: Stabilized video frames
        pipeline_metrics: Metrics from pipeline
        video_name: Name of the video
        fps: Frame rate

    Returns:
        EvaluationResult with all metrics
    """
    evaluator = VideoEvaluator()
    return evaluator.evaluate(
        original_frames, stabilized_frames, pipeline_metrics, video_name, fps
    )
