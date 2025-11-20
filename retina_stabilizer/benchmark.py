"""
Benchmarking module for comparing stabilization methods.

Measures computational performance metrics:
- Execution time (total and per-component)
- Memory usage (peak and average)
- GPU utilization
- Throughput metrics
"""

import gc
import time
import psutil
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import traceback

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class TimingMetrics:
    """Timing breakdown for each pipeline stage."""
    total: float
    preprocessing: float
    reference_selection: float
    vessel_segmentation: float
    motion_estimation: float
    trajectory_smoothing: float
    warping: float
    per_frame_avg: float


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    peak_cpu_mb: float
    avg_cpu_mb: float
    peak_gpu_mb: float
    avg_gpu_mb: float


@dataclass
class ThroughputMetrics:
    """Throughput and efficiency metrics."""
    fps: float
    realtime_factor: float
    pixels_per_second: float
    frames_per_gb_ram: float


@dataclass
class BenchmarkResult:
    """Complete benchmark results for a configuration."""
    method_name: str
    video_name: str
    n_frames: int
    resolution: tuple
    timing: TimingMetrics
    memory: MemoryMetrics
    throughput: ThroughputMetrics

    def to_dict(self) -> Dict:
        return {
            'method_name': self.method_name,
            'video_name': self.video_name,
            'n_frames': self.n_frames,
            'resolution': self.resolution,
            'timing': asdict(self.timing),
            'memory': asdict(self.memory),
            'throughput': asdict(self.throughput)
        }


class MemoryTracker:
    """Track memory usage during execution."""

    def __init__(self):
        self.cpu_samples = []
        self.gpu_samples = []
        self.process = psutil.Process()
        self._tracking = False

    def start(self):
        """Start memory tracking."""
        gc.collect()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        self.cpu_samples = []
        self.gpu_samples = []
        self._tracking = True
        self._sample()

    def sample(self):
        """Take a memory sample."""
        if self._tracking:
            self._sample()

    def _sample(self):
        """Internal sampling method."""
        # CPU memory
        cpu_mb = self.process.memory_info().rss / (1024 * 1024)
        self.cpu_samples.append(cpu_mb)

        # GPU memory
        if HAS_TORCH and torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            self.gpu_samples.append(gpu_mb)
        else:
            self.gpu_samples.append(0)

    def stop(self) -> MemoryMetrics:
        """Stop tracking and return metrics."""
        self._tracking = False
        self._sample()

        peak_gpu = 0
        if HAS_TORCH and torch.cuda.is_available():
            peak_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return MemoryMetrics(
            peak_cpu_mb=max(self.cpu_samples) if self.cpu_samples else 0,
            avg_cpu_mb=np.mean(self.cpu_samples) if self.cpu_samples else 0,
            peak_gpu_mb=peak_gpu,
            avg_gpu_mb=np.mean(self.gpu_samples) if self.gpu_samples else 0
        )


class PipelineBenchmarker:
    """Benchmark stabilization pipeline with detailed timing."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def benchmark(
        self,
        video_path: str,
        stabilizer,
        method_name: str = "default",
        video_fps: float = 30.0,
        warmup_runs: int = 0
    ) -> BenchmarkResult:
        """
        Benchmark a stabilization run with detailed metrics.

        Args:
            video_path: Path to input video
            stabilizer: RetinaStabilizer instance
            method_name: Name identifier for the method
            video_fps: Video frame rate
            warmup_runs: Number of warmup runs (for GPU)

        Returns:
            BenchmarkResult with all metrics
        """
        from retina_stabilizer.preprocessing import Preprocessor

        # Load video
        preprocessor = Preprocessor()
        frames, fps = preprocessor.load_video(video_path)
        if fps > 0:
            video_fps = fps

        n_frames = len(frames)
        resolution = (frames[0].shape[1], frames[0].shape[0])
        video_name = Path(video_path).name

        # Warmup runs
        for _ in range(warmup_runs):
            stabilizer.stabilize_frames(frames[:min(10, n_frames)], video_fps)

        # Initialize tracking
        memory_tracker = MemoryTracker()
        timings = {}

        # Start benchmark
        memory_tracker.start()
        total_start = time.perf_counter()

        # Preprocessing
        t0 = time.perf_counter()
        preprocessed = []
        for frame in frames:
            prep = preprocessor.preprocess_frame(frame)
            preprocessed.append(prep)
        memory_tracker.sample()
        timings['preprocessing'] = time.perf_counter() - t0

        # Reference selection
        t0 = time.perf_counter()
        ref_idx = stabilizer.ref_selector.select_reference(preprocessed)
        memory_tracker.sample()
        timings['reference_selection'] = time.perf_counter() - t0

        # Vessel segmentation
        t0 = time.perf_counter()
        if stabilizer.use_vessel_segmentation:
            vessel_weights = []
            for prep in preprocessed:
                weights = stabilizer.vessel_segmentor.get_confidence_weights(prep)
                vessel_weights.append(weights)
        else:
            vessel_weights = [None] * n_frames
        memory_tracker.sample()
        timings['vessel_segmentation'] = time.perf_counter() - t0

        # Frangi enhancement
        frangi_frames = []
        for prep in preprocessed:
            frangi = stabilizer.vessel_segmentor.create_frangi_vessel_map(prep)
            frangi_frames.append(frangi)

        # Motion estimation
        t0 = time.perf_counter()
        raw_transforms = stabilizer._estimate_motion_progressive(
            preprocessed, vessel_weights, frangi_frames, ref_idx
        )
        memory_tracker.sample()
        timings['motion_estimation'] = time.perf_counter() - t0

        # Trajectory smoothing
        t0 = time.perf_counter()
        smoothed_transforms = stabilizer.smoother.smooth(raw_transforms, video_fps)
        stabilizing_transforms = stabilizer.smoother.compute_stabilizing_transforms(
            raw_transforms, smoothed_transforms
        )
        memory_tracker.sample()
        timings['trajectory_smoothing'] = time.perf_counter() - t0

        # Warping
        t0 = time.perf_counter()
        stabilized_frames, crop_region = stabilizer.warper.stabilize_batch(
            frames, stabilizing_transforms,
            auto_crop=stabilizer.auto_crop,
            inpaint=stabilizer.inpaint_borders
        )
        memory_tracker.sample()
        timings['warping'] = time.perf_counter() - t0

        total_time = time.perf_counter() - total_start
        memory_metrics = memory_tracker.stop()

        # Compute metrics
        timing_metrics = TimingMetrics(
            total=total_time,
            preprocessing=timings['preprocessing'],
            reference_selection=timings['reference_selection'],
            vessel_segmentation=timings['vessel_segmentation'],
            motion_estimation=timings['motion_estimation'],
            trajectory_smoothing=timings['trajectory_smoothing'],
            warping=timings['warping'],
            per_frame_avg=total_time / n_frames
        )

        # Throughput
        video_duration = n_frames / video_fps
        pixels_total = n_frames * resolution[0] * resolution[1]

        throughput_metrics = ThroughputMetrics(
            fps=n_frames / total_time,
            realtime_factor=video_duration / total_time,
            pixels_per_second=pixels_total / total_time,
            frames_per_gb_ram=n_frames / (memory_metrics.peak_cpu_mb / 1024) if memory_metrics.peak_cpu_mb > 0 else 0
        )

        result = BenchmarkResult(
            method_name=method_name,
            video_name=video_name,
            n_frames=n_frames,
            resolution=resolution,
            timing=timing_metrics,
            memory=memory_metrics,
            throughput=throughput_metrics
        )

        self.results.append(result)
        return result

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate benchmark comparison report."""
        if not self.results:
            return "No benchmark results available."

        lines = [
            "=" * 90,
            "BENCHMARK COMPARISON REPORT",
            "=" * 90,
            ""
        ]

        for result in self.results:
            lines.extend([
                f"\nMethod: {result.method_name} | Video: {result.video_name}",
                f"Frames: {result.n_frames} | Resolution: {result.resolution[0]}x{result.resolution[1]}",
                "-" * 50,
                "",
                "TIMING BREAKDOWN:",
                f"  Total Time:          {result.timing.total:.2f}s",
                f"  Per-Frame Average:   {result.timing.per_frame_avg * 1000:.2f}ms",
                f"  Preprocessing:       {result.timing.preprocessing:.2f}s ({result.timing.preprocessing/result.timing.total*100:.1f}%)",
                f"  Reference Selection: {result.timing.reference_selection:.2f}s ({result.timing.reference_selection/result.timing.total*100:.1f}%)",
                f"  Vessel Segmentation: {result.timing.vessel_segmentation:.2f}s ({result.timing.vessel_segmentation/result.timing.total*100:.1f}%)",
                f"  Motion Estimation:   {result.timing.motion_estimation:.2f}s ({result.timing.motion_estimation/result.timing.total*100:.1f}%)",
                f"  Trajectory Smooth:   {result.timing.trajectory_smoothing:.2f}s ({result.timing.trajectory_smoothing/result.timing.total*100:.1f}%)",
                f"  Warping:             {result.timing.warping:.2f}s ({result.timing.warping/result.timing.total*100:.1f}%)",
                "",
                "MEMORY USAGE:",
                f"  Peak CPU Memory:     {result.memory.peak_cpu_mb:.1f} MB",
                f"  Avg CPU Memory:      {result.memory.avg_cpu_mb:.1f} MB",
                f"  Peak GPU Memory:     {result.memory.peak_gpu_mb:.1f} MB",
                f"  Avg GPU Memory:      {result.memory.avg_gpu_mb:.1f} MB",
                "",
                "THROUGHPUT:",
                f"  Processing Speed:    {result.throughput.fps:.1f} FPS",
                f"  Real-time Factor:    {result.throughput.realtime_factor:.2f}x",
                f"  Pixels/Second:       {result.throughput.pixels_per_second/1e6:.1f} MP/s",
                ""
            ])

        # Comparison table if multiple methods
        if len(self.results) > 1:
            lines.extend(self._generate_comparison_table())

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report

    def _generate_comparison_table(self) -> List[str]:
        """Generate comparison table across methods."""
        # Group by method
        methods = {}
        for r in self.results:
            if r.method_name not in methods:
                methods[r.method_name] = []
            methods[r.method_name].append(r)

        lines = [
            "",
            "=" * 90,
            "METHOD COMPARISON (averaged across videos)",
            "=" * 90,
            ""
        ]

        # Metrics to compare
        metrics = [
            ('Total Time (s)', lambda r: r.timing.total),
            ('Per-Frame (ms)', lambda r: r.timing.per_frame_avg * 1000),
            ('FPS', lambda r: r.throughput.fps),
            ('Real-time Factor', lambda r: r.throughput.realtime_factor),
            ('Peak CPU (MB)', lambda r: r.memory.peak_cpu_mb),
            ('Peak GPU (MB)', lambda r: r.memory.peak_gpu_mb),
        ]

        # Header
        method_names = list(methods.keys())
        header = f"{'Metric':<20}"
        for name in method_names:
            header += f" {name:>15}"
        lines.append(header)
        lines.append("-" * (20 + 16 * len(method_names)))

        # Rows
        for metric_name, metric_fn in metrics:
            row = f"{metric_name:<20}"
            for name in method_names:
                values = [metric_fn(r) for r in methods[name]]
                avg = np.mean(values)
                row += f" {avg:>15.2f}"
            lines.append(row)

        # Timing breakdown comparison
        lines.extend([
            "",
            "TIMING BREAKDOWN (%)",
            "-" * (20 + 16 * len(method_names))
        ])

        stages = [
            ('Preprocessing', lambda r: r.timing.preprocessing / r.timing.total * 100),
            ('Vessel Seg.', lambda r: r.timing.vessel_segmentation / r.timing.total * 100),
            ('Motion Est.', lambda r: r.timing.motion_estimation / r.timing.total * 100),
            ('Smoothing', lambda r: r.timing.trajectory_smoothing / r.timing.total * 100),
            ('Warping', lambda r: r.timing.warping / r.timing.total * 100),
        ]

        for stage_name, stage_fn in stages:
            row = f"{stage_name:<20}"
            for name in method_names:
                values = [stage_fn(r) for r in methods[name]]
                avg = np.mean(values)
                row += f" {avg:>15.1f}"
            lines.append(row)

        return lines

    def save_results_json(self, output_path: str):
        """Save results to JSON."""
        data = {
            'results': [r.to_dict() for r in self.results],
            'summary': self._compute_summary()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _compute_summary(self) -> Dict:
        """Compute summary statistics."""
        if not self.results:
            return {}

        # Group by method
        methods = {}
        for r in self.results:
            if r.method_name not in methods:
                methods[r.method_name] = []
            methods[r.method_name].append(r)

        summary = {}
        for method_name, results in methods.items():
            summary[method_name] = {
                'n_videos': len(results),
                'avg_fps': np.mean([r.throughput.fps for r in results]),
                'avg_realtime_factor': np.mean([r.throughput.realtime_factor for r in results]),
                'avg_peak_cpu_mb': np.mean([r.memory.peak_cpu_mb for r in results]),
                'avg_peak_gpu_mb': np.mean([r.memory.peak_gpu_mb for r in results]),
                'avg_total_time': np.mean([r.timing.total for r in results]),
            }

        return summary

    def clear_results(self):
        """Clear stored results."""
        self.results = []


def run_benchmark_comparison(
    video_paths: List[str],
    output_dir: str,
    device: str = 'cuda',
    warmup: int = 1
):
    """
    Run benchmark comparison between methods.

    Args:
        video_paths: List of video file paths
        output_dir: Directory for output files
        device: Device to use ('cuda' or 'cpu')
        warmup: Number of warmup runs
    """
    from retina_stabilizer.pipeline import RetinaStabilizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define methods to compare
    configs = {
        'frangi_classical': {
            'use_vessel_segmentation': False,
        },
        'neural_unet': {
            'use_vessel_segmentation': True,
        },
    }

    benchmarker = PipelineBenchmarker()

    for config_name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking: {config_name}")
        print('='*60)

        stabilizer = RetinaStabilizer(
            device=device,
            use_vessel_segmentation=config['use_vessel_segmentation'],
            auto_crop=True,
            inpaint_borders=True
        )

        for video_path in video_paths:
            print(f"\n  Processing: {Path(video_path).name}")
            try:
                result = benchmarker.benchmark(
                    video_path,
                    stabilizer,
                    method_name=config_name,
                    warmup_runs=warmup
                )
                print(f"    FPS: {result.throughput.fps:.1f}")
                print(f"    Peak Memory: {result.memory.peak_cpu_mb:.0f} MB CPU, {result.memory.peak_gpu_mb:.0f} MB GPU")
            except Exception as e:
                print(f"    Error: {e}")
                traceback.print_exc()

    # Generate reports
    report = benchmarker.generate_report(str(output_dir / 'benchmark_report.txt'))
    print("\n" + report)

    benchmarker.save_results_json(str(output_dir / 'benchmark_results.json'))
    print(f"\nResults saved to {output_dir}")
