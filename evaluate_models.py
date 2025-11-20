#!/usr/bin/env python3
"""
Evaluate retinal video stabilization models on test videos.

Usage:
    python evaluate_models.py --input_dir /path/to/videos --output_dir results
    python evaluate_models.py --input_dir videos --compare_methods
    python evaluate_models.py --video single_video.mp4 --output_dir results
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
import json
import numpy as np
import cv2

from retina_stabilizer.pipeline import RetinaStabilizer
from retina_stabilizer.preprocessing import Preprocessor
from retina_stabilizer.evaluation import VideoEvaluator, EvaluationResult
from retina_stabilizer.benchmark import PipelineBenchmarker, run_benchmark_comparison


def find_videos(input_path: Path) -> List[Path]:
    """Find all video files in directory or return single video."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    if input_path.is_file():
        return [input_path]

    videos = []
    for ext in video_extensions:
        videos.extend(input_path.glob(f'*{ext}'))
        videos.extend(input_path.glob(f'*{ext.upper()}'))

    return sorted(videos)


def load_video_frames(video_path: Path) -> tuple:
    """Load video and return frames and fps."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


def evaluate_single_video(
    video_path: Path,
    stabilizer: RetinaStabilizer,
    evaluator: VideoEvaluator,
    output_dir: Optional[Path] = None,
    save_output: bool = False
) -> EvaluationResult:
    """Evaluate a single video."""
    print(f"\nProcessing: {video_path.name}")

    # Load original frames
    original_frames, fps = load_video_frames(video_path)
    print(f"  Loaded {len(original_frames)} frames at {fps:.1f} FPS")

    # Determine output path
    output_path = None
    if save_output and output_dir:
        output_path = str(output_dir / f"stabilized_{video_path.name}")

    # Run stabilization
    start_time = time.time()
    stabilized_frames, metrics = stabilizer.stabilize(
        str(video_path),
        output_path
    )

    # Evaluate
    result = evaluator.evaluate(
        original_frames=original_frames,
        stabilized_frames=stabilized_frames,
        pipeline_metrics=metrics,
        video_name=video_path.name,
        fps=fps
    )

    # Print quick summary
    print(f"  Motion Reduction: {result.stability.motion_reduction:.1f}%")
    print(f"  FOV Retention: {result.efficiency.fov_retention * 100:.1f}%")
    print(f"  Processing: {result.efficiency.realtime_factor:.2f}x real-time")

    return result


def compare_methods(
    video_paths: List[Path],
    output_dir: Path,
    device: str = 'cuda'
):
    """Compare different stabilization methods/configurations."""

    # Define configurations to compare
    configs = {
        'frangi_only': {
            'use_vessel_segmentation': False,
            'description': 'Frangi filter (classical method)'
        },
        'neural_unet': {
            'use_vessel_segmentation': True,
            'vessel_model_path': None,  # Uses default/random weights
            'description': 'U-Net vessel segmentation'
        },
    }

    all_results = {}

    for config_name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Testing configuration: {config_name}")
        print(f"Description: {config['description']}")
        print('='*60)

        # Create stabilizer with this config
        stabilizer = RetinaStabilizer(
            device=device,
            use_vessel_segmentation=config.get('use_vessel_segmentation', True),
            vessel_model_path=config.get('vessel_model_path'),
            auto_crop=True,
            inpaint_borders=True
        )

        evaluator = VideoEvaluator()

        # Evaluate all videos
        for video_path in video_paths:
            try:
                evaluate_single_video(
                    video_path,
                    stabilizer,
                    evaluator,
                    output_dir / config_name,
                    save_output=False
                )
            except Exception as e:
                print(f"  Error processing {video_path.name}: {e}")

        # Save results for this configuration
        all_results[config_name] = {
            'config': config,
            'results': [r.to_dict() for r in evaluator.results]
        }

        # Generate report
        report = evaluator.generate_report()
        print(report)

        # Save config results
        config_output = output_dir / config_name
        config_output.mkdir(parents=True, exist_ok=True)
        evaluator.save_results_json(str(config_output / 'results.json'))

        with open(config_output / 'report.txt', 'w') as f:
            f.write(report)

    # Generate comparison summary
    generate_comparison_summary(all_results, output_dir)


def generate_comparison_summary(all_results: Dict, output_dir: Path):
    """Generate summary comparing all methods."""
    summary_lines = [
        "\n" + "="*80,
        "METHOD COMPARISON SUMMARY",
        "="*80,
        ""
    ]

    # Compute mean metrics for each configuration
    metrics_comparison = {}

    for config_name, data in all_results.items():
        results = data['results']
        if not results:
            continue

        metrics_comparison[config_name] = {
            'Motion Reduction (%)': np.mean([r['stability']['motion_reduction'] for r in results]),
            'Residual Motion (px)': np.mean([r['stability']['residual_motion'] for r in results]),
            'PSNR (dB)': np.mean([r['quality']['avg_psnr'] for r in results]),
            'SSIM': np.mean([r['quality']['avg_ssim'] for r in results]),
            'FOV Retention (%)': np.mean([r['efficiency']['fov_retention'] * 100 for r in results]),
            'Speed (FPS)': np.mean([r['efficiency']['fps_processed'] for r in results]),
        }

    if not metrics_comparison:
        print("No results to compare")
        return

    # Print comparison table
    metric_names = list(next(iter(metrics_comparison.values())).keys())
    config_names = list(metrics_comparison.keys())

    # Header
    header = f"{'Metric':<25}"
    for name in config_names:
        header += f" {name:>15}"
    summary_lines.append(header)
    summary_lines.append("-" * (25 + 16 * len(config_names)))

    # Rows
    for metric in metric_names:
        row = f"{metric:<25}"
        for config_name in config_names:
            value = metrics_comparison[config_name][metric]
            row += f" {value:>15.2f}"
        summary_lines.append(row)

    summary = "\n".join(summary_lines)
    print(summary)

    # Save comparison
    with open(output_dir / 'comparison_summary.txt', 'w') as f:
        f.write(summary)

    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(metrics_comparison, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate retinal video stabilization models'
    )
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        help='Directory containing test videos'
    )
    parser.add_argument(
        '--video', '-v',
        type=str,
        help='Single video file to evaluate'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='evaluation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )
    parser.add_argument(
        '--compare_methods',
        action='store_true',
        help='Compare different stabilization methods'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run computational benchmark (timing, memory, throughput)'
    )
    parser.add_argument(
        '--save_videos',
        action='store_true',
        help='Save stabilized videos to output directory'
    )
    parser.add_argument(
        '--use_neural',
        action='store_true',
        default=True,
        help='Use neural network vessel segmentation'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to vessel segmentation model weights'
    )

    args = parser.parse_args()

    # Determine input
    if args.video:
        input_path = Path(args.video)
    elif args.input_dir:
        input_path = Path(args.input_dir)
    else:
        print("Error: Please specify --input_dir or --video")
        sys.exit(1)

    # Find videos
    video_paths = find_videos(input_path)
    if not video_paths:
        print(f"No video files found in {input_path}")
        sys.exit(1)

    print(f"Found {len(video_paths)} video(s) to evaluate")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    if args.benchmark:
        # Run computational benchmark
        print("Running computational benchmark...")
        run_benchmark_comparison(
            [str(p) for p in video_paths],
            str(output_dir),
            device=args.device,
            warmup=1
        )
    elif args.compare_methods:
        compare_methods(video_paths, output_dir, args.device)
    else:
        # Single configuration evaluation
        stabilizer = RetinaStabilizer(
            device=args.device,
            use_vessel_segmentation=args.use_neural,
            vessel_model_path=args.model_path,
            auto_crop=True,
            inpaint_borders=True
        )

        evaluator = VideoEvaluator()

        for video_path in video_paths:
            try:
                evaluate_single_video(
                    video_path,
                    stabilizer,
                    evaluator,
                    output_dir,
                    save_output=args.save_videos
                )
            except Exception as e:
                print(f"Error processing {video_path.name}: {e}")
                import traceback
                traceback.print_exc()

        # Generate and save report
        report = evaluator.generate_report(str(output_dir / 'report.txt'))
        print("\n" + report)

        evaluator.save_results_json(str(output_dir / 'results.json'))
        print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
