"""
Video processing service that wraps RetinaStabilizer
"""
import os
import sys
import time
import traceback

# Add parent directory to import retina_stabilizer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from retina_stabilizer import RetinaStabilizer
from retina_stabilizer.evaluation import VideoEvaluator
from retina_stabilizer.benchmark import PipelineBenchmarker


class VideoProcessor:
    """Handles video stabilization with progress tracking"""

    def process(self, job_id, jobs, input_path, output_path, model_type):
        """
        Process video stabilization in background

        Args:
            job_id: Unique job identifier
            jobs: Shared jobs dictionary
            input_path: Path to input video
            output_path: Path for output video
            model_type: 'neural' or 'classical'
        """
        try:
            job = jobs[job_id]

            # Update progress
            def update_progress(progress, stage):
                job['progress'] = progress
                job['stage'] = stage

            # Stage 1: Initialize stabilizer
            update_progress(5, 'Initializing stabilizer')

            # Configure based on model type
            if model_type == 'neural':
                device = 'cuda'
                use_neural = True
            else:
                device = 'cpu'
                use_neural = False

            # Try CUDA, fallback to CPU
            try:
                import torch
                if not torch.cuda.is_available():
                    device = 'cpu'
            except:
                device = 'cpu'

            stabilizer = RetinaStabilizer(
                device=device,
                auto_crop=True,
                inpaint_borders=True
            )

            # Stage 2: Load video
            update_progress(10, 'Loading video')

            # Stage 3-7: Run stabilization with progress updates
            update_progress(15, 'Preprocessing frames')

            # Create a wrapper to track progress during stabilization
            original_stabilize = stabilizer.stabilize

            def stabilize_with_progress(input_path, output_path):
                # We'll update progress based on typical stage timing
                update_progress(20, 'Extracting frames')
                time.sleep(0.5)

                update_progress(30, 'Enhancing frames (CLAHE)')
                time.sleep(0.5)

                update_progress(40, 'Selecting reference frame')
                time.sleep(0.5)

                update_progress(50, 'Computing vessel maps')

                # Run actual stabilization
                result = original_stabilize(input_path, output_path)

                return result

            # Monkey-patch for progress (simplified)
            update_progress(25, 'Processing video')

            # Run stabilization
            start_time = time.time()
            stabilized_frames, pipeline_metrics = stabilizer.stabilize(input_path, output_path)
            processing_time = time.time() - start_time

            update_progress(70, 'Evaluating results')

            # Stage 8: Run evaluation
            evaluator = VideoEvaluator()

            # Load original frames for comparison
            import cv2
            cap = cv2.VideoCapture(input_path)
            original_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                original_frames.append(frame)
            cap.release()

            # Run evaluation
            eval_result = evaluator.evaluate(original_frames, stabilized_frames, pipeline_metrics)

            update_progress(85, 'Computing benchmarks')

            # Stage 9: Compile metrics
            metrics = {
                'stability': {
                    'itf': round(eval_result.stability.itf, 4),
                    'residual_motion': round(eval_result.stability.residual_motion, 4),
                    'jitter_reduction_x': round(eval_result.stability.jitter_reduction_x * 100, 2),
                    'jitter_reduction_y': round(eval_result.stability.jitter_reduction_y * 100, 2),
                    'jitter_reduction_rotation': round(eval_result.stability.jitter_reduction_rotation * 100, 2),
                },
                'quality': {
                    'psnr': round(eval_result.quality.psnr, 2),
                    'ssim': round(eval_result.quality.ssim, 4),
                    'sharpness_retention': round(eval_result.quality.sharpness_retention * 100, 2),
                },
                'efficiency': {
                    'fov_retention': round(eval_result.efficiency.fov_retention * 100, 2),
                    'processing_fps': round(eval_result.efficiency.processing_speed, 2),
                    'realtime_factor': round(eval_result.efficiency.realtime_factor, 2),
                },
                'temporal': {
                    'smoothness': round(eval_result.temporal.temporal_smoothness, 4),
                    'velocity_consistency': round(eval_result.temporal.velocity_consistency, 4),
                    'tremor_reduction': round(eval_result.temporal.tremor_reduction * 100, 2),
                }
            }

            # Benchmark data
            benchmark = {
                'total_time': round(processing_time, 2),
                'frames_processed': len(stabilized_frames),
                'avg_fps': round(len(stabilized_frames) / processing_time, 2),
                'model_used': model_type,
                'device': device,
            }

            # Add pipeline metrics if available
            if pipeline_metrics:
                if hasattr(pipeline_metrics, 'get'):
                    benchmark['preprocessing_time'] = round(pipeline_metrics.get('preprocessing_time', 0), 2)
                    benchmark['motion_estimation_time'] = round(pipeline_metrics.get('motion_estimation_time', 0), 2)
                    benchmark['smoothing_time'] = round(pipeline_metrics.get('smoothing_time', 0), 2)
                    benchmark['warping_time'] = round(pipeline_metrics.get('warping_time', 0), 2)

            update_progress(100, 'Complete')

            # Update job with results
            job['status'] = 'completed'
            job['progress'] = 100
            job['stage'] = 'Complete'
            job['metrics'] = metrics
            job['benchmark'] = benchmark

        except Exception as e:
            traceback.print_exc()
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = str(e)
            jobs[job_id]['stage'] = 'Failed'
