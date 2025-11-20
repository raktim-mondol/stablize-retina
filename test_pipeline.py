#!/usr/bin/env python3
"""
Test script for retinal video stabilization with dummy data.
Tests both Option 1 (Frangi filter) and Option 2 (Neural vessel segmentation).
"""

import numpy as np
import cv2
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_dummy_retinal_video(
    output_path: str = "dummy_retinal.mp4",
    n_frames: int = 60,
    size: tuple = (512, 512),
    fps: float = 30.0
):
    """
    Create a synthetic retinal video with simulated vessels and camera shake.
    """
    print(f"Creating dummy retinal video: {n_frames} frames at {size}")

    # Create base retinal image with vessels
    h, w = size
    base_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Orange/red background (fundus color)
    base_image[:, :] = [40, 80, 180]  # BGR

    # Add circular optic disc
    center = (w // 2, h // 2)
    cv2.circle(base_image, center, 60, (150, 200, 255), -1)

    # Add blood vessels (dark red lines)
    vessel_color = (20, 40, 120)

    # Main vessels from optic disc
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        rad = np.radians(angle)
        end_x = int(center[0] + 200 * np.cos(rad))
        end_y = int(center[1] + 200 * np.sin(rad))
        thickness = np.random.randint(2, 5)
        cv2.line(base_image, center, (end_x, end_y), vessel_color, thickness)

        # Add branching
        mid_x = int(center[0] + 100 * np.cos(rad))
        mid_y = int(center[1] + 100 * np.sin(rad))
        for branch_offset in [-30, 30]:
            branch_rad = np.radians(angle + branch_offset)
            branch_end_x = int(mid_x + 80 * np.cos(branch_rad))
            branch_end_y = int(mid_y + 80 * np.sin(branch_rad))
            cv2.line(base_image, (mid_x, mid_y), (branch_end_x, branch_end_y),
                    vessel_color, max(1, thickness - 1))

    # Add some texture noise
    noise = np.random.randint(-10, 10, base_image.shape, dtype=np.int16)
    base_image = np.clip(base_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Generate frames with simulated camera shake
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Simulate hand tremor trajectory
    t = np.linspace(0, 2 * np.pi, n_frames)

    # Hand tremor: 8-12 Hz oscillation + random drift
    tremor_x = 5 * np.sin(10 * t) + 3 * np.sin(8 * t) + np.cumsum(np.random.randn(n_frames) * 0.5)
    tremor_y = 4 * np.cos(10 * t) + 2 * np.cos(12 * t) + np.cumsum(np.random.randn(n_frames) * 0.5)

    # Add occasional larger jumps (saccades)
    for _ in range(3):
        jump_idx = np.random.randint(10, n_frames - 10)
        tremor_x[jump_idx:] += np.random.randn() * 15
        tremor_y[jump_idx:] += np.random.randn() * 15

    # Create padded image for translation
    pad = 50
    padded = cv2.copyMakeBorder(base_image, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    for i in range(n_frames):
        # Apply translation
        tx = int(tremor_x[i])
        ty = int(tremor_y[i])

        # Extract shifted region
        frame = padded[pad + ty:pad + ty + h, pad + tx:pad + tx + w].copy()

        # Add slight illumination variation
        brightness = 1.0 + 0.05 * np.sin(0.5 * t[i])
        frame = np.clip(frame * brightness, 0, 255).astype(np.uint8)

        writer.write(frame)

    writer.release()
    print(f"Saved dummy video to: {output_path}")
    return output_path


def test_option1_frangi():
    """Test with Frangi filter (no neural network)."""
    print("\n" + "="*60)
    print("OPTION 1: Testing with Frangi Filter (No Training Required)")
    print("="*60 + "\n")

    from retina_stabilizer import RetinaStabilizer

    # Create dummy video
    input_path = "test_input.mp4"
    output_path = "test_output_frangi.mp4"
    create_dummy_retinal_video(input_path, n_frames=30, size=(256, 256))

    # Create stabilizer without neural vessel segmentation
    stabilizer = RetinaStabilizer(
        use_vessel_segmentation=False,  # Use Frangi filter
        device='cpu',  # Use CPU for testing
        verbose=True
    )

    # Run stabilization
    try:
        frames, metrics = stabilizer.stabilize(input_path, output_path)

        print("\n" + "-"*40)
        print("OPTION 1 TEST PASSED!")
        print(f"Output saved to: {output_path}")
        print(f"Frames processed: {len(frames)}")
        print("-"*40)
        return True

    except Exception as e:
        print(f"\nOPTION 1 TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_option2_neural():
    """Test with neural vessel segmentation."""
    print("\n" + "="*60)
    print("OPTION 2: Testing with Neural Vessel Segmentation")
    print("="*60 + "\n")

    from retina_stabilizer import RetinaStabilizer

    # Create dummy video
    input_path = "test_input.mp4"
    output_path = "test_output_neural.mp4"

    if not os.path.exists(input_path):
        create_dummy_retinal_video(input_path, n_frames=30, size=(256, 256))

    # Create stabilizer with neural vessel segmentation
    # Note: Without pre-trained weights, this uses randomly initialized weights
    stabilizer = RetinaStabilizer(
        use_vessel_segmentation=True,  # Use U-Net
        vessel_model_path=None,  # No pre-trained weights
        device='cpu',
        verbose=True
    )

    # Run stabilization
    try:
        frames, metrics = stabilizer.stabilize(input_path, output_path)

        print("\n" + "-"*40)
        print("OPTION 2 TEST PASSED!")
        print(f"Output saved to: {output_path}")
        print(f"Frames processed: {len(frames)}")
        print("-"*40)
        print("\nNOTE: Used randomly initialized U-Net weights.")
        print("For best results, load pre-trained weights (see below).")
        return True

    except Exception as e:
        print(f"\nOPTION 2 TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_modules():
    """Test each module individually."""
    print("\n" + "="*60)
    print("Testing Individual Modules")
    print("="*60 + "\n")

    # Create test image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    # Test preprocessing
    print("Testing Preprocessor...", end=" ")
    from retina_stabilizer import Preprocessor
    prep = Preprocessor()
    enhanced = prep.preprocess_frame(test_image)
    assert enhanced.shape == (256, 256), "Preprocessor output shape mismatch"
    print("OK")

    # Test reference selection
    print("Testing ReferenceFrameSelector...", end=" ")
    from retina_stabilizer import ReferenceFrameSelector
    selector = ReferenceFrameSelector()
    score, _, _ = selector.score_frame(test_gray)
    assert isinstance(score, float), "Score should be float"
    print("OK")

    # Test vessel segmentation
    print("Testing VesselSegmenter...", end=" ")
    from retina_stabilizer import VesselSegmenter
    segmenter = VesselSegmenter(device='cpu')
    prob_map = segmenter.segment(test_gray)
    assert prob_map.shape == (256, 256), "Vessel map shape mismatch"
    print("OK")

    # Test motion estimation
    print("Testing MotionEstimator...", end=" ")
    from retina_stabilizer import MotionEstimator
    estimator = MotionEstimator(device='cpu')
    transform, conf, method = estimator.estimate_motion(test_gray, test_gray)
    assert hasattr(transform, 'tx'), "Transform should have tx attribute"
    print(f"OK (using {method})")

    # Test trajectory smoothing
    print("Testing TrajectorySmoother...", end=" ")
    from retina_stabilizer import TrajectorySmoother
    from retina_stabilizer.motion_estimation import Transform
    smoother = TrajectorySmoother()
    transforms = [Transform.identity() for _ in range(10)]
    smoothed = smoother.smooth(transforms, fps=30.0)
    assert len(smoothed) == 10, "Smoothed trajectory length mismatch"
    print("OK")

    # Test warping
    print("Testing FrameWarper...", end=" ")
    from retina_stabilizer import FrameWarper
    warper = FrameWarper()
    warped = warper.warp_frame(test_image, Transform.identity())
    assert warped.shape == test_image.shape, "Warped frame shape mismatch"
    print("OK")

    print("\nAll module tests passed!")


def cleanup():
    """Remove test files."""
    test_files = [
        "test_input.mp4",
        "test_output_frangi.mp4",
        "test_output_neural.mp4",
        "dummy_retinal.mp4"
    ]
    for f in test_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed: {f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test retinal stabilization pipeline")
    parser.add_argument("--option", type=int, choices=[1, 2],
                       help="Test specific option (1=Frangi, 2=Neural)")
    parser.add_argument("--modules", action="store_true",
                       help="Test individual modules only")
    parser.add_argument("--cleanup", action="store_true",
                       help="Remove test files after running")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests")

    args = parser.parse_args()

    if args.modules:
        test_individual_modules()
    elif args.option == 1:
        test_option1_frangi()
    elif args.option == 2:
        test_option2_neural()
    elif args.all or len(sys.argv) == 1:
        # Run all tests
        test_individual_modules()
        test_option1_frangi()
        test_option2_neural()

    if args.cleanup:
        print("\nCleaning up test files...")
        cleanup()
