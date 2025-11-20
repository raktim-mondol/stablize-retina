# Retinal Video Stabilization Pipeline

State-of-the-art retinal video stabilization using vessel-guided RAFT optical flow with L1-optimized trajectory smoothing (2025 best practice).

## Features

- **Vessel-guided motion estimation**: Uses blood vessels as stable landmarks
- **RAFT optical flow**: Handles >60px motion, occlusions, illumination changes
- **ECC fallback**: Robust estimation on blurry frames using Frangi-enhanced images
- **L1 bundled smoothing**: Minimizes crop while removing jitter
- **Kalman filtering**: Removes 8-12 Hz hand tremor
- **Content-preserving warps**: Telea inpainting for clean borders
- **Memory optimized**: Handles large images (1800x3200+) with reduced memory footprint

## Expected Performance

- **Residual error**: 0.4-0.9 pixels (median)
- **FOV retention**: ≥93% of original
- **Speed**: ~2.8x real-time on modern GPU

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
# Basic usage
python run_stabilization.py input.mp4 output.mp4

# With GPU
python run_stabilization.py input.mp4 output.mp4 --device cuda

# Without auto-crop (keep full frame)
python run_stabilization.py input.mp4 output.mp4 --no-crop

# Use Frangi filter instead of neural vessel segmentation
python run_stabilization.py input.mp4 output.mp4 --no-vessel-seg

# Memory optimization for large images (1800x3200)
python run_stabilization.py input.mp4 output.mp4 --processing-scale 0.25 --vessel-seg-size 512 512
```

### Python API

```python
from retina_stabilizer import RetinaStabilizer

# Create stabilizer
stabilizer = RetinaStabilizer(
    device='cuda',        # or 'cpu'
    auto_crop=True,       # Crop to stable region
    inpaint_borders=True  # Inpaint border artifacts
)

# Stabilize video
stabilized_frames, metrics = stabilizer.stabilize(
    'input.mp4',
    'output.mp4'
)

# Print metrics
print(f"FOV retention: {metrics['fov_retention']*100:.1f}%")
print(f"Residual motion: {metrics['residual_motion']:.2f} px")
```

### Memory Optimization for Large Images

For high-resolution videos (e.g., 1800x3200), use memory optimization to reduce RAM/VRAM usage:

```python
from retina_stabilizer import RetinaStabilizer

# Optimized for large images (reduces memory from ~20GB to ~4GB)
stabilizer = RetinaStabilizer(
    device='cuda',
    processing_scale=0.25,       # Motion estimation at 1/4 resolution
    vessel_seg_size=(512, 512)   # Vessel segmentation at 512x512
)

stabilized_frames, metrics = stabilizer.stabilize('large_video.mp4', 'output.mp4')
```

| Image Size | processing_scale | Memory Usage |
|------------|------------------|--------------|
| 1800x3200  | 1.0 (default)    | ~20GB+       |
| 1800x3200  | 0.5              | ~8GB         |
| 1800x3200  | 0.25             | ~4GB         |

### Module Components

```python
from retina_stabilizer import (
    Preprocessor,           # Green channel + CLAHE
    ReferenceFrameSelector, # Best frame selection
    VesselSegmenter,        # Neural vessel segmentation
    MotionEstimator,        # RAFT + ECC
    TrajectorySmoother,     # L1 + Kalman
    FrameWarper             # Warp + inpaint
)
```

## Pipeline Architecture

```
Input Video → Green Channel + CLAHE → Reference Frame Selection
     ↓
Vessel Probability Maps (U-Net or Frangi)
     ↓
RAFT Optical Flow → Confidence Check → ECC Fallback (if needed)
     ↓
Similarity Transform Fitting (RANSAC)
     ↓
L1 Bundled Path Smoothing + Kalman Filter
     ↓
Compute Stabilizing Warps → Apply with Lanczos-4
     ↓
Telea Inpainting + Auto-Crop → Output Video
```

## Configuration

Key parameters in `RetinaStabilizer`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clahe_clip_limit` | 2.0 | CLAHE contrast limit |
| `confidence_threshold` | 0.85 | RAFT confidence for ECC fallback |
| `smooth_window` | 31 | Trajectory smoothing window |
| `kalman_process_noise` | 0.03 | Kalman filter tuning |
| `auto_crop` | True | Crop to stable region |
| `inpaint_borders` | True | Inpaint border artifacts |
| `processing_scale` | 1.0 | Scale for motion estimation (0.25 = 1/4 res) |
| `vessel_seg_size` | (512, 512) | Target size for vessel segmentation |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- NumPy, SciPy, scikit-image

## License

MIT License
