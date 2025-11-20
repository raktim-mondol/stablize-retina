# Mermaid Diagrams - Retinal Video Stabilization Pipeline

This folder contains Mermaid diagram files (.mmd) that visualize the complete retinal video stabilization pipeline. These diagrams can be viewed directly on GitHub.

## Diagram Index

| File | Description |
|------|-------------|
| [01_main_pipeline.mmd](01_main_pipeline.mmd) | Complete 9-stage processing pipeline overview |
| [02_feature_extraction.mmd](02_feature_extraction.mmd) | Preprocessing and vessel segmentation workflow |
| [03_motion_estimation.mmd](03_motion_estimation.mmd) | RAFT optical flow with ECC fallback |
| [04_trajectory_smoothing.mmd](04_trajectory_smoothing.mmd) | L1 optimization and Kalman filtering |
| [05_performance_metrics.mmd](05_performance_metrics.mmd) | All metrics and what they measure |
| [06_vessel_guided_stabilization.mmd](06_vessel_guided_stabilization.mmd) | Vessel-guided motion estimation strategy |
| [07_comparison_methods.mmd](07_comparison_methods.mmd) | Comparison with traditional methods |

## Viewing Diagrams

### On GitHub
GitHub automatically renders `.mmd` files with mermaid code blocks. Simply click on any file above.

### Locally
Use any Mermaid-compatible viewer:
- VS Code with Mermaid extension
- [Mermaid Live Editor](https://mermaid.live/)
- Any Markdown viewer with Mermaid support

## Key Pipeline Highlights

### Proposed Method Advantages
- **RAFT Optical Flow**: 30-40% lower error than SIFT/ORB
- **L1 Bundled Smoothing**: 93%+ FOV retention (vs 70-80% Gaussian)
- **Vessel-Guided**: Preserves clinically relevant structures
- **Tremor-Specific**: Butterworth bandstop for 8-12 Hz hand tremor

### Performance Targets
| Category | Metric | Target |
|----------|--------|--------|
| Stability | Residual Motion | 0.4-0.9 px |
| Stability | Jitter Reduction | 85-95% |
| Quality | PSNR | 25-35 dB |
| Quality | SSIM | 0.85-0.98 |
| Efficiency | Real-time Factor | > 2.8x |
| Coverage | FOV Retention | > 93% |
