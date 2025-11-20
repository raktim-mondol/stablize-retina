# Retinal Video Stabilization - Architecture Diagrams

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INPUT VIDEO FILE                                  │
│                      (variable FPS, BGR)                                 │
└────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────────┐
                    │  1. FRAME EXTRACTION         │
                    │  Preprocessor.load_video()  │
                    └──────────────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────────┐
                    │  2. PREPROCESSING            │
                    │  • Extract Green Channel     │
                    │  • Apply CLAHE Enhancement  │
                    └──────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
        ┌──────────────────────┐   ┌──────────────────────┐
        │ Sharpness Scoring    │   │ Vessel Scoring       │
        │ (Laplacian Variance) │   │ (Frangi Filter)      │
        └──────────────────────┘   └──────────────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────────┐
                    │  3. REFERENCE SELECTION      │
                    │  Select Best Frame (Combined │
                    │  Sharpness + Vessel Score)   │
                    └──────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
        ┌──────────────────────────┐  ┌──────────────────────────┐
        │ Vessel Segmentation:     │  │ Vessel Segmentation:     │
        │ Neural U-Net             │  │ Frangi Filter (Fallback) │
        │ (if enabled)             │  │ (always available)       │
        │ [Probability Maps]       │  │ [Tubularity Scores]      │
        └──────────────────────────┘  └──────────────────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
        ┌──────────────────────────┐  ┌──────────────────────────┐
        │ RAFT Optical Flow        │  │ Frangi Enhancement       │
        │ [Dense Flow Field]       │  │ [For ECC Fallback]       │
        └──────────────────────────┘  └──────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │ Flow Confidence Check    │
        │ (smoothness + magnitude) │
        └────────┬─────────────────┘
                 │
         ┌───────┴────────┐
         │                │
     Confidence      Confidence
        High             Low
         │                │
         ▼                ▼
    ┌─────────┐      ┌──────────────────┐
    │ Proceed │      │ ECC Fallback     │
    └────┬────┘      │ (on Frangi Image)│
         │           └──────────────────┘
         │                  │
         └──────────┬───────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │ RANSAC Transform Fitting │
        │ (Similarity, 4-DoF)      │
        │ [Weighted by Vessels]    │
        └──────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │ Motion Trajectory        │
        │ (Raw Transforms)         │
        └──────────────────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
    ┌──────────────┐   ┌──────────────────┐
    │ L1 Bundled   │   │ Kalman Filter    │
    │ Path         │   │ (Fine-Smoothing) │
    │ Smoothing    │   │ [Removes 8-12 Hz]│
    └──────────────┘   └──────────────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │ Tremor Removal           │
        │ (Bandstop Filter)        │
        └──────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │ Smoothed Trajectory      │
        └──────────────────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
    ┌──────────────────┐  ┌──────────────────┐
    │ Stable Crop      │  │ Stabilizing      │
    │ Computation      │  │ Transform        │
    │ (FOV Retention)  │  │ = smooth ∘ inv() │
    └──────────────────┘  └──────────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
    ┌──────────────┐    ┌─────────────────────┐
    │ Apply Warp   │    │ Border Mask         │
    │ (Lanczos-4)  │    │ Generation          │
    └──────────────┘    └─────────────────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │ Telea Inpainting         │
        │ (Border Artifacts)       │
        └──────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │ Apply Auto-Crop          │
        │ (Stable Region)          │
        └──────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │ Compute Metrics          │
        │ • Stability              │
        │ • Quality                │
        │ • Efficiency             │
        │ • Temporal               │
        └──────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │ Save Output Video        │
        │ + Metrics Report         │
        └──────────────────────────┘
```

---

## Module Dependency Graph

```
                        ┌──────────────────┐
                        │ RetinaStabilizer │
                        │  (Orchestrator)  │
                        └────────┬─────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
             ▼                   ▼                   ▼
        ┌─────────┐         ┌──────────┐      ┌─────────────┐
        │Preproc- │         │Reference │      │    Motion   │
        │ essor   │         │ Selector │      │  Estimator  │
        └────┬────┘         └────┬─────┘      └─────┬───────┘
             │                   │                   │
      (green+CLAHE)      (sharpness/vessel)    (RAFT+ECC)
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
                ▼                                 ▼
        ┌──────────────────┐          ┌──────────────────┐
        │Vessel Segmenter  │          │Trajectory        │
        │                  │          │Smoother          │
        │ ├─ U-Net         │          │                  │
        │ └─ Frangi Filter │          │ ├─ L1 Smooth    │
        └──────────────────┘          │ ├─ Kalman       │
                                      │ └─ Tremor Remove│
                                      └────────┬────────┘
                                               │
                                     ┌─────────▼────────┐
                                     │                  │
                                     ▼                  ▼
                            ┌──────────────────┐  ┌──────────────┐
                            │  Frame Warper    │  │  Evaluation  │
                            │                  │  │              │
                            │ ├─ Warp Affine   │  │├─Stability   │
                            │ ├─ Inpaint       │  │├─Quality     │
                            │ └─ Crop Compute  │  │├─Efficiency  │
                            └──────────────────┘  │└─Temporal    │
                                    │             └──────────────┘
                                    │
                                    ▼
                            ┌──────────────────┐
                            │  Output Video    │
                            │  + Metrics       │
                            └──────────────────┘
```

---

## Optical Flow Processing Pipeline

```
Frame Pair (t, t+1)
        │
        ▼
    ┌─────────────────────────┐
    │ Convert to RGB (if GS)  │
    │ Normalize                │
    │ Resize to multiple of 8  │
    └──────────┬──────────────┘
               │
               ▼
    ┌─────────────────────────┐
    │ RAFT Network Forward    │
    │ • 20 refinement iters   │
    │ • Dense flow prediction │
    └──────────┬──────────────┘
               │
               ▼
    ┌─────────────────────────┐
    │ Resize Flow to Original │
    │ Scale Flow Values       │
    └──────────┬──────────────┘
               │
               ▼
    ┌─────────────────────────┐
    │ Compute Confidence:     │
    │ smooth = 1/(1+gradients)│
    │ mag_pen = 1-mean/100    │
    │ conf = smooth × mag_pen │
    └──────────┬──────────────┘
               │
        ┌──────┴──────┐
        │             │
    conf≥85%      conf<85%
        │             │
        ▼             ▼
    ┌────────┐   ┌────────────────┐
    │RANSAC  │   │ECC on Frangi   │
    │Fitting │   │ • Init from RAFT
    └────────┘   │ • 100 iter refine
                 └────────────────┘
        │             │
        └──────┬──────┘
               │
               ▼
    ┌─────────────────────────┐
    │ Similarity Transform    │
    │ [tx, ty, angle, scale]  │
    └─────────────────────────┘
```

---

## Trajectory Smoothing Pipeline

```
Raw Transforms (Motion Trajectory)
        │
        ├─ tx (translation X)
        ├─ ty (translation Y)
        ├─ angle (rotation)
        └─ scale (zoom)
        
        ▼
    ┌──────────────────────────┐
    │ L1 Bundled Smoothing     │
    │ for each dimension:      │
    │ • L1 trend filtering     │
    │ • Moving average         │
    │ • Edge padding           │
    │                          │
    │ minimize: ||y-x||² + λ||D²x||₁
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │ Kalman Filtering         │
    │ for each dimension:      │
    │ • State: [pos, vel]      │
    │ • 1D forward filter      │
    │ • Removes 8-12 Hz freq   │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │ Tremor Removal           │
    │ • Bandstop filter        │
    │ • 8-12 Hz notch          │
    │ • Zero-phase (filtfilt)  │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │ Smoothed Transforms      │
    │ [tx', ty', angle', s']   │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │ Compute Stabilizing      │
    │ T_stab = T_smooth × inv  │
    │            (T_raw)       │
    └──────────────────────────┘
```

---

## Vessel Segmentation Comparison

```
┌─────────────────────────────────────────────────────────────┐
│              VESSEL LANDMARK DETECTION                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT: Preprocessed Grayscale Image                       │
│                                                             │
│  ┌──────────────────────┐        ┌──────────────────────┐ │
│  │  NEURAL U-NET        │        │  FRANGI FILTER       │ │
│  │  (If Model Available)│        │  (Always Available)  │ │
│  │                      │        │                      │ │
│  │ Architecture:        │        │ Approach:            │ │
│  │ • Lightweight UNet   │        │ • Multi-scale        │ │
│  │ • 4 encoder layers   │        │ • Hessian analysis   │ │
│  │ • 4 decoder layers   │        │ • σ ∈ {1,2,3}        │ │
│  │ • Skip connections   │        │ • Tubularity detect  │ │
│  │                      │        │                      │ │
│  │ Output:              │        │ Output:              │ │
│  │ Probability Map [0,1]│        │ Vesselness Score [0,1]
│  │                      │        │                      │
│  │ Speed:  25-35% time  │        │ Speed: <1% time      │
│  │ Acc:    ~90-95%      │        │ Acc:   ~80-85%       │
│  └──────────┬───────────┘        └──────────┬───────────┘ │
│             │                               │              │
│             └──────────────┬────────────────┘              │
│                            │                              │
│                            ▼                              │
│         ┌───────────────────────────────────┐             │
│         │ Confidence Weights Generation     │             │
│         │ w = 0.3 + 0.7 × probability      │             │
│         │ • 0.3 ≤ w ≤ 1.0                  │             │
│         │ • Vessel regions weighted higher │             │
│         └───────────────────────────────────┘             │
│                            │                              │
│  OUTPUT: Confidence Weight Map [0.3, 1.0]                │
│          (Used in Motion Estimation)                     │
│                                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance Metrics Hierarchy

```
┌─────────────────────────────────────┐
│   EVALUATION RESULT                 │
├─────────────────────────────────────┤
│ • video_name                        │
│ • n_frames, duration                │
│ • 4 metric categories               │
└──────────┬──────────────────────────┘
           │
    ┌──────┼──────┬──────┬──────┐
    │      │      │      │      │
    ▼      ▼      ▼      ▼      ▼
 ┌──────────────────┐  ┌──────────────────┐
 │ STABILITY        │  │ QUALITY          │
 │ METRICS          │  │ METRICS          │
 ├──────────────────┤  ├──────────────────┤
 │ • ITF            │  │ • PSNR           │
 │ • Residual Motion│  │ • SSIM           │
 │ • Motion Reduc%  │  │ • Sharpness Ret% │
 │ • Jitter (x/y)   │  │                  │
 │ • Jitter (rot)   │  │                  │
 │ • Jitter (scale) │  │                  │
 └──────────────────┘  └──────────────────┘

 ┌──────────────────┐  ┌──────────────────┐
 │ EFFICIENCY       │  │ TEMPORAL         │
 │ METRICS          │  │ METRICS          │
 ├──────────────────┤  ├──────────────────┤
 │ • FOV Retention% │  │ • Smoothness     │
 │ • Crop Ratio     │  │ • Velocity Cons. │
 │ • Processing FPS │  │ • Accel. Cons.   │
 │ • Realtime Fact  │  │ • Tremor Red. %  │
 │ • Memory (CPU)   │  │                  │
 │ • Memory (GPU)   │  │                  │
 └──────────────────┘  └──────────────────┘
```

---

## Transform Composition & Accumulation

```
Frame₀ → Frame₁ → Frame₂ → Frame₃ → ... → Frameₙ

Single Frame-to-Frame Transforms (from motion estimation):
       T₀→₁    T₁→₂    T₂→₃
         │       │       │
         ▼       ▼       ▼

FORWARD ACCUMULATION (Reference Frame = 2):
       T₀→₁    T₁→₂ (ref)
         │       └─ Identity
         ▼
     T₀→₂ = T₁→₂ ∘ T₀→₁

                        T₂→₃
                         │
                         ▼
                    T₂→₃ (identity)
                         └─ ref frame

T₂→₄ = T₃→₄ ∘ T₂→₃

FINAL TRAJECTORY (relative to reference Frame 2):
T₀→₂ (cumulative back)
T₁→₂ (one step back)
T₂→₂ (identity)
T₂→₃ (one step forward)
T₂→₄ (cumulative forward)

→ APPLY SMOOTHING AND STABILIZATION TRANSFORMS
```

---

## Computational Bottleneck Analysis

```
┌────────────────────────────────────────────────────┐
│        TIMING DISTRIBUTION (GPU Processing)       │
├────────────────────────────────────────────────────┤
│                                                    │
│  Preprocessing:        5-8%  ░░░                  │
│  Reference Selection:  2-3%  ░                    │
│  Vessel Segmentation: 25-35% ░░░░░░░              │
│  Motion Estimation:   30-40% ░░░░░░░░             │
│  Trajectory Smooth:    5-10% ░░░                  │
│  Warping & Inpaint:   10-15% ░░░░                │
│                                                    │
│  CRITICAL PATH:                                  │
│  • Motion Estimation is single largest cost      │
│  • RAFT depth inference dominates                │
│  • Parallelizable across frames                  │
│                                                    │
└────────────────────────────────────────────────────┘

OPTIMIZATION OPPORTUNITIES:
• RAFT: Reduce iterations from 20 to 12-15
• Vessel Seg: Skip on blurry frames (Frangi fallback)
• Batch Processing: GPU memory allows multi-frame batches
• Quantization: Model pruning for faster inference
```

---

## Quality Tradeoff Surface

```
             FOV Retention (%) 
                     │
                  95%├─────────────────────┐
                     │                     │
                  93%├─────┐               │ ← L1 Smoothing
                     │     │ RAFT+Kalman   │
                  90%│     ├────────┐      │ Boundary
                     │     │        │      │
                  80%│     │        ├──────┤ ← Gaussian
                     │     │        │      │   Smoothing
                  70%│     │        ├──────┤
                     │     │        │      │
                     ├─────┴────────┴──────┴─── Smoothing
                     0    Light   Strong   Aggressive
                         
        Residual Motion
             (pixels)
            
        ┌─────────────────────────────────┐
        │ Smoothing ↑ → FOV ↓             │
        │ Smoothing ↑ → Residual ↓        │
        │ L1 is Pareto optimal for both   │
        │ Gaussian wastes FOV for gains   │
        └─────────────────────────────────┘
```

