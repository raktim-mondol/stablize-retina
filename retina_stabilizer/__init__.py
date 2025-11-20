"""
State-of-the-Art Retinal Video Stabilization Pipeline (2025)

A hybrid feature-based + dense optical flow method with vessel-guided priors
and L1-optimized trajectory smoothing for handheld retinal video stabilization.
"""

from .pipeline import RetinaStabilizer
from .preprocessing import Preprocessor
from .reference_selection import ReferenceFrameSelector
from .vessel_segmentation import VesselSegmenter
from .motion_estimation import MotionEstimator
from .trajectory_smoothing import TrajectorySmoother
from .warping import FrameWarper

__version__ = "1.0.0"
__all__ = [
    "RetinaStabilizer",
    "Preprocessor",
    "ReferenceFrameSelector",
    "VesselSegmenter",
    "MotionEstimator",
    "TrajectorySmoother",
    "FrameWarper",
]
