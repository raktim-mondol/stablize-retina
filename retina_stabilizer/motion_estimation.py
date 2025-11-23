"""
Motion Estimation Module.

Primary: RAFT optical flow (handles >60px motion, occlusions, illumination changes)
Fallback: ECC on Frangi-enhanced images (robust for blurry frames)

RAFT achieves 30-40% lower residual than DeepFlow/PWC-Net on retinal videos.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class Transform:
    """Similarity transform parameters (4-DoF)."""
    tx: float  # Translation X
    ty: float  # Translation Y
    angle: float  # Rotation angle (radians)
    scale: float  # Scale factor

    def to_matrix(self) -> np.ndarray:
        """Convert to 2x3 affine matrix."""
        cos_a = np.cos(self.angle) * self.scale
        sin_a = np.sin(self.angle) * self.scale
        return np.array([
            [cos_a, -sin_a, self.tx],
            [sin_a, cos_a, self.ty]
        ], dtype=np.float64)

    def to_3x3_matrix(self) -> np.ndarray:
        """Convert to 3x3 homogeneous matrix."""
        mat = np.eye(3, dtype=np.float64)
        mat[:2, :] = self.to_matrix()
        return mat

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> 'Transform':
        """Create Transform from 2x3 or 3x3 matrix."""
        if matrix.shape[0] == 3:
            matrix = matrix[:2, :]

        tx = matrix[0, 2]
        ty = matrix[1, 2]
        scale = np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
        angle = np.arctan2(matrix[1, 0], matrix[0, 0])

        return Transform(tx=tx, ty=ty, angle=angle, scale=scale)

    @staticmethod
    def identity() -> 'Transform':
        """Return identity transform."""
        return Transform(tx=0.0, ty=0.0, angle=0.0, scale=1.0)


class MotionEstimator:
    """
    Estimates motion between frames using RAFT optical flow with ECC fallback.

    Uses similarity transform (4-DoF) which is clinically safest for retinal imaging
    as it avoids artificial vessel bending from higher DoF transforms.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        raft_iterations: int = 20,
        ransac_iterations: int = 5000,
        ransac_threshold: float = 3.0,
        ecc_iterations: int = 100,
        ecc_epsilon: float = 1e-6,
        confidence_threshold: float = 0.85
    ):
        """
        Initialize motion estimator.

        Args:
            device: Compute device for RAFT
            raft_iterations: Number of RAFT refinement iterations
            ransac_iterations: RANSAC iterations for transform fitting
            ransac_threshold: RANSAC inlier threshold (pixels)
            ecc_iterations: Maximum ECC iterations
            ecc_epsilon: ECC convergence threshold
            confidence_threshold: Threshold to trigger ECC fallback
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.raft_iterations = raft_iterations
        self.ransac_iterations = ransac_iterations
        self.ransac_threshold = ransac_threshold
        self.ecc_iterations = ecc_iterations
        self.ecc_epsilon = ecc_epsilon
        self.confidence_threshold = confidence_threshold

        # RAFT model (will be loaded if available)
        self.raft_model = None
        self._load_raft_model()

    def _load_raft_model(self):
        """Load RAFT model from torchvision."""
        try:
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            self.raft_model = raft_small(weights=Raft_Small_Weights.DEFAULT)
            self.raft_model = self.raft_model.to(self.device)
            self.raft_model.eval()
        except Exception as e:
            print(f"Warning: Could not load RAFT model: {e}")
            print("Will use Farneback optical flow as alternative.")
            self.raft_model = None

    def _preprocess_for_raft(self, image: np.ndarray) -> Tuple[torch.Tensor, int, int]:
        """Preprocess image for RAFT input.

        Returns:
            tensor: Preprocessed image tensor
            new_h: Padded height (multiple of 8)
            new_w: Padded width (multiple of 8)
        """
        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # To tensor and normalize
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        tensor = tensor.unsqueeze(0).to(self.device)

        # Resize to multiple of 8
        h, w = tensor.shape[2:]
        new_h = ((h - 1) // 8 + 1) * 8
        new_w = ((w - 1) // 8 + 1) * 8

        if new_h != h or new_w != w:
            tensor = F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)

        return tensor, new_h, new_w

    @torch.no_grad()
    def compute_raft_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute optical flow using RAFT.

        Args:
            frame1: First frame (grayscale or BGR)
            frame2: Second frame (grayscale or BGR)

        Returns:
            flow: Optical flow field (H, W, 2)
            confidence: Mean flow confidence
        """
        if self.raft_model is None:
            return self._compute_farneback_flow(frame1, frame2)

        img1, new_h, new_w = self._preprocess_for_raft(frame1)
        img2, _, _ = self._preprocess_for_raft(frame2)

        # Run RAFT
        flow_predictions = self.raft_model(img1, img2)
        flow = flow_predictions[-1]  # Take final refinement

        # Resize back to original size
        h, w = frame1.shape[:2]
        flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)

        # Scale flow values to account for resolution change
        flow[:, 0] *= w / new_w
        flow[:, 1] *= h / new_h

        flow = flow.squeeze().permute(1, 2, 0).cpu().numpy()

        # Compute confidence based on flow consistency
        confidence = self._compute_flow_confidence(flow)

        return flow, confidence

    def _compute_farneback_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Fallback: Farneback optical flow."""
        # Ensure grayscale
        if len(frame1.shape) == 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if len(frame2.shape) == 3:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            frame1, frame2, None,
            pyr_scale=0.5,
            levels=5,
            winsize=15,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=0
        )

        confidence = self._compute_flow_confidence(flow)
        return flow, confidence

    def _compute_flow_confidence(self, flow: np.ndarray) -> float:
        """
        Compute confidence score for optical flow.

        Based on flow magnitude consistency and spatial smoothness.
        """
        # Flow magnitude
        mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

        # Spatial gradient of flow (smoothness)
        grad_x = np.abs(np.diff(flow[:, :, 0], axis=1))
        grad_y = np.abs(np.diff(flow[:, :, 1], axis=0))

        # High confidence if flow is smooth
        smoothness = 1.0 / (1.0 + np.mean(grad_x) + np.mean(grad_y))

        # Penalize very large flows (potential errors)
        mag_penalty = np.clip(1.0 - np.mean(mag) / 100.0, 0.0, 1.0)

        confidence = smoothness * mag_penalty
        return float(confidence)

    def fit_similarity_transform(
        self,
        flow: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Transform:
        """
        Fit similarity transform from dense optical flow using RANSAC.

        Args:
            flow: Optical flow field (H, W, 2)
            weights: Optional weight map for important regions

        Returns:
            Fitted similarity transform
        """
        h, w = flow.shape[:2]

        # Sample points
        step = max(1, min(h, w) // 50)
        y_coords, x_coords = np.mgrid[0:h:step, 0:w:step]
        y_coords = y_coords.flatten()
        x_coords = x_coords.flatten()

        # Source and destination points
        src_pts = np.column_stack([x_coords, y_coords]).astype(np.float32)
        dst_pts = src_pts + flow[y_coords, x_coords]

        # Apply weights if provided
        if weights is not None:
            point_weights = weights[y_coords, x_coords]
            # Keep points with higher weights
            mask = point_weights > np.percentile(point_weights, 30)
            src_pts = src_pts[mask]
            dst_pts = dst_pts[mask]

        if len(src_pts) < 4:
            return Transform.identity()

        # Estimate similarity transform with RANSAC
        try:
            matrix, inliers = cv2.estimateAffinePartial2D(
                src_pts.reshape(-1, 1, 2),
                dst_pts.reshape(-1, 1, 2),
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                maxIters=self.ransac_iterations
            )

            if matrix is None:
                return Transform.identity()

            return Transform.from_matrix(matrix)

        except Exception:
            return Transform.identity()

    def compute_ecc_transform(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        initial_transform: Optional[Transform] = None
    ) -> Transform:
        """
        Compute transform using Enhanced Correlation Coefficient (ECC).

        Robust fallback for blurry frames or when RAFT fails.

        Args:
            frame1: Reference frame (grayscale)
            frame2: Target frame (grayscale)
            initial_transform: Initial transform estimate

        Returns:
            Refined similarity transform
        """
        # Ensure grayscale
        if len(frame1.shape) == 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if len(frame2.shape) == 3:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Initialize warp matrix
        if initial_transform is not None:
            warp_matrix = initial_transform.to_matrix().astype(np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # ECC criteria
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.ecc_iterations,
            self.ecc_epsilon
        )

        try:
            _, warp_matrix = cv2.findTransformECC(
                frame1.astype(np.float32),
                frame2.astype(np.float32),
                warp_matrix,
                cv2.MOTION_EUCLIDEAN,
                criteria,
                None,
                5
            )
            return Transform.from_matrix(warp_matrix)

        except cv2.error:
            # ECC failed to converge
            if initial_transform is not None:
                return initial_transform
            return Transform.identity()

    def estimate_motion(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        vessel_weights: Optional[np.ndarray] = None,
        frangi_enhanced1: Optional[np.ndarray] = None,
        frangi_enhanced2: Optional[np.ndarray] = None
    ) -> Tuple[Transform, float, str]:
        """
        Estimate motion between two frames.

        Uses RAFT as primary method with ECC fallback.

        Args:
            frame1: Reference frame
            frame2: Target frame
            vessel_weights: Weight map emphasizing vessels
            frangi_enhanced1: Frangi-enhanced reference for ECC fallback
            frangi_enhanced2: Frangi-enhanced target for ECC fallback

        Returns:
            transform: Estimated similarity transform
            confidence: Estimation confidence
            method: Method used ('raft' or 'ecc')
        """
        # Primary: RAFT optical flow
        flow, confidence = self.compute_raft_flow(frame1, frame2)

        if confidence >= self.confidence_threshold:
            transform = self.fit_similarity_transform(flow, vessel_weights)
            return transform, confidence, 'raft'

        # Fallback: ECC on Frangi-enhanced images
        if frangi_enhanced1 is not None and frangi_enhanced2 is not None:
            # Use RAFT result as initial estimate
            initial = self.fit_similarity_transform(flow, vessel_weights)
            transform = self.compute_ecc_transform(
                frangi_enhanced1,
                frangi_enhanced2,
                initial
            )
            return transform, confidence, 'ecc'

        # Final fallback: ECC on original images
        initial = self.fit_similarity_transform(flow, vessel_weights)
        transform = self.compute_ecc_transform(frame1, frame2, initial)
        return transform, confidence, 'ecc'

    def estimate_trajectory(
        self,
        frames: List[np.ndarray],
        vessel_weights: Optional[List[np.ndarray]] = None,
        frangi_frames: Optional[List[np.ndarray]] = None,
        reference_idx: int = 0
    ) -> List[Transform]:
        """
        Estimate motion trajectory for all frames relative to reference.

        Args:
            frames: List of preprocessed frames
            vessel_weights: Optional weight maps for each frame
            frangi_frames: Optional Frangi-enhanced frames for ECC fallback
            reference_idx: Index of reference frame

        Returns:
            List of transforms (each frame -> reference)
        """
        n_frames = len(frames)
        transforms = [Transform.identity() for _ in range(n_frames)]

        ref_frame = frames[reference_idx]
        ref_weights = vessel_weights[reference_idx] if vessel_weights else None
        ref_frangi = frangi_frames[reference_idx] if frangi_frames else None

        # Forward pass (reference to end)
        cumulative = Transform.identity()
        for i in range(reference_idx + 1, n_frames):
            weights = vessel_weights[i] if vessel_weights else None
            frangi = frangi_frames[i] if frangi_frames else None

            transform, _, _ = self.estimate_motion(
                frames[i - 1], frames[i],
                weights,
                frangi_frames[i - 1] if frangi_frames else None,
                frangi
            )

            # Accumulate transforms
            curr_mat = transform.to_3x3_matrix()
            cum_mat = cumulative.to_3x3_matrix()
            new_mat = curr_mat @ cum_mat
            cumulative = Transform.from_matrix(new_mat)
            transforms[i] = cumulative

        # Backward pass (reference to start)
        cumulative = Transform.identity()
        for i in range(reference_idx - 1, -1, -1):
            weights = vessel_weights[i] if vessel_weights else None
            frangi = frangi_frames[i] if frangi_frames else None

            transform, _, _ = self.estimate_motion(
                frames[i + 1], frames[i],
                weights,
                frangi_frames[i + 1] if frangi_frames else None,
                frangi
            )

            # Accumulate transforms
            curr_mat = transform.to_3x3_matrix()
            cum_mat = cumulative.to_3x3_matrix()
            new_mat = curr_mat @ cum_mat
            cumulative = Transform.from_matrix(new_mat)
            transforms[i] = cumulative

        return transforms
