"""
Preprocessing Module: Frame extraction, green channel enhancement, and CLAHE.

Green channel has ~2.5x higher vessel contrast than RGB.
CLAHE fixes illumination drift across frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class Preprocessor:
    """Handles video loading and frame preprocessing for retinal stabilization."""

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        target_size: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize preprocessor.

        Args:
            clahe_clip_limit: Contrast limiting threshold for CLAHE
            clahe_grid_size: Size of grid for histogram equalization
            target_size: Optional resize target (width, height)
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_grid_size
        )

    def load_video(self, video_path: str) -> Tuple[List[np.ndarray], float, Tuple[int, int]]:
        """
        Load video and extract all frames.

        Args:
            video_path: Path to input video file

        Returns:
            frames: List of BGR frames
            fps: Frames per second
            frame_size: (width, height) of frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.target_size is not None:
                frame = cv2.resize(frame, self.target_size)
            frames.append(frame)

        cap.release()

        if self.target_size is not None:
            frame_size = self.target_size
        else:
            frame_size = (width, height)

        return frames, fps, frame_size

    def extract_green_channel(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract green channel from BGR frame.

        Green channel provides best vessel contrast in retinal images.

        Args:
            frame: BGR image

        Returns:
            Green channel as grayscale image
        """
        return frame[:, :, 1]

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.

        Args:
            image: Grayscale image

        Returns:
            CLAHE-enhanced image
        """
        return self.clahe.apply(image)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Full preprocessing: green channel + CLAHE.

        Args:
            frame: BGR image

        Returns:
            Preprocessed grayscale image
        """
        green = self.extract_green_channel(frame)
        enhanced = self.apply_clahe(green)
        return enhanced

    def preprocess_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Preprocess all frames.

        Args:
            frames: List of BGR frames

        Returns:
            List of preprocessed grayscale frames
        """
        return [self.preprocess_frame(f) for f in frames]

    def normalize_for_network(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image for neural network input.

        Args:
            image: Grayscale or BGR image (uint8)

        Returns:
            Normalized float32 image in [0, 1]
        """
        return image.astype(np.float32) / 255.0
