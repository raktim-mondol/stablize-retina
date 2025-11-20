"""
Vessel Segmentation Module.

Provides vessel probability maps using a lightweight U-Net architecture.
Vessels are unique, high-contrast, stable landmarks that persist even in
low-quality frames - the primary reason this pipeline beats general stabilizers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class DoubleConv(nn.Module):
    """Double convolution block for U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class LightweightUNet(nn.Module):
    """
    Lightweight U-Net for vessel segmentation.

    Optimized for speed while maintaining accuracy on retinal vessels.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder (upsampling)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        # Output
        self.out_conv = nn.Conv2d(32, out_channels, 1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


class VesselSegmenter:
    """Handles vessel segmentation for retinal images."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.5
    ):
        """
        Initialize vessel segmenter.

        Args:
            model_path: Path to pre-trained model weights
            device: Compute device ('cuda', 'cpu', or None for auto)
            threshold: Threshold for binary vessel mask
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.threshold = threshold
        self.model = LightweightUNet().to(self.device)

        if model_path is not None:
            self.load_weights(model_path)
        else:
            # Initialize with reasonable defaults for untrained model
            self._initialize_weights()

        self.model.eval()

    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_weights(self, model_path: str):
        """Load pre-trained weights."""
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def save_weights(self, model_path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), model_path)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for network input.

        Args:
            image: Grayscale image (H, W) uint8

        Returns:
            Tensor (1, 1, H, W) float32
        """
        # Normalize to [0, 1]
        img = image.astype(np.float32) / 255.0

        # Pad to multiple of 16 for U-Net
        h, w = img.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16

        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')

        # To tensor
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def postprocess(
        self,
        output: torch.Tensor,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Postprocess network output.

        Args:
            output: Network output tensor
            original_size: (height, width) of original image

        Returns:
            Probability map (H, W) float32 in [0, 1]
        """
        prob_map = output.squeeze().cpu().numpy()

        # Crop to original size
        h, w = original_size
        prob_map = prob_map[:h, :w]

        return prob_map

    @torch.no_grad()
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment vessels in image.

        Args:
            image: Grayscale image (H, W) uint8

        Returns:
            Vessel probability map (H, W) float32 in [0, 1]
        """
        original_size = image.shape[:2]
        input_tensor = self.preprocess(image)
        output = self.model(input_tensor)
        prob_map = self.postprocess(output, original_size)
        return prob_map

    def get_binary_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Get binary vessel mask.

        Args:
            image: Grayscale image

        Returns:
            Binary mask (H, W) uint8
        """
        prob_map = self.segment(image)
        mask = (prob_map > self.threshold).astype(np.uint8) * 255
        return mask

    def get_confidence_weights(self, image: np.ndarray) -> np.ndarray:
        """
        Get confidence weights for motion estimation.

        Higher weights on vessel regions for better alignment.

        Args:
            image: Grayscale image

        Returns:
            Confidence weight map (H, W) float32
        """
        prob_map = self.segment(image)

        # Enhance vessel regions while keeping some background
        weights = 0.3 + 0.7 * prob_map

        return weights.astype(np.float32)


def create_frangi_vessel_map(image: np.ndarray, scales: Tuple[float, ...] = (1, 2, 3)) -> np.ndarray:
    """
    Quick Frangi-based vessel enhancement as fallback.

    Use when no trained model is available.

    Args:
        image: Grayscale image
        scales: Scales for multi-scale analysis

    Returns:
        Vessel-enhanced image
    """
    from skimage.filters import frangi

    # Apply Frangi filter
    vessel_map = frangi(
        image,
        sigmas=scales,
        alpha=0.5,
        beta=0.5,
        gamma=15,
        black_ridges=False
    )

    # Normalize
    vessel_map = (vessel_map - vessel_map.min()) / (vessel_map.max() - vessel_map.min() + 1e-8)

    return vessel_map.astype(np.float32)
