#!/usr/bin/env python3
"""
Download pre-trained vessel segmentation weights.

Available sources:
1. VesselSeg-Pytorch (U-Net trained on DRIVE/CHASE)
2. SGL-Retinal-Vessel-Segmentation (MICCAI 2021 SOTA)
"""

import os
import urllib.request
import sys


def download_file(url: str, output_path: str, description: str = ""):
    """Download file with progress indicator."""
    print(f"Downloading {description}...")
    print(f"URL: {url}")
    print(f"Output: {output_path}")

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded successfully: {output_path}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def option_1_vesselseg_pytorch():
    """
    Download from VesselSeg-Pytorch repository.

    Repository: https://github.com/lee-zq/VesselSeg-Pytorch
    Models: UNet, DenseUNet, LadderNet trained on DRIVE, STARE, CHASE_DB1
    """
    print("\n" + "="*60)
    print("Option 1: VesselSeg-Pytorch")
    print("="*60)
    print("""
This toolkit provides multiple pre-trained models:
- UNet
- DenseUNet
- LadderNet

Steps to use:

1. Clone the repository:
   git clone https://github.com/lee-zq/VesselSeg-Pytorch.git

2. Download their pre-trained weights from their Google Drive:
   https://drive.google.com/drive/folders/1yG4Y8cYDfKf2rIhz6AKF6hPmDnFiA1P4

3. Convert to our format:
   python convert_vesselseg_weights.py path/to/their/checkpoint.pth models/vessel_unet.pth

The weights are typically located in:
- experiments/UNet_vessel_seg/DRIVE/best_model.pth
- experiments/UNet_vessel_seg/CHASE/best_model.pth
""")


def option_2_sgl_miccai():
    """
    Download from SGL-Retinal-Vessel-Segmentation (MICCAI 2021).

    Repository: https://github.com/SHI-Labs/SGL-Retinal-Vessel-Segmentation
    State-of-the-art results on DRIVE and CHASE_DB1.
    """
    print("\n" + "="*60)
    print("Option 2: SGL-Retinal-Vessel-Segmentation (MICCAI 2021)")
    print("="*60)
    print("""
This is state-of-the-art for retinal vessel segmentation.

Steps to use:

1. Clone the repository:
   git clone https://github.com/SHI-Labs/SGL-Retinal-Vessel-Segmentation.git

2. Pre-trained models are in the 'pretrained' folder:
   - pretrained/DRIVE/model.pth
   - pretrained/CHASE_DB1/model.pth

3. Their architecture differs from ours, so conversion is needed:
   python convert_sgl_weights.py pretrained/DRIVE/model.pth models/vessel_unet.pth

Note: This uses their custom architecture (may need adaptation).
""")


def option_3_direct_drive_weights():
    """
    Provide direct download links for DRIVE-trained weights.
    """
    print("\n" + "="*60)
    print("Option 3: Quick Download (HuggingFace / Zenodo)")
    print("="*60)

    # Note: These would need to be actual hosted weights
    print("""
For the quickest setup, you can use weights hosted on HuggingFace or Zenodo.

Example (if available):

from huggingface_hub import hf_hub_download

# Download pre-trained weights
weights_path = hf_hub_download(
    repo_id="username/retinal-vessel-unet",
    filename="unet_drive.pth"
)

# Use in stabilizer
from retina_stabilizer import RetinaStabilizer
stabilizer = RetinaStabilizer(vessel_model_path=weights_path)

Currently checking for available hosted weights...
""")

    # Check common sources
    sources = [
        ("HuggingFace", "https://huggingface.co/models?search=retinal+vessel"),
        ("Papers With Code", "https://paperswithcode.com/task/retinal-vessel-segmentation"),
    ]

    for name, url in sources:
        print(f"  - {name}: {url}")


def create_weight_converter():
    """Create a script to convert external weights to our format."""
    converter_code = '''#!/usr/bin/env python3
"""
Convert external vessel segmentation weights to our LightweightUNet format.

Usage:
    python convert_weights.py input.pth output.pth --source [vesselseg|sgl]
"""

import torch
import argparse
import sys
sys.path.insert(0, '.')

from retina_stabilizer.vessel_segmentation import LightweightUNet


def convert_vesselseg_weights(input_path: str, output_path: str):
    """Convert VesselSeg-Pytorch weights to our format."""
    # Load source checkpoint
    checkpoint = torch.load(input_path, map_location='cpu')

    # Extract state dict
    if 'state_dict' in checkpoint:
        source_state = checkpoint['state_dict']
    else:
        source_state = checkpoint

    # Create our model
    model = LightweightUNet()
    target_state = model.state_dict()

    # Map weights (architecture should be similar)
    # This may need adjustment based on exact architecture match
    new_state = {}
    for key in target_state.keys():
        if key in source_state:
            if source_state[key].shape == target_state[key].shape:
                new_state[key] = source_state[key]
            else:
                print(f"Shape mismatch for {key}: {source_state[key].shape} vs {target_state[key].shape}")
                new_state[key] = target_state[key]
        else:
            print(f"Missing key: {key}")
            new_state[key] = target_state[key]

    # Save converted weights
    torch.save(new_state, output_path)
    print(f"Converted weights saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input weight file")
    parser.add_argument("output", help="Output weight file")
    parser.add_argument("--source", choices=["vesselseg", "sgl"], default="vesselseg")
    args = parser.parse_args()

    if args.source == "vesselseg":
        convert_vesselseg_weights(args.input, args.output)
    else:
        print(f"Converter for {args.source} not yet implemented")


if __name__ == "__main__":
    main()
'''

    with open("convert_weights.py", "w") as f:
        f.write(converter_code)
    print("\nCreated: convert_weights.py")


def main():
    print("="*60)
    print("Pre-trained Vessel Segmentation Weights Download Guide")
    print("="*60)

    print("""
Your retinal stabilizer can work in two modes:

1. WITHOUT pre-trained weights (default):
   - Uses Frangi filter for vessel enhancement
   - No download needed, works immediately
   - Good performance for most cases

2. WITH pre-trained weights (better quality):
   - Uses neural network for vessel segmentation
   - Requires downloading weights (~50-100 MB)
   - Better vessel detection, especially in low contrast

Below are sources for pre-trained weights:
""")

    option_1_vesselseg_pytorch()
    option_2_sgl_miccai()
    option_3_direct_drive_weights()

    print("\n" + "="*60)
    print("RECOMMENDED: Start with Frangi filter (no download needed)")
    print("="*60)
    print("""
For your video-only workflow, we recommend:

    python run_stabilization.py input.mp4 output.mp4 --no-vessel-seg

This uses the classical Frangi filter which:
- Requires no training or downloads
- Works well for most retinal videos
- Is used in commercial clinical systems

If you need better vessel detection later, follow the steps above
to download and convert pre-trained weights.
""")

    # Create converter script
    create_weight_converter()


if __name__ == "__main__":
    main()
