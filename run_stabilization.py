#!/usr/bin/env python3
"""
CLI entry point for retinal video stabilization.

Usage:
    python run_stabilization.py input.mp4 output.mp4
    python run_stabilization.py input.mp4 output.mp4 --device cuda
    python run_stabilization.py input.mp4 output.mp4 --no-crop --no-inpaint
"""

from retina_stabilizer.pipeline import main

if __name__ == "__main__":
    main()
