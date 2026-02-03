#!/usr/bin/env python3
"""
thermal_video_generator.py

Loads raw Y16 .gray frames from a directory, processes them (removes telemetry lines,
applies colormap), and outputs a thermal video in MP4 or other format.

Usage:
    python3 thermal_video_generator.py --input-dir /path/to/gray/frames --output video.mp4 [--fps 9] [--colormap turbo]
"""

import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def load_gray_frame(filepath, width=160, height=122, drop_telemetry=True):
    """
    Load a .gray file as a Y16 (16-bit little-endian grayscale) frame.
    
    Args:
        filepath: path to .gray file
        width: frame width in pixels (default 160)
        height: total frame height including telemetry (default 122)
        drop_telemetry: if True, remove the last 2 telemetry lines (default True)
    
    Returns:
        numpy array of shape (image_height, width) with 16-bit values
    """
    try:
        raw = np.fromfile(filepath, dtype=np.uint16)
        if raw.size == 0:
            print(f"Warning: {filepath} is empty.", file=sys.stderr)
            return None
        
        frame = raw.reshape(height, width)
        
        if drop_telemetry and height > 120:
            # Lepton image is 120 rows; last 2 are telemetry
            frame = frame[:-2]
        
        return frame
    except Exception as e:
        print(f"Error loading {filepath}: {e}", file=sys.stderr)
        return None


def frame_to_8bit_grayscale(frame):
    """
    Convert a 16-bit grayscale frame to an 8-bit grayscale frame for video.
    
    Args:
        frame: 16-bit numpy array
    
    Returns:
        8-bit grayscale frame (repeated to 3 channels for BGR video codec)
    """
    if frame is None:
        return None
    
    # Normalize to 8-bit
    frame_min = frame.min()
    frame_max = frame.max()
    if frame_max > frame_min:
        normalized = ((frame.astype(np.float32) - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(frame, dtype=np.uint8)
    
    # Convert to BGR (3-channel) for video codec: replicate grayscale to R, G, B
    bgr_frame = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    
    return bgr_frame

def plot_greyscale_frame(path_frame):
    """
    Load and display a .gray frame as grayscale image.
    
    Args:
        path_frame: path to .gray file
    """
    frame_16bit = load_gray_frame(path_frame, width=160, height=122, drop_telemetry=True)
    if frame_16bit is None:
        return
    
    frame_8bit = frame_to_8bit_grayscale(frame_16bit)
    if frame_8bit is None:
        return
    
    cv2.imshow("Grayscale Frame", frame_8bit)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_thermal_video(input_dir, output_file, fps=9, width=160, height=120):
    """
    Load .gray frames and encode them into a video (grayscale, no colormap).
    
    Args:
        input_dir: directory containing .gray files
        output_file: output video path (e.g., video.mp4)
        fps: frames per second (default 9, typical for Lepton)
        width: frame width (default 160)
        height: frame height without telemetry (default 120)
    """
    # Find all .gray files sorted by name (which includes timestamp)
    gray_files = sorted(glob.glob(os.path.join(input_dir, "*.gray")))
    
    if not gray_files:
        print(f"Error: No .gray files found in {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(gray_files)} frames in {input_dir}")
    print(f"Output: {output_file} at {fps} fps (grayscale, no colormap)")
    
    # Initialize video writer
    # Use H.264 codec (mp4v) for MP4 container
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Failed to open video writer for {output_file}", file=sys.stderr)
        sys.exit(1)
    
    frame_count = 0
    for i, filepath in enumerate(gray_files):
        print(f"  [{i+1}/{len(gray_files)}] Processing {os.path.basename(filepath)}...", end="\r")
        
        # Load raw frame
        frame_16bit = load_gray_frame(filepath, width=width, height=height+2, drop_telemetry=True)
        if frame_16bit is None:
            continue
        
        # Convert to 8-bit grayscale frame (BGR for video codec)
        frame_8bit_bgr = frame_to_8bit_grayscale(frame_16bit)
        if frame_8bit_bgr is None:
            continue

        # Flip vertically to match desired orientation
        frame_8bit_bgr = cv2.flip(frame_8bit_bgr, 0)
        
        # Write to video
        out.write(frame_8bit_bgr)
        frame_count += 1
    
    out.release()
    print(f"\n Video written: {output_file} ({frame_count} frames)")


def main():

    input_dir = "data/lepton/processed_lepton_data/03022026_3_interpolated"
    output = "data/recorded_videos/thermal_video_from_y16_frames/03022026_3.mp4"
    fps = 8.767717191929298 # based on findings in get_consistent_fps_for_gray_images.py
    width = 160
    height = 120

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.dirname(output)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    #plot_greyscale_frame("data/lepton/processed_lepton_data/29122025_interpolated/388438_interpolated.gray")

    create_thermal_video(
        input_dir,
        output,
        fps=fps,
        width=width,
        height=height
    )


if __name__ == "__main__":
    main()
