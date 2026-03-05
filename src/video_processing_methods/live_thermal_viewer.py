#!/usr/bin/env python3
"""
Live thermal viewer - displays .gray frames as they are captured.

Usage:
    python src/video_processing_methods/live_thermal_viewer.py /home/arbeit/Dokumente/master_thesis/data/lepton/raw_lepton_data/03022026/

This will monitor the directory for new .gray files and display them in real-time.
Press 'q' to quit, 's' to save a screenshot of the current frame.
"""

import cv2
import numpy as np
import sys
import os
import time
from pathlib import Path


def read_gray_frame(filepath, width=160, height=122):
    """Read a .gray file (Y16 format) and return as numpy array."""
    try:
        # Read raw 16-bit data
        with open(filepath, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint16)
        
        # Reshape to image dimensions
        if len(data) == width * height:
            frame = data.reshape(height, width)
            return frame
        else:
            print(f"Warning: unexpected frame size {len(data)} (expected {width * height})")
            return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def convert_to_celsius(frame_uint16):
    """Convert TLinear format (Kelvin * 100) to Celsius."""
    # TLinear format: value / 100 = Kelvin
    kelvin = frame_uint16.astype(np.float32) / 100.0
    celsius = kelvin - 273.15
    return celsius


def normalize_for_display(frame_celsius, vmin=None, vmax=None):
    """Normalize temperature data to 8-bit for display."""
    if vmin is None:
        vmin = np.percentile(frame_celsius, 1)  # Auto-adjust for contrast
    if vmax is None:
        vmax = np.percentile(frame_celsius, 99)
    
    # Clip and normalize to 0-255
    normalized = np.clip((frame_celsius - vmin) / (vmax - vmin) * 255, 0, 255)
    return normalized.astype(np.uint8)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    watch_dir = Path(sys.argv[1])
    if not watch_dir.exists():
        print(f"Error: Directory {watch_dir} does not exist")
        sys.exit(1)
    
    print(f"Monitoring {watch_dir} for new .gray files...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    width = 160
    height = 122
    
    # Track which files we've already displayed
    seen_files = set()
    
    # Temperature range for normalization (auto-adjusts initially)
    temp_min = None
    temp_max = None
    auto_range = True
    
    cv2.namedWindow('Thermal Viewer', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Thermal Viewer', 640, 488)  # 4x scale
    
    last_frame_display = None
    
    while True:
        # Get all .gray files in directory
        gray_files = sorted(watch_dir.glob('*.gray'))
        
        # Find new files
        new_files = [f for f in gray_files if f not in seen_files]
        
        if new_files:
            # Display the newest file
            latest_file = new_files[-1]
            seen_files.update(new_files)
            
            frame_raw = read_gray_frame(latest_file, width, height)
            if frame_raw is not None:
                # Convert to Celsius
                frame_celsius = convert_to_celsius(frame_raw)
                
                # Get temperature range for the image area (exclude telemetry rows)
                image_area = frame_celsius[:120, :]  # First 120 rows are image
                
                if auto_range:
                    temp_min = np.percentile(image_area, 1)
                    temp_max = np.percentile(image_area, 99)
                
                # Normalize for display
                frame_normalized = normalize_for_display(frame_celsius, temp_min, temp_max)
                
                # Apply colormap for better visualization
                frame_colored = cv2.applyColorMap(frame_normalized, cv2.COLORMAP_JET)
                
                frame_display = cv2.flip(frame_colored, 0)

                # Add temperature info overlay (kept upright)
                min_temp = np.min(image_area)
                max_temp = np.max(image_area)
                mean_temp = np.mean(image_area)
                center_row = image_area.shape[0] // 2
                center_col = image_area.shape[1] // 2
                center_temp = image_area[center_row, center_col]
                
                info_text = [
                    f"File: {latest_file.name}",
                    f"Center: {center_temp:.2f}C",
                    f"Range: {temp_min:.1f}C - {temp_max:.1f}C"
                ]
                
                y_offset = 15
                for text in info_text:
                    cv2.putText(frame_display, text, (5, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 12
                cv2.imshow('Thermal Viewer', frame_display)
                last_frame_display = frame_display
        
        # Handle keyboard input
        key = cv2.waitKey(50) & 0xFF  # 50ms refresh rate
        
        if key == ord('q'):
            print("Quit requested")
            break
        elif key == ord('s') and last_frame_display is not None:
            # Save screenshot
            screenshot_path = watch_dir / f"screenshot_{int(time.time())}.png"
            cv2.imwrite(str(screenshot_path), last_frame_display)
            print(f"Screenshot saved to {screenshot_path}")
        elif key == ord('a'):
            # Toggle auto-range
            auto_range = not auto_range
            print(f"Auto-range: {'ON' if auto_range else 'OFF'}")
        elif key == ord('r'):
            # Reset temperature range
            temp_min = None
            temp_max = None
            auto_range = True
            print("Temperature range reset")
        
        # Small delay to avoid CPU spinning
        time.sleep(0.05)
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
