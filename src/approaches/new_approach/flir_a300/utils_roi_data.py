from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

from utils_data_extraction import extract_frame_number_from_csv, read_thermal_csv


def plot_roi_mean_temperature(frame_numbers: list, mean_temperatures: list, fps: float) -> None:
    """
    Plot the mean temperature of ROI pixels over time.
    
    Args:
        frame_numbers: List of frame numbers
        mean_temperatures: List of mean temperatures for each frame
        fps: Frames per second for time conversion
    """
    if fps <= 0:
        raise ValueError("fps must be greater than zero.")
    
    # Convert frame numbers to time
    times = [(frame_num - 1) / fps for frame_num in frame_numbers]
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Mean temperature over time
    ax1.plot(times, mean_temperatures, marker='o', linewidth=1, color='tab:red')
    ax1.set_ylabel('Mean Temperature (°C)')
    ax1.set_title('ROI Mean Temperature Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Temperature difference to previous frame
    temp_diffs = [0] + [mean_temperatures[i] - mean_temperatures[i-1] for i in range(1, len(mean_temperatures))]
    ax2.plot(times, temp_diffs, marker='o', linewidth=1, color='tab:blue')
    ax2.set_ylabel('Δ Mean Temp (°C)')
    ax2.set_title('Difference in Mean Temperature to Previous Frame')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: FFT of mean temperature
    mean_centered = np.array(mean_temperatures, dtype=float) - float(np.mean(mean_temperatures))
    frequencies = np.fft.rfftfreq(len(times), d=1.0 / fps)
    fft_means = np.abs(np.fft.rfft(mean_centered))
    
    ax3.plot(frequencies, fft_means, label='Mean Temp FFT', color='tab:green')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('FFT of Mean Temperature')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    # Set x-axis label for the first plot
    ax1.set_xlabel('Time (s)')
    
    fig.tight_layout()
    plt.show()


def compute_roi_mean_temperature_all_frames(frame_dir: str | Path, roi_pixels: list) -> tuple:
    """
    Compute the mean temperature for ROI pixels across all frames in a directory.
    
    Args:
        frame_dir: Directory containing thermal CSV files
        roi_pixels: List of (x, y) pixel coordinates within the ROI
        
    Returns:
        Tuple of (frame_numbers, mean_temperatures) where:
        - frame_numbers: List of frame numbers in order
        - mean_temperatures: List of mean temperatures for each frame
    """
    frame_dir = Path(frame_dir)
    csv_files = sorted(frame_dir.glob('Rec-temp-*.csv'), key=extract_frame_number_from_csv)
    
    if not csv_files:
        raise ValueError(f"No thermal frames found in {frame_dir}")
    
    frame_numbers = []
    mean_temperatures = []
    
    print(f"Computing ROI mean temperatures for {len(csv_files)} frames...")
    
    for csv_file in csv_files:
        try:
            # Extract frame number
            frame_num = extract_frame_number_from_csv(csv_file)
            
            # Read frame data
            frame_data = read_thermal_csv(csv_file)
            
            # Extract ROI pixel values
            roi_values = []
            for x, y in roi_pixels:
                if 0 <= y < frame_data.shape[0] and 0 <= x < frame_data.shape[1]:
                    roi_values.append(frame_data[y, x])
            
            if roi_values:
                mean_temp = np.mean(roi_values)
                frame_numbers.append(frame_num)
                mean_temperatures.append(mean_temp)
        
        except Exception as e:
            print(f"Warning: Could not process frame from {csv_file.name}: {e}")
            continue
    
    if not mean_temperatures:
        raise ValueError("No valid mean temperatures computed")
    
    print(f"Successfully computed mean temperatures for {len(mean_temperatures)} frames")
    return frame_numbers, mean_temperatures
