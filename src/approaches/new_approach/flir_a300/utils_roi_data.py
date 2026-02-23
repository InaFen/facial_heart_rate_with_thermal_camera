from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

from utils_data_extraction import extract_frame_number_from_csv, read_thermal_csv

def plot_roi_mean_temperature(frame_numbers: list, mean_temperatures: list, fps: float, ecg_signal: np.ndarray = None, ecg_fs: float = None, ecg_units: str = None, offset: float = 6.6) -> None:
    """
    Plot the mean temperature of ROI pixels over time with optional ground truth ECG data.
    
    Args:
        frame_numbers: List of frame numbers
        mean_temperatures: List of mean temperatures for each frame
        fps: Frames per second for time conversion
        ecg_signal: Optional ECG signal array for ground truth comparison
        ecg_fs: Sampling frequency of ECG signal (Hz)
        ecg_units: Units of ECG signal (e.g., 'mV')
    """
    if fps <= 0:
        raise ValueError("fps must be greater than zero.")
    
    # Convert frame numbers to time
    times = [(frame_num - 1) / fps for frame_num in frame_numbers]

    # Cut ECG signal if provided
    if ecg_signal is not None and ecg_fs is not None:
        cut_samples = int(ecg_fs * offset)
        ecg_signal = ecg_signal[cut_samples:]
        ecg_times = np.arange(len(ecg_signal)) / ecg_fs #+ offset
        num_subplots = 4
    else:
        ecg_times = None
        num_subplots = 3

    # Create figure (NO sharex here — we’ll manually share only time plots)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 3 * num_subplots))

    if num_subplots == 4:
        ax_ecg, ax1, ax2, ax3 = axes

        # Share x-axis between ECG and time-domain temperature plots
        ax1.sharex(ax_ecg)
        ax2.sharex(ax_ecg)
    else:
        ax1, ax2, ax3 = axes
        ax_ecg = None

    # -------------------------------------------------
    # Plot 0: ECG
    # -------------------------------------------------
    if ax_ecg is not None:
        ax_ecg.plot(
            ecg_times,
            ecg_signal,
            linewidth=0.5,
            alpha=0.7,
            label="ECG Signal"
        )
        ax_ecg.set_ylabel(f"Amplitude ({ecg_units})")
        ax_ecg.set_title("Ground Truth ECG Signal")
        ax_ecg.legend()
        ax_ecg.grid(True, alpha=0.3)

    # -------------------------------------------------
    # Plot 1: Mean Temperature
    # -------------------------------------------------
    ax1.plot(times, mean_temperatures, marker='o', linewidth=1)
    ax1.set_ylabel("Mean Temp (°C)")
    ax1.set_title("ROI Mean Temperature Over Time")
    ax1.grid(True, alpha=0.3)

    # -------------------------------------------------
    # Plot 2: Frame-to-frame Difference
    # -------------------------------------------------
    temp_diffs = [0] + [
        mean_temperatures[i] - mean_temperatures[i - 1]
        for i in range(1, len(mean_temperatures))
    ]

    ax2.plot(times, temp_diffs, marker='o', linewidth=1)
    ax2.set_ylabel("Δ Mean Temp (°C)")
    ax2.set_title("Difference to Previous Frame")
    ax2.grid(True, alpha=0.3)

    # -------------------------------------------------
    # Plot 3: FFT (independent x-axis)
    # -------------------------------------------------
    mean_centered = np.array(mean_temperatures, dtype=float) - np.mean(mean_temperatures)
    frequencies = np.fft.rfftfreq(len(times), d=1.0 / fps)
    fft_means = np.abs(np.fft.rfft(mean_centered))

    ax3.plot(frequencies, fft_means, label="Mean Temp FFT")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("FFT of Mean Temperature")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # X-label only on lowest time-domain axis
    if ax_ecg is not None:
        ax2.set_xlabel("Time (s)")
    else:
        ax2.set_xlabel("Time (s)")

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
