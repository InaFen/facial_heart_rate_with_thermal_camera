"""
FFT utilities for computing and visualizing frequency domain analysis of ROI signals.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_fft_each_roi(signal_each_roi_seperate, roi_names, fps, show_peaks=True, peak_threshold=0.1):
    """
    Compute and plot FFT for each ROI in separate subplots.

    Parameters
    ----------
    signal_each_roi_seperate : list of arrays or array
        Signal data for each ROI. Can be:
        - List of arrays where each array is shape (T,) for T time samples
        - 2D array of shape (T, N) where T is time samples and N is number of ROIs
    
    roi_names : list of str
        Names of each ROI.
    
    fps : float
        Frames per second (sampling rate).
    
    show_peaks : bool, optional
        If True, highlight the dominant frequency peak. Default is True.
    
    peak_threshold : float, optional
        Threshold for displaying secondary peaks. Default is 0.1 (10% of max).
    """
    
    # Convert to list of arrays if 2D array
    if isinstance(signal_each_roi_seperate, np.ndarray) and signal_each_roi_seperate.ndim == 2:
        signals = [signal_each_roi_seperate[:, i] for i in range(signal_each_roi_seperate.shape[1])]
    else:
        signals = signal_each_roi_seperate
    
    num_rois = len(signals)
    
    # Calculate grid size for subplots
    cols = min(3, num_rois)
    rows = (num_rois + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if num_rois == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (signal, roi_name) in enumerate(zip(signals, roi_names)):
        # Compute FFT
        n = len(signal)
        fft_values = np.fft.fft(signal)
        fft_magnitude = np.abs(fft_values[:n // 2])  # Take only positive frequencies
        
        # Compute frequency axis
        frequencies = np.fft.fftfreq(n, d=1/fps)[:n // 2]
        
        # Plot
        ax = axes[idx]
        ax.plot(frequencies, fft_magnitude, linewidth=1.5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'FFT - {roi_name}')
        ax.grid(True, alpha=0.3)
        
        if show_peaks and len(fft_magnitude) > 0:
            # Find dominant peak
            peak_idx = np.argmax(fft_magnitude)
            peak_freq = frequencies[peak_idx]
            peak_mag = fft_magnitude[peak_idx]
            
            ax.plot(peak_freq, peak_mag, 'r*', markersize=15, label=f'Peak: {peak_freq:.2f} Hz')
            ax.legend()
    
    # Hide unused subplots
    for idx in range(num_rois, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def get_dominant_frequency_per_roi(signal_each_roi_seperate, roi_names, fps):
    """
    Extract the dominant frequency for each ROI.

    Parameters
    ----------
    signal_each_roi_seperate : list of arrays or array
        Signal data for each ROI.
    
    roi_names : list of str
        Names of each ROI.
    
    fps : float
        Frames per second (sampling rate).
    
    Returns
    -------
    dict
        Dictionary with ROI names as keys and dominant frequencies (in Hz) as values.
    
    dict
        Dictionary with ROI names as keys and heart rate estimates (in BPM) as values.
    """
    
    # Convert to list of arrays if 2D array
    if isinstance(signal_each_roi_seperate, np.ndarray) and signal_each_roi_seperate.ndim == 2:
        signals = [signal_each_roi_seperate[:, i] for i in range(signal_each_roi_seperate.shape[1])]
    else:
        signals = signal_each_roi_seperate
    
    dominant_frequencies = {}
    heart_rates = {}
    
    for signal, roi_name in zip(signals, roi_names):
        # Compute FFT
        n = len(signal)
        fft_values = np.fft.fft(signal)
        fft_magnitude = np.abs(fft_values[:n // 2])
        
        # Compute frequency axis
        frequencies = np.fft.fftfreq(n, d=1/fps)[:n // 2]
        
        # Find dominant peak
        peak_idx = np.argmax(fft_magnitude)
        peak_freq = frequencies[peak_idx]
        
        dominant_frequencies[roi_name] = peak_freq
        heart_rates[roi_name] = peak_freq * 60  # Convert Hz to BPM
    
    return dominant_frequencies, heart_rates


def print_dominant_frequencies(dominant_frequencies, heart_rates):
    """
    Print dominant frequencies and heart rate estimates for all ROIs.

    Parameters
    ----------
    dominant_frequencies : dict
        Dictionary with ROI names as keys and dominant frequencies (Hz) as values.
    
    heart_rates : dict
        Dictionary with ROI names as keys and heart rate estimates (BPM) as values.
    """
    print("\n" + "="*60)
    print("Dominant Frequencies and Heart Rate Estimates per ROI")
    print("="*60)
    
    for roi_name in dominant_frequencies.keys():
        freq = dominant_frequencies[roi_name]
        hr = heart_rates[roi_name]
        print(f"{roi_name:30s} | Freq: {freq:7.3f} Hz | HR: {hr:7.2f} BPM")
    
    print("="*60 + "\n")
