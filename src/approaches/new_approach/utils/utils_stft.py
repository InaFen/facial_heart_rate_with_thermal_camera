import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt


def get_frequencies_over_time_spectrogram(signal_each_roi_seperate, fps, window_seconds, roi_names=None, save_path=None, plot=True):
    """
    Calculate frequencies for each second (or time window) of the signal using STFT.
    
    Args:
        signal_each_roi_seperate: Array of shape (num_timepoints, num_rois)
        fps: Frames per second
        window_seconds: Size of sliding window in seconds (default 1.0)
        roi_names: Optional list of ROI names for plotting
        save_path: Optional path to save the plot
        plot: Whether to generate and display the spectrogram (default True)
    
    Returns:
        time_points: Array of time points in seconds
        frequencies: Array of frequency bins in Hz
        spectrograms: List of spectrograms (one per ROI), each of shape (num_freqs, num_times)
        dominant_frequencies_over_time: Array of shape (num_rois, num_windows) with dominant frequencies
    """
    num_timepoints, num_rois = signal_each_roi_seperate.shape
    
    # Calculate window size in frames
    nperseg = int(window_seconds * fps)
    noverlap = nperseg // 2  # 50% overlap
    
    if nperseg > num_timepoints:
        raise ValueError(f"Window size ({nperseg} frames) is larger than signal length ({num_timepoints} frames).")
    
    # Initialize storage for spectrograms and dominant frequencies
    spectrograms = []
    dominant_frequencies_over_time = []
    
    # Process each ROI
    for roi_idx in range(num_rois):
        signal_roi = signal_each_roi_seperate[:, roi_idx]
        
        # Apply high-pass filter to remove very low frequencies (< 0.4 Hz)
        print("[WARNING STFT]: Applying high-pass filter at 0.4 Hz to remove DC and very low frequencies.")
        sos_hp = signal.butter(4, 0.4, btype='high', fs=fps, output='sos')
        signal_roi = signal.sosfilt(sos_hp, signal_roi)
        
        # Compute STFT
        frequencies, time_points, Zxx = signal.stft(
            signal_roi,
            fs=fps,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=True
        )
        
        # Compute magnitude spectrogram
        spectrogram = np.abs(Zxx)
        spectrograms.append(spectrogram)
        
        # Find dominant frequency at each time point
        # Exclude DC component (frequency 0) to avoid always selecting it
        # argmax along axis 0 (frequency axis) to find frequency bin with max power
        spectrogram_no_dc = spectrogram[1:, :]  # Exclude first frequency (0 Hz)
        frequencies_no_dc = frequencies[1:]
        
        # Debug: Print info about spectrogram
        #print(f"ROI {roi_idx}: Signal shape={signal_roi.shape}, Spectrogram shape={spectrogram.shape}")
        #print(f"  Frequency range: {frequencies_no_dc.min():.3f} - {frequencies_no_dc.max():.3f} Hz")
        #print(f"  Power range: {spectrogram_no_dc.min():.2e} - {spectrogram_no_dc.max():.2e}")
        
        dominant_freq_indices = np.argmax(spectrogram_no_dc, axis=0)
        dominant_freqs = frequencies_no_dc[dominant_freq_indices]
        dominant_frequencies_over_time.append(dominant_freqs)
        
        print(f"  Dominant frequencies: min={dominant_freqs.min():.3f}, max={dominant_freqs.max():.3f}, mean={dominant_freqs.mean():.3f} Hz")
    
    # Convert to arrays
    dominant_frequencies_over_time = np.array(dominant_frequencies_over_time)  # shape: (num_rois, num_times)
    
    # Plot if requested
    if plot:
        plot_spectrogram_and_dominant_frequency(
            time_points,
            frequencies, 
            spectrograms,
            dominant_frequencies_over_time,
            roi_names=roi_names, 
            save_path=save_path,
            plot_type='spectrogram'
        )
    
    return time_points, frequencies, spectrograms, dominant_frequencies_over_time


def plot_spectrogram_and_dominant_frequency(time_points, frequencies, spectrograms, dominant_frequencies_over_time, roi_names=None, save_path=None, plot_type='spectrogram'):
    """
    Plot frequency-time spectrogram or scalogram with dominant frequency overlay for each ROI.
    
    Args:
        time_points: Array of time points in seconds
        frequencies: Array of frequency bins in Hz
        spectrograms: List of spectrograms/scalograms (one per ROI), each of shape (num_freqs, num_times)
        dominant_frequencies_over_time: Array of shape (num_rois, num_times) with dominant frequencies
        roi_names: Optional list of ROI names
        save_path: Optional path to save the plot
        plot_type: Either 'spectrogram' (STFT) or 'scalogram' (CWT) for appropriate labeling
    """
    num_rois = len(spectrograms)
    
    fig, axes = plt.subplots(num_rois, 1, figsize=(14, 4 * num_rois), sharex=True)
    
    # Handle single ROI case
    if num_rois == 1:
        axes = [axes]
    
    for roi_idx in range(num_rois):
        ax = axes[roi_idx]
        
        # Plot spectrogram
        spectrogram = spectrograms[roi_idx]
        
        # Use log scale for better visualization
        spectrogram_db = 10 * np.log10(spectrogram + 1e-10)
        
        # Create spectrogram plot
        im = ax.pcolormesh(
            time_points, 
            frequencies, 
            spectrogram_db,
            shading='gouraud',
            cmap='viridis'
        )
        
        # Overlay dominant frequency as a line
        ax.plot(
            time_points, 
            dominant_frequencies_over_time[roi_idx, :], 
            'r-', 
            linewidth=2, 
            label='Dominant Frequency',
            alpha=0.8
        )
        
        # Set labels and title
        ax.set_ylabel('Frequency (Hz)', fontsize=11)
        
        # Set frequency limits (adjust based on data range)
        freq_max_data = dominant_frequencies_over_time[roi_idx, :].max()
        if freq_max_data < 5:
            ax.set_ylim(0, min(5, frequencies.max()))
        else:
            ax.set_ylim(0, frequencies.max())
        
        # Set title
        if roi_names is not None and roi_idx < len(roi_names):
            title = f"{plot_type.capitalize()}: {roi_names[roi_idx]}"
        else:
            title = f"{plot_type.capitalize()}: ROI {roi_idx + 1}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', fontsize=10)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
        
        # Grid
        ax.grid(True, alpha=0.2, color='white', linestyle=':')
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"{plot_type.capitalize()} saved to {save_path}")
    
    plt.show()


def get_frequencies_over_time_cwt(signal_each_roi_seperate, fps, roi_names=None, save_path=None, plot=True, 
                                  wavelet='morl', freq_min=0.3, freq_max=5.0, num_freqs=60):
    """
    Calculate a time-frequency spectrogram (scalogram) using Continuous Wavelet Transform (CWT) via PyWavelets.

    Args:
        signal_each_roi_seperate: Array of shape (num_timepoints, num_rois)
        fps: Frames per second
        roi_names: Optional list of ROI names for plotting
        save_path: Optional path to save the plot
        plot: Whether to generate and display the scalogram (default True)
        wavelet: Continuous wavelet name for PyWavelets (e.g., 'morl' for Morlet)
        freq_min: Minimum frequency (Hz) for analysis
        freq_max: Maximum frequency (Hz) for analysis
        num_freqs: Number of frequency bins between freq_min and freq_max

    Returns:
        time_points: Array of time points in seconds
        frequencies: Array of frequency bins in Hz (PyWavelets pseudo-frequencies)
        scalograms: List of scalograms (one per ROI), each of shape (num_freqs, num_times)
        dominant_frequencies_over_time: Array of shape (num_rois, num_times) with dominant frequencies
    """

    num_timepoints, num_rois = signal_each_roi_seperate.shape

    # Time axis in seconds for CWT output (one coefficient per sample)
    time_points = np.arange(num_timepoints) / fps

    # Define target frequency axis (Hz)
    frequencies = np.linspace(freq_min, freq_max, num_freqs)

    # Map desired frequencies to PyWavelets scales using scale2frequency
    sampling_period = 1.0 / fps
    scales = pywt.scale2frequency(wavelet, 1.0) / (frequencies * sampling_period)

    scalograms = []
    dominant_frequencies_over_time = []

    for roi_idx in range(num_rois):
        sig_roi = signal_each_roi_seperate[:, roi_idx]

        # Optional high-pass to suppress drift and DC
        print("[WARNING CWT]: Applying high-pass filter at 0.4 Hz to reduce drift and DC.")
        sos_hp = signal.butter(4, 0.4, btype='high', fs=fps, output='sos')
        sig_roi = signal.sosfilt(sos_hp, sig_roi)

        # Compute CWT via PyWavelets: returns coeffs shape (num_scales, num_samples) and frequency array (Hz)
        cwt_coeffs, pywt_freqs = pywt.cwt(sig_roi, scales, wavelet, sampling_period=sampling_period)

        # Magnitude scalogram
        scalogram = np.abs(cwt_coeffs)
        scalograms.append(scalogram)

        # Dominant frequency per time column
        dominant_idx = np.argmax(scalogram, axis=0)  # over frequency axis
        dominant_freqs = pywt_freqs[dominant_idx]

        print(f"  [CWT ROI {roi_idx}] Dominant frequencies: min={dominant_freqs.min():.3f}, max={dominant_freqs.max():.3f}, mean={dominant_freqs.mean():.3f} Hz")
        dominant_frequencies_over_time.append(dominant_freqs)

    dominant_frequencies_over_time = np.array(dominant_frequencies_over_time)  # (num_rois, num_times)

    if plot:
        plot_spectrogram_and_dominant_frequency(
            time_points,
            pywt_freqs,
            scalograms,
            dominant_frequencies_over_time,
            roi_names=roi_names,
            save_path=save_path,
            plot_type='scalogram'
        )

    return time_points, pywt_freqs, scalograms, dominant_frequencies_over_time

