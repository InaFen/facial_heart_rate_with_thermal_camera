import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Force interactive backend
import matplotlib.pyplot as plt
import sys
import os
from scipy import signal as scipy_signal
from scipy.signal import find_peaks

PROJECT_ROOT = "/home/arbeit/Dokumente/master_thesis/"
sys.path.insert(0, PROJECT_ROOT)
from src.approaches.new_approach.utils.utils_landmark_detection import apply_facemesh_to_thermal_video, apply_roi_masks_to_video, plot_roi_mean_signals, apply_roi_masks_to_gray_frames, plot_roi_temperature_means
from src.approaches.new_approach.utils.utils_detrend import detrend_signal
from src.approaches.new_approach.utils.utils_add_ground_truth import estimate_signal_offset_via_xcorr, load_ground_truth_signal_and_fs, plot_denoised_with_ground_truth_offset



def detect_face_mask(path_video, visualize=True, brightness_threshold=40):
    """
    Create a mask excluding black/very dark pixels.
    
    Args:
        path_video: Path to the video file
        visualize: If True, display the mask overlay
        brightness_threshold: Pixels below this brightness value are excluded (0-255)
        
    Returns:
        face_mask: Binary array of shape (height, width) where 1 = non-black region, 0 = black
        frame_shape: (height, width) of video frames
    """
    cap = cv2.VideoCapture(path_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path_video}")
    
    # Read first frame
    ret, first_frame = cap.read()
    cap.release()
    
    if not ret:
        raise RuntimeError(f"Cannot read first frame from: {path_video}")
    
    height, width = first_frame.shape[:2]
    
    # Convert to grayscale for brightness analysis
    if first_frame.ndim == 3:
        gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    else:
        gray_frame = first_frame
        rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2RGB)
    
    print(f"Creating mask from non-black pixels...")
    print(f"  Frame shape: {gray_frame.shape}, dtype: {gray_frame.dtype}, range: [{gray_frame.min()}, {gray_frame.max()}]")
    
    # Create mask: 1 for pixels above threshold, 0 for black/dark pixels
    face_mask = (gray_frame > brightness_threshold).astype(np.uint8)
    
    num_valid_pixels = np.sum(face_mask)
    num_black_pixels = (height * width) - num_valid_pixels
    print(f"  Valid pixels: {num_valid_pixels} ({num_valid_pixels/(height*width)*100:.1f}%)")
    print(f"  Black pixels removed: {num_black_pixels} ({num_black_pixels/(height*width)*100:.1f}%)")
    
    # Visualize if requested
    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original frame
        axes[0].imshow(rgb_frame)
        axes[0].set_title('Original Frame', fontsize=12)
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(face_mask, cmap='gray')
        axes[1].set_title(f'Non-Black Mask (threshold={brightness_threshold})', fontsize=12)
        axes[1].axis('off')
        
        # Overlay
        overlay = rgb_frame.copy()
        overlay[face_mask == 0] = [0, 0, 0]  # Make excluded regions black
        axes[2].imshow(overlay)
        axes[2].set_title('Valid Region Overlay', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('debug_mask.png', dpi=100)
        print(f"  Saved visualization to: debug_mask.png")
        plt.show(block=False)
    
    return face_mask, (height, width)


def extract_pixel_signals_from_video(path_video, fps, face_mask=None):
    """
    Extract temporal signals for each pixel from a video.
    
    Args:
        path_video: Path to the video file
        face_mask: Optional binary mask (height, width) to restrict extraction to face region
        
    Returns:
        pixel_signals: Array of shape (height, width, num_frames) containing
                      the intensity value of each pixel over time
        fps: Frames per second of the video
        frame_shape: (height, width, channels) of video frames
    """
    cap = cv2.VideoCapture(path_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path_video}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read first frame to get dimensions
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read first frame from: {path_video}")
    
    height, width = first_frame.shape[:2]
    channels = first_frame.shape[2] if first_frame.ndim == 3 else 1
    
    # Convert to grayscale if RGB
    if channels == 3:
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    else:
        first_gray = first_frame
    
    # Apply face mask if provided
    if face_mask is not None:
        first_gray = first_gray * face_mask
    
    # Preallocate array for all pixel signals
    pixel_signals = np.zeros((height, width, frame_count), dtype=np.float32)
    pixel_signals[:, :, 0] = first_gray.astype(np.float32)
    
    # Read remaining frames
    frame_idx = 1
    
    if face_mask is not None:
        num_face_pixels = np.sum(face_mask)
        print(f"Extracting pixel signals from {frame_count} frames (face region only: {num_face_pixels} pixels)...")
    else:
        print(f"Extracting pixel signals from {frame_count} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Apply face mask if provided
        if face_mask is not None:
            gray = gray * face_mask
        
        pixel_signals[:, :, frame_idx] = gray.astype(np.float32)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{frame_count} frames")
    
    cap.release()
    
    # Trim if fewer frames were read than expected
    if frame_idx < frame_count:
        pixel_signals = pixel_signals[:, :, :frame_idx]
        print(f"  Warning: Expected {frame_count} frames, got {frame_idx}")
    
    print(f"Extraction complete: {pixel_signals.shape}")
    print(f"  Video: {width}×{height} pixels, {frame_idx} frames at {fps:.2f} fps")
    
    return pixel_signals, fps, (height, width, channels)


def normalize_pixel_signals(pixel_signals, method='zscore', face_mask=None):
    """
    Normalize pixel signals along the temporal dimension.
    
    Args:
        pixel_signals: Array of shape (height, width, num_frames)
        method: Normalization method:
                'zscore' - zero mean, unit variance (default)
                'minmax' - scale to [0, 1] range
                'mean' - subtract mean (center around 0)
        face_mask: Optional binary mask to normalize only face pixels
                
    Returns:
        normalized_signals: Array of same shape with normalized values
    """
    height, width, num_frames = pixel_signals.shape
    normalized_signals = np.zeros_like(pixel_signals, dtype=np.float32)
    
    print(f"Normalizing pixel signals using '{method}' method...")
    
    if method == 'zscore':
        # Z-score normalization: (x - mean) / std
        for y in range(height):
            for x in range(width):
                # Skip non-face pixels if mask is provided
                if face_mask is not None and face_mask[y, x] == 0:
                    continue
                    
                signal = pixel_signals[y, x, :]
                mean = np.mean(signal)
                std = np.std(signal)
                if std > 0:
                    normalized_signals[y, x, :] = (signal - mean) / std
                else:
                    normalized_signals[y, x, :] = signal - mean
                    
    elif method == 'minmax':
        # Min-max normalization: (x - min) / (max - min)
        for y in range(height):
            for x in range(width):
                # Skip non-face pixels if mask is provided
                if face_mask is not None and face_mask[y, x] == 0:
                    continue
                    
                signal = pixel_signals[y, x, :]
                min_val = np.min(signal)
                max_val = np.max(signal)
                if max_val > min_val:
                    normalized_signals[y, x, :] = (signal - min_val) / (max_val - min_val)
                else:
                    normalized_signals[y, x, :] = 0
                    
    elif method == 'mean':
        # Mean subtraction: x - mean
        for y in range(height):
            for x in range(width):
                # Skip non-face pixels if mask is provided
                if face_mask is not None and face_mask[y, x] == 0:
                    continue
                    
                signal = pixel_signals[y, x, :]
                normalized_signals[y, x, :] = signal - np.mean(signal)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    print(f"  Normalization complete")
    
    return normalized_signals


def apply_fft_to_pixel_signals(pixel_signals, fps):
    """
    Apply FFT to each pixel's temporal signal.
    
    Args:
        pixel_signals: Array of shape (height, width, num_frames)
        fps: Frames per second
        
    Returns:
        fft_magnitudes: Array of shape (height, width, num_freqs) with FFT magnitude
        frequencies: 1D array of frequency bins (Hz)
    """
    height, width, num_frames = pixel_signals.shape
    
    print(f"Applying FFT to {height}×{width} pixel signals...")
    
    # Compute FFT for each pixel
    fft_result = np.fft.rfft(pixel_signals, axis=2)
    fft_magnitudes = np.abs(fft_result)
    
    # Frequency bins
    frequencies = np.fft.rfftfreq(num_frames, d=1.0/fps)
    
    print(f"  FFT complete: {fft_magnitudes.shape[2]} frequency bins up to {frequencies[-1]:.2f} Hz")
    
    return fft_magnitudes, frequencies


def plot_average_frequency_spectrum(fft_magnitudes, frequencies, freq_range=(0, 5), face_mask=None):
    """
    Plot the average frequency spectrum across all pixels.
    
    Args:
        fft_magnitudes: Array of shape (height, width, num_freqs)
        frequencies: 1D array of frequency bins (Hz)
        freq_range: Tuple (min_freq, max_freq) to display
        face_mask: Optional binary mask to average only face pixels
    """
    # Average magnitude across all pixels (or only face pixels)
    if face_mask is not None:
        # Create 3D mask
        mask_3d = face_mask[:, :, np.newaxis]
        num_face_pixels = np.sum(face_mask)
        avg_spectrum = np.sum(fft_magnitudes * mask_3d, axis=(0, 1)) / num_face_pixels
        region_desc = f"face region ({num_face_pixels} pixels)"
    else:
        avg_spectrum = np.mean(fft_magnitudes, axis=(0, 1))
        region_desc = "all pixels"
    
    # Filter frequency range for plotting
    freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    freqs_plot = frequencies[freq_mask]
    spectrum_plot = avg_spectrum[freq_mask]
    
    # Find dominant frequency
    dominant_idx = np.argmax(spectrum_plot)
    dominant_freq = freqs_plot[dominant_idx]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(freqs_plot, spectrum_plot, linewidth=1.5)
    plt.axvline(dominant_freq, color='red', linestyle='--', 
                label=f'Dominant: {dominant_freq:.3f} Hz ({dominant_freq*60:.1f} bpm)')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Average Magnitude', fontsize=12)
    plt.title(f'Average Frequency Spectrum ({region_desc})', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    
    print(f"\nDominant frequency: {dominant_freq:.3f} Hz ({dominant_freq*60:.1f} bpm)")


def get_frequency_range_for_rois(roi_pixel_data, fps, freq_range=(0.5, 4.0), visualize=True, ground_truth_signal=None, gt_fs=None):
    """
    Get the frequency range (min and max) for each ROI based on FFT analysis.
    
    Args:
        roi_pixel_data: List of dictionaries (one per frame) where each dict contains
                       ROI data with structure: {roi_name: {"values": [...], "mean_value": float}}
        fps: Frames per second of the video
        freq_range: Tuple (min_freq, max_freq) to consider for analysis (Hz)
        visualize: If True, plot the frequency spectrum for each ROI
        ground_truth_signal: Optional ground truth signal array for comparison
        gt_fs: Sampling frequency of ground truth signal (Hz)
        
    Returns:
        frequency_ranges: Dictionary with ROI names as keys and tuples (min_freq, max_freq, dominant_freq) as values
    """
    frequency_ranges = {}
    
    print(f"\nAnalyzing frequency range for each ROI...")
    print(f"  Frequency analysis range: {freq_range[0]:.2f} - {freq_range[1]:.2f} Hz")
    
    # First, reorganize data from list of frame dicts to dict of ROI time series
    roi_time_series = {}
    
    if not roi_pixel_data or len(roi_pixel_data) == 0:
        print("  No ROI data available")
        return frequency_ranges
    
    # Get ROI names from first frame
    roi_names = list(roi_pixel_data[0].keys())
    
    # Collect mean values for each ROI across all frames
    for roi_name in roi_names:
        roi_time_series[roi_name] = []
        
    for frame_data in roi_pixel_data:
        for roi_name in roi_names:
            if roi_name in frame_data:
                roi_time_series[roi_name].append(frame_data[roi_name]["mean_value"])
    
    # Convert to numpy arrays
    for roi_name in roi_names:
        roi_time_series[roi_name] = np.array(roi_time_series[roi_name])
    
    num_rois = len(roi_names)
    if visualize:
        # If ground truth is provided, add one more subplot for it
        num_plots = num_rois + (1 if ground_truth_signal is not None else 0)
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4*num_plots))
        if num_plots == 1:
            axes = [axes]
    
    # Process ground truth signal if provided
    gt_dominant_freq = None
    if ground_truth_signal is not None and gt_fs is not None:
        print("\nAnalyzing ground truth signal...")
        
        # Detrend ground truth
        gt_signal_detrended = ground_truth_signal - np.mean(ground_truth_signal)
        
        # Apply FFT to ground truth
        gt_fft_result = np.fft.rfft(gt_signal_detrended)
        gt_fft_magnitude = np.abs(gt_fft_result)
        gt_frequencies = np.fft.rfftfreq(len(gt_signal_detrended), d=1.0/gt_fs)
        
        # Filter to specified frequency range
        gt_freq_mask = (gt_frequencies >= freq_range[0]) & (gt_frequencies <= freq_range[1])
        gt_freqs_filtered = gt_frequencies[gt_freq_mask]
        gt_magnitude_filtered = gt_fft_magnitude[gt_freq_mask]
        
        # Find dominant frequency
        if len(gt_magnitude_filtered) > 0:
            gt_dominant_idx = np.argmax(gt_magnitude_filtered)
            gt_dominant_freq = gt_freqs_filtered[gt_dominant_idx]
            
            print(f"  Ground Truth:")
            print(f"    Dominant frequency: {gt_dominant_freq:.3f} Hz ({gt_dominant_freq*60:.1f} bpm)")
            print(f"    Sampling rate: {gt_fs:.2f} Hz")
            print(f"    Number of samples: {len(gt_signal_detrended)}")
    
    for idx, roi_name in enumerate(roi_names):
        roi_mean_signal = roi_time_series[roi_name]
        
        if roi_mean_signal is None or len(roi_mean_signal) == 0:
            print(f"\n  {roi_name}: No data available")
            frequency_ranges[roi_name] = (None, None, None)
            continue
        
        num_frames = len(roi_mean_signal)
        
        # Detrend the signal: remove mean and linear trend
        roi_mean_signal_detrended = roi_mean_signal - np.mean(roi_mean_signal)
        # Optionally apply linear detrending: roi_mean_signal_detrended = scipy_signal.detrend(roi_mean_signal_detrended, type='linear')
        
        # Apply FFT
        fft_result = np.fft.rfft(roi_mean_signal_detrended)
        fft_magnitude = np.abs(fft_result)
        frequencies = np.fft.rfftfreq(num_frames, d=1.0/fps)
        
        # Filter to specified frequency range
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        freqs_filtered = frequencies[freq_mask]
        magnitude_filtered = fft_magnitude[freq_mask]
        
        # Find dominant frequency
        if len(magnitude_filtered) > 0:
            dominant_idx = np.argmax(magnitude_filtered)
            dominant_freq = freqs_filtered[dominant_idx]
            
            # Calculate frequency range (e.g., frequencies with magnitude > 10% of max)
            threshold = 0.1 * np.max(magnitude_filtered)
            significant_freqs = freqs_filtered[magnitude_filtered > threshold]
            
            if len(significant_freqs) > 0:
                min_freq = np.min(significant_freqs)
                max_freq = np.max(significant_freqs)
            else:
                min_freq = dominant_freq
                max_freq = dominant_freq
        else:
            dominant_freq = None
            min_freq = None
            max_freq = None
        
        frequency_ranges[roi_name] = (min_freq, max_freq, dominant_freq)
        
        # Print results
        if dominant_freq is not None:
            print(f"\n  {roi_name}:")
            print(f"    Dominant frequency: {dominant_freq:.3f} Hz ({dominant_freq*60:.1f} bpm)")
            print(f"    Frequency range: {min_freq:.3f} - {max_freq:.3f} Hz ({min_freq*60:.1f} - {max_freq*60:.1f} bpm)")
        else:
            print(f"\n  {roi_name}: No significant frequency found")
        
        # Visualize
        if visualize and dominant_freq is not None:
            ax = axes[idx]
            ax.plot(freqs_filtered, magnitude_filtered, linewidth=1.5, label=f'{roi_name}')
            ax.axvline(dominant_freq, color='red', linestyle='--', 
                      label=f'Dominant: {dominant_freq:.3f} Hz ({dominant_freq*60:.1f} bpm)')
            if min_freq != max_freq:
                ax.axvspan(min_freq, max_freq, alpha=0.2, color='green', 
                          label=f'Range: {min_freq:.3f}-{max_freq:.3f} Hz')
            
            # Add ground truth dominant frequency as reference line
            if gt_dominant_freq is not None:
                ax.axvline(gt_dominant_freq, color='blue', linestyle=':', linewidth=2,
                          label=f'GT: {gt_dominant_freq:.3f} Hz ({gt_dominant_freq*60:.1f} bpm)')
            
            ax.set_xlabel('Frequency (Hz)', fontsize=11)
            ax.set_ylabel('Magnitude', fontsize=11)
            ax.set_title(f'{roi_name} - Frequency Spectrum', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    # Plot ground truth spectrum in last subplot
    if visualize and ground_truth_signal is not None and gt_fs is not None:
        ax = axes[-1]
        ax.plot(gt_freqs_filtered, gt_magnitude_filtered, linewidth=1.5, color='blue', label='Ground Truth')
        ax.axvline(gt_dominant_freq, color='red', linestyle='--',
                  label=f'Dominant: {gt_dominant_freq:.3f} Hz ({gt_dominant_freq*60:.1f} bpm)')
        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('Magnitude', fontsize=11)
        ax.set_title('Ground Truth - Frequency Spectrum', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    if visualize:
        plt.tight_layout()
        plt.savefig('roi_frequency_ranges.png', dpi=100)
        print(f"\n  Saved visualization to: roi_frequency_ranges.png")
        plt.show(block=False)
    
    return frequency_ranges


def add_random_baseline_roi(roi_pixel_data, video_path=None, roi_mask_log=None, fps=None, roi_size=(50, 50), seed=42, amplitude_scale=1.0):
    """
    Add a synthetic random baseline ROI to the roi_pixel_data for comparison.
    Generates pure Gaussian noise similar to add_random_baseline_signal, independent of actual video content.
    
    Args:
        roi_pixel_data: List of dictionaries containing ROI data per frame
        video_path: Path to the video file (not used, kept for compatibility)
        roi_mask_log: List of ROI masks per frame (not used, kept for compatibility)
        fps: Frames per second (not used, kept for compatibility)
        roi_size: Tuple (height, width) for the random ROI region
        seed: Random seed for reproducibility
        amplitude_scale: Multiplier for the baseline standard deviation (e.g., 0.2 lowers amplitude)
        
    Returns:
        roi_pixel_data: Updated list with random baseline ROI added
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_frames = len(roi_pixel_data)
    roi_h, roi_w = roi_size
    num_pixels = roi_h * roi_w
    
    # Calculate mean and std from existing ROI data
    all_mean_values = []
    for frame_data in roi_pixel_data:
        for roi_name, roi_info in frame_data.items():
            all_mean_values.append(roi_info["mean_value"])
    
    signal_mean = np.mean(all_mean_values)
    signal_std = np.std(all_mean_values)
    
    print(f"\nAdding synthetic random baseline ROI (pure Gaussian noise)")
    print(f"  Signal statistics: mean={signal_mean:.2f}, std={signal_std:.2f}")
    print(f"  Amplitude scale: {amplitude_scale}")
    
    # Generate random baseline signal for each frame
    for frame_idx in range(num_frames):
        # Generate random noise for this frame's mean value
        random_mean_value = np.random.normal(signal_mean, signal_std * amplitude_scale)
        
        # Generate individual pixel values with some variance
        pixel_values = np.random.normal(random_mean_value, signal_std * amplitude_scale * 0.1, size=num_pixels)
        
        # Create fake coordinates (not used but kept for consistency)
        coords = [(x, y) for y in range(roi_h) for x in range(roi_w)]
        
        # Add to roi_pixel_data
        roi_pixel_data[frame_idx]["random_baseline"] = {
            "coords": coords,
            "values": pixel_values.tolist(),
            "mean_value": float(random_mean_value)
        }
    
    print(f"  Random baseline ROI added for {num_frames} frames (independent white noise)")
    
    return roi_pixel_data


def plot_roi_mean_signals_with_ground_truth(roi_pixel_data, fps, ground_truth_signal=None, gt_fs=None, plot_in_secs=True):
    """
    Plot the mean signal for each ROI across all frames, with optional ground truth overlay.
    
    Args:
        roi_pixel_data: List of dictionaries containing ROI data per frame
        fps: Frames per second of the video
        ground_truth_signal: Optional ground truth signal array for comparison
        gt_fs: Sampling frequency of ground truth signal (Hz)
        plot_in_secs: If True, plot x-axis in seconds; otherwise in frames
    """
    if len(roi_pixel_data) == 0:
        print("Error: roi_pixel_data is empty.")
        return

    # Get list of ROI names from first frame
    roi_names = list(roi_pixel_data[0].keys())

    # Prepare time axis for ROI data
    num_frames = len(roi_pixel_data)
    if plot_in_secs:
        roi_x_axis = np.array([frame / fps for frame in range(num_frames)])
    else:
        roi_x_axis = np.arange(num_frames)

    # Prepare data container
    roi_means = {roi: [] for roi in roi_names}

    # Extract mean values per frame
    for frame_data in roi_pixel_data:
        for roi in roi_names:
            val = frame_data[roi]["mean_value"]
            roi_means[roi].append(val)

    # Create figure with subplots - one for each ROI
    num_rois = len(roi_names)
    fig, axes = plt.subplots(num_rois, 1, figsize=(14, 4*num_rois))
    if num_rois == 1:
        axes = [axes]

    # Prepare ground truth time axis if provided
    if ground_truth_signal is not None and gt_fs is not None:
        num_gt_samples = len(ground_truth_signal)
        if plot_in_secs:
            gt_x_axis = np.array([sample / gt_fs for sample in range(num_gt_samples)])
        else:
            # Scale GT to match frame indices
            gt_x_axis = np.linspace(0, num_frames-1, num_gt_samples)
        
        # Normalize ground truth to match ROI scale for better visualization
        gt_normalized = ground_truth_signal.copy()
        gt_normalized = (gt_normalized - np.mean(gt_normalized)) / np.std(gt_normalized)

    # Plot each ROI
    for idx, roi in enumerate(roi_names):
        ax = axes[idx]
        roi_signal = np.array(roi_means[roi])
        
        # Plot ROI signal
        ax.plot(roi_x_axis, roi_signal, label=f'{roi}', linewidth=1.5, color='blue')
        
        # Overlay ground truth if provided (normalized to ROI scale)
        if ground_truth_signal is not None and gt_fs is not None:
            # Scale GT to match ROI mean and std for visualization
            roi_mean = np.mean(roi_signal)
            roi_std = np.std(roi_signal)
            gt_scaled = gt_normalized * roi_std + roi_mean
            
            ax.plot(gt_x_axis, gt_scaled, label='Ground Truth (scaled)', 
                   linewidth=1.0, color='red', alpha=0.7, linestyle='--')
        
        # Labels and styling
        if plot_in_secs:
            ax.set_xlabel("Time (seconds)", fontsize=11)
        else:
            ax.set_xlabel("Frame Index", fontsize=11)
        ax.set_ylabel("Mean Value", fontsize=11)
        ax.set_title(f'{roi} - Mean Signal Over Time', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('roi_signals_with_ground_truth.png', dpi=100)
    print(f"Saved ROI signals with ground truth to: roi_signals_with_ground_truth.png")
    plt.show(block=False)


def plot_roi_mean_signals_with_ground_truth_subplot(roi_pixel_data, fps, ground_truth_signal=None, gt_fs=None, plot_in_secs=True):
    """
    Plot each ROI with ground truth in separate subplots.
    Each subplot contains one ROI signal and the ground truth signal for direct comparison.
    
    Args:
        roi_pixel_data: List of dictionaries containing ROI data per frame
        fps: Frames per second of the video
        ground_truth_signal: Optional ground truth signal array for comparison
        gt_fs: Sampling frequency of ground truth signal (Hz)
        plot_in_secs: If True, plot x-axis in seconds; otherwise in frames
    """
    if len(roi_pixel_data) == 0:
        print("Error: roi_pixel_data is empty.")
        return

    # Get list of ROI names from first frame
    roi_names = list(roi_pixel_data[0].keys())

    # Prepare time axis for ROI data
    num_frames = len(roi_pixel_data)
    if plot_in_secs:
        roi_x_axis = np.array([frame / fps for frame in range(num_frames)])
    else:
        roi_x_axis = np.arange(num_frames)

    # Prepare data container
    roi_means = {roi: [] for roi in roi_names}

    # Extract mean values per frame
    for frame_data in roi_pixel_data:
        for roi in roi_names:
            val = frame_data[roi]["mean_value"]
            roi_means[roi].append(val)

    # Prepare ground truth time axis if provided
    if ground_truth_signal is not None and gt_fs is not None:
        num_gt_samples = len(ground_truth_signal)
        if plot_in_secs:
            gt_x_axis = np.array([sample / gt_fs for sample in range(num_gt_samples)])
        else:
            # Scale GT to match frame indices
            gt_x_axis = np.linspace(0, num_frames-1, num_gt_samples)
        
        # Normalize ground truth for scaling
        gt_normalized = (ground_truth_signal - np.mean(ground_truth_signal)) / np.std(ground_truth_signal)

    # Create subplots - one for each ROI
    num_rois = len(roi_names)
    fig, axes = plt.subplots(num_rois, 1, figsize=(14, 4*num_rois))
    if num_rois == 1:
        axes = [axes]

    # Plot each ROI with ground truth
    for idx, roi in enumerate(roi_names):
        ax = axes[idx]
        roi_signal = np.array(roi_means[roi])
        
        # Plot ROI signal
        ax.plot(roi_x_axis, roi_signal, label=roi, linewidth=1.5, color='blue')
        
        # Plot ground truth if provided (scaled to ROI range)
        if ground_truth_signal is not None and gt_fs is not None:
            # Scale GT to match this ROI's mean and std
            roi_mean = np.mean(roi_signal)
            roi_std = np.std(roi_signal)
            gt_scaled = gt_normalized * roi_std + roi_mean
            
            ax.plot(gt_x_axis, gt_scaled, label='Ground Truth (scaled)', 
                   linewidth=1.5, color='red', linestyle='--', alpha=0.8)
        
        # Labels and styling
        if plot_in_secs:
            ax.set_xlabel("Time (seconds)", fontsize=11)
        else:
            ax.set_xlabel("Frame Index", fontsize=11)
        ax.set_ylabel("Mean Value", fontsize=11)
        ax.set_title(f'{roi} vs Ground Truth', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('roi_signals_with_ground_truth_subplot.png', dpi=100)
    print(f"Saved ROI signals with ground truth subplot to: roi_signals_with_ground_truth_subplot.png")
    plt.show(block=False)


def subdivide_roi_into_sub_rois(roi_pixel_data, video_path, roi_mask_log, fps, sub_roi_size=8):
    """
    Subdivide each ROI into smaller sub-ROIs (default 8x8 pixels).
    Extract mean signals for each sub-ROI.
    Only includes pixels that are within the ROI mask.
    
    Args:
        roi_pixel_data: List of dictionaries containing ROI data per frame
        video_path: Path to the video file
        roi_mask_log: List of ROI masks per frame
        fps: Frames per second
        sub_roi_size: Size of sub-ROI (default 8x8 pixels)
        
    Returns:
        sub_roi_data: List of dictionaries with sub-ROI data per frame
        sub_roi_names: List of sub-ROI names (roi_name_x_y format)
        sub_roi_coords: Dictionary with sub-ROI coordinates
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nSubdividing ROIs into {sub_roi_size}x{sub_roi_size} pixel sub-ROIs...")
    
    # Get ROI names from first frame of roi_pixel_data
    if not roi_pixel_data or len(roi_pixel_data) == 0:
        print("  Error: No ROI data available")
        return [], [], {}
    
    roi_names = list(roi_pixel_data[0].keys())
    
    # Create sub-ROI data structure
    sub_roi_data = []
    sub_roi_coords = {}  # Store coordinates of each sub-ROI for later extraction
    
    # First pass: identify all sub-ROI regions from roi_mask_log
    for roi_name in roi_names:
        sub_roi_coords[roi_name] = []
        
        # Get mask for this ROI from first frame
        if len(roi_mask_log) > 0 and roi_name in roi_mask_log[0]:
            mask = roi_mask_log[0][roi_name]
            
            # Find bounding box of the ROI
            ys, xs = np.where(mask == 255)
            if len(xs) > 0 and len(ys) > 0:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                
                # Subdivide into sub-ROIs and only keep regions with ROI pixels
                for y_start in range(int(y_min), int(y_max), sub_roi_size):
                    for x_start in range(int(x_min), int(x_max), sub_roi_size):
                        y_end = min(y_start + sub_roi_size, int(y_max) + 1)
                        x_end = min(x_start + sub_roi_size, int(x_max) + 1)
                        
                        # Check if this sub-ROI region contains any ROI pixels
                        sub_roi_mask_region = mask[y_start:y_end, x_start:x_end]
                        roi_pixel_count = np.sum(sub_roi_mask_region == 255)
                        
                        # Only create sub-ROI if it contains ROI pixels
                        if roi_pixel_count > 0:
                            sub_roi_name = f"{roi_name}_{x_start}_{y_start}"
                            sub_roi_coords[roi_name].append({
                                "name": sub_roi_name,
                                "x_start": x_start,
                                "y_start": y_start,
                                "x_end": x_end,
                                "y_end": y_end,
                                "roi_pixel_count": roi_pixel_count
                            })
    
    # Second pass: extract pixel values for each sub-ROI from all frames
    frame_idx = 0
    while frame_idx < len(roi_pixel_data):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale if needed
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        frame_sub_roi_data = {}
        
        # Extract sub-ROI data
        for roi_name in roi_names:
            frame_sub_roi_data[roi_name] = {}
            
            for sub_roi_info in sub_roi_coords[roi_name]:
                sub_roi_name = sub_roi_info["name"]
                x_start = sub_roi_info["x_start"]
                y_start = sub_roi_info["y_start"]
                x_end = sub_roi_info["x_end"]
                y_end = sub_roi_info["y_end"]
                
                # Extract sub-ROI region
                sub_roi_region = gray[y_start:y_end, x_start:x_end]
                
                # Calculate mean and store
                mean_value = np.mean(sub_roi_region) if sub_roi_region.size > 0 else 0.0
                
                frame_sub_roi_data[roi_name][sub_roi_name] = {
                    "mean_value": float(mean_value),
                    "x_start": x_start,
                    "y_start": y_start,
                    "x_end": x_end,
                    "y_end": y_end
                }
        
        sub_roi_data.append(frame_sub_roi_data)
        frame_idx += 1
        
        if frame_idx % 20 == 0:
            print(f"  Processed {frame_idx}/{len(roi_pixel_data)} frames")
    
    cap.release()
    
    # Generate list of all sub-ROI names
    all_sub_roi_names = []
    for roi_name in roi_names:
        for sub_roi_info in sub_roi_coords[roi_name]:
            all_sub_roi_names.append(sub_roi_info["name"])
    
    print(f"  Created {len(all_sub_roi_names)} sub-ROIs from {len(roi_names)} ROIs")
    
    return sub_roi_data, all_sub_roi_names, sub_roi_coords


def save_video_with_sub_roi_visualization(video_path, roi_mask_log, sub_roi_coords, output_path, fps):
    """
    Save a video with sub-ROI rectangles overlaid on each frame.
    
    Args:
        video_path: Path to input video
        roi_mask_log: List of ROI masks per frame
        sub_roi_coords: Dictionary with sub-ROI coordinates (from subdivide_roi_into_sub_rois)
        output_path: Path to save output video
        fps: Frames per second
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\nSaving sub-ROI visualization video to: {output_path}")
    
    # Color palette for different ROIs
    colors = {
        "forehead": (255, 0, 0),      # Blue
        "left_cheek": (0, 255, 0),    # Green
        "right_cheek": (0, 0, 255),   # Red
        "chin": (255, 255, 0),        # Cyan
        "random_baseline": (255, 0, 255)  # Magenta
    }
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(roi_mask_log):
            break
        
        # Convert to BGR if needed for display
        if frame.dtype != np.uint8:
            frame_display = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if frame_display.ndim == 2:
                frame_display = cv2.cvtColor(frame_display, cv2.COLOR_GRAY2BGR)
        else:
            if frame.ndim == 2:
                frame_display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_display = frame
        
        # Draw sub-ROI rectangles
        for roi_name, sub_rois in sub_roi_coords.items():
            color = colors.get(roi_name, (128, 128, 128))
            
            for sub_roi_info in sub_rois:
                x_start = sub_roi_info["x_start"]
                y_start = sub_roi_info["y_start"]
                x_end = sub_roi_info["x_end"]
                y_end = sub_roi_info["y_end"]
                
                # Draw rectangle
                cv2.rectangle(frame_display, (x_start, y_start), (x_end, y_end), color, 1)
        
        # Write frame to video
        out.write(frame_display)
        frame_idx += 1
        
        if frame_idx % 20 == 0:
            print(f"  Processed {frame_idx}/{len(roi_mask_log)} frames")
    
    cap.release()
    out.release()
    
    print(f"  Sub-ROI visualization video saved successfully")


def analyze_and_plot_top_sub_rois_by_frequency(sub_roi_data, sub_roi_names, fps, freq_range=(1.0, 2.0), top_n=5, ground_truth_signal=None, gt_fs=None):
    """
    Analyze frequency spectrum of all sub-ROIs and plot the top N with highest amplitude in specified frequency range.
    
    Args:
        sub_roi_data: List of dictionaries with sub-ROI data per frame
        sub_roi_names: List of all sub-ROI names
        fps: Frames per second
        freq_range: Tuple of (min_freq, max_freq) in Hz to analyze
        top_n: Number of top sub-ROIs to plot
        ground_truth_signal: Optional ground truth signal array for comparison
        gt_fs: Sampling frequency of ground truth signal (Hz)
        
    Returns:
        top_sub_rois: List of tuples (sub_roi_name, max_amplitude, dominant_freq)
    """
    print(f"\nAnalyzing frequency spectrum for {len(sub_roi_names)} sub-ROIs...")
    print(f"  Frequency range of interest: {freq_range[0]:.2f} - {freq_range[1]:.2f} Hz")
    
    # Extract mean signal for each sub-ROI
    sub_roi_signals = {}
    
    for sub_roi_name in sub_roi_names:
        signal_values = []
        
        # Find which ROI this sub-ROI belongs to
        parent_roi = None
        for roi_name in sub_roi_data[0].keys():
            if sub_roi_name in sub_roi_data[0][roi_name]:
                parent_roi = roi_name
                break
        
        if parent_roi is None:
            continue
        
        # Extract signal across all frames
        for frame_data in sub_roi_data:
            if parent_roi in frame_data and sub_roi_name in frame_data[parent_roi]:
                signal_values.append(frame_data[parent_roi][sub_roi_name]["mean_value"])
        
        if len(signal_values) > 0:
            sub_roi_signals[sub_roi_name] = np.array(signal_values)
    
    print(f"  Extracted signals for {len(sub_roi_signals)} sub-ROIs")
    
    # Analyze frequency spectrum for each sub-ROI
    sub_roi_freq_analysis = []
    
    for sub_roi_name, signal in sub_roi_signals.items():
        # Detrend signal (remove mean)
        signal_detrended = signal - np.mean(signal)
        
        # Apply FFT
        n = len(signal_detrended)
        fft_result = np.fft.rfft(signal_detrended)
        fft_freq = np.fft.rfftfreq(n, d=1.0/fps)
        fft_magnitude = np.abs(fft_result)
        
        # Find max amplitude in the frequency range of interest
        freq_mask = (fft_freq >= freq_range[0]) & (fft_freq <= freq_range[1])
        
        if np.sum(freq_mask) > 0:
            max_amp_in_range = np.max(fft_magnitude[freq_mask])
            max_amp_idx = np.argmax(fft_magnitude[freq_mask])
            dominant_freq = fft_freq[freq_mask][max_amp_idx]
            
            sub_roi_freq_analysis.append({
                'name': sub_roi_name,
                'max_amplitude': max_amp_in_range,
                'dominant_freq': dominant_freq,
                'signal': signal_detrended,
                'fft_freq': fft_freq,
                'fft_magnitude': fft_magnitude
            })
    
    # Sort by max amplitude in descending order
    sub_roi_freq_analysis.sort(key=lambda x: x['max_amplitude'], reverse=True)
    
    # Get top N sub-ROIs
    top_sub_rois = sub_roi_freq_analysis[:top_n]
    
    print(f"\nTop {top_n} sub-ROIs by amplitude in {freq_range[0]:.2f}-{freq_range[1]:.2f} Hz range:")
    for i, roi_info in enumerate(top_sub_rois, 1):
        print(f"  {i}. {roi_info['name']}: amplitude={roi_info['max_amplitude']:.2f}, dominant_freq={roi_info['dominant_freq']:.3f} Hz ({roi_info['dominant_freq']*60:.1f} bpm)")
    
    # Helper function for HR computation from peaks
    def compute_hr_from_peaks(sig, fs, min_distance_s=0.3, prominence=None):
        """Detect peaks and compute instantaneous HR (bpm) per beat.
        Returns: hr_times (seconds), hr_bpm (array), peak_times (seconds), peak_indices (array)
        """
        distance = int(min_distance_s * fs)
        peaks, _ = find_peaks(sig, distance=max(1, distance), prominence=prominence)
        peak_times = peaks / fs
        if len(peaks) < 2:
            return np.array([]), np.array([]), peak_times, peaks
        ibi = np.diff(peak_times)  # seconds between consecutive peaks
        hr_bpm = 60.0 / ibi
        hr_times = peak_times[1:]  # Assign HR to the latter peak time
        return hr_times, hr_bpm, peak_times, peaks
    
    # Compute HR for ground truth if provided
    gt_hr_times, gt_hr_bpm, gt_peak_times, gt_peaks = None, None, None, None
    if ground_truth_signal is not None and gt_fs is not None:
        gt_hr_times, gt_hr_bpm, gt_peak_times, gt_peaks = compute_hr_from_peaks(
            ground_truth_signal, gt_fs, min_distance_s=0.5, prominence=0.5
        )
    
    # Create visualization with 3 columns: time domain with peaks, frequency, HR comparison
    fig, axes = plt.subplots(top_n, 3, figsize=(18, 3*top_n))
    
    if top_n == 1:
        axes = axes.reshape(1, -1)
    
    for i, roi_info in enumerate(top_sub_rois):
        # Compute peaks and HR for this sub-ROI
        roi_hr_times, roi_hr_bpm, roi_peak_times, roi_peaks = compute_hr_from_peaks(
            roi_info['signal'], fps, min_distance_s=0.3, prominence=None
        )
        
        # Column 0: Time domain plot with peaks
        ax_time = axes[i, 0]
        time_axis = np.arange(len(roi_info['signal'])) / fps
        
        # Plot sub-ROI signal on primary axis
        ax_time.plot(time_axis, roi_info['signal'], linewidth=1.5, color='C0', label='Sub-ROI Signal')
        if roi_peaks.size > 0:
            ax_time.plot(time_axis[roi_peaks], roi_info['signal'][roi_peaks], 'rx', 
                        label=f'Sub-ROI Peaks (n={len(roi_peaks)})', markersize=8)
        ax_time.set_ylabel('Sub-ROI Amplitude', color='C0')
        ax_time.tick_params(axis='y', labelcolor='C0')
        
        # Add ground truth on secondary axis if provided
        if ground_truth_signal is not None and gt_fs is not None:
            ax_gt = ax_time.twinx()
            gt_time_axis = np.arange(len(ground_truth_signal)) / gt_fs
            ax_gt.plot(gt_time_axis, ground_truth_signal, linewidth=1.0, color='C1', 
                      linestyle='--', alpha=0.7, label='Ground Truth')
            if gt_peaks is not None and gt_peaks.size > 0:
                ax_gt.plot(gt_time_axis[gt_peaks], ground_truth_signal[gt_peaks], 'g^', 
                          label=f'GT Peaks (n={len(gt_peaks)})', markersize=6, alpha=0.7)
            ax_gt.set_ylabel('GT Amplitude', color='C1')
            ax_gt.tick_params(axis='y', labelcolor='C1')
            
            # Combine legends
            lines0, labels0 = ax_time.get_legend_handles_labels()
            lines1, labels1 = ax_gt.get_legend_handles_labels()
            ax_time.legend(lines0 + lines1, labels0 + labels1, loc='upper right', fontsize=7)
        else:
            ax_time.legend(fontsize=8)
        
        ax_time.set_xlabel('Time (s)')
        ax_time.set_title(f"{i+1}. {roi_info['name']} - Signals with Peaks")
        ax_time.grid(True, alpha=0.3)
        
        # Column 1: Frequency domain plot
        ax_freq = axes[i, 1]
        ax_freq.plot(roi_info['fft_freq'], roi_info['fft_magnitude'], linewidth=1, color='red')
        ax_freq.axvline(roi_info['dominant_freq'], color='green', linestyle='--', linewidth=2, 
                       label=f"Dominant: {roi_info['dominant_freq']:.3f} Hz ({roi_info['dominant_freq']*60:.1f} bpm)")
        ax_freq.axvspan(freq_range[0], freq_range[1], alpha=0.2, color='yellow', label='Analysis Range')
        ax_freq.set_xlabel('Frequency (Hz)')
        ax_freq.set_ylabel('Magnitude')
        ax_freq.set_title(f"{i+1}. {roi_info['name']} - Frequency Spectrum")
        ax_freq.set_xlim(0, 4.0)
        ax_freq.legend(fontsize=8)
        ax_freq.grid(True, alpha=0.3)
        
        # Column 2: HR comparison plot
        ax_hr = axes[i, 2]
        if roi_hr_bpm.size > 0:
            ax_hr.plot(roi_hr_times, roi_hr_bpm, 'o-', label='Sub-ROI HR', markersize=4, color='C0')
            mean_hr_roi = np.mean(roi_hr_bpm)
            ax_hr.axhline(mean_hr_roi, color='C0', linestyle=':', alpha=0.5, 
                         label=f'Mean: {mean_hr_roi:.1f} bpm')
        
        if gt_hr_bpm is not None and gt_hr_bpm.size > 0:
            ax_hr.plot(gt_hr_times, gt_hr_bpm, 'x-', label='GT HR', markersize=4, color='C1')
            mean_hr_gt = np.mean(gt_hr_bpm)
            ax_hr.axhline(mean_hr_gt, color='C1', linestyle=':', alpha=0.5, 
                         label=f'GT Mean: {mean_hr_gt:.1f} bpm')
            
            # Calculate MAE and RMSE if both signals have HR data
            if roi_hr_bpm.size > 0 and gt_hr_bpm.size > 0:
                # Interpolate GT HR onto ROI HR times for comparison
                gt_hr_interp = np.interp(roi_hr_times, gt_hr_times, gt_hr_bpm, left=np.nan, right=np.nan)
                mask = ~np.isnan(gt_hr_interp)
                if mask.any():
                    mae = np.mean(np.abs(roi_hr_bpm[mask] - gt_hr_interp[mask]))
                    rmse = np.sqrt(np.mean((roi_hr_bpm[mask] - gt_hr_interp[mask])**2))
                    ax_hr.set_title(f"{i+1}. HR Comparison (MAE={mae:.2f}, RMSE={rmse:.2f} bpm)", fontsize=9)
                else:
                    ax_hr.set_title(f"{i+1}. HR Comparison (no overlap)", fontsize=9)
            else:
                ax_hr.set_title(f"{i+1}. Heart Rate Comparison", fontsize=9)
        else:
            ax_hr.set_title(f"{i+1}. Heart Rate (no GT)", fontsize=9)
        
        ax_hr.set_xlabel('Time (s)')
        ax_hr.set_ylabel('HR (bpm)')
        ax_hr.legend(fontsize=7)
        ax_hr.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)
    
    # Return summary
    return [(roi['name'], roi['max_amplitude'], roi['dominant_freq']) for roi in top_sub_rois]


def analyze_and_plot_top_sub_rois_per_roi(sub_roi_data, sub_roi_names, fps, freq_range=(1.0, 2.0), top_n=5, ground_truth_signal=None, gt_fs=None):
    """
    Analyze frequency spectrum of sub-ROIs and plot the top N for each parent ROI separately.
    Creates n_rois x top_n subplots.
    
    Args:
        sub_roi_data: List of dictionaries with sub-ROI data per frame
        sub_roi_names: List of all sub-ROI names
        fps: Frames per second
        freq_range: Tuple of (min_freq, max_freq) in Hz to analyze
        top_n: Number of top sub-ROIs to plot per parent ROI
        ground_truth_signal: Optional ground truth signal array for comparison
        gt_fs: Sampling frequency of ground truth signal (Hz)
        
    Returns:
        roi_top_sub_rois: Dictionary mapping ROI name to list of top sub-ROIs
    """
    print(f"\nAnalyzing frequency spectrum for sub-ROIs by parent ROI...")
    print(f"  Frequency range of interest: {freq_range[0]:.2f} - {freq_range[1]:.2f} Hz")
    
    # Get parent ROI names
    parent_roi_names = list(sub_roi_data[0].keys())
    
    # Organize sub-ROIs by parent ROI
    sub_rois_by_parent = {roi_name: [] for roi_name in parent_roi_names}
    
    for sub_roi_name in sub_roi_names:
        # Find parent ROI
        for roi_name in parent_roi_names:
            if sub_roi_name in sub_roi_data[0][roi_name]:
                sub_rois_by_parent[roi_name].append(sub_roi_name)
                break
    
    # Analyze each parent ROI's sub-ROIs
    roi_top_sub_rois = {}
    
    for roi_name in parent_roi_names:
        print(f"\n  Analyzing {roi_name}: {len(sub_rois_by_parent[roi_name])} sub-ROIs")
        
        # Extract signals for this ROI's sub-ROIs
        sub_roi_freq_analysis = []
        
        for sub_roi_name in sub_rois_by_parent[roi_name]:
            signal_values = []
            
            # Extract signal across all frames
            for frame_data in sub_roi_data:
                if roi_name in frame_data and sub_roi_name in frame_data[roi_name]:
                    signal_values.append(frame_data[roi_name][sub_roi_name]["mean_value"])
            
            if len(signal_values) == 0:
                continue
            
            signal = np.array(signal_values)
            signal_detrended = signal - np.mean(signal)
            
            # Apply FFT
            n = len(signal_detrended)
            fft_result = np.fft.rfft(signal_detrended)
            fft_freq = np.fft.rfftfreq(n, d=1.0/fps)
            fft_magnitude = np.abs(fft_result)
            
            # Find max amplitude in the frequency range of interest
            freq_mask = (fft_freq >= freq_range[0]) & (fft_freq <= freq_range[1])
            
            if np.sum(freq_mask) > 0:
                max_amp_in_range = np.max(fft_magnitude[freq_mask])
                max_amp_idx = np.argmax(fft_magnitude[freq_mask])
                dominant_freq = fft_freq[freq_mask][max_amp_idx]
                
                sub_roi_freq_analysis.append({
                    'name': sub_roi_name,
                    'parent_roi': roi_name,
                    'max_amplitude': max_amp_in_range,
                    'dominant_freq': dominant_freq,
                    'signal': signal_detrended,
                    'fft_freq': fft_freq,
                    'fft_magnitude': fft_magnitude
                })
        
        # Sort by amplitude and get top N
        sub_roi_freq_analysis.sort(key=lambda x: x['max_amplitude'], reverse=True)
        top_for_this_roi = sub_roi_freq_analysis[:top_n]
        roi_top_sub_rois[roi_name] = top_for_this_roi
        
        print(f"    Top {len(top_for_this_roi)} sub-ROIs:")
        for i, roi_info in enumerate(top_for_this_roi, 1):
            print(f"      {i}. {roi_info['name']}: amplitude={roi_info['max_amplitude']:.2f}, dominant_freq={roi_info['dominant_freq']:.3f} Hz ({roi_info['dominant_freq']*60:.1f} bpm)")
    
    # Create visualization: n_rois rows x (top_n * 2) columns
    n_rois = len(parent_roi_names)
    fig, axes = plt.subplots(n_rois, top_n * 2, figsize=(5*top_n, 3*n_rois))
    
    # Ensure axes is 2D
    if n_rois == 1 and top_n == 1:
        axes = axes.reshape(1, -1)
    elif n_rois == 1:
        axes = axes.reshape(1, -1)
    elif top_n == 1:
        axes = axes.reshape(-1, 2)
    
    for roi_idx, roi_name in enumerate(parent_roi_names):
        top_sub_rois = roi_top_sub_rois[roi_name]
        
        # Prepare ground truth if provided
        if ground_truth_signal is not None and gt_fs is not None:
            gt_normalized = (ground_truth_signal - np.mean(ground_truth_signal)) / np.std(ground_truth_signal)
        
        for sub_roi_idx, roi_info in enumerate(top_sub_rois):
            # Time domain plot
            col_time = sub_roi_idx * 2
            ax_time = axes[roi_idx, col_time] if n_rois > 1 else axes[col_time]
            
            time_axis = np.arange(len(roi_info['signal'])) / fps
            ax_time.plot(time_axis, roi_info['signal'], linewidth=1, color='blue', label='Sub-ROI')
            
            # Add ground truth if provided
            if ground_truth_signal is not None and gt_fs is not None:
                roi_mean = np.mean(roi_info['signal'])
                roi_std = np.std(roi_info['signal'])
                gt_scaled = gt_normalized * roi_std + roi_mean
                num_gt_samples = len(ground_truth_signal)
                gt_time_axis = np.arange(num_gt_samples) / gt_fs
                ax_time.plot(gt_time_axis, gt_scaled, linewidth=1, color='red', linestyle='--', alpha=0.7, label='GT')
            
            ax_time.set_xlabel('Time (s)', fontsize=8)
            ax_time.set_ylabel('Signal', fontsize=8)
            ax_time.set_title(f"{roi_name} - {roi_info['name'].split('_')[-2]}_{roi_info['name'].split('_')[-1]} (Time)", fontsize=9)
            ax_time.grid(True, alpha=0.3)
            ax_time.legend(fontsize=6)
            ax_time.tick_params(labelsize=7)
            
            # Frequency domain plot
            col_freq = sub_roi_idx * 2 + 1
            ax_freq = axes[roi_idx, col_freq] if n_rois > 1 else axes[col_freq]
            
            ax_freq.plot(roi_info['fft_freq'], roi_info['fft_magnitude'], linewidth=1, color='red')
            ax_freq.axvline(roi_info['dominant_freq'], color='green', linestyle='--', linewidth=1.5, 
                           label=f"{roi_info['dominant_freq']:.2f} Hz")
            ax_freq.axvspan(freq_range[0], freq_range[1], alpha=0.2, color='yellow')
            ax_freq.set_xlabel('Frequency (Hz)', fontsize=8)
            ax_freq.set_ylabel('Magnitude', fontsize=8)
            ax_freq.set_title(f"{roi_name} - {roi_info['name'].split('_')[-2]}_{roi_info['name'].split('_')[-1]} (Freq)", fontsize=9)
            ax_freq.set_xlim(0, 4.0)
            ax_freq.legend(fontsize=6)
            ax_freq.grid(True, alpha=0.3)
            ax_freq.tick_params(labelsize=7)
        
        # Hide unused subplots if fewer than top_n sub-ROIs
        for sub_roi_idx in range(len(top_sub_rois), top_n):
            col_time = sub_roi_idx * 2
            col_freq = sub_roi_idx * 2 + 1
            if n_rois > 1:
                axes[roi_idx, col_time].axis('off')
                axes[roi_idx, col_freq].axis('off')
            else:
                axes[col_time].axis('off')
                axes[col_freq].axis('off')
    
    plt.tight_layout()
    plt.show(block=True)
    
    return roi_top_sub_rois


def create_manual_roi_from_video(
    video_path: str,
    frame_index: int = 0,
    roi_name: str = "manual_roi",
    mode: str = "polygon",
    use_colormap: bool = False,
    mask_save_path: str = None,
) -> list:
    """
    Create a manual ROI mask by selecting a region on a video frame.
    
    Args:
        video_path: Path to the video file
        frame_index: Index of frame to display for ROI selection
        roi_name: Name of the ROI
        mode: Selection mode - "rectangle" or "polygon"
        use_colormap: If True, apply colormap to the frame for display
        mask_save_path: Optional path to save the mask
        
    Returns:
        roi_mask_log: List of masks (same mask repeated for all frames)
    """
    # Open video and read frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Navigate to the requested frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if not ret:
        raise ValueError(f"Cannot read frame {frame_index} from video: {video_path}")
    
    # Convert frame to grayscale
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Normalize for display
    height, width = frame_gray.shape
    vmin, vmax = np.percentile(frame_gray, [1, 99])
    frame_clipped = np.clip(frame_gray, vmin, vmax)
    frame_norm = cv2.normalize(frame_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if use_colormap:
        frame_display = cv2.applyColorMap(frame_norm, cv2.COLORMAP_TURBO)
    else:
        frame_display = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)
    
    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if mode == "rectangle":
        cv2.namedWindow("Draw ROI (press ENTER to confirm)", cv2.WINDOW_NORMAL)
        roi_rect = cv2.selectROI("Draw ROI (press ENTER to confirm)", frame_display, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Draw ROI (press ENTER to confirm)")
        
        x, y, w, h = roi_rect
        if w <= 0 or h <= 0:
            raise ValueError("No ROI selected. Please draw a rectangle ROI and press ENTER.")
        
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)
    
    elif mode == "polygon":
        points = []
        display = frame_display.copy()
        instructions = "Left click: add point | Right click: undo | ENTER: finish | c: clear"
        
        def draw_overlay(img, pts):
            canvas = img.copy()
            if pts:
                cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (0, 255, 255), 1)
                for p in pts:
                    cv2.circle(canvas, p, 2, (0, 255, 255), -1)
            cv2.putText(canvas, instructions, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 255), 1)
            return canvas
        
        def on_mouse(event, x, y, flags, param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and points:
                points.pop()
        
        cv2.namedWindow("Draw ROI (press ENTER to confirm)", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Draw ROI (press ENTER to confirm)", on_mouse)
        
        while True:
            canvas = draw_overlay(display, points)
            cv2.imshow("Draw ROI (press ENTER to confirm)", canvas)
            key = cv2.waitKey(20) & 0xFF
            
            if key == 13:  # Enter
                break
            if key == ord('c'):
                points = []
        
        cv2.destroyWindow("Draw ROI (press ENTER to confirm)")
        
        if len(points) < 3:
            raise ValueError("Polygon ROI needs at least 3 points. Please try again.")
        
        poly = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
    else:
        raise ValueError(f"Unknown manual ROI mode: {mode}")
    
    # Save mask if path provided
    if mask_save_path:
        os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
        np.save(mask_save_path, mask)
        print(f"Saved manual ROI mask to: {mask_save_path}")
    
    # Create mask log for all frames (same mask for each frame)
    roi_mask_log = [{roi_name: mask} for _ in range(num_frames)]
    
    return roi_mask_log


def save_video_with_top_sub_rois_only(video_path, sub_roi_coords, top_sub_rois, roi_top_sub_rois, fps, output_path):
    """
    Save a video showing only the top sub-ROIs (overall top 5) overlaid on frames.
    Draws rectangles for top sub-ROIs only, not all sub-ROIs.
    
    Args:
        video_path: Path to input video
        sub_roi_coords: Dictionary with sub-ROI coordinates from subdivide_roi_into_sub_rois
        top_sub_rois: List of tuples (sub_roi_name, max_amplitude, dominant_freq) overall top 5
        roi_top_sub_rois: Dictionary mapping ROI name to list of top sub-ROIs
        fps: Frames per second
        output_path: Path to save output video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\nSaving video with top sub-ROIs only to: {output_path}")
    
    # Build lookup for top sub-ROI coordinates
    top_coords_lookup = {}
    for parent_roi, sub_roi_list in sub_roi_coords.items():
        for sub_roi_info in sub_roi_list:
            top_coords_lookup[sub_roi_info["name"]] = (sub_roi_info, parent_roi)
    
    # Color palette for ROI types (BGR format for OpenCV)
    roi_colors = {
        'forehead': (255, 0, 0),        # Blue in BGR
        'left_cheek': (0, 255, 0),      # Green
        'right_cheek': (0, 0, 255),     # Red
        'chin': (0, 255, 255),          # Yellow
        'random_baseline': (255, 0, 255) # Magenta
    }
    
    frame_idx = 0
    while frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to BGR if needed for display
        if frame.dtype != np.uint8:
            frame_display = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if frame_display.ndim == 2:
                frame_display = cv2.cvtColor(frame_display, cv2.COLOR_GRAY2BGR)
        else:
            if frame.ndim == 2:
                frame_display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_display = frame.copy()
        
        # Draw only top sub-ROIs overall (with rank labels)
        for roi_idx, (sub_roi_name, amplitude, freq) in enumerate(top_sub_rois):
            if sub_roi_name in top_coords_lookup:
                sub_roi_info, parent_roi = top_coords_lookup[sub_roi_name]
                x_start = sub_roi_info["x_start"]
                y_start = sub_roi_info["y_start"]
                x_end = sub_roi_info["x_end"]
                y_end = sub_roi_info["y_end"]
                
                color = roi_colors.get(parent_roi, (128, 128, 128))
                # Draw rectangle for top ROIs
                cv2.rectangle(frame_display, (x_start, y_start), (x_end, y_end), color, 1)
                # Add small rank label
                rank_text = f"{roi_idx+1}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.3
                thickness = 1
                text_size = cv2.getTextSize(rank_text, font, font_scale, thickness)[0]
                # Draw background rectangle for text
                cv2.rectangle(frame_display, 
                            (x_start, y_start - text_size[1] - 2),
                            (x_start + text_size[0] + 2, y_start),
                            color, -1)
                # Draw text
                cv2.putText(frame_display, rank_text, (x_start + 1, y_start - 1),
                           font, font_scale, (255, 255, 255), thickness)
        
        # Write frame to video
        out.write(frame_display)
        frame_idx += 1
        
        if frame_idx % 20 == 0:
            print(f"  Processed {frame_idx}/{num_frames} frames")
    
    cap.release()
    out.release()
    
    print(f"  Top sub-ROI video saved successfully")


if __name__ == "__main__":
    path_video_original = "data/temp/03022026_3_thermal_cut.mp4"
    fps_original = 8.767704384613774

    path_video_color_magnification = path_video_original #"data/recorded_videos/new_approach/color_amplification/03022026_3__1_2_Hz_80amp.mp4"
    fps_color_magnification = fps_original#9

    ######
    path_video = path_video_original
    fps = fps_original

    path_video_to_get_values_from = path_video_color_magnification
    fps_to_get_values_from = fps_color_magnification

    path_pixels_log = "data/saved_data/new_approach/roi_pixel_log.npy"
    path_video_facemesh = "data/recorded_videos/new_approach/color_amplification/facemesh_overlay.mp4"
    path_video_thermal_with_roi_masks = "data/recorded_videos/new_approach/color_amplification/video_thermal_with_roi_masks.mp4"

    # Manual ROI configuration (set to True to use manual ROI instead of automatic detection)
    manual_roi_enabled = True
    manual_roi_name = "manual_roi"
    manual_roi_mode = "polygon"  # "rectangle" or "polygon"
    manual_roi_frame_index = 0  # Frame index in the video to use for manual ROI selection
    manual_roi_mask_save_path = "data/saved_data/new_approach/manual_roi_mask_color_amp.npy"
    manual_roi_selection_colormap = False

    roi_landmarks_subgrid_face = {
    "forehead": [
      10, 338, 297, 332, 284, 251, 389, 368, 300, 293, 334, 296, 336, 9, 
      107, 66, 105, 63, 71, 162, 21, 54, 103, 67, 109
    ],
    "left_cheek": [
      245, 233, 232, 231, 230, 229, 228, 31, 111, 116, 123, 147, 187, 205, 
      36, 142, 126, 217, 174, 188
    ],
    "right_cheek": [
      453, 357, 343, 420, 360, 266, 280, 352, 345, 449, 451
    ],
    "chin": [
      202, 210, 169, 170, 140, 171, 175, 262, 432, 422, 424, 418, 421, 200, 
      201, 194, 204
    ]
    }
    use_ema = True
    ema_alpha = 0.15

    #######
    timespan_signal = [180,210]
    offset_seconds = 26.5 

    path_ground_truth = "data/ground_truth/opensignals/extracted_npy_files/03022026/0302_3/opensignals_0007800F80FA_2026-02-03_12-07-12.npz"

    freq_range_for_analysis_subrois = (1, 2)
    sub_roi_size = 8

    ##########
    # Get ROI data
    ##########
    # Detect face or use manual ROI
    if manual_roi_enabled:
        roi_pixel_log = create_manual_roi_from_video(
            video_path=path_video_to_get_values_from,
            frame_index=manual_roi_frame_index,
            roi_name=manual_roi_name,
            mode=manual_roi_mode,
            use_colormap=manual_roi_selection_colormap,
            mask_save_path=manual_roi_mask_save_path
        )
    else:
        roi_pixel_log = apply_facemesh_to_thermal_video(input_path=path_video, output_path=path_video_facemesh, roi_landmarks=roi_landmarks_subgrid_face, fps = fps, use_ema = use_ema, ema_alpha = ema_alpha)
    
    roi_pixel_data = apply_roi_masks_to_video(video_path=path_video_to_get_values_from, roi_mask_log=roi_pixel_log, output_path=path_video_thermal_with_roi_masks, fps = fps_to_get_values_from)

    # Add random baseline ROI for comparison (pure Gaussian noise, independent of video content)
    roi_pixel_data = add_random_baseline_roi(roi_pixel_data=roi_pixel_data, roi_size=(50, 50), seed=42, amplitude_scale=0.5)

    plot_roi_mean_signals(roi_pixel_data=roi_pixel_data, fps=fps_to_get_values_from, plot_in_secs=True)
    
    ##########
    # Get ground truth signal
    ##########
    ground_truth_signal, gt_fs, gt_units = load_ground_truth_signal_and_fs(path_ground_truth)

    # Apply timespan filter to ground truth if configured
    start_time, end_time = timespan_signal
    # Adjust for ground truth offset: GT time = signal_time + (gt_start - signal_start)
    time_offset = offset_seconds
    if time_offset < 0:
        print("Warning: Negative time offset between signal and ground truth.")
    gt_start_time = start_time + time_offset
    gt_end_time = end_time + time_offset
    gt_start_sample = int(gt_start_time * gt_fs)
    gt_end_sample = int(gt_end_time * gt_fs)
    ground_truth_signal = ground_truth_signal[gt_start_sample:gt_end_sample]
    print(f"Applied GT timespan filter: {gt_start_time:.2f}s to {gt_end_time:.2f}s (samples {gt_start_sample} to {gt_end_sample})")

    # Invert ground truth signal for better peak detection (ECG valleys become peaks)
    #ground_truth_signal = -ground_truth_signal
    #print("Inverted ground truth signal for peak detection")

    # Calculate actual duration and correct FPS if needed
    num_signal_frames = len(roi_pixel_data)
    num_gt_samples = len(ground_truth_signal)
    
    # The actual duration should be the same for both signals
    actual_duration_gt = num_gt_samples / gt_fs
    
    # Correct the video FPS to match actual duration
    fps_corrected = num_signal_frames / actual_duration_gt
    
    print(f"\nFPS Correction:")
    print(f"  Original FPS: {fps_to_get_values_from:.4f} Hz")
    print(f"  Corrected FPS: {fps_corrected:.4f} Hz")
    print(f"  Signal frames: {num_signal_frames}")
    print(f"  GT samples: {num_gt_samples}")
    print(f"  Actual duration: {actual_duration_gt:.2f}s")
    
    # Ensure ground truth is not longer than signal duration
    max_gt_samples = int(num_signal_frames / fps_corrected * gt_fs)
    if num_gt_samples > max_gt_samples:
        print(f"  Warning: Ground truth too long, trimming from {num_gt_samples} to {max_gt_samples} samples")
        ground_truth_signal = ground_truth_signal[:max_gt_samples]
        num_gt_samples = len(ground_truth_signal)
    
    # Use corrected FPS for all subsequent analysis
    fps_to_use = fps_corrected


    # Plot ROI signals using original function (all ROIs together)
    plot_roi_mean_signals(roi_pixel_data=roi_pixel_data, fps=fps_to_use, plot_in_secs=True)

    # Plot ROI signals with ground truth in separate subplots
    plot_roi_mean_signals_with_ground_truth_subplot(
        roi_pixel_data=roi_pixel_data, 
        fps=fps_to_use, 
        ground_truth_signal=ground_truth_signal,
        gt_fs=gt_fs,
        plot_in_secs=True
    )
    
    ##########
    # Subdivide ROIs and analyze frequency ranges
    ##########
    # Subdivide ROIs into nxn pixel sub-ROIs
    sub_roi_data, sub_roi_names, sub_roi_coords = subdivide_roi_into_sub_rois(
        roi_pixel_data=roi_pixel_data,
        video_path=path_video_to_get_values_from,
        roi_mask_log=roi_pixel_log,
        fps=fps_to_use,
        sub_roi_size=sub_roi_size
    )
    
    # Save video visualization of sub-ROIs
    print("Saving sub-ROI video visualization...")
    save_video_with_sub_roi_visualization(
        video_path=path_video_to_get_values_from,
        roi_mask_log=roi_pixel_log,
        sub_roi_coords=sub_roi_coords,
        output_path="data/recorded_videos/new_approach/color_amplification/sub_roi_visualization.mp4",
        fps=fps_to_use
    )
    
    # Get frequency range for each ROI
    frequency_ranges = get_frequency_range_for_rois(
        roi_pixel_data=roi_pixel_data, 
        fps=fps_to_get_values_from, 
        freq_range=(0.5, 4.0), 
        visualize=True,
        ground_truth_signal=ground_truth_signal,
        gt_fs=gt_fs
    )
    
    # Analyze sub-ROI frequency spectra and plot top 5 in wanted range (overall)
    top_sub_rois = analyze_and_plot_top_sub_rois_by_frequency(
        sub_roi_data=sub_roi_data,
        sub_roi_names=sub_roi_names,
        fps=fps_to_use,
        freq_range=freq_range_for_analysis_subrois,
        top_n=30,
        ground_truth_signal=ground_truth_signal,
        gt_fs=gt_fs
    )
    
    # Analyze and plot top 5 sub-ROIs for each parent ROI separately
    roi_top_sub_rois = analyze_and_plot_top_sub_rois_per_roi(
        sub_roi_data=sub_roi_data,
        sub_roi_names=sub_roi_names,
        fps=fps_to_use,
        freq_range=freq_range_for_analysis_subrois,
        top_n=5,
        ground_truth_signal=ground_truth_signal,
        gt_fs=gt_fs
    )
    
    # Save video with only top sub-ROIs overlaid
    save_video_with_top_sub_rois_only(
        video_path=path_video_to_get_values_from,
        sub_roi_coords=sub_roi_coords,
        top_sub_rois=top_sub_rois,
        roi_top_sub_rois=roi_top_sub_rois,
        fps=fps_to_use,
        output_path="data/recorded_videos/new_approach/color_amplification/sub_rois_top_only.mp4"
    )





"""
    # Detect face region (with visualization)
    face_mask, frame_shape = detect_face_mask(path_video, visualize=True)

    # Extract pixel signals (only from face region)
    pixel_signals, fps, frame_shape = extract_pixel_signals_from_video(path_video, fps, face_mask=face_mask)
    normalized_signals = normalize_pixel_signals(pixel_signals, method='mean', face_mask=face_mask)

    # Apply FFT and plot frequency spectrum (averaged over face pixels only)
    fft_magnitudes, frequencies = apply_fft_to_pixel_signals(normalized_signals, fps)
    plot_average_frequency_spectrum(fft_magnitudes, frequencies, freq_range=(0, 5), face_mask=face_mask)
"""