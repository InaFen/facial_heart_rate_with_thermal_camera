import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Force interactive backend
import matplotlib.pyplot as plt
import sys
from scipy import signal as scipy_signal

PROJECT_ROOT = "/home/arbeit/Dokumente/master_thesis/"
sys.path.insert(0, PROJECT_ROOT)
from src.approaches.new_approach.utils.utils_landmark_detection import apply_facemesh_to_thermal_video, apply_roi_masks_to_video, plot_roi_mean_signals, apply_roi_masks_to_gray_frames, plot_roi_adc_means
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
        plt.show(block=True)
    
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
    plt.show(block=True)
    
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
        plt.show(block=True)
    
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
    plt.show(block=True)


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
    plt.show(block=True)


if __name__ == "__main__":

    path_video_color_magnification = "data/recorded_videos/new_approach/0_1_3_5_Hz_40amp.mp4"
    fps_color_magnification = 9

    path_video_original = "data/temp/19012026_thermal_cut.mp4"
    fps_original = 8.767687772484802

    path_last = "data/recorded_videos/new_approach/1_2_Hz_100amp.mp4"
    fps_last = 9

    ######
    path_video = path_video_original
    fps = fps_original

    path_video_to_get_values_from = path_last
    fps_to_get_values_from = fps_last

    path_pixels_log = "data/saved_data/new_approach/roi_pixel_log.npy"
    path_video_facemesh = "data/recorded_videos/new_approach/color_amplification/facemesh_overlay.mp4"
    path_video_thermal_with_roi_masks = "data/recorded_videos/new_approach/color_amplification/video_thermal_with_roi_masks.mp4"

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
    timespan_signal = [21,30]
    offset_seconds = 10.3  

    path_ground_truth = "data/ground_truth/opensignals/extracted_npy_files/19012026/opensignals_0007800F80FA_2026-01-19_13-53-34.npz"


    ##########
    # Get ROI data
    ##########
    # Detect face
    roi_pixel_log = apply_facemesh_to_thermal_video(input_path=path_video, output_path=path_video_facemesh, roi_landmarks=roi_landmarks_subgrid_face, fps = fps, use_ema = use_ema, ema_alpha = ema_alpha)
    #os.makedirs(os.path.dirname(configs.path_pixels_log), exist_ok=True)
    #np.save(configs.path_pixels_log, roi_pixel_log, allow_pickle=True)

    roi_pixel_data = apply_roi_masks_to_video(video_path=path_video_to_get_values_from, roi_mask_log=roi_pixel_log, output_path=path_video_thermal_with_roi_masks, fps = fps_to_get_values_from)

    # Add random baseline ROI for comparison (pure Gaussian noise, independent of video content)
    roi_pixel_data = add_random_baseline_roi(roi_pixel_data=roi_pixel_data, roi_size=(50, 50), seed=42, amplitude_scale=0.5)

    plot_roi_mean_signals(roi_pixel_data=roi_pixel_data, fps=fps_to_get_values_from, plot_in_secs=True)
    ##########
    # Get ground truth signal
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


    # Plot ROI signals with ground truth in separate subplot
    plot_roi_mean_signals_with_ground_truth_subplot(
        roi_pixel_data=roi_pixel_data, 
        fps=fps_to_use, 
        ground_truth_signal=ground_truth_signal,
        gt_fs=gt_fs,
        plot_in_secs=True
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

# add maybe bandpass filter


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