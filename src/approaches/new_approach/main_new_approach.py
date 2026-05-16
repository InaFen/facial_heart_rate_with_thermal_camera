import sys
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import pywt


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)

from src.approaches.new_approach.classes.user_input import Config
from src.approaches.yang_et_al.utils_post_processing import apply_butterworth_bandpass_filter

from src.approaches.new_approach.pipelines.get_roi_data import get_roi_data_pipeline

CONFIGS_PATH = "src/approaches/new_approach/configs_new_approach.yaml"

configs = Config.from_yaml(CONFIGS_PATH)
output_dir = os.path.dirname(configs.path_video_thermal_with_roi_masks)


roi_pixel_data, _ = get_roi_data_pipeline(configs=configs)

# Spatial averaging of each ROI (should be very small ROIs)
# to average random noise out (by sqrt(n))
# Create matrix with rows: subregions, columns: time frames, values: mean pixel intensity of the subregion in the frame
result = defaultdict(list)
subregions_names = []

for frame in roi_pixel_data:
    for roi, values in frame.items():
        if roi not in result:
            subregions_names.append(roi)  # remember first-seen order
        result[roi].append(values["mean_kelvin_multiplied_100"])

mean_values_subregions = [result[roi] for roi in subregions_names]
raw_before_averaging = np.array(mean_values_subregions)

# Temporal filtering over ROI to remove NETD noise while preserving heart rate
# Can use either moving average or Savitzky-Golay filter (preserves peaks better)
input_fps = float(configs.fps)

# Choose filter type: 'moving_average' or 'savitzky_golay'
temporal_filter_type = 'moving_average'  # Change to 'moving_average' to use simple moving average

# Savitzky-Golay acts as a low-pass filter. Set target cutoff frequency in Hz
# Larger window = lower cutoff frequency (more smoothing)
# For heart rate detection: 3-5 Hz is typical
target_cutoff_hz = 8.0  # Hz (tune this for desired noise reduction)

# Approximate window size for target cutoff frequency
# Rule of thumb: window_length ≈ fps / cutoff_hz
filtering_window_size_frames = max(3, int(input_fps / target_cutoff_hz))
# Make odd for savgol_filter
if filtering_window_size_frames % 2 == 0:
    filtering_window_size_frames += 1

if filtering_window_size_frames > 1:
    temporally_filtered_values_subregions = []
    for roi_signal_values in mean_values_subregions:
        roi_signal_array = np.asarray(roi_signal_values)
        if roi_signal_array.size == 0:
            temporally_filtered_values_subregions.append(roi_signal_array.tolist())
            continue
        
        if temporal_filter_type == 'savitzky_golay':
            # Savitzky-Golay filter: preserves peaks and valleys
            # window_length must be odd and <= signal length
            window_length = min(len(roi_signal_array), filtering_window_size_frames)
            if window_length % 2 == 0:  # Make odd if even
                window_length = max(3, window_length - 1)
            polyorder = min(3, window_length - 1)  # polynomial order must be < window_length
            filtered_roi_signal = savgol_filter(roi_signal_array, window_length, polyorder)
        else:  # moving_average
            # Simple moving average
            averaging_kernel = np.ones(filtering_window_size_frames) / filtering_window_size_frames
            filtered_roi_signal = np.convolve(roi_signal_array, averaging_kernel, mode="same")
        
        temporally_filtered_values_subregions.append(filtered_roi_signal.tolist())
    mean_values_subregions = temporally_filtered_values_subregions

if configs.plot_steps:
    raw_after_averaging = np.array(mean_values_subregions)

    t = np.arange(raw_before_averaging.shape[1])

    fig, ax = plt.subplots(figsize=(10, 4))

    for i in range(raw_before_averaging.shape[0]):
        ax.plot(
            t,
            raw_before_averaging[i],
            label=f"{subregions_names[i]} (before)",
            alpha=0.4,
        )
        ax.plot(
            t,
            raw_after_averaging[i],
            label=f"{subregions_names[i]} (after)",
            linewidth=2,
        )

    ax.set_title("Mean values subregions (before vs after averaging)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout()
    plt.show()

# Butterworth Bandpass filter to isolate heart rate frequencies
#TODO fix
bandpass = False
if bandpass:
    # TODO check again based on NETD values
    # After Savitzky-Golay at 4Hz cutoff, effective fs is 4Hz
    # Nyquist frequency is target_cutoff_hz/2 = 2Hz
    bandpass_lowcut = 0.5  # Hz (min heart rate: 30 BPM)
    bandpass_highcut = 1.9  # Hz (stay safely below Nyquist of 2Hz)
    bandpass_order = 4

    # Apply bandpass filter to all ROI signals
    # Create matrix with rows: subregions, columns: time frames, values: mean pixel intensity of the subregion in the frame
    result = defaultdict(list)
    subregions_names = []

    for frame in roi_pixel_data:
        for roi, values in frame.items():
            if roi not in result:
                subregions_names.append(roi)  # remember first-seen order
            result[roi].append(values["mean_kelvin_multiplied_100"])

    mean_values_subregions = [result[roi] for roi in subregions_names]
    bandpassed_mean_values_subregions = np.zeros_like(mean_values_subregions, dtype=np.float32)

    for subregion in range(len(subregions_names)):
        bandpassed_mean_values_subregions[subregion] = apply_butterworth_bandpass_filter(signal = mean_values_subregions[subregion], lowcut = configs.bandpass_lowcut, highcut = configs.bandpass_highcut, fs = configs.fps, order = configs.bandpass_order)

        if configs.plot_steps:
            # Plot signal before and after bandpass filtering for all subregions
            bandpassed = np.array(bandpassed_mean_values_subregions)
            raw = np.array(mean_values_subregions)
            
            t = np.arange(raw.shape[1])

            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # ---- Axis 0: raw ----
            for i in range(raw.shape[0]):
                axes[0].plot(t, raw[i], label=subregions_names[i])

            axes[0].set_title("Mean values subregions")
            axes[0].set_ylabel("Amplitude")
            axes[0].legend(loc='upper right', fontsize=8)

            # ---- Axis 1: bandpassed ----
            for i in range(bandpassed.shape[0]):
                axes[1].plot(t, bandpassed[i], label=subregions_names[i])

            axes[1].set_title("Bandpassed mean values subregions")
            axes[1].set_xlabel("Time")
            axes[1].set_ylabel("Amplitude")
            axes[1].legend(loc='upper right', fontsize=8)

            plt.tight_layout()
            plt.show()

# Wavelet denoising
def wavelet_denoise_signal(sig, wavelet='db4', level=None, mode='soft'):
    """Denoise a 1D signal using wavelet thresholding.

    Handles NaNs by linear interpolation before denoising.
    """
    sig = np.asarray(sig, dtype=float)
    if sig.size == 0:
        return sig

    # Interpolate NaNs if present
    if np.isnan(sig).any():
        x = np.arange(sig.size)
        mask = ~np.isnan(sig)
        if mask.sum() == 0:
            return sig
        if mask.sum() == 1:
            sig[:] = sig[mask][0]
        else:
            sig_interp = np.copy(sig)
            sig_interp[~mask] = np.interp(x[~mask], x[mask], sig[mask])
            sig = sig_interp

    # Determine maximum useful decomposition level
    try:
        max_level = pywt.dwt_max_level(len(sig), pywt.Wavelet(wavelet).dec_len)
    except Exception:
        max_level = None
    if level is None:
        if max_level is None:
            level = 3
        else:
            level = min(3, max_level) if max_level >= 1 else 1

    # Wavelet decomposition
    coeffs = pywt.wavedec(sig, wavelet, level=level)

    # Estimate noise sigma using the median absolute deviation of the finest detail coeffs
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745 if detail_coeffs.size > 0 else 0.0
    # Universal threshold
    uthresh = sigma * np.sqrt(2 * np.log(len(sig))) if sigma > 0 else 0.0

    # Threshold detail coefficients
    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        new_coeffs.append(pywt.threshold(c, uthresh, mode=mode))

    # Reconstruction
    denoised = pywt.waverec(new_coeffs, wavelet)
    denoised = denoised[: len(sig)]
    return denoised


# Apply wavelet denoising to each subregion (operates on `mean_values_subregions` which
# should be an array-like with shape (n_subregions, T) after previous steps)
try:
    mv = np.asarray(mean_values_subregions)
except Exception:
    mv = None

if mv is not None and mv.size:
    n_sub, T = mv.shape
    wavelet = getattr(configs, 'wavelet_name', 'db4')
    wavelet_level = getattr(configs, 'wavelet_level', None)
    wavelet_mode = getattr(configs, 'wavelet_thresh_mode', 'soft')

    denoised_subregions = np.zeros((n_sub, T), dtype=np.float32)
    for i in range(n_sub):
        denoised = wavelet_denoise_signal(mv[i], wavelet=wavelet, level=wavelet_level, mode=wavelet_mode)
        if denoised.size < T:
            padded = np.zeros(T, dtype=np.float32)
            padded[: denoised.size] = denoised
            denoised = padded
        elif denoised.size > T:
            denoised = denoised[:T]
        denoised_subregions[i] = denoised

    # Replace mean_values_subregions with denoised version for downstream processing
    mean_values_subregions = denoised_subregions

    if configs.plot_steps:
        t = np.arange(T)
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(n_sub):
            ax.plot(t, mv[i], color='C0', alpha=0.4)
            ax.plot(t, denoised_subregions[i], color='C1', linewidth=1.2)
        ax.set_title('Wavelet denoised signals')
        ax.set_xlabel('Frames')
        ax.set_ylabel('Amplitude')
        plt.tight_layout()
        plt.show()

# Peak detection, Peakt ot peak
# TODO peak detection
# TODO understand wavelet
# TODO bandpass filter