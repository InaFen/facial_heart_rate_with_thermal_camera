import sys
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
import pywt


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)

from src.approaches.new_approach.classes.user_input import Config
from src.approaches.yang_et_al.utils_post_processing import apply_butterworth_bandpass_filter

from src.approaches.new_approach.pipelines.get_roi_data import get_roi_data_pipeline

CONFIGS_PATH = "src/approaches/new_approach/configs_new_approach.yaml"

configs = Config.from_yaml(CONFIGS_PATH)
output_dir = os.path.dirname(configs.path_video_thermal_with_roi_masks)

TEMPORAL_FILTER_TYPE = 'savitzky_golay' #'moving_average' or 'savitzky_golay'
FILTER_TYPE = 'highpass'  # 'none', 'highpass', or 'bandpass'
HIGHPASS_CUTOFF_HZ = 0.7
BANDPASS_LOW_HZ = 0.7
BANDPASS_HIGH_HZ = 4.0
FILTER_ORDER = 3

# Get ROIs----------------------------------------------------------------------------
roi_pixel_data, _ = get_roi_data_pipeline(configs=configs)

# Spatial averaging of each ROI (should be very small ROIs)----------------------------
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

# TODO look at details of temporal filtering

# Savitzky-Golay acts as a low-pass filter. Set target cutoff frequency in Hz
# Larger window = lower cutoff frequency (more smoothing)
# For heart rate detection: 3-5 Hz is typical
target_cutoff_hz = 8.0  # Hz (tune this for desired noise reduction)

# Approximate window size for target cutoff frequency
# Rule of thumb: window_length ≈ fps / cutoff_hz
filtering_window_size_frames = max(3, int(configs.fps / target_cutoff_hz))
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
        
        if TEMPORAL_FILTER_TYPE == 'savitzky_golay':
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

# Interpolate NaNs per subregion over time (fallback to global mean if a subregion is fully NaN)
mean_values_series = [np.asarray(series, dtype=float) for series in mean_values_subregions]
flat_values = np.concatenate([series for series in mean_values_series if series.size])
global_mean = np.nanmean(flat_values) if flat_values.size else 0.0
if np.isnan(global_mean):
    global_mean = 0.0

interpolated_series = []
for series in mean_values_series:
    if series.size == 0:
        interpolated_series.append(series)
        continue

    mask = ~np.isnan(series)
    if mask.sum() == 0:
        series = np.full_like(series, fill_value=global_mean, dtype=float)
    elif mask.sum() == 1:
        series[:] = series[mask][0]
    else:
        x = np.arange(series.size)
        series[~mask] = np.interp(x[~mask], x[mask], series[mask])

    interpolated_series.append(series)

mean_values_subregions = [series.tolist() for series in interpolated_series]


def apply_butterworth_highpass_filter(signal, cutoff, fs, order):
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive. Got fs={fs}.")

    nyquist = 0.5 * fs
    if not (0 < cutoff < nyquist):
        raise ValueError(
            "Highpass cutoff must satisfy 0 < cutoff < Nyquist. "
            f"Got cutoff={cutoff}, Nyquist={nyquist}."
        )

    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype='high')
    return filtfilt(b, a, signal, axis=0)


if FILTER_TYPE != 'none':
    filtered_series = [np.asarray(series, dtype=float) for series in mean_values_subregions]
    if not filtered_series:
        raise ValueError("No data available for filtering.")

    min_length = min(series.size for series in filtered_series)
    if min_length < 2:
        raise ValueError("Not enough samples for FFT/filtering.")

    filtered_input = np.vstack([series[:min_length] for series in filtered_series])

    signal_for_filter = filtered_input.T
    if configs.plot_steps:
        n_samples = signal_for_filter.shape[0]
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / configs.fps)
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(filtered_input.shape[0]):
            spectrum = np.abs(np.fft.rfft(filtered_input[i]))
            ax.plot(freqs, spectrum, alpha=0.4)
        ax.set_title("FFT before filtering")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        plt.tight_layout()
        plt.show()
    if FILTER_TYPE == 'highpass':
        filtered_signal = apply_butterworth_highpass_filter(
            signal_for_filter, HIGHPASS_CUTOFF_HZ, configs.fps, FILTER_ORDER
        )
    elif FILTER_TYPE == 'bandpass':
        filtered_signal = apply_butterworth_bandpass_filter(
            signal_for_filter, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ, configs.fps, FILTER_ORDER
        )
    else:
        raise ValueError(f"Unsupported FILTER_TYPE: {FILTER_TYPE}")

    mean_values_subregions = filtered_signal.T

    if configs.plot_steps:
        t = np.arange(mean_values_subregions.shape[1])
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(filtered_input.shape[0]):
            ax.plot(t, filtered_input[i], alpha=0.4, color='C0')
            ax.plot(t, mean_values_subregions[i], color='C1', linewidth=1.5)
        ax.set_title(f"Filter ({FILTER_TYPE}) before vs after")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

# Butterworth Bandpass filter to isolate heart rate frequencies
#TODO fix

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