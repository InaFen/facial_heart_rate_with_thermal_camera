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
from src.metrics.metric_pipeline import evaluate_instantaneous_hr_against_ground_truth


CONFIGS_PATH = "src/approaches/new_approach/configs_new_approach.yaml"

configs = Config.from_yaml(CONFIGS_PATH)
output_dir = os.path.dirname(configs.path_video_thermal_with_roi_masks)

print("Expects very small ROIs")

FILTER_TYPE = 'bandpass'  # 'none', 'highpass', or 'bandpass' 
HIGHPASS_CUTOFF_HZ = 0.4
BANDPASS_LOW_HZ = 0.4
BANDPASS_HIGH_HZ = 4.0
FILTER_ORDER = 3
TARGET_HZ = 8.0  # Hz # "Goal frequency length" for which is averaged to have after temporal filtering, has to be big enough to detect heart rate
WAVELET = False

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


# Temporal filtering over ROI to remove NETD noise while preserving heart rate----------------------------
# Can use either moving average or Savitzky-Golay filter (preserves peaks better)

filtering_window_size_frames = int(configs.fps / TARGET_HZ)

if filtering_window_size_frames > 1:
    temporally_filtered_values_subregions = []
    for roi_signal_values in mean_values_subregions:
        roi_signal_array = np.asarray(roi_signal_values)
        # moving_average
        # Average over a sliding window: The result should have target_cutoff_hz (e.g. 8 Hz)
        window_length = min(len(roi_signal_array), filtering_window_size_frames)
        if window_length < 1:
            temporally_filtered_values_subregions.append(roi_signal_array.tolist())
            continue
        averaging_kernel = np.ones(window_length) / window_length
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
    mean_values_subregions_transpose = np.asarray(mean_values_subregions, dtype=float).T  # (T, N)

    # Apply highpass to remove low-frequency nois that was found in NETD analysis
    if FILTER_TYPE == 'highpass':
        filtered_signal = apply_butterworth_highpass_filter(mean_values_subregions_transpose, HIGHPASS_CUTOFF_HZ, configs.fps, FILTER_ORDER)
    # Apply bandpass to isolate heart rate frequencies
    elif FILTER_TYPE == 'bandpass':
        filtered_signal = apply_butterworth_bandpass_filter(
            mean_values_subregions_transpose, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ, configs.fps, FILTER_ORDER
        )
    else:
        raise ValueError(f"Unsupported FILTER_TYPE: {FILTER_TYPE}")

    mean_values_subregions = filtered_signal.T

    if configs.plot_steps:
        t = np.arange(mean_values_subregions.shape[1])
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for i in range(mean_values_subregions_transpose.shape[1]):
            label = subregions_names[i] if i < len(subregions_names) else f"ROI {i}"
            axes[0].plot(t, mean_values_subregions_transpose[:, i], alpha=0.4, label=label)
            axes[1].plot(t, mean_values_subregions[i], linewidth=1.5, label=label)
        axes[0].set_title(f"Filter ({FILTER_TYPE}) before")
        axes[1].set_title(f"Filter ({FILTER_TYPE}) after")
        axes[1].set_xlabel("Frames")
        axes[0].set_ylabel("Amplitude")
        axes[1].set_ylabel("Amplitude")
        axes[0].legend(loc='upper right', fontsize=8, ncol=2)
        axes[1].legend(loc='upper right', fontsize=8, ncol=2)
        plt.tight_layout()
        plt.show()

# Wavelet denoising-------------------
if WAVELET:
    def wavelet_denoise_signal(sig, wavelet='db4', level=None, mode='soft'):
        """Denoise a 1D signal using wavelet thresholding.

        """
        sig = np.asarray(sig, dtype=float)
        if sig.size == 0:
            return sig

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


#outlier detection 
#verteilung rauschen - gaussian noise rausfiltern -> which type of gaussian noise?
# 1. noise type -> eigene methode


# FFT-----------------------------------
# For each signal, compute and plot the FFT.
if configs.plot_steps:
    fft_signals = np.asarray(mean_values_subregions, dtype=float)
    if fft_signals.size:
        fft_min_len = min(series.size for series in fft_signals)
        if fft_min_len > 1:
            fft_signals = np.asarray([series[:fft_min_len] for series in fft_signals], dtype=float)
            fft_freqs = np.fft.rfftfreq(fft_min_len, d=1.0 / float(configs.fps))
            fig, ax = plt.subplots(figsize=(10, 4))
            for i in range(fft_signals.shape[0]):
                fft_mag = np.abs(np.fft.rfft(fft_signals[i]))
                label = subregions_names[i] if i < len(subregions_names) else f"ROI {i}"
                ax.plot(fft_freqs, fft_mag, alpha=0.6, label=label)
            ax.set_title("FFT magnitude per signal")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")
            ax.legend(loc='upper right', fontsize=8, ncol=2)
            plt.tight_layout()
            plt.show()


# Peak detection, Metrics
# TODO how to decide on final signal?
ihr_eval = evaluate_instantaneous_hr_against_ground_truth(
		predicted_signal=mean_values_subregions,
		predicted_fs=float(configs.fps),
		path_ground_truth=configs.path_ground_truth,
		timespan=configs.timespan_signal_to_look_at,
		time_first_frame_signal=float(configs.time_first_frame_signal or 0.0),
		time_first_frame_ecg=float(configs.time_first_frame_ecg or 0.0),
		flip_ground_truth=False,
		tolerance_ms=150.0,
		plot_steps=bool(configs.plot_steps),
		plot_bland_altman=bool(configs.plot_steps),
	)


# TODO understand wavelet
# TODO one signal: use ICA? PCA? Or just pick one with best SNR in FFT? Or look at peaks?
# Adern fließen nihct überall, evtl nicht dominant

# decided on bandpass bc filters out high-, low-frequency noise
# decided on moving average bc with random NETD noise its better than savitzky-golay which preserves peaks but also preserves noise peaks, while moving average smooths out noise more (but also smooths out peaks more, so need to choose window size carefully to not smooth out heart rate peaks too much)