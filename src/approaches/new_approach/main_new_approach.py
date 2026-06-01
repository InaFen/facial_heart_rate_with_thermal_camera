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
        result[roi].append(values["mean_value"])

mean_values_subregions = [result[roi] for roi in subregions_names]
raw_before_averaging = np.array(mean_values_subregions)


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



# Show raw signal
# Apply spatial filter: Gaussian kernel (size given by user)
# Show signal after spatial filter
# Bandpass filter/ Highpass and notch filter
# Decide on signal for peak detection: Relative vs absoulte FFT, see notes
