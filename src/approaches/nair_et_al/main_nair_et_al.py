import sys
from collections import defaultdict
import numpy as np
import os
import json
import re
from scipy.signal import stft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from utils import generate_ppg_signal, extract_curve
from src.approaches.new_approach.classes.user_input import Config

project_root = "/home/arbeit/Dokumente/master_thesis/"
sys.path.insert(0, project_root)

from src.approaches.new_approach.pipelines.get_roi_data import get_roi_data_pipeline
from src.approaches.yang_et_al.utils_post_processing import apply_butterworth_bandpass_filter
from src.approaches.yang_et_al.utils_ground_truth import (
	load_ground_truth_signal,
	get_ground_truth_segment
)

CONFIGS_PATH = os.environ.get("NAIR_CONFIG_PATH", "src/approaches/nair_et_al/configs_nair_et_al.yaml")
PLOTS_SAVE_DIR = os.environ.get("NAIR_PLOTS_SAVE_DIR", None)
PLOTS_OUTPUT_DIR = Path(PLOTS_SAVE_DIR) if PLOTS_SAVE_DIR else Path(project_root) / "saved_results" / "nair_et_al" / "plots"

configs = Config.from_yaml(CONFIGS_PATH)
output_dir = os.path.dirname(configs.path_video_thermal_with_roi_masks)

APPLY_DLIB_LANDMARK_DETECTION = True
STFT_WINDOW_SIZE = 64
STFT_OVERLAP = 56
SMOOTHNESS_PENALTY = 0.1 # used in curve extraction, higher values lead to smoother curve but less accurate, lower values lead to more accurate but less smooth curve
NORMAL_HR_FREQ = 1.3 #Hz, 80 bpm # used for signal quality indices, only consider frequencies around normal heart rate frequency

print("Hardcoded parameters in main_nair_et_al.py:")
print(f"APPLY_DLIB_LANDMARK_DETECTION: {APPLY_DLIB_LANDMARK_DETECTION}")
print(f"STFT_WINDOW_SIZE: {STFT_WINDOW_SIZE}")
print(f"STFT_OVERLAP: {STFT_OVERLAP}")
print(f"SMOOTHNESS_PENALTY: {SMOOTHNESS_PENALTY}")
print(f"NORMAL_HR_FREQ: {NORMAL_HR_FREQ}")
print(f"ROI selection name: {getattr(configs, 'roi_name', None)}")
print(f"ROI selection used_rois: {getattr(configs, 'used_rois', None)}")

def save_figure(fig: plt.Figure, filename: str) -> None:
    PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

if not getattr(configs, "used_rois", None):
    raise ValueError("configs.used_rois must be defined in configs_nair_et_al.yaml, similar to the Yang-style YAML.")
# ROI landmark definitions live in the shared file
# `src/approaches/new_approach/roi_landmarks_options.yaml` (merged with any inline
# configs.roi_landmarks_options). Verify the requested ROIs resolve.
from src.approaches.new_approach.pipelines.get_roi_data import (
    get_roi_landmarks_options,
    select_roi_landmarks,
)
_roi_options = get_roi_landmarks_options(configs)
if not _roi_options:
    raise ValueError(
        "No ROI landmark definitions found. Define them in the shared file "
        "'src/approaches/new_approach/roi_landmarks_options.yaml' (or inline in configs_nair_et_al.yaml)."
    )
# Raises if any name in used_rois is unknown.
select_roi_landmarks(configs.used_rois, _roi_options)

############
# Preprocessing: Landmark detection & denoising 
############
print("IMPORTANT: If mediapipe is applied, each ROI will not be split further so needs to be defined accordingly in configs.")

if APPLY_DLIB_LANDMARK_DETECTION:
    roi_pixel_data, _ = get_roi_data_pipeline(configs=configs, roi_detection_method="dlib")
else:
    # Apply Facemesh landmark detection
    roi_pixel_data, _ = get_roi_data_pipeline(configs=configs)


# Apply bandpass filter
print("Applying bandpass filter to mean values of subregions...")

# Create matrix with rows: subregions, columns: time frames, values: mean pixel intensity of the subregion in the frame
result = defaultdict(list)
subregions_names = []

for frame in roi_pixel_data:
    for roi, values in frame.items():
        if roi not in result:
            subregions_names.append(roi)  # remember first-seen order
        result[roi].append(values["mean_value"])

mean_values_subregions = [result[roi] for roi in subregions_names]

# Interpolate NaNs per subregion over time (fallback to global mean if fully NaN)
flat_values = np.concatenate([np.asarray(values, dtype=float) for values in mean_values_subregions])
global_mean = np.nanmean(flat_values) if flat_values.size else 0.0
if np.isnan(global_mean):
    global_mean = 0.0

interpolated_values_subregions = []
for values in mean_values_subregions:
    signal = np.asarray(values, dtype=float)
    if signal.size == 0:
        interpolated_values_subregions.append(signal.tolist())
        continue

    mask = ~np.isnan(signal)
    if mask.sum() == 0:
        signal = np.full_like(signal, fill_value=global_mean, dtype=float)
    elif mask.sum() == 1:
        signal[:] = signal[mask][0]
    else:
        x = np.arange(signal.size)
        signal[~mask] = np.interp(x[~mask], x[mask], signal[mask])

    interpolated_values_subregions.append(signal.tolist())

mean_values_subregions = interpolated_values_subregions
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
        save_figure(fig, "01_bandpass_before_after.png")

# Apply SVD
# Should be:
# Rows = regions -> weight of the signals in the spatial location in U (Nair et al.)
# Columns = time frames -> siginificant temporal signals in V (Nair et al.)
U, S, Vh = np.linalg.svd(bandpassed_mean_values_subregions, full_matrices=False)

if configs.plot_steps:
    num_components = Vh.shape[0]
    fig = plt.figure(figsize=(12, 6))
    for i in range(num_components):
        plt.subplot(num_components, 1, i + 1)
        plt.plot(Vh[i, :])
        plt.title(f'Component {i + 1}')
        plt.xlabel('Frame')
        plt.ylabel('Amplitude')
        plt.grid()
    plt.tight_layout()
    save_figure(fig, "02_svd_components.png")

# Apply signal quality indices to select the best accumalated components
optimal_vh_components = generate_ppg_signal(signals = Vh, fs = configs.fps, normal_hr_freq = NORMAL_HR_FREQ)

if configs.plot_steps:
    fig = plt.figure(figsize=(12, 6))
    plt.plot(optimal_vh_components[:])
    plt.title(f'Optimal Component')
    plt.xlabel('Frame')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.tight_layout()
    save_figure(fig, "03_optimal_component.png")

# Short-Time-Fourier-Transform
frequencies, times, short_fourier_ppg = stft(optimal_vh_components, fs=configs.fps, nperseg=STFT_WINDOW_SIZE,noverlap=STFT_OVERLAP)
short_time_fourier_ppg_magnitude = np.abs(short_fourier_ppg).T # Return shape (time_frames, magnitude at frequencies) 

# Convert STFT frequencies (Hz) -> BPM
heart_rate_candidates = frequencies * 60

# Choose smooth, dominant curve in the STFT magnitude as heart rate curve
hr_curve = extract_curve(short_time_fourier_ppg_magnitude, heart_rate_candidates, SMOOTHNESS_PENALTY)

if configs.plot_steps:
    fig = plt.figure(figsize=(12, 6))
    plt.plot(hr_curve[:])
    plt.title(f'Extracted Heart Rate Curve')
    plt.xlabel('Time Windows')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.tight_layout()
    save_figure(fig, "04_extracted_hr_curve.png")

# HR Prediction
# Calculate the time duration (T) of the signal
T = len(optimal_vh_components) / configs.fps

# Convert the curve to heart rate (HR)
hr = (hr_curve * configs.fps) / T
print("Predicted Heart Rate:", hr)


print("Finished Nair et al. approach pipeline.")

# Ground truth steps---------------------------------------------------------
if configs.timespan_signal_to_look_at is not None:
	start_time, end_time = configs.timespan_signal_to_look_at
	signal_offset = float(configs.time_first_frame_signal or 0.0)
	ecg_offset = float(configs.time_first_frame_ecg or 0.0)
	ground_truth_signal, ground_truth_fs = load_ground_truth_signal(
		configs.path_ground_truth,
		flip=False,
	)
	ground_truth_segment, ground_truth_time = get_ground_truth_segment(
		ground_truth_signal,
		ground_truth_fs,
		start_time,
		end_time,
		signal_offset=signal_offset,
		ecg_offset=ecg_offset,
	)


# Save results at end of pipeline
try:
    # Directory for saved results inside project
    save_root = Path(project_root) / "saved_results" / "nair_et_al"
    save_root.mkdir(parents=True, exist_ok=True)

    # Build descriptive filename: signalName_roiName_timespan
    signal_path_cfg = getattr(configs, "path_gray_frames", "") or ""
    signal_name = Path(signal_path_cfg).stem if signal_path_cfg else "signal"
    roi_name_cfg = getattr(configs, "roi_name", "roi") or "roi"
    timespan_cfg = getattr(configs, "timespan_signal_to_look_at", None)
    if timespan_cfg:
        try:
            start_s, end_s = timespan_cfg
            timespan_str = f"{float(start_s):.2f}-{float(end_s):.2f}s"
        except Exception:
            timespan_str = str(timespan_cfg)
    else:
        timespan_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_name = f"{signal_name}_{roi_name_cfg}_{timespan_str}"
    safe_base = re.sub(r"[^A-Za-z0-9_-]", "_", base_name)
    save_file = save_root / f"{safe_base}.npz"

    metadata = {
        "pipeline": "nair_et_al",
        "path_signal": getattr(configs, "path_gray_frames", ""),
        "path_ground_truth": getattr(configs, "path_ground_truth", ""),
        "timespan_signal_to_look_at": getattr(configs, "timespan_signal_to_look_at", None),
        "roi_name": getattr(configs, "roi_name", ""),
        "heart_rate_estimate_bpm": float(np.mean(hr)),
        "fps": float(configs.fps),
        "fps_ground_truth": float(ground_truth_fs) if "ground_truth_fs" in globals() else None,
    }

    if "ground_truth_segment" in globals():
        np.savez_compressed(
            save_file,
            predicted_signal=np.asarray(optimal_vh_components[:], dtype=float),
            ground_truth_segment=np.asarray(ground_truth_segment, dtype=float),
            heart_rate_estimate=float(np.mean(hr)),
            metadata=json.dumps(metadata),
        )
    else:
        np.savez_compressed(
            save_file,
            predicted_signal=np.asarray(optimal_vh_components[:], dtype=float),
            heart_rate_estimate=float(hr),
            metadata=json.dumps(metadata),
        )

    try:
        meta_path = save_file.with_suffix('.json')
        with open(meta_path, 'w', encoding='utf-8') as jf:
            json.dump(metadata, jf, ensure_ascii=False, indent=2)
        print(f"Saved pipeline results to: {save_file}")
        print(f"Saved metadata JSON to: {meta_path}")
    except Exception as e:
        print(f"Warning: failed to save metadata JSON: {e}")
except Exception as e:
    print(f"Warning: failed to save results: {e}")
    