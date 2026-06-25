import sys
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import json
import re
from pathlib import Path
from datetime import datetime
import numpy as np

project_root = "/home/arbeit/Dokumente/master_thesis/"
sys.path.insert(0, project_root)

from src.approaches.yang_et_al.utils_face_tracking import track_face, filter_out_pixels_below_threshold_in_roi_pixel_data, save_threshold_snapshot
from src.approaches.yang_et_al.utils_spatial_temporal_graph import build_block_features_from_roi_pixel_data, build_spatiotemporal_graph, add_centroids_to_roi_data, denoise_via_laplacian
from src.approaches.yang_et_al.utils_plotting import plot_signal_each_roi_seperately, plot_flattened_signal_each_roi, plot_denoised_signal_unflat, plot_original_vs_denoised, plot_principal_components_in_seperate_plots, plot_chosen_component
from src.approaches.yang_et_al.utils_post_processing import pca, apply_butterworth_bandpass_filter, transform_flatten_signal_to_original_shape, find_most_periodic_pca_component
from src.approaches.yang_et_al.utils_ground_truth import (
	load_ground_truth_signal,
	get_ground_truth_segment,
	detect_peaks_in_signal,
	mean_hr_from_peak_times,
)
from src.approaches.yang_et_al.utils_random_baseline import run_random_baseline
from src.approaches.new_approach.classes.user_input import Config
from src.approaches.new_approach.pipelines.get_roi_data import get_roi_data_pipeline
from src.approaches.new_approach.utils_pixels import maybe_split_single_roi_and_save_overlay
from src.approaches.new_approach.utils_water_experiment import (
	load_high_temp_bands,
	get_water_timestamps_path,
)

CONFIGS_PATH = os.environ.get("YANG_CONFIG_PATH", "src/approaches/yang_et_al/configs_yang_et_al.yaml")
# Where plots are written. run_batch sets YANG_PLOTS_SAVE_DIR per run; when running
# this script directly the env var is unset, so fall back to a default directory so
# plots are ONLY saved (never shown via plt.show()), matching the batch behaviour.
PLOTS_SAVE_DIR = os.environ.get(
	"YANG_PLOTS_SAVE_DIR",
	str(Path(project_root) / "saved_results" / "yang_et_al" / "plots"),
)

# Get configurations
configs = Config.from_yaml(CONFIGS_PATH)

RUN_RANDOM_PARALLEL = False
RANDOM_SEED = 42
# Water experiment: when True, the chosen PCA component is saved twice — once plain
# (chosen_component.png) and once with the high-temperature time spans from the
# new-approach water experiment shaded a slight blue (chosen_component_highlighted.png).
WATER_EXPERIMENT = True


def _env_flag(name: str, default: bool) -> bool:
	"""Read a boolean flag from the environment (1/true/yes -> True)."""
	value = os.environ.get(name)
	if value is None:
		return default
	return value.strip().lower() in ("1", "true", "yes")


def _parse_threshold(value: str):
	"""Parse the threshold env var, returning an int for whole numbers."""
	number = float(value)
	return int(number) if number.is_integer() else number


# Whether to use the tracker-based ROI detection (else MediaPipe facemesh). When
# launched by run_batch this is set via YANG_APPLY_TRACKER_LANDMARK_DETECTION
# (1 when the roi name starts with "tracker" and the pipeline is "yang").
APPLY_TRACKER_LANDMARK_DETECTION = _env_flag("YANG_APPLY_TRACKER_LANDMARK_DETECTION", True)
# Temperature threshold (only keep pixels above it) used with tracker landmark
# detection. Camera-dependent; run_batch sets YANG_THRESHOLD_VALUE per camera.
THRESHOLD_VALUE = _parse_threshold(os.environ.get("YANG_THRESHOLD_VALUE", "29"))
print("Hardcoded parameters in main_yang_et_al.py:")
print(f"RUN_RANDOM_PARALLEL: {RUN_RANDOM_PARALLEL}")
print(f"RANDOM_SEED: {RANDOM_SEED}")
print(f"APPLY_TRACKER_LANDMARK_DETECTION: {APPLY_TRACKER_LANDMARK_DETECTION}")
print(f"THRESHOLD_VALUE: {THRESHOLD_VALUE}")
print(f"PLOTS_SAVE_DIR: {PLOTS_SAVE_DIR}")

# If exactly one landmark ROI from this map is used, it is split into a
# rows x rows grid of sub-ROIs (each sub-ROI becomes its own graph building
# block) and a sub-ROI overlay video + first-frame image are saved
# (mirroring main_new_approach).
SUBROI_GRID_ROWS_BY_ROI = {
	"forehead": 3,
	"left_forehead": 2,
	"right_forehead": 2,
	"glabella": 2,
	"manual_rectangle_front_view": 6,
	"manual_rectangle": 2,
	"roi_signal_water": 2
}

# IMPORTANT: Each ROI represents one building block in the graph
# If ROIs are detected with mediapipe, they have to be set accordingly (e.g. 16 ROIs for 16 building blocks)
# If ROI is manually defined, manual_roi_split_enabled = True so the ROI is split in sub-ROIs

# Face detection and ROI extraction--------------------------------------------------------------------
if APPLY_TRACKER_LANDMARK_DETECTION:
	roi_pixel_data, _ = get_roi_data_pipeline(configs=configs, roi_detection_method="tracker")
	roi_pixel_data = filter_out_pixels_below_threshold_in_roi_pixel_data(
		roi_pixel_data,
		threshold_value=THRESHOLD_VALUE,
		output_video_path=configs.path_video_thresholded_pixel,
		fps=configs.fps,
	)

	# Save a snapshot of what is left after the threshold: the full thermal frame
	# stays visible (grayscale) and the kept (surviving) pixels are colored,
	# mirroring the roi_frame_*.png saved by the ROI pipeline.
	threshold_snapshot_dir = (
		os.path.dirname(configs.path_video_thresholded_pixel)
		or os.path.dirname(configs.path_video_tracked_face)
	)
	threshold_snapshot_path = os.path.join(
		threshold_snapshot_dir, f"roi_frame_threshold_{THRESHOLD_VALUE}.png"
	)
	save_threshold_snapshot(
		configs=configs,
		roi_pixel_data=roi_pixel_data,
		threshold_value=THRESHOLD_VALUE,
		output_path=threshold_snapshot_path,
	)

else:
	# Get face with mediapipe
	roi_pixel_data, _ = get_roi_data_pipeline(configs=configs)


# If a single ROI from SUBROI_GRID_ROWS_BY_ROI is used, split it into sub-ROIs so
# each sub-ROI becomes its own graph building block; also save the sub-ROI overlay
# video + first-frame image (as in main_new_approach).
overlay_output_dir = PLOTS_SAVE_DIR or str(Path(project_root) / "saved_results" / "yang_et_al" / "plots")
roi_pixel_data, _was_split_into_subrois = maybe_split_single_roi_and_save_overlay(
	roi_pixel_data,
	configs,
	overlay_output_dir,
	SUBROI_GRID_ROWS_BY_ROI,
)


# Calculate centroids of each ROI in each frame (change in place)
_ = add_centroids_to_roi_data(roi_pixel_data)


# Create graph and get laplacian--------------------------------------------------------------------
# Each block is one ROI (see information above), and features for each block include:
# - Mean pixel value
# - Centroid x-coordinate
# - Centroid y-coordinate
block_features, roi_names = build_block_features_from_roi_pixel_data(roi_pixel_data)

adjacency_matrix, laplacian = build_spatiotemporal_graph(block_features=block_features)

signal_each_roi_seperate = [f[:, 0:1] for f in block_features] # shape = (T, N, 1)
# plot each ROI signal seperately
plot_signal_each_roi_seperately(block_features=signal_each_roi_seperate, roi_names=roi_names, fps=configs.fps, title="Mean Pixel Value of Each ROI Over Time", show_seconds=True, save_path=PLOTS_SAVE_DIR)

signal_all_rois = np.vstack(signal_each_roi_seperate)      # shape = (T*N, 1)
#plot_flattened_signal_each_roi(signal_flattened=signal_all_rois, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="Before denoising in changed format")

# compute mean ignoring NaNs (used as fallback if a ROI is fully NaN)
mean_val = np.nanmean(signal_all_rois)
if np.isnan(mean_val):
	mean_val = 0.0

signal_all_rois_unflat = transform_flatten_signal_to_original_shape(
	block_features=signal_each_roi_seperate,
	flattened_signal=signal_all_rois,
)
signal_all_rois_unflat = signal_all_rois_unflat.astype(float)

# Replace NaNs via per-ROI temporal interpolation (fallback to global mean if a ROI is fully NaN)
for roi_index in range(signal_all_rois_unflat.shape[1]):
	y = signal_all_rois_unflat[:, roi_index]
	mask = ~np.isnan(y)
	if mask.sum() == 0:
		signal_all_rois_unflat[:, roi_index] = mean_val
	elif mask.sum() == 1:
		signal_all_rois_unflat[:, roi_index] = y[mask][0]
	else:
		x = np.arange(y.size)
		signal_all_rois_unflat[:, roi_index] = np.interp(x, x[mask], y[mask])

signal_all_rois_without_nans = signal_all_rois_unflat.reshape(-1, 1)
#plot_flattened_signal_each_roi(signal_flattened=signal_all_rois_without_nans, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="Before denoising with NaNs replaced")


# Denoise signal via laplacian
denoised_signal_all_rois = denoise_via_laplacian(Y=signal_all_rois_without_nans, laplacian=laplacian)
#plot_flattened_signal_each_roi(signal_flattened=denoised_signal_all_rois, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="After denoising")
plot_original_vs_denoised(signal_flattened_orig=signal_all_rois_without_nans, signal_flattened_denoised=denoised_signal_all_rois, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="Original vs Denoised Signals per ROI", save_path=PLOTS_SAVE_DIR)

random_selected_pca_component = None
if RUN_RANDOM_PARALLEL:
		random_selected_pca_component = run_random_baseline(
		signal_all_rois_without_nans=signal_all_rois_without_nans,
		signal_each_roi_seperate=signal_each_roi_seperate,
		roi_names=roi_names,
		laplacian=laplacian,
		fps=configs.fps,
			plot_steps=bool(configs.plot_steps),
		random_seed=RANDOM_SEED,
	)

# Bandpass filter--------------------------------------------------------------
denoised_signal_each_roi_seperate = transform_flatten_signal_to_original_shape(block_features=signal_each_roi_seperate, flattened_signal=denoised_signal_all_rois)
#plot_denoised_signal_unflat(denoised_signal_unflat=denoised_signal_each_roi_seperate, roi_names=roi_names)

bandpassed_signal_each_roi_seperate = apply_butterworth_bandpass_filter(signal=denoised_signal_each_roi_seperate, lowcut=configs.bandpass_lowcut, highcut=configs.bandpass_highcut, fs=configs.fps, order=configs.bandpass_order)
#plot_denoised_signal_unflat(denoised_signal_unflat=bandpassed_signal_each_roi_seperate, roi_names=roi_names, fps = configs.fps, title="Denoised and Bandpassed Signals per ROI")


# PCA ---------------------------------------------------------------------
prinicpal_components = pca(data=bandpassed_signal_each_roi_seperate, num_components=4)
plot_principal_components_in_seperate_plots(principal_components=prinicpal_components, save_path=PLOTS_SAVE_DIR)

# Find most periodic component
selected_pca_component = find_most_periodic_pca_component(principal_components=prinicpal_components)

if random_selected_pca_component is not None:
	selected_scale = float(np.std(selected_pca_component))
	random_selected_scale = float(np.std(random_selected_pca_component))
	if random_selected_scale > 0:
		random_selected_pca_component = random_selected_pca_component * (selected_scale / random_selected_scale)
plot_chosen_component(chosen_component=selected_pca_component, fps=configs.fps, save_path=PLOTS_SAVE_DIR)
if WATER_EXPERIMENT:
	# Reuse the high-temperature time spans saved by the new-approach water experiment
	# so the chosen component is highlighted over the exact same periods.
	water_bands = load_high_temp_bands(fps=configs.fps)
	if not water_bands:
		print(
			"[yang] WATER_EXPERIMENT is True but no high-temperature bands were found at "
			f"{get_water_timestamps_path()} (run the new-approach water experiment first)."
		)
	plot_chosen_component(
		chosen_component=selected_pca_component,
		fps=configs.fps,
		save_path=PLOTS_SAVE_DIR,
		highlight_bands=water_bands,
		plot_name="chosen_component_highlighted",
	)

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

	signal_time = np.arange(len(selected_pca_component)) / configs.fps + start_time

	min_distance_signal = int(0.3 * configs.fps)
	signal_peaks, _ = find_peaks(selected_pca_component, distance=max(1, min_distance_signal))
	signal_peak_times = signal_time[signal_peaks]

	ground_truth_peaks, ground_truth_peak_times = detect_peaks_in_signal(
		ground_truth_segment,
		ground_truth_fs,
		min_distance_s=0.5,
		prominence=1.0,
	)
	ground_truth_peak_times = ground_truth_peak_times + start_time

	gt_hr_mean = mean_hr_from_peak_times(ground_truth_peak_times)
	if gt_hr_mean is not None:
		print(f"Ground truth mean HR in timespan: {gt_hr_mean:.2f} bpm")
	else:
		print("Ground truth mean HR in timespan: insufficient peaks")

	signal_hr_mean = mean_hr_from_peak_times(signal_peak_times)
	if signal_hr_mean is not None:
		print(f"Signal mean HR in timespan before PCA: {signal_hr_mean:.2f} bpm")
	else:
		print("Signal mean HR in timespan before PCA: insufficient peaks")

	plt.figure(figsize=(10, 5))
	signal_line, = plt.plot(signal_time, selected_pca_component, label="Selected PCA component")
	ground_truth_line, = plt.plot(ground_truth_time, ground_truth_segment, label="Ground truth", alpha=0.7)
	if bool(configs.plot_steps) and random_selected_pca_component is not None:
		random_time = np.arange(len(random_selected_pca_component)) / configs.fps + start_time
		random_line, = plt.plot(random_time, random_selected_pca_component, label="Random pipeline component", alpha=0.7)
		min_distance_random = int(0.3 * configs.fps)
		random_peaks, _ = find_peaks(
			random_selected_pca_component,
			distance=max(1, min_distance_random),
		)
		random_peak_times = random_time[random_peaks]
		plt.scatter(
			random_peak_times,
			random_selected_pca_component[random_peaks],
			label="_nolegend_",
			s=25,
			color=random_line.get_color(),
		)

		random_hr_mean = mean_hr_from_peak_times(random_peak_times)
		if random_hr_mean is not None:
			print(f"Random mean HR in timespan: {random_hr_mean:.2f} bpm")
		else:
			print("Random mean HR in timespan: insufficient peaks")
	plt.scatter(
		signal_peak_times,
		selected_pca_component[signal_peaks],
		label="_nolegend_",
		s=25,
		color=signal_line.get_color(),
	)
	plt.scatter(
		ground_truth_peak_times,
		ground_truth_segment[ground_truth_peaks],
		label="_nolegend_",
		s=25,
		color=ground_truth_line.get_color(),
	)
	plt.xlabel("Time (s)")
	plt.ylabel("Amplitude")
	plt.title("Selected PCA Component vs Ground Truth")
	plt.legend()
	plt.tight_layout()
	if PLOTS_SAVE_DIR is not None:
		save_path = Path(PLOTS_SAVE_DIR)
		save_path.mkdir(parents=True, exist_ok=True)
		save_file = save_path / "pca_component_vs_ground_truth.png"
		plt.savefig(save_file, dpi=100, bbox_inches='tight')
		print(f"Saved plot to: {save_file}")
		plt.close()
	else:
		plt.show()



# Apply FFT to the selected PCA component (positive frequencies only)---------------------------------------
fft_selected = np.abs(np.fft.rfft(selected_pca_component, n=502))

# Calculate the frequencies corresponding to the FFT bins (non-negative)
frequencies = np.fft.rfftfreq(502, d=1/configs.fps)

# Find the dominant frequency----------------------------------------------------------------------------------
dominant_frequency_index = np.argmax(fft_selected)
dominant_frequency = frequencies[dominant_frequency_index]

if configs.plot_steps:
	fig = plt.figure(figsize=(10, 5))
	plt.plot(frequencies, fft_selected, label="FFT magnitude")
	plt.axvline(
		dominant_frequency,
		color="red",
		linestyle="--",
		label=f"Dominant: {dominant_frequency:.3f} Hz",
	)
	plt.xlabel("Frequency (Hz)")
	plt.ylabel("Magnitude")
	plt.title("FFT of Selected PCA Component")
	plt.legend()
	plt.tight_layout()
	if PLOTS_SAVE_DIR is not None:
		save_path = Path(PLOTS_SAVE_DIR)
		save_path.mkdir(parents=True, exist_ok=True)
		save_file = save_path / "fft_selected_pca_component.png"
		fig.savefig(save_file, dpi=100, bbox_inches='tight')
		print(f"Saved plot to: {save_file}")
		plt.close(fig)
	else:
		plt.show()

# Estimate heart rate--------------------------------------------------------------------------------
heart_rate_estimate = dominant_frequency * 60
print(f"Estimated heart rate: {heart_rate_estimate:.2f} beats per minute")



print("Finished Yang et al. approach pipeline.")

# Save results at end of pipeline
try:
	# Directory for saved results inside project, grouped per camera
	# (e.g. saved_results/nair_et_al/results/vario). The camera name is provided
	# by run_batch via the CAMERA_NAME env var; falls back to "unknown".
	camera_name = os.environ.get("CAMERA_NAME", "unknown") or "unknown"
	save_root = Path(project_root) / "saved_results" / "yang_et_al" / "results" / camera_name
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
	# sanitize to safe filename (keep alnum, dash and underscore)
	safe_base = re.sub(r"[^A-Za-z0-9_-]", "_", base_name)
	save_file = save_root / f"{safe_base}.npz"

	metadata = {
		"pipeline": "yang_et_al",
		"path_signal": getattr(configs, "path_gray_frames", ""),
		"path_ground_truth": getattr(configs, "path_ground_truth", ""),
		"timespan_signal_to_look_at": getattr(configs, "timespan_signal_to_look_at", None),
		"roi_name": getattr(configs, "roi_name", ""),
		"heart_rate_estimate_bpm": float(heart_rate_estimate),
		"fps": float(configs.fps),
		"fps_ground_truth": float(ground_truth_fs) if configs.timespan_signal_to_look_at else None,
	}

	if "ground_truth_segment" in globals():
		np.savez_compressed(
			save_file,
			predicted_signal=np.asarray(selected_pca_component, dtype=float),
			ground_truth_segment=np.asarray(ground_truth_segment, dtype=float),
			heart_rate_estimate=float(heart_rate_estimate),
			metadata=json.dumps(metadata),
		)
	else:
		np.savez_compressed(
			save_file,
			predicted_signal=np.asarray(selected_pca_component, dtype=float),
			heart_rate_estimate=float(heart_rate_estimate),
			metadata=json.dumps(metadata),
		)


	# Also write metadata as a separate, human-readable JSON file next to the .npz
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
