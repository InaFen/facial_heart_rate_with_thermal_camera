import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

project_root = "/home/arbeit/Dokumente/master_thesis/"
sys.path.insert(0, project_root)

from src.approaches.yang_et_al.utils_face_tracking import track_face, filter_out_pixels_below_threshold_in_roi_pixel_data
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

CONFIGS_PATH = "src/approaches/yang_et_al/configs_yang_et_al.yaml"

# Get configurations
configs = Config.from_yaml(CONFIGS_PATH)

RUN_RANDOM_PARALLEL = False
RANDOM_SEED = 42
APPLY_TRACKER_LANDMARK_DETECTION = True
THRESHOLD_VALUE = 160
print("Hardcoded parameters in main_yang_et_al.py:")
print(f"RUN_RANDOM_PARALLEL: {RUN_RANDOM_PARALLEL}")
print(f"RANDOM_SEED: {RANDOM_SEED}")
print(f"APPLY_TRACKER_LANDMARK_DETECTION: {APPLY_TRACKER_LANDMARK_DETECTION}")
print(f"THRESHOLD_VALUE: {THRESHOLD_VALUE}")

# IMPORTANT: Each ROI represents one building block in the graph
# If ROIs are detected with mediapipe, they have to be set accordingly (e.g. 16 ROIs for 16 building blocks)
# If ROI is manually defined, manual_roi_split_enabled = True so the ROI is split in sub-ROIs


# Face detection and ROI extraction--------------------------------------------------------------------
# original pipeline for face detection # TODO
if APPLY_TRACKER_LANDMARK_DETECTION:
	roi_pixel_data, _ = get_roi_data_pipeline(configs=configs, roi_detection_method="tracker")
	roi_pixel_data = filter_out_pixels_below_threshold_in_roi_pixel_data(roi_pixel_data, threshold_value=THRESHOLD_VALUE)
else:
	# Get face with mediapipe
	roi_pixel_data, _ = get_roi_data_pipeline(configs=configs)


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
plot_signal_each_roi_seperately(block_features=signal_each_roi_seperate, roi_names=roi_names, fps=configs.fps, title="Mean Pixel Value of Each ROI Over Time", show_seconds=True)

signal_all_rois = np.vstack(signal_each_roi_seperate)      # shape = (T*N, 1)
#plot_flattened_signal_each_roi(signal_flattened=signal_all_rois, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="Before denoising in changed format")

# compute mean ignoring NaNs
mean_val = np.nanmean(signal_all_rois)
# replace NaNs
signal_all_rois_without_nans = np.nan_to_num(signal_all_rois, nan=mean_val)
#plot_flattened_signal_each_roi(signal_flattened=signal_all_rois_without_nans, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="Before denoising with NaNs replaced")


# Denoise signal via laplacian
denoised_signal_all_rois = denoise_via_laplacian(Y=signal_all_rois_without_nans, laplacian=laplacian)
#plot_flattened_signal_each_roi(signal_flattened=denoised_signal_all_rois, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="After denoising")
plot_original_vs_denoised(signal_flattened_orig=signal_all_rois_without_nans, signal_flattened_denoised=denoised_signal_all_rois, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="Original vs Denoised Signals per ROI")

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
plot_principal_components_in_seperate_plots(principal_components=prinicpal_components)

# Find most periodic component
selected_pca_component = find_most_periodic_pca_component(principal_components=prinicpal_components)

if random_selected_pca_component is not None:
	selected_scale = float(np.std(selected_pca_component))
	random_selected_scale = float(np.std(random_selected_pca_component))
	if random_selected_scale > 0:
		random_selected_pca_component = random_selected_pca_component * (selected_scale / random_selected_scale)
#plot_chosen_component(chosen_component=selected_pca_component, fps = configs.fps)

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
	plt.show()



# Apply FFT to the selected PCA component (positive frequencies only)---------------------------------------
fft_selected = np.abs(np.fft.rfft(selected_pca_component, n=502))

# Calculate the frequencies corresponding to the FFT bins (non-negative)
frequencies = np.fft.rfftfreq(502, d=1/configs.fps)

# Find the dominant frequency----------------------------------------------------------------------------------
dominant_frequency_index = np.argmax(fft_selected)
dominant_frequency = frequencies[dominant_frequency_index]

if configs.plot_steps:
	plt.figure(figsize=(10, 5))
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
	plt.show()

# Estimate heart rate--------------------------------------------------------------------------------
heart_rate_estimate = dominant_frequency * 60
print(f"Estimated heart rate: {heart_rate_estimate:.2f} beats per minute")



print("Finished Yang et al. approach pipeline.")
