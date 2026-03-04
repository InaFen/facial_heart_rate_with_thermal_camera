import sys
import os
import numpy as np

project_root = "/home/arbeit/Dokumente/master_thesis/"
sys.path.insert(0, project_root)

from src.approaches.new_approach.utils.utils_landmark_detection import apply_facemesh_to_thermal_video, apply_roi_masks_to_video, plot_roi_mean_signals, apply_roi_masks_to_gray_frames, plot_roi_temperature_means
from src.approaches.yang_et_al.utils_face_tracking import track_face, filter_out_pixels_below_threshold
from src.approaches.yang_et_al.utils_spatial_temporal_graph import build_block_features_from_roi_pixel_data, build_spatiotemporal_graph_2, add_centroids_to_roi_data, denoise_via_laplacian
from src.approaches.yang_et_al.utils_spatial_temporal_graph_with_optimal_feature_metric import *
from src.approaches.yang_et_al.utils_plotting import plot_signal_each_roi_seperately, plot_flattened_signal_each_roi, plot_denoised_signal_unflat, plot_original_vs_denoised, plot_principal_components_in_seperate_plots, plot_chosen_component
from src.approaches.yang_et_al.utils_post_processing import pca, apply_butterworth_bandpass_filter, transform_flatten_signal_to_original_shape, find_most_periodic_pca_component
from src.approaches.new_approach.classes.user_input import Config
from src.approaches.new_approach.pipelines.get_roi_data import get_roi_data_pipeline

CONFIGS_PATH = "src/approaches/yang_et_al/configs_yang_et_al.json"

configs = Config.from_json(CONFIGS_PATH)

#IMPORTANT: Each ROI represents one building block in the graph

# original pipeline for face detection
#track_face(in_path= path_video, out_path=path_video_tracked_face)
#filter_out_pixels_below_threshold(video_path=path_video, output_path=path_video_thresholded_pixel, threshold_value=160)

# get face with mediapipe (not original pipeline)
roi_pixel_data, _ = get_roi_data_pipeline(configs=configs)


# Calculate centroids of each ROI in each frame
roi_pixel_data_with_centroids = add_centroids_to_roi_data(roi_pixel_data)


# Create graph and get laplacian
block_features, roi_names = build_block_features_from_roi_pixel_data(roi_pixel_data)

adjacency_matrix, laplacian = build_spatiotemporal_graph_2(block_features=block_features)

#TODO graph creation with optimal feature metric

signal_each_roi_seperate = [f[:, 0:1] for f in block_features] # shape = (T, N, 1)
# plot each ROI signal seperately
plot_signal_each_roi_seperately(block_features=signal_each_roi_seperate, roi_names=roi_names, fps=configs.fps, title="Mean Pixel Value of Each ROI Over Time", show_seconds=True)

signal_all_rois = np.vstack(signal_each_roi_seperate)      # shape = (T*N, 1)
plot_flattened_signal_each_roi(signal_flattened=signal_all_rois, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="Before denoising in changed format")

# compute mean ignoring NaNs
mean_val = np.nanmean(signal_all_rois)
# replace NaNs
signal_all_rois_without_nans = np.nan_to_num(signal_all_rois, nan=mean_val)
plot_flattened_signal_each_roi(signal_flattened=signal_all_rois_without_nans, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="Before denoising with NaNs replaced")


# Denoise signal via laplacian
denoised_signal_all_rois = denoise_via_laplacian(Y=signal_all_rois_without_nans, L=laplacian)
#plot_flattened_signal_each_roi(signal_flattened=denoised_signal_all_rois, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="After denoising")
plot_original_vs_denoised(signal_flattened_orig=signal_all_rois_without_nans, signal_flattened_denoised=denoised_signal_all_rois, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="Original vs Denoised Signals per ROI")

# Bandpass filter
denoised_signal_each_roi_seperate = transform_flatten_signal_to_original_shape(block_features=signal_each_roi_seperate, flattened_signal=denoised_signal_all_rois)
#plot_denoised_signal_unflat(denoised_signal_unflat=denoised_signal_each_roi_seperate, roi_names=roi_names)

bandpassed_signal_each_roi_seperate = apply_butterworth_bandpass_filter(signal=denoised_signal_each_roi_seperate, lowcut=0.667, highcut=1.667, fs=configs.fps, order=2)
plot_denoised_signal_unflat(denoised_signal_unflat=bandpassed_signal_each_roi_seperate, roi_names=roi_names, fps = configs.fps, title="Denoised and Bandpassed Signals per ROI")


# PCA 
prinicpal_components = pca(data=bandpassed_signal_each_roi_seperate, num_components=4)
plot_principal_components_in_seperate_plots(principal_components=prinicpal_components)

# Find most periodic component
selected_pca_component = find_most_periodic_pca_component(principal_components=prinicpal_components)
plot_chosen_component(chosen_component=selected_pca_component, fps = configs.fps)

# Apply FFT to the selected PCA component
fft_selected = np.abs(np.fft.fft(selected_pca_component, n=502))

# Calculate the frequencies corresponding to the FFT bins
frequencies = np.fft.fftfreq(502, d=1/configs.fps)

# Find the dominant frequency
dominant_frequency_index = np.argmax(fft_selected)
dominant_frequency = frequencies[dominant_frequency_index]

# Estimate heart rate
heart_rate_estimate = dominant_frequency * 60
print(f"Estimated heart rate: {heart_rate_estimate:.2f} beats per minute")



print("Finished Yang et al. approach pipeline.")
