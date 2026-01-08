import sys
import os
import numpy as np

project_root = "/home/arbeit/Dokumente/master_thesis/"
sys.path.insert(0, project_root)

from approaches.new_approach.utils_landmark_detection import *
from approaches.yang_et_al.utils_face_tracking import track_face, filter_out_pixels_below_threshold
from approaches.yang_et_al.utils_spatial_temporal_graph import *
from approaches.yang_et_al.utils_spatial_temporal_graph_with_optimal_feature_metric import *
from approaches.yang_et_al.utils_plotting import *
from approaches.yang_et_al.utils_post_processing import *

from approaches.new_approach.utils_add_ground_truth import *
from ground_truth_methods.extract_edf import read_edf, save_signal_to_npy, plot_signal

path_video = 'recorded_videos/thermal_video_12_03.avi'
path_video_tracked_face = 'recorded_videos/yang_et_al/output_tracked_video.mp4'
path_video_thresholded_pixel="recorded_videos/yang_et_al/output_thresholded_video.mp4"
path_video_facemesh = "recorded_videos/yang_et_al/output_facemesh_video.mp4"
path_pixels_log = "saved_data/yang_et_al/roi_pixel_log.npy"
path_video_thermal_with_roi_masks = "recorded_videos/yang_et_al/output_video_with_roi_masks.mp4"
path_roi_pixel_data = "saved_data/yang_et_al/roi_pixel_data.npy"

overwrite_existing_files = False
fps = 8.7

#TODO the code should be similar to Yang et al. but double-check before using

#track_face(in_path= path_video, out_path=path_video_tracked_face)

#filter_out_pixels_below_threshold(video_path=path_video, output_path=path_video_thresholded_pixel, threshold_value=160)

# get face with mediapipe (not original pipeline)
roi_landmarks = {"whole face": [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
    127, 162, 21, 54, 103, 67, 109
]}

roi_landmarks_subgrid_face = {"forehead": [10, 338, 297, 332, 284, 251, 389, 368, 300, 293, 334, 296, 336, 9, 107, 66, 105, 63, 71, 162, 21, 54, 103, 67, 109],#[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 300, 293, 334, 296, 336, 107, 66, 105, 63, 70, 127,103],
"left_cheek": [245, 233, 232, 231, 230, 229, 228, 31, 111, 116, 123, 147, 187, 205, 36, 142, 126, 217, 174, 188],
"right_cheek": [453, 357, 343, 420, 360, 266, 280, 352, 345, 449, 451],
"chin": [202,210, 169, 170, 140, 171, 175, 262, 432, 422, 424, 418, 421, 200, 201, 194, 204]
}

file_path_ground_truth="saved_data/ground_truth/beurer/extracted_npy_files/202512_17_153829.npz"
#file_path_ground_truth="saved_data/ground_truth/opensignals/extracted_npy_files/opensignals_0007800f80fa_2025-12-22_17-19-20.npz"


if not os.path.isfile(path_pixels_log) or overwrite_existing_files:
    roi_pixel_log = apply_facemesh_to_thermal_video(input_path=path_video, output_path=path_video_facemesh, roi_landmarks=roi_landmarks_subgrid_face)
    os.makedirs(os.path.dirname(path_pixels_log), exist_ok=True)
    np.save(path_pixels_log, roi_pixel_log, allow_pickle=True)
else:
    roi_pixel_log = np.load(path_pixels_log, allow_pickle=True)

if not os.path.isfile(path_roi_pixel_data) or overwrite_existing_files:
    roi_pixel_data = apply_roi_masks_to_video(video_path=path_video, roi_mask_log=roi_pixel_log, output_path= path_video_thermal_with_roi_masks)
    os.makedirs(os.path.dirname(path_roi_pixel_data), exist_ok=True)
    np.save(path_roi_pixel_data, roi_pixel_data, allow_pickle=True)
else:
    roi_pixel_data = np.load(path_roi_pixel_data, allow_pickle=True)

#plot_roi_mean_signals(roi_pixel_data=roi_pixel_data) 


# Calculate centroids of each ROI in each frame
roi_pixel_data_with_centroids = add_centroids_to_roi_data(roi_pixel_data)


# Create graph and get laplacian
block_features, roi_names = build_block_features_from_roi_pixel_data(roi_pixel_data)

adjacency_matrix, laplacian = build_spatiotemporal_graph_2(block_features=block_features)

#TODO graph creation with optimal feature metric

signal_each_roi_seperate = [f[:, 0:1] for f in block_features] # shape = (T, N, 1)
# plot each ROI signal seperately
#plot_signal_each_roi_seperately(block_features=signal_each_roi_seperate, roi_names=roi_names)

signal_all_rois = np.vstack(signal_each_roi_seperate)      # shape = (T*N, 1)
#plot_flattened_signal_each_roi(signal_flattened=signal_all_rois, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="Before denoising in changed format")

# compute mean ignoring NaNs
mean_val = np.nanmean(signal_all_rois)
# replace NaNs
signal_all_rois_without_nans = np.nan_to_num(signal_all_rois, nan=mean_val)
#plot_flattened_signal_each_roi(signal_flattened=signal_all_rois_without_nans, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="Before denoising with NaNs replaced")


# Denoise signal via laplacian
denoised_signal_all_rois = denoise_via_laplacian(Y=signal_all_rois_without_nans, L=laplacian)
#plot_flattened_signal_each_roi(signal_flattened=denoised_signal_all_rois, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="After denoising")
#plot_original_vs_denoised(signal_flattened_orig=signal_all_rois_without_nans, signal_flattened_denoised=denoised_signal_all_rois, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="Original vs Denoised Signals per ROI")

# Bandpass filter
denoised_signal_each_roi_seperate = transform_flatten_signal_to_original_shape(block_features=signal_each_roi_seperate, flattened_signal=denoised_signal_all_rois)
#plot_denoised_signal_unflat(denoised_signal_unflat=denoised_signal_each_roi_seperate, roi_names=roi_names)

bandpassed_signal_each_roi_seperate = apply_butterworth_bandpass_filter(signal=denoised_signal_each_roi_seperate, lowcut=0.667, highcut=1.667, fs=fps, order=2)
#plot_denoised_signal_unflat(denoised_signal_unflat=bandpassed_signal_each_roi_seperate, roi_names=roi_names)

# Get ground truth signal
ground_truth_signal, gt_fs, gt_units = load_ground_truth_signal_and_fs(file_path_ground_truth)
plot_signal(title="Ground Truth ECG Signal", signal=ground_truth_signal, xlabel="Samples", ylabel=f"Amplitude ({gt_units})")
ground_truth_signal_denoised = apply_butterworth_bandpass_filter(signal=ground_truth_signal, lowcut=0.667, highcut=1.667, fs=gt_fs, order=2)
#plot_signal(title="Denoised Ground Truth ECG Signal", signal=ground_truth_signal_denoised, xlabel="Samples", ylabel=f"Amplitude ({gt_units})")

plot_denoised_with_ground_truth_offset(denoised_signal_unflat=bandpassed_signal_each_roi_seperate,
                                       roi_names=roi_names,
                                       gt_signal=ground_truth_signal,
                                       fps_denoised=fps,
                                       fps_gt=gt_fs,
                                       gt_start_time=9.4)  

# TODO do they all have the same unit? Ground truth and own signal
"""
# PCA 
prinicpal_components = pca(data=bandpassed_signal_each_roi_seperate, num_components=4)
#plot_principal_components_in_seperate_plots(principal_components=prinicpal_components)

# Find most periodic component
selected_pca_component = find_most_periodic_pca_component(principal_components=prinicpal_components)
#plot_chosen_component(chosen_component=selected_pca_component, fps = fps)

# Apply FFT to the selected PCA component
fft_selected = np.abs(np.fft.fft(selected_pca_component, n=502))

# Calculate the frequencies corresponding to the FFT bins
frequencies = np.fft.fftfreq(502, d=1/fps)

# Find the dominant frequency
dominant_frequency_index = np.argmax(fft_selected)
dominant_frequency = frequencies[dominant_frequency_index]

# Estimate heart rate
heart_rate_estimate = dominant_frequency * 60
print(f"Estimated heart rate: {heart_rate_estimate:.2f} beats per minute")
"""


print("Finished Yang et al. approach pipeline.")
