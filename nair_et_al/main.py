from utils import *
from helpers_landmark_detection import *
import numpy as np
import os

edges_bandpass_filter = [0.1, 4.4]  # in Hz
video_path = 'recorded_videos/thermal_video_12_03.avi'  # Path to the recorded thermal video
filtered_signal_matrix_path = f'filtered_signal_matrix_{edges_bandpass_filter[0]}_{edges_bandpass_filter[1]}.npy'
normalized_signal_matrix_path = 'normalized_signal_matrix.npy'
first_derivative_matrix_path = 'first_derivative_matrix.npy'
second_derivative_matrix_path = 'second_derivative_matrix.npy'
mean_of_subregions_path = 'mean_values_subregions.npy'
fps = 9.0

path_save_video_with_landmarks = 'recorded_videos/video_with_landmarks.avi'  # Path to Dlib landmark model
filtered_mean_values_subregions_matrix_path = f'filtered_mean_values_subregions.npy'

PREPROCESSING = False
#############
# Preprocessing
#############

if PREPROCESSING:
    # Load video and get raw signal
    raw_signal_matrix = get_raw_signal(video_path)

    x_visualization = 60
    y_visualization = 80


    # Bandpass filter (according to paper, here 0.1 - 4.4 Hz to stay in possible limits)
    if not(os.path.exists(filtered_signal_matrix_path)):
        print("Applying bandpass filter...")
        frames, H, W = raw_signal_matrix.shape
        filtered_signal_matrix = np.zeros_like(raw_signal_matrix, dtype=np.float32)

        for y in range(H):
            for x in range(W):
                filtered_signal_matrix[:, y, x] = bandpass_filter(raw_signal_matrix[:, y, x], edges=edges_bandpass_filter, sample_rate=9.0, poles=4)

        # Save uncompressed
        np.save(filtered_signal_matrix_path, filtered_signal_matrix)
    else:
        filtered_signal_matrix = np.load(filtered_signal_matrix_path)


    # Normalization of signal (min-max normalization)
    #TODO maybe change to scikit-learn min-max scaler
    if not(os.path.exists(normalized_signal_matrix_path)):
        print("Applying min-max normalization...")
        frames, H, W = filtered_signal_matrix.shape
        normalized_signal_matrix = np.zeros_like(filtered_signal_matrix, dtype=np.float32)

        for y in range(H):
            for x in range(W):
                normalized_signal_matrix[:, y, x] = min_max_normalization(filtered_signal_matrix[:, y, x])

        # Save uncompressed
        np.save(normalized_signal_matrix_path, normalized_signal_matrix)
    else:
        normalized_signal_matrix = np.load(normalized_signal_matrix_path)


    # First and Second Order Derivative
    if not(os.path.exists(first_derivative_matrix_path)) or not(os.path.exists(second_derivative_matrix_path)):
        print("Calculating first and second order derivatives...")
        frames, H, W = normalized_signal_matrix.shape
        first_derivative_matrix = np.zeros_like(normalized_signal_matrix, dtype=np.float32)
        second_derivative_matrix = np.zeros_like(normalized_signal_matrix, dtype=np.float32)

        for y in range(H):
            for x in range(W):
                first_derivative, second_derivative = calculate_first_second_derivative(normalized_signal_matrix[:, y, x])
                first_derivative_matrix[:, y, x] = first_derivative
                second_derivative_matrix[:, y, x] = second_derivative

        # Save uncompressed
        np.save(first_derivative_matrix_path, first_derivative_matrix)
        np.save(second_derivative_matrix_path, second_derivative_matrix)
    else:
        first_derivative_matrix = np.load(first_derivative_matrix_path)
        second_derivative_matrix = np.load(second_derivative_matrix_path)

    show_preprocessed_signal_as_video(signal_matrix=normalized_signal_matrix, fps=fps)

    #plot_multiple_1d_signals([normalized_signal_matrix[:, x_visualization, y_visualization], first_derivative_matrix[:, x_visualization, y_visualization], second_derivative_matrix[:, x_visualization, y_visualization]], labels=['Normalized Signal', 'First Derivative', 'Second Derivative'])

    ## Feature Detection # TODO
    ## Dissolution and irregular cycle removal #TODO
    ## Feature Extraction # TODO

    print("Done")
############
# Landmark detection & denoising 
############

if not(os.path.exists(mean_of_subregions_path)):
    mean_values_subregions_matrix = get_facial_dlib_landmarks_regions_and_save_their_mean(video_path="external_library/thermal-facial-landmarks-detection/test_video/thermal_video_12_03.avi", path_save_video_with_landmarks=path_save_video_with_landmarks, upsample=2, resize_frame = True)
    np.save(mean_of_subregions_path, mean_values_subregions_matrix)
else:
    mean_values_subregions_matrix = np.load(mean_of_subregions_path)

# Apply bandpass filter
if not(os.path.exists(filtered_mean_values_subregions_matrix_path)):
    print("Applying bandpass filter to mean values of subregions...")
    subregions, frames = mean_values_subregions_matrix.shape
    filtered_mean_values_subregions_matrix = np.zeros_like(mean_values_subregions_matrix, dtype=np.float32)

    for subregion in range(subregions):
        filtered_mean_values_subregions_matrix[subregion, :] = bandpass_filter(mean_values_subregions_matrix[subregion, :], edges=[0.4,4.4], sample_rate=9.0, poles=5)

    # Save uncompressed
    np.save(filtered_mean_values_subregions_matrix_path, filtered_mean_values_subregions_matrix)
else:
    filtered_mean_values_subregions_matrix = np.load(filtered_mean_values_subregions_matrix_path)


# Dimensionality Reduction
# Apply SVD
# Should be:
# Rows = regions -> weight of the signals in the spatial location in U
# Columns = time frames -> siginificant temporal signals in V
U, S, Vh = np.linalg.svd(filtered_mean_values_subregions_matrix, full_matrices=False)

# Plot the first temporal component
plt.figure()
plt.plot(Vh[0, :])
plt.title("Vh[0, :] — Temporal Component 1")
plt.xlabel("Time (frame)")
plt.ylabel("Amplitude")
plt.show()


print("done")
# PPG Signal Reconstruction

# STFT

# HR Prediction