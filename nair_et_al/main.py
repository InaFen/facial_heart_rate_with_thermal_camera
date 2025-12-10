from utils import *
from helpers_landmark_detection import *
import numpy as np
import os
from scipy.signal import stft
from preprocessing import preprocessing_pipeline


edges_bandpass_filter = [0.1, 4.4]  # in Hz
video_path = 'recorded_videos/thermal_video_12_03.avi'  # Path to the recorded thermal video
filtered_signal_matrix_path = f'saved_data/filtered_signal_matrix_{edges_bandpass_filter[0]}_{edges_bandpass_filter[1]}.npy'
normalized_signal_matrix_path = 'saved_data/normalized_signal_matrix.npy'
first_derivative_matrix_path = 'saved_data/first_derivative_matrix.npy'
second_derivative_matrix_path = 'saved_data/second_derivative_matrix.npy'
mean_of_subregions_path = 'saved_data/mean_values_subregions.npy'
fps = 9.0

path_save_video_with_landmarks = 'recorded_videos/video_with_landmarks.avi'  # Path to Dlib landmark model
filtered_mean_values_subregions_matrix_path = f'saved_data/filtered_mean_values_subregions.npy'

#############
# Preprocessing TODO apply after landmark detection/ apply face detection on thisß
#############

preprocessing_pipeline(video_path, filtered_signal_matrix_path, normalized_signal_matrix_path, first_derivative_matrix_path, second_derivative_matrix_path, fps, edges_bandpass_filter)

############
# Landmark detection & denoising 
############
#apply_facemesh_to_thermal_video(input_path=video_path, output_path=path_save_video_with_landmarks)


if not(os.path.exists(mean_of_subregions_path)):
    mean_values_subregions_matrix = get_facial_dlib_landmarks_regions_and_save_their_mean(video_path=video_path, path_save_video_with_landmarks=path_save_video_with_landmarks, upsample=2, resize_frame = True)
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

#plot_Vh_components(Vh, num_components=5)

# Apply signal quality indices to select the best component
normal_hr_freq = 1.683  # in Hz (101 bpm) - was during testing

optimal_vh_components = generate_ppg_signal(signals = Vh, fs = fps, normal_hr_freq = normal_hr_freq, quality_threshold=0.379)

#plot_Vh_components(optimal_vh_components[np.newaxis, :], num_components=1)

# Short-Time-Fourier-Transform
frequencies, times, short_fourier_ppg = stft(optimal_vh_components, fs=fps)
short_time_fourier_ppg_magnitude = np.abs(short_fourier_ppg) 

# TODO following steps
#hr_curve = extract_curve(short_fourier_ppg_magnitude)

# HR Prediction
# Calculate the time duration (T) of the signal
#T = len(optimal_vh_components) / fps

# Convert the curve to heart rate (HR)
#HR = (hr_curve * fps) / T
#print("Predicted Heart Rate:", HR)
