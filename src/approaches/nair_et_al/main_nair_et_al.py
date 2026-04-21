import sys
from collections import defaultdict
import numpy as np
import os
from scipy.signal import stft

from utils import *
from helpers_landmark_detection import *
from src.approaches.new_approach.classes.user_input import Config

project_root = "/home/arbeit/Dokumente/master_thesis/"
sys.path.insert(0, project_root)

from src.approaches.new_approach.pipelines.get_roi_data import get_roi_data_pipeline


CONFIGS_PATH = "src/approaches/nair_et_al/configs_nair_et_al.yaml"

configs = Config.from_yaml(CONFIGS_PATH)
output_dir = os.path.dirname(configs.path_video_thermal_with_roi_masks)
apply_dlib_landmark_detection = False

filtered_signal_matrix_path = "data/saved_data/nair_et_al/filtered_signal_matrix.npy"
normalized_signal_matrix_path = "data/saved_data/nair_et_al/normalized_signal_matrix.npy"
first_derivative_matrix_path = "data/saved_data/nair_et_al/first_derivative_matrix.npy"
second_derivative_matrix_path = "data/saved_data/nair_et_al/second_derivative_matrix.npy"
mean_of_subregions_path = "data/saved_data/nair_et_al/mean_values_subregions.npy"
path_save_video_with_landmarks = "data/recorded_videos/video_with_landmarks.avi"
filtered_mean_values_subregions_matrix_path = "data/saved_data/nair_et_al/filtered_mean_values_subregions.npy"


############
# Preprocessing: Landmark detection & denoising 
############
print("IMPORTANT: If mediapipe is applied, each ROI will not be split further so needs to be defined accordingly in configs.")

if apply_dlib_landmark_detection:
    # TODO try face detection with dlib but with image transferred to RGB, see yang et al.
    # TODO see whether works
    if not(os.path.exists(mean_of_subregions_path)):
        mean_values_subregions_matrix = get_facial_dlib_landmarks_regions_and_save_their_mean(video_path=video_path, path_save_video_with_landmarks=path_save_video_with_landmarks, upsample=2, resize_frame = True)
        np.save(mean_of_subregions_path, mean_values_subregions_matrix)
    else:
        mean_values_subregions_matrix = np.load(mean_of_subregions_path)
else:
    # Apply Facemesh landmark detection
    roi_pixel_data, _ = get_roi_data_pipeline(configs=configs)


# Apply bandpass filter
if not(os.path.exists(filtered_mean_values_subregions_matrix_path)):
        #TODO continue code preparation here
        #TODO and then follow pipeline as described here VI. HEART RATE AND BP ESTIMATION in paper

    print("Applying bandpass filter to mean values of subregions...")

    # Create matrix with columns: subregions, rows: time frames
    result = defaultdict(list)
    subregions = []

    for frame in roi_pixel_data:
        for roi, values in frame.items():
            if roi not in result:
                subregions.append(roi)  # remember first-seen order
            result[roi].append(values["mean_kelvin_multiplied_100"])

    # build matrix in the same order
    mean_values_subregions_matrix = [result[roi] for roi in subregions]
    filtered_mean_values_subregions_matrix = np.zeros_like(mean_values_subregions_matrix, dtype=np.float32)

    for subregion in range(len(subregions)):
        filtered_mean_values_subregions_matrix[subregion, :] = bandpass_filter(mean_values_subregions_matrix[subregion, :], edges=[0.4,4.4], sample_rate=9.0, poles=5)

    # Save uncompressed
    np.save(filtered_mean_values_subregions_matrix_path, filtered_mean_values_subregions_matrix)
else:
    filtered_mean_values_subregions_matrix = np.load(filtered_mean_values_subregions_matrix_path)

# TODO plot signal before SVD

# Dimensionality Reduction
# Apply SVD
# Should be:
# Rows = regions -> weight of the signals in the spatial location in U
# Columns = time frames -> siginificant temporal signals in V
U, S, Vh = np.linalg.svd(filtered_mean_values_subregions_matrix, full_matrices=False)

plot_Vh_components(Vh, num_components=5) # TODO double-check why way smoother than in line 68 plot

# Apply signal quality indices to select the best component
normal_hr_freq = 1.683  # in Hz (101 bpm) - was during testing

optimal_vh_components = generate_ppg_signal(signals = Vh, fs = configs.fps, normal_hr_freq = normal_hr_freq, quality_threshold=0.379)

plot_Vh_components(optimal_vh_components[np.newaxis, :], num_components=1)

# Short-Time-Fourier-Transform
frequencies, times, short_fourier_ppg = stft(optimal_vh_components, fs=configs.fps)
short_time_fourier_ppg_magnitude = np.abs(short_fourier_ppg) 

# TODO following steps
#hr_curve = extract_curve(short_fourier_ppg_magnitude)

# HR Prediction
# Calculate the time duration (T) of the signal
#T = len(optimal_vh_components) / fps

# Convert the curve to heart rate (HR)
#HR = (hr_curve * fps) / T
#print("Predicted Heart Rate:", HR)
