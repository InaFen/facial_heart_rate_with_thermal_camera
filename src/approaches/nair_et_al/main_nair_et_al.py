import sys
from collections import defaultdict
import numpy as np
import os
from scipy.signal import stft
import matplotlib.pyplot as plt
import numpy as np

from utils import generate_ppg_signal, extract_curve
from helpers_landmark_detection import get_facial_dlib_landmarks_regions_and_save_their_mean
from src.approaches.new_approach.classes.user_input import Config

project_root = "/home/arbeit/Dokumente/master_thesis/"
sys.path.insert(0, project_root)

from src.approaches.new_approach.pipelines.get_roi_data import get_roi_data_pipeline
from src.approaches.yang_et_al.utils_post_processing import apply_butterworth_bandpass_filter

CONFIGS_PATH = "src/approaches/nair_et_al/configs_nair_et_al.yaml"

configs = Config.from_yaml(CONFIGS_PATH)
output_dir = os.path.dirname(configs.path_video_thermal_with_roi_masks)
apply_dlib_landmark_detection = True

filtered_signal_matrix_path = "data/saved_data/nair_et_al/filtered_signal_matrix.npy"
normalized_signal_matrix_path = "data/saved_data/nair_et_al/normalized_signal_matrix.npy"
first_derivative_matrix_path = "data/saved_data/nair_et_al/first_derivative_matrix.npy"
second_derivative_matrix_path = "data/saved_data/nair_et_al/second_derivative_matrix.npy"
mean_of_subregions_path = "data/saved_data/nair_et_al/mean_values_subregions.npy"
path_save_video_with_landmarks = "data/recorded_videos/video_with_landmarks.avi"
bandpassed_mean_values_subregions_path = "data/saved_data/nair_et_al/bandpassed_mean_values_subregions.npy"

STFT_WINDOW_SIZE = 64
STFT_OVERLAP = 56
SMOOTHNESS_PENALTY = 0.1
NORMAL_HR_FREQ = 1.3 #Hz, 80 bpm


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
if not(os.path.exists(bandpassed_mean_values_subregions_path)):
    print("Applying bandpass filter to mean values of subregions...")

    # Create matrix with rows: subregions, columns: time frames, values: mean pixel intensity of the subregion in the frame
    result = defaultdict(list)
    subregions_names = []

    for frame in roi_pixel_data:
        for roi, values in frame.items():
            if roi not in result:
                subregions_names.append(roi)  # remember first-seen order
            result[roi].append(values["mean_kelvin_multiplied_100"])

    mean_values_subregions = [result[roi] for roi in subregions_names]
    bandpassed_mean_values_subregions = np.zeros_like(mean_values_subregions, dtype=np.float32)

    for subregion in range(len(subregions_names)):
        bandpassed_mean_values_subregions[subregion] = apply_butterworth_bandpass_filter(signal = mean_values_subregions[subregion], lowcut = configs.bandpass_lowcut, highcut = configs.bandpass_highcut, fs = configs.fps, order = configs.bandpass_order)

    np.save(bandpassed_mean_values_subregions_path, bandpassed_mean_values_subregions)


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
        plt.show()

else:
    bandpassed_mean_values_subregions = np.load(bandpassed_mean_values_subregions_path)

# Apply SVD
# Should be:
# Rows = regions -> weight of the signals in the spatial location in U (Nair et al.)
# Columns = time frames -> siginificant temporal signals in V (Nair et al.)
U, S, Vh = np.linalg.svd(bandpassed_mean_values_subregions, full_matrices=False)

if configs.plot_steps:
    num_components = Vh.shape[0]
    plt.figure(figsize=(12, 6))
    for i in range(num_components):
        plt.subplot(num_components, 1, i + 1)
        plt.plot(Vh[i, :])
        plt.title(f'Component {i + 1}')
        plt.xlabel('Frame')
        plt.ylabel('Amplitude')
        plt.grid()
    plt.tight_layout()
    plt.show()

# Apply signal quality indices to select the best accumalated components
optimal_vh_components = generate_ppg_signal(signals = Vh, fs = configs.fps, normal_hr_freq = NORMAL_HR_FREQ)

if configs.plot_steps:
    plt.figure(figsize=(12, 6))
    plt.plot(optimal_vh_components[:])
    plt.title(f'Optimal Component')
    plt.xlabel('Frame')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.tight_layout()
    plt.show()

# Short-Time-Fourier-Transform
frequencies, times, short_fourier_ppg = stft(optimal_vh_components, fs=configs.fps, nperseg=STFT_WINDOW_SIZE,noverlap=STFT_OVERLAP)
short_time_fourier_ppg_magnitude = np.abs(short_fourier_ppg).T # Return shape (time_frames, magnitude at frequencies) 

# Convert STFT frequencies (Hz) -> BPM
heart_rate_candidates = frequencies * 60

# Choose smooth, dominant curve in the STFT magnitude as heart rate curve
hr_curve = extract_curve(short_time_fourier_ppg_magnitude, heart_rate_candidates, SMOOTHNESS_PENALTY)

if configs.plot_steps:
    plt.figure(figsize=(12, 6))
    plt.plot(hr_curve[:])
    plt.title(f'Extracted Heart Rate Curve')
    plt.xlabel('Time Windows')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.tight_layout()
    plt.show()

# HR Prediction
# Calculate the time duration (T) of the signal
T = len(optimal_vh_components) / configs.fps

# Convert the curve to heart rate (HR)
hr = (hr_curve * configs.fps) / T
print("Predicted Heart Rate:", hr)
    