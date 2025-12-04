from utils import get_raw_signal, plot_pixel_signal, bandpass_filter, plot_multiple_1d_signals, min_max_normalization
import numpy as np
import os

edges_bandpass_filter = [0.1, 4.4]  # in Hz
video_path = 'recorded_videos/thermal_video_12_03.avi'  # Path to the recorded thermal video
filtered_signal_matrix_path = f'filtered_signal_matrix_{edges_bandpass_filter[0]}_{edges_bandpass_filter[1]}.npy'
normalized_signal_matrix_path = f'normalized_signal_matrix_{edges_bandpass_filter[0]}_{edges_bandpass_filter[1]}.npy'

#############
# Preprocessing
#############


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

plot_multiple_1d_signals([raw_signal_matrix[:,x_visualization, y_visualization], filtered_signal_matrix[:, x_visualization, y_visualization], normalized_signal_matrix[:, x_visualization, y_visualization]], labels=['Raw Signal', 'Filtered Signal', 'Normalized Signal'])


# First and Second Order Derivative - Do

## Feature Detection
## Dissolution and irregular cycle removal 
## Feature Extraction 

print("Done")
############
# Landmark detection & denoising - (Do)
############

# Data Extraction

# Dimensionality Reduction

# PPG Signal Reconstruction

# STFT

# HR Prediction