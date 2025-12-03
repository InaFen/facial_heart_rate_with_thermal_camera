from utils import get_raw_signal, plot_pixel_signal, bandpass_filter
import numpy as np
import os

video_path = 'recorded_videos/thermal_video_12_03.avi'  # Path to the recorded thermal video
filtered_signal_matrix_path = 'filtered_signal_matrix.npy'


#############
# Preprocessing
#############


# Load video and get raw signal
raw_signal_matrix = get_raw_signal(video_path)

#plot_pixel_signal(raw_signal_matrix, x=60, y=80)

# Bandpass filter (according to paper, here 0.1 - 4.4 Hz to stay in possible limits)
if not(os.path.exists(filtered_signal_matrix_path)):
    frames, H, W = raw_signal_matrix.shape
    filtered_signal_matrix = np.zeros_like(raw_signal_matrix, dtype=np.float32)

    for y in range(H):
        for x in range(W):
            filtered_signal_matrix[:, y, x] = bandpass_filter(raw_signal_matrix[:, y, x], edges=[0.1, 4.4], sample_rate=9.0, poles=4)

    # Save uncompressed
    np.save('filtered_signal_matrix.npy', filtered_signal_matrix)
else:
    filtered_signal_matrix = np.load(filtered_signal_matrix_path)

print("Done")
# Normalization of signal

# First and Second Order Derivative

## Feature Detection
## Dissolution and irregular cycle removal 
## Feature Extraction 

############
# Landmark detection & denoising
############

# Data Extraction

# Dimensionality Reduction

# PPG Signal Reconstruction

# STFT

# HR Prediction