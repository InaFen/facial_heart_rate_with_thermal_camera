import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io.wavfile



def get_raw_signal(video_path: str) -> np.ndarray:
    """
    Extracts raw thermal signal from a thermal video.

    :param video_path: Path to the thermal video file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    print("Extracts raw thermal signal from the video...")  

    cap = cv2.VideoCapture(video_path)

    raw_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        raw_frames.append(frame)

    cap.release()

    raw_signal = np.stack(raw_frames)  
    # Shape: (num_frames, height, width)

    print("Raw signal shape (frames, height, width):", raw_signal.shape)
    print("Data type:", raw_signal.dtype)
    print("Min:", raw_signal.min(), "Max:", raw_signal.max())

    return raw_signal


def plot_pixel_signal(signal: np.ndarray, x: int, y: int):
    """
    Plots the thermal signal over time for a specific pixel.

    :param signal: 3D numpy array of shape (num_frames, height, width).
    :param x: X-coordinate of the pixel.
    :param y: Y-coordinate of the pixel.
    """
    if signal.ndim == 1:
        pixel_signal = signal
    else:   
        pixel_signal = signal[:, y, x]  

    plt.figure(figsize=(10, 4))
    plt.plot(pixel_signal)
    plt.title(f'Thermal Signal at Pixel ({x}, {y})')
    plt.xlabel('Frame')
    plt.ylabel('Intensity')
    plt.grid()
    plt.show()


def plot_multiple_1d_signals(signals: list[np.ndarray], labels: list[str] | None = None):
    """
    Plots multiple 1D signals in one figure.

    :param signals: List of 1D numpy arrays (each one is a signal).
    :param labels: Optional list of labels for the legend.
    """

    plt.figure(figsize=(10, 4))

    for i, sig in enumerate(signals):
        label = labels[i] if labels is not None else f"Signal {i+1}"
        plt.plot(sig, label=label)

    plt.title("Thermal Signals of a Pixel")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.grid()
    plt.legend()
    plt.show()


def bandpass_filter(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    """Based on example from scipy documentation.
    Applies a Butterworth bandpass filter to the data.

    :param data: 1D numpy array of the signal to be filtered.
    :param edges: List containing the low and high cutoff frequencies [low, high].
    :param sample_rate: Sampling rate of the signal.
    :param poles: Number of poles for the Butterworth filter.
    :return: Filtered signal as a 1D numpy array."""

    assert data.ndim == 1
    assert len(data) > 2 * poles 

    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


def min_max_normalization(signal: np.ndarray) -> np.ndarray:
    """
    Applies min-max normalization to the signal.

    :param signal: 1D numpy array of the signal to be normalized.
    :return: Normalized signal as a 1D numpy array.
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    normalized_signal = (signal - min_val) / (max_val - min_val)
    return normalized_signal

def calculate_first_second_derivative(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the first and second order derivatives of a 1D signal.

    :param signal: 1D numpy array of the signal.
    :return: Tuple containing first and second order derivatives as 1D numpy arrays.
    """
    first_derivative = np.gradient(signal)
    second_derivative = np.gradient(first_derivative)
    return first_derivative, second_derivative

