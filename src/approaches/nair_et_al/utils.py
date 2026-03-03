import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io.wavfile
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import numpy as np
import argparse
import time
import dlib
import cv2
import sys
import os
from typing import List, Dict

PROJECT_ROOT = "/home/arbeit/Dokumente/master_thesis/"
sys.path.insert(0, PROJECT_ROOT)

from src.approaches.new_approach.utils.utils_gray_files import sort_grayfiles_with_interpolations
from src.approaches.new_approach.utils.utils_landmark_detection import load_gray_frame

def play_recorded_video(video_path: str):
    """
    Plays a recorded video from the specified path.

    :param video_path: Path to the video file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)  # ms per frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Video Playback', frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


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


def extract_pixel_data_from_gray_frames(
    gray_dir: str,
    output_npz: str = None,
) -> List[Dict[str, object]]:
    """Extract all pixel data from each .gray frame (no ROI).

    Parameters:
        gray_dir: directory containing .gray frames
        output_npz: optional path to save extracted data as npz
        always stores per-pixel (x, y) coordinates

    Returns:
        frame_pixel_data: list (per frame) of dicts
            {
                "adc_values": 2D array as nested lists,
                "shape": (height, width),
                "coords": list of (x, y) tuples
            }
    """
    # Collect and sort frame filenames (with any interpolations)
    _, gray_files = sort_grayfiles_with_interpolations(gray_dir)
    if not gray_files:
        raise ValueError(f"No .gray files found in {gray_dir}")

    frame_pixel_data: List[Dict[str, object]] = []

    for idx, fname in enumerate(gray_files):
        # Load raw 16-bit frame from disk
        frame_path = os.path.join(gray_dir, fname)
        frame = load_gray_frame(frame_path)

        # Store full-frame data and summary stats for this frame
        data_entry: Dict[str, object] = {
            "adc_values": frame.astype(np.float64).tolist(),
            "shape": frame.shape,
        }

        # Build full coordinate list (x, y) for the frame
        height, width = frame.shape
        ys, xs = np.indices((height, width))
        coords = list(zip(xs.ravel().tolist(), ys.ravel().tolist()))
        data_entry["coords"] = coords

        frame_pixel_data.append(data_entry)

        # Progress logging for long sequences
        if (idx + 1) % 100 == 0 or idx == len(gray_files) - 1:
            print(f"Processed {idx+1}/{len(gray_files)} frames")

    if output_npz:
        # Persist results if requested
        os.makedirs(os.path.dirname(output_npz), exist_ok=True)
        np.savez(output_npz, frame_pixel_data=frame_pixel_data)
        print(f"Saved full-frame pixel data to {output_npz}")

    return frame_pixel_data


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

def get_facial_dlib_landmarks(video_path: str, path_save_video_with_landmarks: str, upsample: int, resize: bool):
    """
    Code taken from https://github.com/IS2AI/thermal-facial-landmarks-detection/tree/main
    """


    # load the face detector (HOG-SVM)
    print("[INFO] loading dlib thermal face detector...")
    detector = dlib.simple_object_detector("external_library/thermal-facial-landmarks-detection/models/dlib_face_detector.svm")

    # load the facial landmarks predictor
    print("[INFO] loading facial landmark predictor...")
    predictor = dlib.shape_predictor("external_library/thermal-facial-landmarks-detection/models/dlib_landmark_predictor.dat")

    # initialize the video stream
    vs = cv2.VideoCapture(video_path)

    input_fps = vs.get(cv2.CAP_PROP_FPS)
    print("[INFO] Input FPS:", input_fps)
    delay = int(1000 / input_fps)  # ms per frame


    # initialize the video writer
    writer = None
    (W, H) = (None, None)

    frame_counter = 0
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        frame_counter += 1

        # break the loop if the frame 
        # was not grabbed
        if not grabbed:
            frame_counter -= 1
            print("Frame not grabbed")
            break
        
        if resize:
            # resize the frame (might hurt performance if original is smaller, maybe only upsize better then)
            frame = imutils.resize(frame, width=300)

        # copy the frame
        frame_thermal = frame.copy()

        # convert the frame to grayscale
        frame_greyscale = cv2.cvtColor(frame_thermal, cv2.COLOR_BGRA2GRAY)

        # detect faces in the frame
        rects = detector(frame_greyscale, upsample_num_times=upsample)

        # loop over the bounding boxes
        for rect in rects:
            # convert the dlib rectangle into an OpenCV bounding box and
            # draw a bounding box surrounding the face
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame_thermal, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # predict the location of facial landmark coordinates then 
            # convert the prediction to an easily parsable NumPy array
            shape = predictor(frame_greyscale, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates from our dlib shape
            # predictor model draw them on the image
            for (sx, sy) in shape:
                cv2.circle(frame_thermal, (sx, sy), 2, (255, 0, 0), -1)

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame_greyscale.shape[:2]

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(path_save_video_with_landmarks, fourcc, input_fps,
                (frame_greyscale.shape[1], frame_greyscale.shape[0]), True)

        # push the frame to the writer 
        writer.write(frame_thermal)

        # show the image
        cv2.imshow("Frame", frame_thermal)
        key = cv2.waitKey(delay) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit cleanup
    print("[INFO] Total frames processed:", frame_counter) 
    cv2.destroyAllWindows()
    vs.release()
    writer.release()


def show_preprocessed_signal_as_video(signal_matrix: np.ndarray, fps: float, apply_colormap: bool = False):
    """
    Displays the preprocessed signal matrix as a video.

    :param signal_matrix: 3D numpy array of shape (num_frames, height, width).
    :param fps: Frames per second for video playback.
    """
    num_frames, _, _ = signal_matrix.shape
    delay = int(1000 / fps)

    for i in range(num_frames):
        frame = signal_matrix[i]

        # Convert to 8-bit
        frame_uint8 = frame#(frame * 255).astype(np.uint8)

        # Apply color map if requested
        if apply_colormap:
            frame_display = cv2.applyColorMap(frame_uint8, cv2.COLORMAP_JET)
        else:
            frame_display = frame_uint8

        cv2.imshow("Thermal Video", frame_display)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def plot_Vh_components(Vh: np.ndarray, num_components: int):
    """
    Plots the first 'num_components' temporal components from Vh matrix.

    :param Vh: 2D numpy array from SVD (shape: num_components, num_frames).
    :param num_components: Number of components to plot.
    """
    plt.figure(figsize=(12, 6))
    for i in range(num_components):
        plt.subplot(num_components, 1, i + 1)
        plt.plot(Vh[i, :])
        plt.title(f'Temporal Component {i + 1}')
        plt.xlabel('Frame')
        plt.ylabel('Amplitude')
        plt.grid()
    plt.tight_layout()
    plt.show()


def compute_quality_index(signal, fs, normal_hr_freq):
    """
    Compute the quality index Q(z) for a given PPG signal.

    Parameters:
    - signal: Input PPG signal segment.
    - fs: Sampling frequency.
    - normal_hr_freq: Normal heart rate frequency (scalar) in Hz.

    Returns:
    - Q: Quality index for the signal.
    """
    # Compute the Fourier Transform
    n = len(signal)
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, 1/fs)

    # Compute the magnitude of the Fourier Transform
    magnitude = np.abs(fft_signal)

    # Only consider positive frequencies
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    magnitude_pos = magnitude[pos_mask]

    # Find the indices corresponding to the normal heart rate range
    # Numerator: 3/4 to 5/4 of normal_hr_freq
    # Denominator: 1/2 to 2 times normal_hr_freq
    idx_numerator = np.where((freqs_pos >= (0.75 * normal_hr_freq)) & (freqs_pos <= (1.25 * normal_hr_freq)))
    idx_denominator = np.where((freqs_pos >= (0.5 * normal_hr_freq)) & (freqs_pos <= (2 * normal_hr_freq)))

    # Compute the numerator: integral over the heart rate band
    numerator = np.trapz(magnitude_pos[idx_numerator], freqs_pos[idx_numerator])

    # Compute the denominator: integral over the entire positive frequency range
    denominator = np.trapz(magnitude_pos[idx_denominator], freqs_pos[idx_denominator])

    # Compute the quality index
    quality_index = numerator / denominator

    return quality_index


def generate_ppg_signal(signals, fs, normal_hr_freq, quality_threshold):
    """
    Generate PPG signal by selecting and summing high-quality components.

    Parameters:
    - signals: list or array of candidate signals (e.g., from SVD components)
    - fs: sampling frequency
    - normal_hr_freq: normal heart rate frequency in Hz
    - quality_threshold: minimum quality index to stop summing signals

    Returns:
    - ppg_signal: combined PPG signal
    """
    # Step 1: Compute quality index for each signal
    quality_indexs = [compute_quality_index(s, fs, normal_hr_freq) for s in signals]

    for i, q in enumerate(quality_indexs):
        print(f"Component {i}: Quality Index = {q}")

    # Step 2: Rank signals by quality (descending)
    sorted_indices = np.argsort(quality_indexs)[::-1]

    # Step 3: Sum signals until a good quality is reached
    cumulative_signal = np.zeros_like(signals[0])
    cumulative_quality = 0
    for idx in sorted_indices:
        cumulative_signal += signals[idx]
        cumulative_quality = compute_quality_index(cumulative_signal, fs, normal_hr_freq)
        print(f"Added component {idx}, Cumulative Quality Index = {cumulative_quality}")
        if cumulative_quality >= quality_threshold:
            break

    return cumulative_signal


def extract_curve(S_PPG_abs, lambda_reg=0.1, max_iter=100):
    n_freq, n_time = S_PPG_abs.shape
    cc = np.zeros(n_time)

    # Initialize cc(t) with the frequency of maximum magnitude at each time
    for t in range(n_time):
        cc[t] = np.argmax(S_PPG_abs[:, t])

    # Iterative optimization to find the best curve
    for _ in range(max_iter):
        new_cc = np.zeros(n_time)
        for t in range(n_time):
            if t > 0:
                # Cost function: log|S_PPG| - λ * |cc(t) - cc(t-1)|
                cost = np.log(S_PPG_abs[:, t] + 1e-10) - lambda_reg * np.abs(np.arange(n_freq) - cc[t-1])
            else:
                cost = np.log(S_PPG_abs[:, t] + 1e-10)
            new_cc[t] = np.argmax(cost)
        if np.allclose(cc, new_cc):
            break
        cc = new_cc

    return cc