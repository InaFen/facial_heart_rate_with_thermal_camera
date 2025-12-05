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
import os

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

def get_facial_dlib_landmarks(video_path: str, path_save_video_with_landmarks: str, upsample: int):
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
        
        # resize the frame
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