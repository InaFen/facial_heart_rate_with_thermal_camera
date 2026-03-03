from utils import *
import numpy as np

def preprocessing_pipeline(video_path, filtered_signal_matrix_path, normalized_signal_matrix_path, first_derivative_matrix_path, second_derivative_matrix_path, fps, edges_bandpass_filter, frames_dir):
    
    get_signal_from_video = True

    if get_signal_from_video:
        # Load video and get raw signal
        raw_signal_matrix = get_raw_signal(video_path)
    else:
        frames_pixel_data = extract_pixel_data_from_gray_frames(gray_dir = frames_dir)
        raw_signal_matrix = np.array([frame_data['adc_values'] for frame_data in frames_pixel_data])

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

    #show_preprocessed_signal_as_video(signal_matrix=normalized_signal_matrix, fps=fps)

    #plot_multiple_1d_signals([normalized_signal_matrix[:, x_visualization, y_visualization], first_derivative_matrix[:, x_visualization, y_visualization], second_derivative_matrix[:, x_visualization, y_visualization]], labels=['Normalized Signal', 'First Derivative', 'Second Derivative'])

    ## Feature Detection # TODO
    ## Dissolution and irregular cycle removal #TODO
    ## Feature Extraction # TODO

    print("Preprocessing pipeline completed. [TODO: First, second derivative]")