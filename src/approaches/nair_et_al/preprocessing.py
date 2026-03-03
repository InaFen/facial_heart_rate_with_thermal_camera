import os
import matplotlib.pyplot as plt
from utils import extract_pixel_data_from_gray_frames, bandpass_filter, min_max_normalization, calculate_first_second_derivative, show_preprocessed_signal_as_video, plot_multiple_1d_signals, detect_ppg_features, remove_irregular_cycles
import numpy as np

def preprocessing_pipeline(video_path, filtered_signal_matrix_path, normalized_signal_matrix_path, first_derivative_matrix_path, second_derivative_matrix_path, fps, edges_bandpass_filter, frames_dir, raw_pixel_data_path, output_dir):
    
    get_signal_from_video = False

    if get_signal_from_video:
        # Load video and get raw signal
        raw_signal_matrix = get_raw_signal(video_path)
    else:
        if raw_pixel_data_path is not None and os.path.exists(raw_pixel_data_path):
            npz = np.load(raw_pixel_data_path, allow_pickle=True)
            frames_pixel_data = npz["frame_pixel_data"].tolist()
        else:
            frames_pixel_data = extract_pixel_data_from_gray_frames(gray_dir = frames_dir, output_npz=raw_pixel_data_path)
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
                filtered_signal_matrix[:, y, x] = bandpass_filter(raw_signal_matrix[:, y, x], edges=edges_bandpass_filter, sample_rate=fps, poles=4)

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

    # Save a visualization video of the normalized signal.
    show_preprocessed_signal_as_video(signal_matrix=normalized_signal_matrix, fps=fps, output_dir=output_dir)

    # Quick sanity plot of the signal and its derivatives at one pixel.
    plot_multiple_1d_signals([normalized_signal_matrix[:, x_visualization, y_visualization], first_derivative_matrix[:, x_visualization, y_visualization], second_derivative_matrix[:, x_visualization, y_visualization]], labels=['Normalized Signal', 'First Derivative', 'Second Derivative'])

    # Feature detection and cycle removal are performed on a representative pixel.
    ppg = normalized_signal_matrix[:, x_visualization, y_visualization]
    ppg_first = first_derivative_matrix[:, x_visualization, y_visualization]
    ppg_second = second_derivative_matrix[:, x_visualization, y_visualization]

    # Extract systolic peaks, valleys, slope extrema, and dicrotic notch candidates.
    features = detect_ppg_features(ppg=ppg, ppg_first=ppg_first, ppg_second=ppg_second, fs=fps)
    # Remove cycles that are too short/long or missing a clear systolic peak.
    retained_cycles, cycle_mask = remove_irregular_cycles(
        cycles=features["cycles"],
        fs=fps,
    )

    # Plot before and after irregular cycle removal.
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(ppg, label="PPG", color="black")
    axes[0].scatter(features["systolic_peaks"], ppg[features["systolic_peaks"]], color="red", s=10, label="Systolic peaks")
    axes[0].scatter(features["valleys"], ppg[features["valleys"]], color="blue", s=10, label="Valleys")
    axes[0].set_title("Before irregular cycle removal")
    axes[0].legend(loc="upper right")

    axes[1].plot(ppg, label="PPG", color="black")
    for cycle in retained_cycles:
        v_start = cycle["valley_start"]
        v_end = cycle["valley_end"]
        axes[1].axvspan(v_start, v_end, color="green", alpha=0.15)
    axes[1].scatter(features["systolic_peaks"], ppg[features["systolic_peaks"]], color="red", s=10, label="Systolic peaks")
    axes[1].scatter(features["valleys"], ppg[features["valleys"]], color="blue", s=10, label="Valleys")
    axes[1].set_title("After irregular cycle removal")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "ppg_cycles_before_after.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()

    feature_output_path = os.path.join(output_dir, "ppg_feature_detection.npz")
    # Persist feature indices and filtered cycle list for later analysis.
    np.savez(
        feature_output_path,
        systolic_peaks=features["systolic_peaks"],
        valleys=features["valleys"],
        cycles=np.array(features["cycles"], dtype=object),
        retained_cycles=np.array(retained_cycles, dtype=object),
        cycle_mask=cycle_mask,
        x_visualization=x_visualization,
        y_visualization=y_visualization,
    )

    print(f"Saved feature detection output to {feature_output_path}")

    print("Preprocessing pipeline completed. [TODO: First, second derivative irregular cycle detection and removal should be applied to all pixels, not just one for visualization. This is just a proof of concept.]")