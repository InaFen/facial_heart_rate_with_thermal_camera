from src.approaches.new_approach.utils.utils_add_ground_truth import estimate_signal_offset_via_xcorr, load_ground_truth_signal_and_fs, plot_denoised_with_ground_truth_offset
from src.ground_truth_methods.extract_edf import plot_signal
from src.approaches.yang_et_al.utils_post_processing import apply_butterworth_bandpass_filter


def compare_signal_to_ground_truth_pipeline(configs, bandpassed_signal_each_roi_seperate, roi_names):

    # Get ground truth signal
    ground_truth_signal, gt_fs, gt_units = load_ground_truth_signal_and_fs(configs.path_ground_truth)

    if configs.plot_steps:
        plot_signal(title="Ground Truth ECG Signal", signal=ground_truth_signal, xlabel="Samples", ylabel=f"Amplitude ({gt_units})")
    
    # Apply bandpass filter to ground truth signal
    ground_truth_signal_denoised = apply_butterworth_bandpass_filter(signal=ground_truth_signal, lowcut=configs.bandpass_lowcut, highcut=configs.bandpass_highcut, fs=gt_fs, order=configs.bandpass_order)
    
    if configs.plot_steps:
        plot_signal(title="Denoised Ground Truth ECG Signal", signal=ground_truth_signal_denoised, xlabel="Samples", ylabel=f"Amplitude ({gt_units})")

    # Estimate time offset via cross-correlation per ROI (positive => denoised starts later than GT)
    # TODO check if this correlation makes sense as measurment
    offsets_seconds, best_lag_samples, peak_corrs, best_roi_idx = estimate_signal_offset_via_xcorr(
        denoised_signal_unflat=bandpassed_signal_each_roi_seperate,
        gt_signal=ground_truth_signal_denoised,
        fps_denoised=configs.fps,
        fps_gt=gt_fs
    )
    print("Per-ROI offsets (s):", offsets_seconds)
    print("Per-ROI lag samples:", best_lag_samples)
    print("Per-ROI peak correlations:", peak_corrs)
    print(f"Selected ROI for alignment (max peak corr): {roi_names[best_roi_idx]} (idx {best_roi_idx})")
    print("Currently NOT used, manually set")

    # Use the best ROI's offset for alignment
    offset_seconds = offsets_seconds[best_roi_idx]


    plot_denoised_with_ground_truth_offset(denoised_signal_unflat=bandpassed_signal_each_roi_seperate,
                                        roi_names=roi_names,
                                        gt_signal=ground_truth_signal,
                                        fps_denoised=configs.fps,
                                        fps_gt=gt_fs,
                                        gt_start_time=0.0,
                                        signal_start_time=13.0)  