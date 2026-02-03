from src.approaches.new_approach.utils.utils_add_ground_truth import estimate_signal_offset_via_xcorr, load_ground_truth_signal_and_fs, plot_denoised_with_ground_truth_offset
from src.ground_truth_methods.extract_edf import plot_signal
from src.approaches.yang_et_al.utils_post_processing import apply_butterworth_bandpass_filter
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def compare_signal_to_ground_truth_pipeline(configs, bandpassed_signal_each_roi_seperate, roi_names, timespan_filter=None):

    # Get ground truth signal
    ground_truth_signal, gt_fs, gt_units = load_ground_truth_signal_and_fs(configs.path_ground_truth)

    # Apply timespan filter to ground truth if configured
    if timespan_filter is not None:
        start_time, end_time = timespan_filter
        # Adjust for ground truth offset: GT time = signal_time + (gt_start - signal_start)
        time_offset = configs.time_first_frame_signal - configs.time_first_frame_ecg
        if time_offset < 0:
            print("Warning: Negative time offset between signal and ground truth.")
        gt_start_time = start_time + time_offset
        gt_end_time = end_time + time_offset
        gt_start_sample = int(gt_start_time * gt_fs)
        gt_end_sample = int(gt_end_time * gt_fs)
        ground_truth_signal = ground_truth_signal[gt_start_sample:gt_end_sample]
        print(f"Applied GT timespan filter: {gt_start_time:.2f}s to {gt_end_time:.2f}s (samples {gt_start_sample} to {gt_end_sample})")

    #if configs.plot_steps:
    #    plot_signal(title="Ground Truth ECG Signal", signal=ground_truth_signal, xlabel="Samples", ylabel=f"Amplitude ({gt_units})")
    
    # Apply bandpass filter to ground truth signal
    #ground_truth_signal_denoised = apply_butterworth_bandpass_filter(signal=ground_truth_signal, lowcut=configs.bandpass_lowcut, highcut=configs.bandpass_highcut, fs=gt_fs, order=configs.bandpass_order)
    print("[Warning ground truth] No bandpass filtering applied to ground truth ECG signal and flipped ground truth.")
    ground_truth_signal_denoised = ground_truth_signal  # No additional filtering for ECG
    ground_truth_signal_denoised = -ground_truth_signal_denoised  # Flip ECG signal for correct orientation

    #if configs.plot_steps:
    #    plot_signal(title="Denoised Ground Truth ECG Signal", signal=ground_truth_signal_denoised, xlabel="Samples", ylabel=f"Amplitude ({gt_units})")

    # Determine effective start times (accounting for timespan filter)
    effective_signal_start_time = configs.time_first_frame_signal
    effective_gt_start_time = configs.time_first_frame_ecg
    if timespan_filter is not None:
        start_time, end_time = timespan_filter
        effective_signal_start_time = start_time
        effective_gt_start_time = start_time # offset already applied above

    # Optional: estimate time offset via cross-correlation per ROI (informational)
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

    # Plot signals roughly aligned using configured start times 
    """
    plot_denoised_with_ground_truth_offset(
        denoised_signal_unflat=bandpassed_signal_each_roi_seperate,
        roi_names=roi_names,
        gt_signal=ground_truth_signal,
        fps_denoised=configs.fps,
        fps_gt=gt_fs,
        gt_start_time=effective_gt_start_time,
        signal_start_time=effective_signal_start_time
    )
    """

    # --- Peak-based HR estimation and comparison ---
    def compute_hr_series_from_peaks(sig: np.ndarray, fs: float, min_distance_s: float , prominence: float, configs):
        """Detect peaks and compute instantaneous HR (bpm) per beat.

        Returns: hr_times (seconds), hr_bpm (array), peak_times (seconds), peak_indices (array)
        """
        distance = int(min_distance_s * fs)
        peaks, _ = find_peaks(sig, distance=max(1, distance), prominence=prominence)
        peak_times = peaks / fs
        if len(peaks) < 2:
            return np.array([]), np.array([]), peak_times, peaks
        ibi = np.diff(peak_times)  # seconds between consecutive peaks
        hr_bpm = 60.0 / ibi
        # Assign HR to the latter peak time (or mid-point); here we use latter peak time
        hr_times = peak_times[1:]
        return hr_times, hr_bpm, peak_times, peaks

    # Compute HR series for ground truth
    gt_hr_times, gt_hr_bpm, gt_peak_times, gt_peaks = compute_hr_series_from_peaks(
        ground_truth_signal_denoised, gt_fs, min_distance_s=0.5, prominence=0.5, configs=configs
    )

    # Compute HR per ROI
    num_timepoints, num_rois = bandpassed_signal_each_roi_seperate.shape
    t_den = np.arange(num_timepoints) / configs.fps

    results_summary = []
    for roi_idx in range(num_rois):
        sig_roi = bandpassed_signal_each_roi_seperate[:, roi_idx]
        roi_hr_times_rel, roi_hr_bpm, roi_peak_times_rel, roi_peaks = compute_hr_series_from_peaks(
            sig_roi, configs.fps, min_distance_s=0.3, prominence=None, configs=configs
        )

        # Align times using effective start times (already computed above)
        roi_hr_times = roi_hr_times_rel + effective_signal_start_time
        gt_hr_times_aligned = gt_hr_times + effective_gt_start_time

        # Interpolate GT HR onto ROI HR times for comparison
        if len(gt_hr_times_aligned) >= 2 and len(roi_hr_times) >= 1:
            gt_hr_interp = np.interp(roi_hr_times, gt_hr_times_aligned, gt_hr_bpm, left=np.nan, right=np.nan)
            # Compute metrics on overlapping (non-NaN) points
            mask = ~np.isnan(gt_hr_interp)
            if mask.any():
                mae = float(np.mean(np.abs(roi_hr_bpm[mask] - gt_hr_interp[mask])))
                rmse = float(np.sqrt(np.mean((roi_hr_bpm[mask] - gt_hr_interp[mask])**2)))
            else:
                mae = float('nan')
                rmse = float('nan')
        else:
            gt_hr_interp = np.array([])
            mae = float('nan')
            rmse = float('nan')

        results_summary.append({
            'roi': roi_names[roi_idx] if roi_idx < len(roi_names) else f'ROI {roi_idx}',
            'num_beats': int(len(roi_hr_bpm)),
            'mae_bpm': mae,
            'rmse_bpm': rmse,
        })

        # Plot per ROI: signal with peaks (top) and HR comparison (bottom)
        roi_times = np.arange(len(sig_roi)) / configs.fps
        roi_times_aligned = roi_times + effective_signal_start_time
        gt_times = np.arange(len(ground_truth_signal_denoised)) / gt_fs
        gt_times_aligned = gt_times + effective_gt_start_time
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
        
        # Top subplot: bandpassed ROI signal + GT signal with detected peaks
        ax0_roi = axes[0]
        ax0_roi.plot(roi_times_aligned, sig_roi, label='Bandpassed ROI Signal', linewidth=1.0, color='C0')
        if roi_peaks.size > 0:
            axes[0].plot(roi_times_aligned[roi_peaks], sig_roi[roi_peaks], 'rx', label='ROI Detected Peaks', markersize=6)
        ax0_roi.set_ylabel('ROI Amplitude (a.u.)', color='C0')
        ax0_roi.tick_params(axis='y', labelcolor='C0')
        ax0_roi.grid(True, alpha=0.3)
        
        # Overlay GT signal on secondary y-axis
        ax0_gt = ax0_roi.twinx()
        ax0_gt.plot(gt_times_aligned, ground_truth_signal_denoised, label='GT ECG', linewidth=1.0, color='C1', alpha=0.7)
        if gt_peaks.size > 0:
            ax0_gt.plot(gt_times_aligned[gt_peaks], ground_truth_signal_denoised[gt_peaks], 'g^', label='GT Detected Peaks', markersize=5, alpha=0.7)
        ax0_gt.set_ylabel(f'GT Amplitude ({gt_units})', color='C1')
        ax0_gt.tick_params(axis='y', labelcolor='C1')
        
        axes[0].set_xlabel('Time (s)')
        axes[0].set_title(f"Bandpassed Signals with Peaks: {results_summary[-1]['roi']}")
        
        # Combine legends
        lines0, labels0 = ax0_roi.get_legend_handles_labels()
        lines1, labels1 = ax0_gt.get_legend_handles_labels()
        axes[0].legend(lines0 + lines1, labels0 + labels1, loc='upper right')
        
        # Bottom subplot: HR comparison
        axes[1].plot(roi_hr_times, roi_hr_bpm, 'o-', label='ROI HR (bpm)', markersize=4)
        if len(gt_hr_times_aligned) > 0:
            axes[1].plot(gt_hr_times_aligned, gt_hr_bpm, 'x-', label='GT HR (bpm)', markersize=4)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('HR (bpm)')
        axes[1].set_title(f"HR Comparison (MAE={mae:.2f}, RMSE={rmse:.2f} bpm)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if configs.plot_steps or True:
            plt.show()
        else:
            plt.close()

    print("HR comparison summary per ROI:")
    for r in results_summary:
        print(f"  {r['roi']}: beats={r['num_beats']}, MAE={r['mae_bpm']:.2f} bpm, RMSE={r['rmse_bpm']:.2f} bpm")
