import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def load_ground_truth_signal_and_fs(file_path: str) -> tuple[npt.NDArray[np.float64], float, str]:
    """Load ground truth signal and its sampling frequency from a .npz file.
    Args:
        file_path (str): Path to the .npz file containing the signal data.
    Returns:
        tuple[npt.NDArray[np.float64], float, str]: A tuple containing the signal data as a NumPy array, the sampling frequency, and the units of the signal."""
    with np.load(file_path) as data:
        signal = data['signal']
        fs = data['fs'].item()  # Extract as a scalar
        units = data['units'].item()  # Extract as a scalar
    return signal, fs, units

def plot_denoised_with_ground_truth_offset(
    denoised_signal_unflat: np.ndarray,
    roi_names: list[str],
    gt_signal: np.ndarray,
    fps_denoised: float,
    fps_gt: float,
    gt_start_time: float,  # in seconds
    signal_start_time: float = 0.0  # in seconds
):
    """
    Plot denoised signals per ROI with a ground truth signal
    that starts later in time.

    Parameters
    ----------
    denoised_signal_unflat : array, shape (T_d, N)
        Denoised signals (T_d samples, N ROIs)
    roi_names : list of str
        Names of ROIs
    gt_signal : array, shape (T_gt,)
        Ground truth signal (single channel)
    fps_denoised : float
        Sampling frequency of denoised signals
    fps_gt : float
        Sampling frequency of ground truth
    gt_start_time : float
        Start time of GT in seconds relative to time zero
    signal_start_time : float
        Start time of denoised signal in seconds relative to time zero (default: 0.0)
    """

    N = denoised_signal_unflat.shape[1]

    # Time axis for denoised signals, offset by start time
    t_denoised = np.arange(denoised_signal_unflat.shape[0]) / fps_denoised + signal_start_time

    # Time axis for GT, offset by start time
    t_gt = np.arange(len(gt_signal)) / fps_gt + gt_start_time

    scale_factor = np.max(np.abs(denoised_signal_unflat)) / np.max(np.abs(gt_signal))
    gt_scaled = gt_signal * scale_factor

    plt.figure(figsize=(12, 6))

    # Plot each ROI
    for n in range(N):
        plt.plot(t_denoised,
                 denoised_signal_unflat[:, n],
                 label=f'{roi_names[n]} - Denoised',
                 alpha=0.8)

    # Plot GT
    plt.plot(t_gt,
             gt_scaled,
             '--k',
             label='Ground Truth',
             alpha=0.9,
             linewidth=2)

    plt.xlabel("Seconds")
    plt.ylabel("Signal Value")
    plt.title("Denoised Signals per ROI with Ground Truth (Offset)")
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)



def estimate_signal_offset_via_xcorr(
    denoised_signal_unflat: np.ndarray,
    gt_signal: np.ndarray,
    fps_denoised: float,
    fps_gt: float
) -> tuple[list[float], list[int], list[float], int]:
    """Estimate start-time offset per ROI between own signal and ground truth via cross-correlation.

    Returns lists (per ROI):
      offsets_seconds (positive -> denoised starts later than GT),
      best_lags (samples at fps_denoised),
      peak_corrs (correlation peak values),
      and the index of the ROI with the highest peak correlation.
    """

    num_rois = denoised_signal_unflat.shape[1]
    offsets_seconds: list[float] = []
    best_lags: list[int] = []
    peak_corrs: list[float] = []

    # Determine common duration across signals, so shortest signal length is used
    duration = min(denoised_signal_unflat.shape[0] / fps_denoised, len(gt_signal) / fps_gt)
    if duration <= 0:
        raise ValueError("Signals have zero duration")

    t_common = np.arange(0, duration, 1.0 / fps_denoised)
    t_gt = np.arange(len(gt_signal)) / fps_gt
    gt_resamp = np.interp(t_common, t_gt, gt_signal)
    gt_z = gt_resamp - np.mean(gt_resamp)

    for roi_idx in range(num_rois):
        own_sig = denoised_signal_unflat[:, roi_idx]
        own_sig = np.nan_to_num(own_sig, nan=np.nanmean(own_sig))

        t_own = np.arange(len(own_sig)) / fps_denoised
        own_resamp = np.interp(t_common, t_own, own_sig)
        own_z = own_resamp - np.mean(own_resamp)

        corr = np.correlate(gt_z, own_z, mode="full")
        lags = np.arange(-len(own_z) + 1, len(gt_z))
        best_idx = int(np.argmax(corr))
        best_lag = int(lags[best_idx])  # samples at fps_denoised
        peak_corr = float(corr[best_idx])

        offset_seconds = -best_lag / fps_denoised  # positive -> own starts later

        offsets_seconds.append(offset_seconds)
        best_lags.append(best_lag)
        peak_corrs.append(peak_corr)

    best_roi_idx = int(np.argmax(peak_corrs))
    return offsets_seconds, best_lags, peak_corrs, best_roi_idx
