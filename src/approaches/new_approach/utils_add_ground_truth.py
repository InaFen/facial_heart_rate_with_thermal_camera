import numpy as np
import numpy.typing as npt


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

import numpy as np
import matplotlib.pyplot as plt

def plot_denoised_with_ground_truth_offset(
    denoised_signal_unflat: np.ndarray,
    roi_names: list[str],
    gt_signal: np.ndarray,
    fps_denoised: float,
    fps_gt: float,
    gt_start_time: float  # in seconds
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
        Start time of GT in seconds relative to denoised signal
    """

    N = denoised_signal_unflat.shape[1]

    # Time axis for denoised signals
    t_denoised = np.arange(denoised_signal_unflat.shape[0]) / fps_denoised

    # Time axis for GT, offset by start time
    t_gt = np.arange(len(gt_signal)) / fps_gt + gt_start_time

    scale_factor = np.max(np.abs(denoised_signal_unflat)) / np.max(np.abs(gt_signal))
    gt_scaled = gt_signal #* scale_factor

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
