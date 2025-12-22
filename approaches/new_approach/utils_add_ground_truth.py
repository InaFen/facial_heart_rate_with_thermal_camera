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

