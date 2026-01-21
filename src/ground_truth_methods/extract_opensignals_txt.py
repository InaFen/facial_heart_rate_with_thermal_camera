import numpy as np
import numpy.typing as npt
import json
import os

import os, sys


SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, SRC_PATH)

from src.ground_truth_methods.extract_edf import plot_signal


def read_opensignals_txt(file_name: str) -> tuple[npt.NDArray[np.float64], float]:
    """Read OpenSignals text file and extract ECG signal data.
    
    Parameters:
        file_name (str): Path to the OpenSignals .txt file.
    
    Returns:
        tuple[npt.NDArray[np.float64], float]: A tuple containing:
            - ECG signal data as a NumPy array
            - Sampling frequency in Hz
    """
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    # Parse header (first two lines)
    header_line = lines[1]  # Line 1 contains JSON metadata
    header_json = header_line[2:]  # Remove "# " prefix
    
    # Extract metadata
    metadata = json.loads(header_json)
    # Get the first (and usually only) device key
    device_key = list(metadata.keys())[0]
    device_info = metadata[device_key]
    
    sampling_rate = device_info.get("sampling rate", 1000)
    
    # Parse data lines (skip header and "# EndOfHeader")
    data_lines = []
    for line in lines[3:]:  # Skip first 3 lines (metadata, EndOfHeader comment, blank)
        if line.strip() and not line.startswith('#'):
            data_lines.append(line.strip())
    
    # Extract ECG signal (CH1 column, which is the 3rd column)
    signal_data = []
    for line in data_lines:
        parts = line.split('\t')
        if len(parts) >= 3:
            try:
                ch1_value = float(parts[2])
                signal_data.append(ch1_value)
            except ValueError:
                continue
    
    signal_array = np.array(signal_data, dtype=np.float64)
    
    return signal_array, sampling_rate


def get_info_opensignals_txt(file_name: str) -> dict:
    """Get information about the signals in an OpenSignals text file.
    
    Parameters:
        file_name (str): Path to the OpenSignals .txt file.
    
    Returns:
        dict: Dictionary containing metadata about the signal.
    """
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header_line = lines[1]
    header_json = header_line[2:]
    metadata = json.loads(header_json)
    
    device_key = list(metadata.keys())[0]
    device_info = metadata[device_key]
    
    info = {
        "sampling_rate": device_info.get("sampling rate", 1000),
        "channels": device_info.get("channels", []),
        "sensor": device_info.get("sensor", []),
        "label": device_info.get("label", []),
        "device": device_info.get("device", "unknown"),
        "date": device_info.get("date", "unknown"),
        "time": device_info.get("time", "unknown"),
        "resolution": device_info.get("resolution", [])
    }
    
    return info


def save_signal_to_npz(target_path: str, filename: str, signal: npt.NDArray[np.float64], sampling_rate: float, units: str = "mV"):
    """Save signal data to a NumPy .npz file, including sampling frequency.
    
    Parameters:
        target_path (str): The path where the .npz file will be saved.
        filename (str): The name of the file to save.
        signal (npt.NDArray[np.float64]): The signal data to save.
        sampling_rate (float): The sampling frequency of the signal.
        units (str): The units of the signal data (default: "mV").
    """
    os.makedirs(target_path, exist_ok=True)
    np.savez(os.path.join(target_path, filename),
             signal=signal,
             fs=sampling_rate,
             units=units)
    
    print("Signal saved to", os.path.join(target_path, filename))


if __name__ == "__main__":
    signal_path = "data/ground_truth/opensignals/original_signal/19012026/opensignals_0007800F80FA_2026-01-19_13-53-34.txt"
    npz_filename = "opensignals_0007800F80FA_2026-01-19_13-53-34.npz"
    npz_path = "data/ground_truth/opensignals/extracted_npy_files/19012026/"
    
    # Read signal
    record_data, fs = read_opensignals_txt(signal_path)
    
    # Get info
    info = get_info_opensignals_txt(signal_path)
    print(f"Device: {info['device']}")
    print(f"Sensor: {info['sensor']}")
    print(f"Sampling rate: {info['sampling_rate']} Hz")
    print(f"Signal length: {len(record_data)} samples")
    print(f"Signal duration: {len(record_data) / fs:.2f} seconds")

    print("[Warning] Assumes that the unit is uV (as stated in the edf), transforms in mV")
    # Convert from microvolts to millivolts (more common unit)
    record_data = record_data / 1000.0 

    # Plot signal
    plot_signal(record_data, 'ECG Signal from OpenSignals', 'Samples', 'Amplitude (mV)')
    
    # Save to npz
    save_signal_to_npz(target_path=npz_path, filename=npz_filename, signal=record_data, sampling_rate=fs, units="mV")
