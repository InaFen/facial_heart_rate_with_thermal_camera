import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import json
import os
import sys

DETECTION_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, DETECTION_PATH)


def _load_opensignals_metadata_and_data_lines(file_name: str) -> tuple[dict, list[str]]:
    """Load OpenSignals metadata and data lines from a text file."""
    with open(file_name, 'r', encoding='utf-8') as file_handle:
        lines = file_handle.readlines()

    metadata_line = next(
        (
            line for line in lines
            if line.lstrip().startswith('# {') and line.rstrip().endswith('}')
        ),
        None,
    )
    if metadata_line is None:
        raise ValueError(f"Could not find OpenSignals JSON metadata in file: {file_name}")

    metadata = json.loads(metadata_line.lstrip()[2:].strip())

    end_header_index = next((index for index, line in enumerate(lines) if line.strip() == '# EndOfHeader'), None)
    if end_header_index is None:
        raise ValueError(f"Could not find '# EndOfHeader' marker in file: {file_name}")

    data_lines = [
        line.strip()
        for line in lines[end_header_index + 1:]
        if line.strip() and not line.lstrip().startswith('#')
    ]
    return metadata, data_lines


def read_opensignals_txt_converted(file_name: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], float]:
    """Read converted OpenSignals text files with `# EndOfHeader`.

    Expected rows are tab-separated and usually contain `nSeq`, `DI`, and a signal
    column such as `CH1`.
    """
    metadata, data_lines = _load_opensignals_metadata_and_data_lines(file_name)

    device_key = list(metadata.keys())[0]
    device_info = metadata[device_key]
    sampling_rate = float(device_info.get("sampling rate", 1000))

    columns = device_info.get("column", ["nSeq", "DI", "CH1"])
    if not isinstance(columns, list) or len(columns) < 3:
        columns = ["nSeq", "DI", "CH1"]

    try:
        di_index = columns.index("DI")
    except ValueError:
        di_index = 1

    signal_index = next(
        (index for index, column_name in enumerate(columns) if isinstance(column_name, str) and column_name.upper().startswith("CH")),
        2,
    )

    signal_data = []
    button_data = []
    for line in data_lines:
        parts = [part for part in line.split('\t') if part != '']
        if len(parts) <= max(di_index, signal_index):
            parts = line.split()
        if len(parts) <= max(di_index, signal_index):
            continue

        try:
            button_value = int(float(parts[di_index]))
            signal_value = float(parts[signal_index])
        except ValueError:
            continue

        signal_data.append(signal_value)
        button_data.append(button_value)

    return np.array(signal_data, dtype=np.float64), np.array(button_data, dtype=np.int32), sampling_rate


def read_opensignals_txt(file_name: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], float]:
    """Read OpenSignals text file and extract ECG signal data and button pressed data.
    
    Parameters:
        file_name (str): Path to the OpenSignals .txt file.
    
    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], float]: A tuple containing:
            - ECG signal data as a NumPy array
            - Button pressed data (0 or 1) as a NumPy array
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
    
    # Extract ECG signal (CH1 column, which is the 3rd column) and button pressed (DI, 2nd column)
    signal_data = []
    button_data = []
    for line in data_lines:
        parts = line.split('\t')
        if len(parts) >= 3:
            try:
                button_value = int(parts[1])  # DI column (button pressed)
                ch1_value = float(parts[2])    # CH1 column (signal)
                signal_data.append(ch1_value)
                button_data.append(button_value)
            except ValueError:
                continue
    
    signal_array = np.array(signal_data, dtype=np.float64)
    button_array = np.array(button_data, dtype=np.int32)
    
    return signal_array, button_array, sampling_rate


def get_info_opensignals_txt(file_name: str) -> dict:
    """Get information about the signals in an OpenSignals text file.
    
    Parameters:
        file_name (str): Path to the OpenSignals .txt file.
    
    Returns:
        dict: Dictionary containing metadata about the signal.
    """
    return get_info_opensignals_txt_converted(file_name)


def get_info_opensignals_txt_converted(file_name: str) -> dict:
    """Get information from converted OpenSignals TXT files.

    Supports files containing comment headers, a JSON metadata line starting
    with `# { ... }`, and optionally `# EndOfHeader` before data rows.
    """
    with open(file_name, 'r') as f:
        lines = f.readlines()

    metadata_line = next(
        (
            line for line in lines
            if line.lstrip().startswith('# {') and line.rstrip().endswith('}')
        ),
        None,
    )
    if metadata_line is None:
        raise ValueError(f"Could not find OpenSignals JSON metadata in file: {file_name}")

    metadata = json.loads(metadata_line.lstrip()[2:].strip())
    
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

def offset_and_saving_pipeline(signal_path: str, npz_path: str, npz_filename: str):
    """Pipeline to read OpenSignals text file, extract signal and button data, and save to .npz format."""
    # Read signal and button pressed data
    record_data, button_data, fs = read_opensignals_txt(signal_path)
    
    # Get info
    info = get_info_opensignals_txt(signal_path)
    print(f"Device: {info['device']}")
    print(f"Sensor: {info['sensor']}")
    print(f"Sampling rate: {info['sampling_rate']} Hz")
    print(f"Signal length: {len(record_data)} samples")
    print(f"Signal duration: {len(record_data) / fs:.2f} seconds")
    print(f"Button pushed count: {np.sum(button_data)} times")

    print("[Warning] Assumes that the unit is uV (as stated in the edf), transforms in mV")
    # Convert from microvolts to millivolts (more common unit)
    record_data = record_data / 1000.0 

    # Plot signal and button pressed data
    time_axis = np.arange(len(record_data)) / fs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    ax1.plot(time_axis, record_data)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (mV)')
    ax1.set_title('ECG Signal from OpenSignals')
    ax1.grid(True)
    
    ax2.plot(time_axis, button_data, 'r-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Button Pressed')
    ax2.set_title('Button Pushed Status (0 or 1)')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Save to npz
    save_signal_to_npz(target_path=npz_path, filename=npz_filename, signal=record_data, sampling_rate=fs, units="mV")


if __name__ == "__main__":
    signal_path = "data/ground_truth/opensignals/original_signal/31032026/Julian_31_03_2/opensignals_0007800F7623_2026-03-31_16-37-56.txt"
    npz_filename = "opensignals_0007800F7623_2026-03-31_16-37-56.npz"
    npz_path = "data/ground_truth/opensignals/extracted_npy_files/31032026/Julian_31_03_2/"

    offset_and_saving_pipeline(signal_path=signal_path, npz_path=npz_path, npz_filename=npz_filename)
    
    
    