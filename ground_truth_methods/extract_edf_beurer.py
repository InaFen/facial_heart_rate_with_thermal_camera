import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pyedflib import EdfReader
import pyedflib
import os


def edfread(reader: pyedflib.EdfReader, label: str) -> tuple[float,npt.NDArray[np.float64]]:
    """Read a single signal from an EDF file.
    Args:
        reader (pyedflib.EdfReader): An open EDF reader object.
        label (str): The label of the signal to read.
    Returns:
        tuple[float,npt.NDArray[np.float64]]: A tuple containing the sampling frequency and the signal data as a NumPy array."""
    sig_labels = reader.getSignalLabels()
    idx = sig_labels.index(label)
    buf = np.zeros(reader.getNSamples()[idx])
    buf[:] = reader.readSignal(idx)
    f_samp: float = reader.getSampleFrequencies()[idx]
    return f_samp, buf

def read_edf(file_name):
    """Read EDF file and extract ECG signal data.
    Args:
        file_name (str): Path to the EDF file.
    Returns:
        npt.NDArray[np.float64]: Extracted ECG signal data as a NumPy array."""
    # Lese Header und Daten mit edfread
    with EdfReader(file_name) as reader:
        hdr, record = edfread(reader, label='ECG')
    record_data = np.array(record)
    return record_data, hdr

def plot_signal(signal: npt.NDArray[np.float64], title: str, xlabel: str, ylabel: str):
    """Plot a signal using Matplotlib.
    Args:
        signal (npt.NDArray[np.float64]): The signal data to plot.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis."""
    plt.plot(signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def save_signal_to_npy(target_path: str, filename: str, signal: npt.NDArray[np.float64], sampling_rate: float, units: str):
    """Save signal data to a NumPy .npy file, including sampling frequency.
    Args:
        target_path (str): The path where the .npy file will be saved.
        filename (str): The name of the file to save.
        signal (npt.NDArray[np.float64]): The signal data to saved
        sampling_rate (float): The sampling frequency of the signal.
        units (str): The units of the signal data."
    """
    os.makedirs(target_path, exist_ok=True)
    np.savez(os.path.join(target_path, filename),
         signal=signal,
         fs=sampling_rate,
         units=units)
    
    print("Signal saved to", os.path.join(target_path, filename))
    
if __name__ == "__main__":
    signal_path = "saved_data/ground_truth/beurer/original_edf/Name Name/ECG/202512/17/153829.edf"

    record_data, fs = read_edf(signal_path)

    # Convert from microvolts (EDF storage unit) to millivolts (more common unit)
    record_data = record_data / 1000.0 

    #plot_signal(record_data, 'Beurer ECG signal', 'Samples', 'Amplitude (mV)')

    save_signal_to_npy("saved_data/ground_truth/beurer/extracted_npy_files", filename= "202512_17_153829.npz",signal=record_data, sampling_rate=fs, units="mV")