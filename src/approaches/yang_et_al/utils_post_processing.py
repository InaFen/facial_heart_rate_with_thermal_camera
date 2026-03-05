from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

def transform_flatten_signal_to_original_shape(block_features, flattened_signal):
    """
    Transform a flattened signal back to its original (T, N) shape.
    Parameters
    ----------
    block_features : list of arrays
        Used to infer T (frames) and N (ROIs)
    flattened_signal : array, shape (T*N, 1)
    """
    T = len(block_features)
    N = block_features[0].shape[0]

    # Reshape signals to (T, N)
    signal_unflatten = flattened_signal.reshape(T, N)

    return signal_unflatten


def apply_butterworth_bandpass_filter(signal, lowcut, highcut, fs, order):
    """ Apply a Butterworth bandpass filter to the signal.  

    Parameters
    ----------
    signal : array, shape (T, N)
        Input signal to be filtered.
    lowcut : float
        Low cutoff frequency in Hz.
    highcut : float
        High cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Order of the Butterworth filter.
    Returns
    -------
    filtered_signal : array, shape (T, N)
        Filtered signal."""

    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive. Got fs={fs}.")

    nyquist = 0.5 * fs
    if not (0 < lowcut < nyquist and 0 < highcut < nyquist):
        raise ValueError(
            "Bandpass frequencies must satisfy 0 < lowcut, highcut < Nyquist. "
            f"Got lowcut={lowcut}, highcut={highcut}, Nyquist={nyquist}. "
            "Lower the cutoff values or increase the sampling rate."
        )

    # Normalize cutoff frequencies to match expected input format for butter
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal, axis=0)
    return y


def pca(data, num_components):
    """
    Perform PCA on the data and return the top principal components.

    Parameters
    ----------
    data : array, shape (T, N)
        Input data where T is the number of samples and N is the number of features.
    num_components : int
        Number of principal components to return.   
    Returns
    -------
    principal_components : array, shape (T, num_components)
        The top principal components of the data.
    """ 
    # Center the data
    data_meaned = data - np.mean(data, axis=0)

    # Compute covariance matrix
    cov_matrix = np.cov(data_meaned, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top 'num_components' eigenvectors
    principal_components = sorted_eigenvectors[:, :num_components]

    # Project the data onto the principal components
    transformed_data = np.dot(data_meaned, principal_components)

    return transformed_data

def find_most_periodic_pca_component(principal_components, num_harmonics=3):
    """
    Identify the PCA component with the most periodic behavior using FFT.

    Parameters
    ----------
    principal_components : array, shape (T, num_components)
        The principal components to analyze.
    num_harmonics : int, optional
        Number of harmonics (including the dominant frequency) to use when
        computing the proportion p.
    
    Returns
    -------
    selected_pca_component : array, shape (T,)
        The PCA component with the most periodic behavior.
    """
    if num_harmonics < 1:
        raise ValueError("num_harmonics must be >= 1.")

    # Use real FFT to obtain the single-sided spectrum (non-negative frequencies)
    # for every PCA component. This matches the paper's single-sided power use.
    fft_components = np.fft.rfft(principal_components, axis=0)
    power_spectrum = np.abs(fft_components) ** 2

    # Exclude the DC bin (index 0) when searching for the dominant frequency.
    # The dominant frequency is the largest peak among non-zero frequencies.
    if power_spectrum.shape[0] < 2:
        raise ValueError("Signal length too short for FFT-based periodicity.")

    p_values = []
    for comp_idx in range(power_spectrum.shape[1]):
        # Extract the power spectrum for one PCA component
        spectrum = power_spectrum[:, comp_idx]
        # Dominant frequency index (largest non-DC power)
        dominant_index = np.argmax(spectrum[1:]) + 1
        # Collect indices for the dominant frequency and its first harmonics
        harmonic_indices = [dominant_index * i for i in range(1, num_harmonics + 1)]
        # Keep only harmonics that are within the FFT spectrum
        harmonic_indices = [idx for idx in harmonic_indices if idx < spectrum.shape[0]]

        # P1 is the dominant-frequency power; harmonic_sum is P1 + higher harmonics
        p1 = spectrum[dominant_index]
        harmonic_sum = np.sum(spectrum[harmonic_indices])
        # p in [0, 1]; higher values indicate stronger periodicity at harmonics
        p = p1 / harmonic_sum if harmonic_sum > 0 else 0.0
        p_values.append(p)

    # Select the PCA component with the largest proportion p
    dominant_component_index = int(np.argmax(p_values))

    # Extract the most periodic PCA component
    selected_pca_component = principal_components[:, dominant_component_index]

    return selected_pca_component
    