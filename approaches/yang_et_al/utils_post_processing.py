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

    # Normalize cutoff frequencies to match expected input format for butter
    nyquist = 0.5 * fs
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

def find_most_periodic_pca_component(principal_components):
    """
    Identify the PCA component with the most periodic behavior using FFT.

    Parameters
    ----------
    principal_components : array, shape (T, num_components)
        The principal components to analyze.
    
    Returns
    -------
    selected_pca_component : array, shape (T,)
        The PCA component with the most periodic behavior.
    """
    # Apply FFT to each PCA component
    fft_components = np.abs(np.fft.fft(principal_components, axis=0))

    # Calculate the power spectrum
    power_spectrum = fft_components**2

    # Calculate the proportion of power for each component
    p = power_spectrum[1:, :] / power_spectrum[0, :]

    # Find the index of the PCA component with the largest proportion of power
    dominant_component_index = np.argmax(np.max(p, axis=0))

    # Extract the most periodic PCA component
    selected_pca_component = principal_components[:, dominant_component_index]

    return selected_pca_component
    