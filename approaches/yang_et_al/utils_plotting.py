import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from approaches.yang_et_al.utils_post_processing import transform_flatten_signal_to_original_shape

def plot_signal_each_roi_seperately(block_features, roi_names):
    """
    Plot the mean signal of each ROI on a single plot over time.

    Parameters
    ----------
    block_features : list of arrays
        block_features[t] is an NxD matrix (N blocks, D features)
        one feature vector per subregion for each frame.
    
    roi_names : list of str
        Names of the ROIs corresponding to the blocks.
    """
    T = len(block_features)
    N = block_features[0].shape[0]

    # Collect signals for each ROI
    signals = [np.array([block_features[t][n, 0] for t in range(T)]) for n in range(N)]

    # Plot all ROI signals on a single plot
    plt.figure(figsize=(12, 6))
    for n in range(N):
        plt.plot(signals[n], label=f'ROI: {roi_names[n]}')
    
    plt.xlabel('Frame')
    plt.ylabel('Mean Pixel Value')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_flattened_signal_each_roi(signal_flattened, block_features, roi_names, title_info):
    """
    Reshape a flattened signal (T*N, 1) back to (T, N) and plot each ROI on one plot.

    Parameters
    ----------
    signal_flattened : array, shape (T*N, 1)
        Flattened signal (stacked ROIs over frames).

    block_features : list of arrays
        Used only to infer T (frames) and N (ROIs).

    roi_names : list of str
        Names of each ROI (length N).
    """

    # Infer dimensions
    T = len(block_features)
    N = block_features[0].shape[0]

    # Reshape back to (T, N)
    signal_matrix = signal_flattened.reshape(T, N)

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    for n in range(N):
        plt.plot(signal_matrix[:, n], label=f'ROI: {roi_names[n]}')

    plt.xlabel("Frame")
    plt.ylabel("Signal Value")
    plt.title(f"Signal for Each ROI: {title_info}")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    return signal_matrix  # optionally return the reconstructed matrix


def plot_original_vs_denoised(signal_flattened_orig, signal_flattened_denoised, block_features, roi_names, title_info="Signal Comparison"):
    """
    Plot original vs denoised signals per ROI in one figure.

    Parameters
    ----------
    signal_flattened_orig : array, shape (T*N, 1)
        Original signal (flattened)
    signal_flattened_denoised : array, shape (T*N, 1)
        Denoised signal (flattened)
    block_features : list of arrays
        Used to infer T (frames) and N (ROIs)
    roi_names : list of str
        Names of ROIs
    title_info : str
        Title for the plot
    """

    N = block_features[0].shape[0]

    # Reshape signals to (T, N)
    orig_matrix = transform_flatten_signal_to_original_shape(block_features=block_features, flattened_signal=signal_flattened_orig)
    denoised_matrix = transform_flatten_signal_to_original_shape(block_features=block_features, flattened_signal=signal_flattened_denoised)

    plt.figure(figsize=(12, 6))

    for n in range(N):
        plt.plot(orig_matrix[:, n], label=f'{roi_names[n]} - Original', alpha=0.6)
        plt.plot(denoised_matrix[:, n], label=f'{roi_names[n]} - Denoised', alpha=0.9)

    plt.xlabel("Frame")
    plt.ylabel("Signal Value")
    plt.title(title_info)
    plt.legend()
    plt.tight_layout()
    plt.show(block=True) 


def plot_denoised_signal_unflat(denoised_signal_unflat, roi_names):
    """
    Plot denoised signals per ROI in one figure.

    Parameters
    ----------
    denoised_signal_unflat : array, shape (T, N)
        Denoised signal in unflattened form.
    roi_names : list of str
        Names of ROIs
    """

    N = denoised_signal_unflat.shape[1]

    plt.figure(figsize=(12, 6))

    fps = 8.7  
    time_seconds = np.arange(denoised_signal_unflat.shape[0]) / fps

    for n in range(N):
        plt.plot(time_seconds, denoised_signal_unflat[:, n], label=f'{roi_names[n]} - Denoised', alpha=0.9)

    plt.xlabel("Seconds")
    plt.ylabel("Signal Value")
    plt.title("Denoised Signals per ROI")
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)

def plot_principal_components_in_seperate_plots(principal_components):
    """
    Plot each principal component in a separate plot.

    Parameters
    ----------
    principal_components : array, shape (T, num_components)
        The principal components to plot.
    """
    T, num_components = principal_components.shape

    plt.figure(figsize=(12, 3 * num_components))
    for i in range(num_components):
        plt.subplot(num_components, 1, i + 1)
        plt.plot(principal_components[:, i])
        plt.title(f'Principal Component {i + 1}')
        plt.xlabel('Frame')
        plt.ylabel('Amplitude')
        plt.tight_layout()
    plt.show()

def plot_chosen_component(chosen_component, fps):
    plt.figure(figsize=(10, 4))
    time_seconds = np.arange(chosen_component.shape[0]) / fps
    plt.plot(time_seconds, chosen_component, label='Most Periodic PCA Component', color='blue')
    plt.xlabel("Seconds")
    plt.ylabel("Signal Value")
    plt.title("Most Periodic PCA Component Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)