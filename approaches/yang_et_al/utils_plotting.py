import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


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

    T = len(block_features)
    N = block_features[0].shape[0]

    # Reshape signals to (T, N)
    orig_matrix = signal_flattened_orig.reshape(T, N)
    denoised_matrix = signal_flattened_denoised.reshape(T, N)

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

