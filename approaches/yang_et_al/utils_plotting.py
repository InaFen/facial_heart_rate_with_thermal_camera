import matplotlib.pyplot as plt
import numpy as np

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