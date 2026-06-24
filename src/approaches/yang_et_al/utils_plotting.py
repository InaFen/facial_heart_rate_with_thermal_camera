import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.approaches.yang_et_al.utils_post_processing import transform_flatten_signal_to_original_shape

def _save_or_show_plot(fig, save_path=None, plot_name="plot"):
    """Helper function to save plot to file or display it.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save or display
    save_path : str or Path, optional
        If provided, save the plot to this directory instead of showing
    plot_name : str
        Name for the saved plot file (without extension)
    """
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        save_file = save_path / f"{plot_name}.png"
        fig.savefig(save_file, dpi=100, bbox_inches='tight')
        print(f"Saved plot to: {save_file}")
        plt.close(fig)
    else:
        plt.show()

def plot_signal_each_roi_seperately(block_features, roi_names, title, fps, show_seconds=False, save_path=None):
    """
    Plot the mean signal of each ROI on a single plot over time.

    Parameters
    ----------
    block_features : list of arrays
        block_features[t] is an NxD matrix (N blocks, D features)
        one feature vector per subregion for each frame.
    
    roi_names : list of str
        Names of the ROIs corresponding to the blocks.
    
    fps : float
        Frames per second.
    
    show_seconds : bool, optional
        If True, x-axis will be in seconds instead of frames. Default is False.
    
    save_path : str or Path, optional
        If provided, save the plot to this directory instead of showing
    """
    T = len(block_features)
    N = block_features[0].shape[0]

    # Collect signals for each ROI
    signals = [np.array([block_features[t][n, 0] for t in range(T)]) for n in range(N)]

    # Create time axis
    if show_seconds:
        x_axis = np.arange(T) / fps
        x_label = 'Time (seconds)'
    else:
        x_axis = np.arange(T)
        x_label = 'Frame'

    # Plot all ROI signals on a single plot
    fig = plt.figure(figsize=(12, 6))
    for n in range(N):
        plt.plot(x_axis, signals[n], label=f'ROI: {roi_names[n]}')
    
    plt.xlabel(x_label)
    plt.ylabel('Mean Kelvin * 100')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    _save_or_show_plot(fig, save_path, "signal_each_roi_separately")


def plot_flattened_signal_each_roi(signal_flattened, block_features, roi_names, title_info, save_path=None):
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
    
    save_path : str or Path, optional
        If provided, save the plot to this directory instead of showing
    """

    # Infer dimensions
    T = len(block_features)
    N = block_features[0].shape[0]

    # Reshape back to (T, N)
    signal_matrix = signal_flattened.reshape(T, N)

    # --- Plot ---
    fig = plt.figure(figsize=(12, 6))
    for n in range(N):
        plt.plot(signal_matrix[:, n], label=f'ROI: {roi_names[n]}')

    plt.xlabel("Frame")
    plt.ylabel("Signal Value")
    plt.title(f"Signal for Each ROI: {title_info}")
    plt.legend()
    plt.tight_layout()
    _save_or_show_plot(fig, save_path, "flattened_signal_each_roi")

    return signal_matrix  # optionally return the reconstructed matrix


def plot_original_vs_denoised(signal_flattened_orig, signal_flattened_denoised, block_features, roi_names, title_info="Signal Comparison", save_path=None):
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
    save_path : str or Path, optional
        If provided, save the plot to this directory instead of showing
    """

    N = block_features[0].shape[0]

    # Reshape signals to (T, N)
    orig_matrix = transform_flatten_signal_to_original_shape(block_features=block_features, flattened_signal=signal_flattened_orig)
    denoised_matrix = transform_flatten_signal_to_original_shape(block_features=block_features, flattened_signal=signal_flattened_denoised)

    fig = plt.figure(figsize=(12, 6))

    for n in range(N):
        plt.plot(orig_matrix[:, n], label=f'{roi_names[n]} - Original', alpha=0.6)
        plt.plot(denoised_matrix[:, n], label=f'{roi_names[n]} - Denoised', alpha=0.9)

    plt.xlabel("Frame")
    plt.ylabel("Signal Value")
    plt.title(title_info)
    plt.legend()
    plt.tight_layout()
    _save_or_show_plot(fig, save_path, "original_vs_denoised") 


def plot_denoised_signal_unflat(denoised_signal_unflat, roi_names, title, fps, show_seconds=False, save_path=None):
    """
    Plot denoised signals per ROI in one figure.

    Parameters
    ----------
    denoised_signal_unflat : array, shape (T, N)
        Denoised signal in unflattened form.
    roi_names : list of str
        Names of ROIs
    fps : float
        Frames per second.
    show_seconds : bool, optional
        If True, x-axis will be in seconds instead of frames. Default is False.
    save_path : str or Path, optional
        If provided, save the plot to this directory instead of showing
    """

    N = denoised_signal_unflat.shape[1]

    fig = plt.figure(figsize=(12, 6))

    # Create time axis
    if show_seconds:
        x_axis = np.arange(denoised_signal_unflat.shape[0]) / fps
        x_label = "Time (seconds)"
    else:
        x_axis = np.arange(denoised_signal_unflat.shape[0])
        x_label = "Frame"

    for n in range(N):
        plt.plot(x_axis, denoised_signal_unflat[:, n], label=f'{roi_names[n]} - Denoised', alpha=0.9)

    plt.xlabel(x_label)
    plt.ylabel("Signal Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    _save_or_show_plot(fig, save_path, "denoised_signal_unflat")


def plot_signal_before_after(signal_before, signal_after, roi_names, title="Signal Before vs After Processing", fps=8.767687772484802, save_path=None):
    """
    Plot signals before and after processing for all ROIs in one plot.

    Parameters
    ----------
    signal_before : array, shape (T, N)
        Signal before processing
    signal_after : array, shape (T, N)
        Signal after processing
    roi_names : list of str
        Names of ROIs
    title : str
        Title for the plot
    fps : float
        Frames per second for time axis
    save_path : str or Path, optional
        If provided, save the plot to this directory instead of showing
    """
    N = signal_before.shape[1]
    time_seconds = np.arange(signal_before.shape[0]) / fps

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Use distinct colors for each ROI
    colors = plt.cm.tab10(np.linspace(0, 1, N))

    for n in range(N):
        ax1.plot(time_seconds, signal_before[:, n], label=roi_names[n], alpha=0.8, linewidth=1.5, color=colors[n])
        ax2.plot(time_seconds, signal_after[:, n], label=roi_names[n], alpha=0.8, linewidth=1.5, color=colors[n])

    ax1.set_ylabel("Signal Value", fontsize=11)
    ax1.set_title("Before Processing", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Signal Value", fontsize=11)
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_title("After Processing", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    _save_or_show_plot(fig, save_path, "signal_before_after")


def plot_principal_components_in_seperate_plots(principal_components, save_path=None):
    """
    Plot each principal component in a separate plot.

    Parameters
    ----------
    principal_components : array, shape (T, num_components)
        The principal components to plot.
    save_path : str or Path, optional
        If provided, save the plot to this directory instead of showing
    """
    T, num_components = principal_components.shape

    fig = plt.figure(figsize=(12, 3 * num_components))
    for i in range(num_components):
        plt.subplot(num_components, 1, i + 1)
        plt.plot(principal_components[:, i])
        plt.title(f'Principal Component {i + 1}')
        plt.xlabel('Frame')
        plt.ylabel('Amplitude')
        plt.tight_layout()
    _save_or_show_plot(fig, save_path, "principal_components")

def plot_chosen_component(chosen_component, fps, save_path=None):
    fig = plt.figure(figsize=(10, 4))
    time_seconds = np.arange(chosen_component.shape[0]) / fps
    plt.plot(time_seconds, chosen_component)#, label='Most Periodic PCA Component', color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (arbitrary unit)")
    #plt.title("Most Periodic PCA Component Over Time")
    #plt.legend()
    plt.tight_layout()
    _save_or_show_plot(fig, save_path, "chosen_component")