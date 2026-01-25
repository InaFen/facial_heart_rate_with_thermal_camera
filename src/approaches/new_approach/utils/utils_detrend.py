import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from src.approaches.yang_et_al.utils_plotting import plot_signal_before_after


def detrend_signal(signal_each_roi_seperate, roi_names):
    """
    Detrend the signal for each ROI by removing the mean and linear trend.
    Args:
        signal_each_roi_seperate: Array of shape (num_timepoints, num_rois)
        roi_names: List of ROI names for plotting
    Returns:
        detrended_signal: Array of same shape as input with detrended signals
    """

    print("[Detrends signal]: Currently only removing mean, detrending commented out as no trend visible.")

    _, num_rois = signal_each_roi_seperate.shape
    
    # Create a copy to avoid modifying the original
    detrended_signal = signal_each_roi_seperate.copy()

    for roi_idx in range(num_rois):
        signal_roi = detrended_signal[:, roi_idx]

        # Preprocess: Remove mean (DC offset) and detrend
        signal_roi = signal_roi - np.mean(signal_roi)
        #signal_roi = signal.detrend(signal_roi, type='linear')
        
        # Save the detrended signal back to the array
        detrended_signal[:, roi_idx] = signal_roi
    
    # Plot before and after in one figure
    plot_signal_before_after(
        signal_before=signal_each_roi_seperate,
        signal_after=detrended_signal,
        roi_names=roi_names,
        title="Signal Before vs After Detrending"
    )

    return detrended_signal
