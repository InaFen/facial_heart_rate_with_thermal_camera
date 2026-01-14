"""
Utility to add synthetic baseline (control) signals to denoised/processed signals.
"""

import numpy as np
from typing import List


def add_random_baseline_signal(
    signal_each_roi_separate: np.ndarray,
    seed: int = None,
    amplitude_scale: float = 1.0,
) -> tuple[np.ndarray, List[str]]:
    """
    Add a synthetic random baseline signal to denoised/bandpassed signals.
    
    The baseline is independent white noise, simulating a non-physiological signal
    for validation that unrelated signals don't correlate with ECG.
    
    Parameters:
        signal_each_roi_separate: Array of shape (T, N) where T=frames, N=ROIs
        seed: Optional random seed for reproducibility
        amplitude_scale: Multiplier for the baseline standard deviation (e.g., 0.2 lowers amplitude)
        
    Returns:
        signal_with_baseline: Extended array of shape (T, N+1) with baseline appended
        roi_names_with_baseline: List of ROI names including 'random_baseline' at end
    """
    if seed is not None:
        np.random.seed(seed)
    
    T, N = signal_each_roi_separate.shape
    
    # Generate random baseline signal matching signal range
    signal_mean = np.nanmean(signal_each_roi_separate)
    signal_std = np.nanstd(signal_each_roi_separate)
    baseline_signal = np.random.normal(signal_mean, signal_std * amplitude_scale, size=(T, 1))
    
    # Append to signals
    signal_with_baseline = np.hstack([signal_each_roi_separate, baseline_signal])
    
    return signal_with_baseline, ["random_baseline"]
