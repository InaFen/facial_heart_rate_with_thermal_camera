"""
Utility to add synthetic baseline (control) signals to ROI data.
"""

import numpy as np
from typing import List, Dict


def add_random_baseline_roi(
    roi_pixel_data: List[Dict[str, Dict[str, object]]],
    num_pixels_in_baseline: int = 100,
    adc_range: tuple = (28000, 30000),
    seed: int = None,
) -> List[Dict[str, Dict[str, object]]]:
    """
    Add a synthetic random baseline ROI to roi_pixel_data.
    
    The baseline signal is independent noise/random values, simulating a
    non-facial region (background/non-physiological).
    
    Parameters:
        roi_pixel_data: List of per-frame dicts with ROI data.
                       Each frame: {roi_name: {"coords": [...], "adc_values": [...], "mean_adc": float}}
        num_pixels_in_baseline: Number of fake pixel coordinates in the baseline ROI
        adc_range: Tuple (min, max) for random ADC values
        seed: Optional random seed for reproducibility
        
    Returns:
        roi_pixel_data with "baseline" ROI added to each frame
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_frames = len(roi_pixel_data)
    adc_min, adc_max = adc_range
    
    # Generate one set of baseline pixel coordinates (same for all frames)
    baseline_coords = [(np.random.randint(0, 160), np.random.randint(0, 122)) 
                       for _ in range(num_pixels_in_baseline)]
    
    # Process each frame
    for frame_idx in range(num_frames):
        # Generate random ADC values for this frame's baseline
        baseline_values = np.random.uniform(adc_min, adc_max, num_pixels_in_baseline)
        
        roi_pixel_data[frame_idx]["random_baseline"] = {
            "coords": baseline_coords,
            "adc_values": baseline_values.tolist(),
            "mean_adc": float(np.mean(baseline_values)),
        }
    
    return roi_pixel_data
