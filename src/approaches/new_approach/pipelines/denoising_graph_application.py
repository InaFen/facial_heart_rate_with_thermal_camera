import numpy as np

from src.approaches.yang_et_al.utils_spatial_temporal_graph import build_block_features_from_roi_pixel_data, build_spatiotemporal_graph_2, denoise_via_laplacian
from src.approaches.yang_et_al.utils_plotting import plot_signal_each_roi_seperately, plot_flattened_signal_each_roi, plot_original_vs_denoised
from src.approaches.new_approach.classes.user_input import Config


def apply_denoising_graph_pipeline(roi_pixel_data, configs: Config):
    """Apply denoising graph pipeline to ROI pixel data.
    Parameters:
        roi_pixel_data: List of Dicts with ROI pixel data per frame
        configs: Configuration object with processing parameters
    Returns:
        denoised_signal_all_rois: Flattened denoised signal for all ROIs
        signal_each_roi_seperate: List of per-ROI signals before flattening and nan replacement (steps were only needed for laplacian denoising)
        roi_names: List of ROI names corresponding to the signals"""
    
    #TODO graph creation with optimal feature metric, see utils_spatial_temporal_graph_with_optimal_feature_metric.py

    # Create graph and get laplacian
    block_features, roi_names = build_block_features_from_roi_pixel_data(roi_pixel_data)

    adjacency_matrix, laplacian = build_spatiotemporal_graph_2(block_features=block_features)

    # Prepare signal for denoising: Take block features and flatten
    signal_each_roi_seperate = [f[:, 0:1] for f in block_features] # shape = (T, N, 1)

    # plot each ROI signal, all in one plot
    if configs.plot_steps:
        plot_signal_each_roi_seperately(block_features=signal_each_roi_seperate, roi_names=roi_names, title = "Before Denoising - Each ROI Signal", fps = configs.fps, show_seconds=True)

    signal_all_rois = np.vstack(signal_each_roi_seperate)      # shape = (T*N, 1)

    #if configs.plot_steps:
    #    plot_flattened_signal_each_roi(signal_flattened=signal_all_rois, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="Before denoising")

    # Remove NaNs to be able to apply Laplacian denoising
    # compute mean ignoring NaNs
    mean_val = np.nanmean(signal_all_rois)
    # replace NaNs
    signal_all_rois_without_nans = np.nan_to_num(signal_all_rois, nan=mean_val)

    #if configs.plot_steps:
    #    plot_flattened_signal_each_roi(signal_flattened=signal_all_rois_without_nans, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="Before denoising with NaNs replaced")

    # Denoise signal via laplacian
    #denoised_signal_all_rois = denoise_via_laplacian(Y=signal_all_rois_without_nans, L=laplacian)

    #if configs.plot_steps:
    #    plot_flattened_signal_each_roi(signal_flattened=denoised_signal_all_rois, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="After denoising")
    #    plot_original_vs_denoised(signal_flattened_orig=signal_all_rois_without_nans, signal_flattened_denoised=denoised_signal_all_rois, block_features= signal_each_roi_seperate, roi_names=roi_names, title_info="Original vs Denoised Signals per ROI")

    print("[WARNING Graph Application]: Currently graph not applied and not all plots shown.")
    return signal_all_rois_without_nans, signal_each_roi_seperate, roi_names