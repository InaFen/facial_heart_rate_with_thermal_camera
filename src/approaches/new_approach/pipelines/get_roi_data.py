
from typing import Dict, List
import os
import numpy as np
from src.approaches.new_approach.classes.user_input import Config
from src.approaches.new_approach.utils.utils_landmark_detection import apply_facemesh_to_thermal_video, apply_roi_masks_to_video, plot_roi_mean_signals, apply_roi_masks_to_gray_frames, plot_roi_adc_means

def get_roi_data_pipeline(configs: Config, extract_from_video_instead_of_gray_files = False) -> List[Dict[str, Dict[str, object]]]:
    
    # Detect face
    if not os.path.isfile(configs.path_pixels_log) or configs.overwrite_existing_files:
        roi_pixel_log = apply_facemesh_to_thermal_video(input_path=configs.path_video, output_path=configs.path_video_facemesh, roi_landmarks=configs.roi_landmarks_subgrid_face, fps = configs.fps)
        os.makedirs(os.path.dirname(configs.path_pixels_log), exist_ok=True)
        np.save(configs.path_pixels_log, roi_pixel_log, allow_pickle=True)
    else:
        roi_pixel_log = np.load(configs.path_pixels_log, allow_pickle=True)

    # Extract ROI pixel data
    if extract_from_video_instead_of_gray_files:
        # Extract ROI pixel data from video instead of gray frames
        if not os.path.isfile(configs.path_roi_pixel_data) or configs.overwrite_existing_files:
            roi_pixel_data = apply_roi_masks_to_video(video_path=configs.path_video, roi_mask_log=roi_pixel_log, output_path=configs.path_video_thermal_with_roi_masks)
            os.makedirs(os.path.dirname(configs.path_roi_pixel_data), exist_ok=True)
            np.save(configs.path_roi_pixel_data, roi_pixel_data, allow_pickle=True)
        else:
            roi_pixel_data = np.load(configs.path_roi_pixel_data, allow_pickle=True)
    else:
        # Extract ROI pixel data from gray frames
        if not os.path.isfile(configs.path_roi_pixel_data) or configs.overwrite_existing_files:
            roi_pixel_data = apply_roi_masks_to_gray_frames(gray_dir=configs.path_gray_frames, roi_mask_log=roi_pixel_log, output_npz = None) # saved as npy in step below (has only one array)
            os.makedirs(os.path.dirname(configs.path_roi_pixel_data), exist_ok=True)
            np.save(configs.path_roi_pixel_data, roi_pixel_data, allow_pickle=True)
        else:
            roi_pixel_data = np.load(configs.path_roi_pixel_data, allow_pickle=True)

    # Plot
    if configs.plot_steps and extract_from_video_instead_of_gray_files:
        plot_roi_mean_signals(roi_pixel_data=roi_pixel_data) 
    elif configs.plot_steps and not extract_from_video_instead_of_gray_files:
        plot_roi_adc_means(roi_pixel_data=roi_pixel_data)

    return roi_pixel_data