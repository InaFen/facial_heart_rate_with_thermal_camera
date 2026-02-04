
from typing import Dict, List
import os
import numpy as np
from src.approaches.new_approach.classes.user_input import Config
from src.approaches.new_approach.utils.utils_landmark_detection import (
    apply_facemesh_to_thermal_video,
    apply_roi_masks_to_video,
    plot_roi_mean_signals,
    apply_roi_masks_to_gray_frames,
    plot_roi_temperature_means,
    create_manual_roi_mask_log_from_gray_frames,
)

def get_roi_data_pipeline(configs: Config, extract_from_video_instead_of_gray_files = False) -> tuple:
    """
    Extract ROI data from thermal video and gray frames.
    
    Returns:
        Tuple of (roi_pixel_data, roi_pixel_log) where:
        - roi_pixel_data: The extracted pixel intensity data
        - roi_pixel_log: The detected pixel locations per ROI per frame
    """
    
    # Detect face or use manual ROI
    if configs.manual_roi_enabled:
        if not os.path.isfile(configs.path_pixels_log) or configs.overwrite_existing_files:
            frame_index = 0
            if configs.timespan_signal_to_look_at is not None:
                start_time, _ = configs.timespan_signal_to_look_at
                frame_index = int(start_time * configs.fps)

            roi_pixel_log = create_manual_roi_mask_log_from_gray_frames(
                gray_dir=configs.path_gray_frames,
                roi_name=configs.manual_roi_name,
                mode=configs.manual_roi_mode,
                frame_index=frame_index,
                mask_load_path=configs.manual_roi_mask_load_path,
                mask_save_path=configs.manual_roi_mask_save_path,
                use_colormap=configs.manual_roi_selection_colormap,
            )
            os.makedirs(os.path.dirname(configs.path_pixels_log), exist_ok=True)
            np.save(configs.path_pixels_log, roi_pixel_log, allow_pickle=True)
        else:
            roi_pixel_log = np.load(configs.path_pixels_log, allow_pickle=True)
    else:
        if not os.path.isfile(configs.path_pixels_log) or configs.overwrite_existing_files:
            roi_pixel_log = apply_facemesh_to_thermal_video(
                input_path=configs.path_video,
                output_path=configs.path_video_facemesh,
                roi_landmarks=configs.roi_landmarks_subgrid_face,
                fps=configs.fps,
                use_ema=configs.use_ema,
                ema_alpha=configs.ema_alpha,
            )
            os.makedirs(os.path.dirname(configs.path_pixels_log), exist_ok=True)
            np.save(configs.path_pixels_log, roi_pixel_log, allow_pickle=True)
        else:
            roi_pixel_log = np.load(configs.path_pixels_log, allow_pickle=True)

    # Extract ROI pixel data
    if extract_from_video_instead_of_gray_files:
        # Extract ROI pixel data from video instead of gray frames
        if not os.path.isfile(configs.path_roi_pixel_data) or configs.overwrite_existing_files:
            roi_pixel_data = apply_roi_masks_to_video(video_path=configs.path_video, roi_mask_log=roi_pixel_log, output_path=configs.path_video_thermal_with_roi_masks, fps = configs.fps)
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

    if configs.manual_roi_enabled and configs.timespan_signal_to_look_at is not None:
        # Save ROI masked video for the selected timespan for verification
        start_time, end_time = configs.timespan_signal_to_look_at
        start_frame = int(start_time * configs.fps)
        end_frame = int(end_time * configs.fps)
        os.makedirs(os.path.dirname(configs.path_video_tracked_face), exist_ok=True)
        _ = apply_roi_masks_to_video(
            video_path=configs.path_video,
            roi_mask_log=roi_pixel_log,
            output_path=configs.path_video_tracked_face,
            fps=configs.fps,
            start_frame=start_frame,
            end_frame=end_frame,
        )

    # Plot
    if configs.plot_steps and extract_from_video_instead_of_gray_files:
        plot_roi_mean_signals(roi_pixel_data=roi_pixel_data) 
    #elif configs.plot_steps and not extract_from_video_instead_of_gray_files:
    #    plot_roi_temperature_means(roi_pixel_data=roi_pixel_data)

    return roi_pixel_data, roi_pixel_log