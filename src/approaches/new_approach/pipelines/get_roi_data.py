
from typing import Dict, List
import os
import cv2
import numpy as np
from src.approaches.new_approach.classes.user_input import Config
from src.approaches.new_approach.utils.utils_landmark_detection import (
    apply_facemesh_to_thermal_video,
    apply_roi_masks_to_video,
    plot_roi_mean_signals,
    apply_roi_masks_to_gray_frames,
    apply_roi_masks_to_asc_frames,
    plot_roi_temperature_means,
    create_manual_roi_mask_log_from_gray_frames,
    create_manual_roi_mask_log_from_asc_frames,
    _parse_asc_file,
    _asc_to_kelvin_x100,
    load_gray_frame,
    write_roi_tracked_video_from_gray_frames,
    split_manual_roi_into_subrois
)
from src.approaches.nair_et_al.helpers_landmark_detection import (
    apply_dlib_to_thermal_video,
)
from src.approaches.new_approach.utils.utils_gray_files import sort_grayfiles_with_interpolations
from src.approaches.yang_et_al.utils_face_tracking import track_face


def get_roi_data_pipeline(configs: Config, roi_detection_method: str = 'facemesh', extract_from_video_instead_of_gray_files: bool = False) -> tuple:
    """
    Extract ROI data from thermal video and gray frames (raw data).
    ROI can be defined via manual user input (drawing on first frame) or via facemesh-based landmark detection.
    The ROI masks are applied to either the video frames or the original gray frames to extract pixel data for each ROI over time.

    Parameters:
    - configs: Config object containing all necessary paths and parameters
    - roi_detection_method: Method to use for ROI detection ('dlib', 'facemesh', or 'tracker')
    - extract_from_video_instead_of_gray_files: If True, applies ROI masks to video, else applies to gray frames (default: False)
    
    Returns:
        Tuple of (roi_pixel_data, roi_pixel_log) where:
        - roi_pixel_data: The extracted pixel intensity data
        - roi_pixel_log: The detected pixel locations per ROI per frame
    """
    
    # Limit processing to specific timespan if specified in configs
    start_frame = None
    end_frame = None
    limit_to_timespan = configs.timespan_signal_to_look_at is not None
    if limit_to_timespan:
        start_time, end_time = configs.timespan_signal_to_look_at
        start_frame = int(start_time * configs.fps)
        end_frame = int(end_time * configs.fps)

    ##########
    # Get ROI (manual ROI with user input or facemesh-based ROI)
    ##########

    # If manual ROI is enabled, user draws ROI on first frame and it is applied to all frames
    # Check if we need to split a single manual ROI into sub-ROIs based on configs
    split_single_manual_roi = (
        configs.manual_roi_enabled
        and configs.manual_roi_split_enabled
        and len(configs.manual_roi_names) == 1
    )
    if configs.manual_roi_enabled:
        if not os.path.isfile(configs.path_pixels_log) or configs.overwrite_existing_files:
            frame_index = 0
            if configs.timespan_signal_to_look_at is not None:
                start_time, _ = configs.timespan_signal_to_look_at
                frame_index = int(start_time * configs.fps)

            files_in_dir = [
                f for f in os.listdir(configs.path_gray_frames)
                if os.path.isfile(os.path.join(configs.path_gray_frames, f))
            ]
            if not files_in_dir:
                raise ValueError(f"No files found in {configs.path_gray_frames}")

            sample_ext = os.path.splitext(files_in_dir[0])[1].lower()

            roi_mask_logs = []
            for i, roi_name in enumerate(configs.manual_roi_names):
                if sample_ext == ".asc":
                    roi_mask_logs.append(create_manual_roi_mask_log_from_asc_frames(
                        asc_dir=configs.path_gray_frames,
                        roi_name=roi_name,
                        display_title=f"Draw ROI '{roi_name}' (press ENTER to confirm)",
                        mode=configs.manual_roi_modes[i] if i < len(configs.manual_roi_modes) else configs.manual_roi_modes[0],
                        frame_index=frame_index,
                        mask_load_path=configs.manual_roi_mask_load_paths[i] if i < len(configs.manual_roi_mask_load_paths) else None,
                        mask_save_path=configs.manual_roi_mask_save_paths[i] if i < len(configs.manual_roi_mask_save_paths) else None,
                        use_colormap=configs.manual_roi_selection_colormap,
                        unit=configs.asc_unit,
                    ))
                else:
                    roi_mask_logs.append(create_manual_roi_mask_log_from_gray_frames(
                        gray_dir=configs.path_gray_frames,
                        roi_name=roi_name,
                        display_title=f"Draw ROI '{roi_name}' (press ENTER to confirm)",
                        mode=configs.manual_roi_modes[i] if i < len(configs.manual_roi_modes) else configs.manual_roi_modes[0],
                        frame_index=frame_index,
                        mask_load_path=configs.manual_roi_mask_load_paths[i] if i < len(configs.manual_roi_mask_load_paths) else None,
                        mask_save_path=configs.manual_roi_mask_save_paths[i] if i < len(configs.manual_roi_mask_save_paths) else None,
                        use_colormap=configs.manual_roi_selection_colormap,
                    ))
            # Merge masks for each frame into a single dict per frame
            roi_pixel_log = []
            for frame_idx in range(len(roi_mask_logs[0])):
                frame_dict = {}
                for roi_idx, roi_name in enumerate(configs.manual_roi_names):
                    frame_entry = roi_mask_logs[roi_idx][frame_idx]
                    for key, value in frame_entry.items():
                        frame_dict[key] = value
                roi_pixel_log.append(frame_dict)
            os.makedirs(os.path.dirname(configs.path_pixels_log), exist_ok=True)
            np.save(configs.path_pixels_log, roi_pixel_log, allow_pickle=True)
        else:
            roi_pixel_log = np.load(configs.path_pixels_log, allow_pickle=True)

        # Split single manual ROI into sub-ROIs if enabled in configs
        if split_single_manual_roi:
            first_frame = roi_pixel_log[0] if len(roi_pixel_log) > 0 else {}
            if configs.manual_roi_names[0] in first_frame:
                grid_rows, grid_cols = configs.manual_roi_subroi_grid
                roi_pixel_log, _ = split_manual_roi_into_subrois(
                    roi_pixel_log=roi_pixel_log,
                    roi_name=configs.manual_roi_names[0],
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                )
    
    # If manual ROI not enabled, use facemesh-based ROI detection: 
    # detects landmarks on thermal video, applies ROI mask then on gray frames (so raw data)
    else:
        if not os.path.isfile(configs.path_pixels_log) or configs.overwrite_existing_files:
            if roi_detection_method == "tracker":
                roi_pixel_log = track_face(
                    in_path=configs.path_video,
                    out_path=configs.path_video_tracked_face,
                    fps=configs.fps,
                    flip_upside_down=True,
                    start_frame=start_frame if limit_to_timespan else 0,
                    end_frame=end_frame if limit_to_timespan else None,
                )
            elif roi_detection_method == "dlib":
                print("Use Dlib-based landmark detection for ROI definition (instead of Facemesh).")
                roi_pixel_log = apply_dlib_to_thermal_video(
                    video_path=configs.path_video,
                    path_save_video_with_landmarks=configs.path_video_facemesh,
                    upsample=0,
                    resize_frame=False,
                    fps=configs.fps,
                    show_display=True,
                    start_frame=start_frame if limit_to_timespan else 0,
                    end_frame=end_frame if limit_to_timespan else None,
                    flip_upside_down=True,
                )
            else: # Standard method is facemesh-based
                # Build ROI landmarks to pass to facemesh: prefer explicit selection via configs.used_rois
                # and configs.roi_landmarks_options. Fall back to the older roi_landmarks_subgrid_face for compatibility.
                if getattr(configs, 'used_rois', None):
                    roi_landmarks_selected = {}
                    for rn in configs.used_rois:
                        if rn in getattr(configs, 'roi_landmarks_options', {}):
                            roi_landmarks_selected[rn] = configs.roi_landmarks_options[rn]
                        else:
                            raise ValueError(
                                f"Requested ROI '{rn}' not found in 'roi_landmarks_options'.\n"
                                "Please add the ROI definition to 'roi_landmarks_options' in the config."
                            )
                    if not roi_landmarks_selected:
                        raise ValueError(
                            "No valid ROIs selected from 'used_rois'. Ensure 'used_rois' names exist in 'roi_landmarks_options'."
                        )
                else:
                    raise ValueError(
                        "No ROI definitions found in config. Please provide 'roi_landmarks_options' mapping in the YAML."
                    )

                roi_pixel_log = apply_facemesh_to_thermal_video(
                    input_path=configs.path_video,
                    output_path=configs.path_video_facemesh,
                    roi_landmarks=roi_landmarks_selected,
                    fps=configs.fps,
                    use_ema=configs.use_ema,
                    ema_alpha=configs.ema_alpha,
                    start_frame=start_frame if limit_to_timespan else 0,
                    end_frame=end_frame if limit_to_timespan else None,
                )
            os.makedirs(os.path.dirname(configs.path_pixels_log), exist_ok=True)
            np.save(configs.path_pixels_log, roi_pixel_log, allow_pickle=True)
        else:
            roi_pixel_log = np.load(configs.path_pixels_log, allow_pickle=True)

    ##########
    # Extract ROI pixel data
    ##########

    # Extract ROI pixel data from video instead of gray frames
    if extract_from_video_instead_of_gray_files:
        if (
            not os.path.isfile(configs.path_roi_pixel_data)
            or configs.overwrite_existing_files
            or split_single_manual_roi
        ):
            roi_pixel_data = apply_roi_masks_to_video(
                video_path=configs.path_video,
                roi_mask_log=roi_pixel_log,
                output_path=configs.path_video_thermal_with_roi_masks,
                fps=configs.fps,
                start_frame=start_frame if limit_to_timespan else 0,
                end_frame=end_frame if limit_to_timespan else None,
            )
            os.makedirs(os.path.dirname(configs.path_roi_pixel_data), exist_ok=True)
            np.save(configs.path_roi_pixel_data, roi_pixel_data, allow_pickle=True)
        else:
            roi_pixel_data = np.load(configs.path_roi_pixel_data, allow_pickle=True)
    
    # Extract ROI pixel data from gray or asc frames
    else:
        if (
            not os.path.isfile(configs.path_roi_pixel_data)
            or configs.overwrite_existing_files
            or split_single_manual_roi # Each ROI is treated as a sub-ROI so ROIs need to be defined accordingly in config
        ):
            files_in_dir = [f for f in os.listdir(configs.path_gray_frames) if os.path.isfile(os.path.join(configs.path_gray_frames, f))]
            if not files_in_dir:
                raise ValueError(f"No files found in {configs.path_gray_frames}")

            sample_ext = os.path.splitext(files_in_dir[0])[1].lower()
            # extract from asc files
            if sample_ext == ".asc":
                asc_paths = sorted(
                    [os.path.join(configs.path_gray_frames, f) for f in files_in_dir if f.lower().endswith(".asc")]
                )
                roi_pixel_data = apply_roi_masks_to_asc_frames(
                    asc_paths=asc_paths,
                    roi_mask_log=roi_pixel_log,
                    unit=configs.asc_unit,
                )
            else:
            # extract from gray files
                roi_pixel_data = apply_roi_masks_to_gray_frames(
                    gray_dir=configs.path_gray_frames,
                    roi_mask_log=roi_pixel_log,
                    output_npz=None,
                    start_frame=start_frame if limit_to_timespan else 0,
                    end_frame=end_frame if limit_to_timespan else None,
                )
            os.makedirs(os.path.dirname(configs.path_roi_pixel_data), exist_ok=True)
            np.save(configs.path_roi_pixel_data, roi_pixel_data, allow_pickle=True)
        else:
            roi_pixel_data = np.load(configs.path_roi_pixel_data, allow_pickle=True)

    roi_pixel_data_for_return = roi_pixel_data

    if configs.timespan_signal_to_look_at is not None:
        # Save ROI masked video for the selected timespan for verification
        start_time, end_time = configs.timespan_signal_to_look_at
        start_frame = int(start_time * configs.fps)
        end_frame = int(end_time * configs.fps)
        os.makedirs(os.path.dirname(configs.path_video_tracked_face), exist_ok=True)
        write_roi_tracked_video_from_gray_frames(
            gray_dir=configs.path_gray_frames,
            roi_mask_log=roi_pixel_log,
            output_path=configs.path_video_tracked_face,
            fps=configs.fps,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        roi_pixel_data_for_return = roi_pixel_data

        local_start_idx = 0 if limit_to_timespan else start_frame
        if 0 <= local_start_idx < len(roi_pixel_log):
            roi_masks = roi_pixel_log[local_start_idx]
            output_dir = os.path.dirname(configs.path_video_tracked_face) or configs.path_gray_frames
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"roi_first_frame_{start_frame:06d}.png")

            # If extracting from video, read the first frame of the video for visualization, else read the corresponding gray frame
            if extract_from_video_instead_of_gray_files:
                cap = cv2.VideoCapture(configs.path_video)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        if frame.dtype != np.uint8:
                            vmin, vmax = np.percentile(frame, [1, 99])
                            frame_clipped = np.clip(frame, vmin, vmax)
                            display_frame = cv2.normalize(frame_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        else:
                            display_frame = frame.copy()
                    cap.release()
            else:
                files_in_dir = [
                    f for f in os.listdir(configs.path_gray_frames)
                    if os.path.isfile(os.path.join(configs.path_gray_frames, f))
                ]
                if not files_in_dir:
                    raise ValueError(f"No files found in {configs.path_gray_frames}")

                sample_ext = os.path.splitext(files_in_dir[0])[1].lower()
                if sample_ext == ".asc":
                    # Read the corresponding ASC frame for visualization
                    asc_files = sorted(
                        [f for f in files_in_dir if f.lower().endswith(".asc")]
                    )
                    if 0 <= start_frame < len(asc_files):
                        frame_path = os.path.join(configs.path_gray_frames, asc_files[start_frame])
                        if roi_masks:
                            frame = _parse_asc_file(frame_path)
                            frame = _asc_to_kelvin_x100(frame, configs.asc_unit)
                            vmin, vmax = np.percentile(frame, [1, 99])
                            frame_clipped = np.clip(frame, vmin, vmax)
                            display_frame = cv2.normalize(frame_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                else:
                    # Read the corresponding gray frame for visualization
                    _, gray_files = sort_grayfiles_with_interpolations(configs.path_gray_frames)
                    if 0 <= start_frame < len(gray_files):
                        frame_path = os.path.join(configs.path_gray_frames, gray_files[start_frame])
                        if roi_masks:
                            first_mask = next(iter(roi_masks.values()))
                            drop_telemetry = first_mask.shape[0] <= 120
                            raw_height = first_mask.shape[0] + (2 if drop_telemetry else 0)
                            raw_width = first_mask.shape[1]
                            frame = load_gray_frame(
                                frame_path,
                                width=raw_width,
                                height=raw_height,
                                drop_telemetry=drop_telemetry,
                            )
                            vmin, vmax = np.percentile(frame, [1, 99])
                            frame_clipped = np.clip(frame, vmin, vmax)
                            display_frame = cv2.normalize(frame_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Overlay ROI masks on the frame for visualization and save
            if display_frame is not None and roi_masks:
                if display_frame.ndim == 2:
                    overlay = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
                else:
                    overlay = display_frame.copy()

                colors = [
                    (0, 255, 0),
                    (0, 0, 255),
                    (255, 0, 0),
                    (255, 255, 0),
                    (255, 0, 255),
                    (0, 255, 255),
                ]
                for idx, (roi_name, mask) in enumerate(roi_masks.items()):
                    if mask.shape != overlay.shape[:2]:
                        print(
                            f"Warning: ROI mask shape {mask.shape} does not match frame shape {overlay.shape[:2]} for {roi_name}"
                        )
                        continue
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    color = colors[idx % len(colors)]
                    cv2.drawContours(overlay, contours, -1, color, 1)

                cv2.imwrite(output_path, overlay)
                print(f"Saved ROI first-frame snapshot to: {output_path}")

    # Plot
    if configs.plot_steps and extract_from_video_instead_of_gray_files:
        plot_roi_mean_signals(roi_pixel_data=roi_pixel_data_for_return) 
    elif configs.plot_steps and not extract_from_video_instead_of_gray_files:
        plot_roi_temperature_means(roi_pixel_data=roi_pixel_data_for_return, fps=configs.fps)

    return roi_pixel_data_for_return, roi_pixel_log