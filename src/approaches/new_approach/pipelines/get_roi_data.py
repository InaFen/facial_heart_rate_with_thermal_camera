
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
    plot_roi_temperature_means,
    create_manual_roi_mask_log_from_gray_frames,
    load_gray_frame,
    write_roi_tracked_video_from_gray_frames,
)
from src.approaches.new_approach.utils.utils_gray_files import sort_grayfiles_with_interpolations


def split_manual_roi_into_subrois(
    roi_pixel_log: List[Dict[str, np.ndarray]],
    roi_name: str,
    grid_rows: int,
    grid_cols: int,
) -> tuple[List[Dict[str, np.ndarray]], List[str]]:
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("manual_roi_subroi_grid must contain positive integers")

    subroi_names = [f"{roi_name}_r{row + 1}c{col + 1}" for row in range(grid_rows) for col in range(grid_cols)]
    new_log: List[Dict[str, np.ndarray]] = []

    for frame_dict in roi_pixel_log:
        mask = frame_dict.get(roi_name)
        if mask is None:
            new_log.append(frame_dict)
            continue

        ys, xs = np.where(mask > 0)
        if ys.size == 0 or xs.size == 0:
            empty_frame = {name: np.zeros_like(mask) for name in subroi_names}
            new_log.append(empty_frame)
            continue

        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())

        y_edges = np.linspace(y_min, y_max + 1, grid_rows + 1).astype(int)
        x_edges = np.linspace(x_min, x_max + 1, grid_cols + 1).astype(int)

        frame_subrois: Dict[str, np.ndarray] = {}
        name_idx = 0
        for row in range(grid_rows):
            for col in range(grid_cols):
                y0, y1 = y_edges[row], y_edges[row + 1]
                x0, x1 = x_edges[col], x_edges[col + 1]
                sub_mask = np.zeros_like(mask)
                sub_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
                frame_subrois[subroi_names[name_idx]] = sub_mask
                name_idx += 1

        new_log.append(frame_subrois)

    return new_log, subroi_names

def get_roi_data_pipeline(configs: Config, extract_from_video_instead_of_gray_files = False) -> tuple:
    """
    Extract ROI data from thermal video and gray frames.
    
    Returns:
        Tuple of (roi_pixel_data, roi_pixel_log) where:
        - roi_pixel_data: The extracted pixel intensity data
        - roi_pixel_log: The detected pixel locations per ROI per frame
    """
    
    start_frame = None
    end_frame = None
    limit_to_timespan = configs.timespan_signal_to_look_at is not None
    if limit_to_timespan:
        start_time, end_time = configs.timespan_signal_to_look_at
        start_frame = int(start_time * configs.fps)
        end_frame = int(end_time * configs.fps)

    # Detect face or use manual ROI (support dual ROI selection)
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

            roi_mask_logs = []
            for i, roi_name in enumerate(configs.manual_roi_names):
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
                    frame_dict[roi_name] = roi_mask_logs[roi_idx][frame_idx][roi_name]
                roi_pixel_log.append(frame_dict)
            os.makedirs(os.path.dirname(configs.path_pixels_log), exist_ok=True)
            np.save(configs.path_pixels_log, roi_pixel_log, allow_pickle=True)
        else:
            roi_pixel_log = np.load(configs.path_pixels_log, allow_pickle=True)

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
    else:
        if not os.path.isfile(configs.path_pixels_log) or configs.overwrite_existing_files:
            roi_pixel_log = apply_facemesh_to_thermal_video(
                input_path=configs.path_video,
                output_path=configs.path_video_facemesh,
                roi_landmarks=configs.roi_landmarks_subgrid_face,
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

    # Extract ROI pixel data
    if extract_from_video_instead_of_gray_files:
        # Extract ROI pixel data from video instead of gray frames
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
    else:
        # Extract ROI pixel data from gray frames
        if (
            not os.path.isfile(configs.path_roi_pixel_data)
            or configs.overwrite_existing_files
            or split_single_manual_roi
        ):
            roi_pixel_data = apply_roi_masks_to_gray_frames(
                gray_dir=configs.path_gray_frames,
                roi_mask_log=roi_pixel_log,
                output_npz=None,
                start_frame=start_frame if limit_to_timespan else 0,
                end_frame=end_frame if limit_to_timespan else None,
            ) # saved as npy in step below (has only one array)
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

            display_frame = True
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