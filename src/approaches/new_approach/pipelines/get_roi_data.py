
from typing import Dict, List
import os
import cv2
import numpy as np
import yaml
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
    create_manual_roi_mask_log_from_npy_frames,
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


# Top forehead landmarks (left side: 103/67/109, center: 10, right side: 338/297/332).
# If a chosen ROI contains ANY of these it reaches the top of the forehead and is
# eligible for upward extension towards the hairline.
FOREHEAD_EXTEND_LANDMARKS = {103, 67, 109, 10, 338, 297, 332}
FOREHEAD_TEMP_TOLERANCE_PCT = 0.05  # Allow 5% deviation from mean
FOREHEAD_MAX_EXTEND_PIXELS = 100  # Max pixels to extend upward

# ROI names for which the upward forehead extension is activated. Each ROI is
# expanded independently using only its own mask, so using e.g. only
# "right_forehead" expands just the right forehead, not the whole forehead.
FOREHEAD_ROI_NAMES = {"forehead", "left_forehead", "right_forehead"}


def _normalize_roi_name(name: str) -> str:
    """Normalize an ROI name for matching (case/space/hyphen insensitive)."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def is_forehead_roi(name: str) -> bool:
    """Return True if the ROI name is a forehead-type ROI eligible for extension."""
    return _normalize_roi_name(name) in FOREHEAD_ROI_NAMES


# Path to the shared ROI landmark definitions file (single source of truth).
# Located in the new_approach package root, next to the per-approach configs.
ROI_LANDMARKS_OPTIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "roi_landmarks_options.yaml",
)

_ROI_LANDMARKS_OPTIONS_CACHE: Dict[str, List[int]] = None


def load_roi_landmarks_options(path: str = None) -> Dict[str, List[int]]:
    """
    Load the named ROI landmark definitions from the shared options file.

    The result is cached when the default path is used. Returns a mapping of
    {roi_name: [landmark indices]}.
    """
    global _ROI_LANDMARKS_OPTIONS_CACHE
    if path is None and _ROI_LANDMARKS_OPTIONS_CACHE is not None:
        return _ROI_LANDMARKS_OPTIONS_CACHE

    file_path = path or ROI_LANDMARKS_OPTIONS_PATH
    options: Dict[str, List[int]] = {}
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            data = yaml.safe_load(f) or {}
        # Support both a top-level 'roi_landmarks_options' key and a bare mapping.
        options = data.get("roi_landmarks_options", data) or {}

    if path is None:
        _ROI_LANDMARKS_OPTIONS_CACHE = options
    return options


def get_roi_landmarks_options(configs: Config) -> Dict[str, List[int]]:
    """
    Return the effective ROI landmark options for a run.

    The shared external definitions are used as the base; any inline
    `configs.roi_landmarks_options` (e.g. still defined in a per-approach YAML)
    is merged on top and takes precedence, preserving backward compatibility.
    """
    options = dict(load_roi_landmarks_options())
    inline = getattr(configs, "roi_landmarks_options", None) or {}
    options.update(inline)
    return options


def select_roi_landmarks(
    used_rois: List[str],
    options: Dict[str, List[int]],
    strict: bool = True,
) -> Dict[str, List[int]]:
    """
    Select the landmark definitions for the requested ROI names.

    Parameters:
        used_rois: list of ROI names to select.
        options: mapping of all available ROI landmark definitions.
        strict: if True, raise on an unknown ROI name; if False, silently skip it.

    Returns:
        Mapping {roi_name: [landmark indices]} for the requested ROIs.
    """
    selected: Dict[str, List[int]] = {}
    for rn in used_rois:
        if rn in options:
            selected[rn] = options[rn]
        elif strict:
            raise ValueError(
                f"Requested ROI '{rn}' not found in 'roi_landmarks_options'.\n"
                f"Please add the ROI definition to {ROI_LANDMARKS_OPTIONS_PATH} "
                "(or to an inline 'roi_landmarks_options' mapping in the config)."
            )
    return selected


def extend_forehead_roi_upward(
    frame: np.ndarray,
    mask: np.ndarray,
    landmark_indices: List[int],
    tolerance_pct: float = FOREHEAD_TEMP_TOLERANCE_PCT,
    max_extend_px: int = FOREHEAD_MAX_EXTEND_PIXELS,
) -> np.ndarray:
    """
    Extend the forehead ROI upward (decreasing Y) following the hairline shape.

    Instead of growing the whole forehead width as a single rectangular block,
    each column of the forehead is grown independently upward. Because every
    column stops at its own hairline, the resulting upper boundary follows the
    natural (curved) shape of the forehead rather than producing a square.

    Algorithm:
    1. Check that the required forehead landmarks are present; otherwise skip.
    2. Compute a robust reference temperature from the base forehead ROI
       (median + MAD) which represents skin temperature ("mean temperature x").
    3. For each column within the forehead's horizontal span, start at the
       topmost masked pixel of that column and walk upward pixel by pixel.
    4. Keep a pixel while its (locally smoothed) temperature stays similar to
       the skin reference; stop that column as soon as it deviates significantly
       (assumed to be the hairline / end of the forehead).
    5. Clean up the result with a light morphological closing to remove small
       holes / jagged single-pixel artifacts while preserving the overall shape.

    Parameters:
        frame: 2D numpy array (height, width) with temperature/intensity values
        mask: 2D binary mask (height, width) with 255 indicating ROI pixels
        landmark_indices: list of mediapipe landmark indices that define the base forehead
        tolerance_pct: allowed deviation as fraction of the skin reference value
        max_extend_px: maximum pixels to extend upward per column

    Returns:
        Extended mask as 2D binary array
    """
    # The ROI must reach the top of the forehead to be extendable. For the full
    # forehead all top landmarks are present; for a left/right forehead only the
    # corresponding side's top landmarks are present. Hence we trigger on ANY
    # overlap with the top-forehead landmark set rather than requiring all of them.
    trigger_set = set(FOREHEAD_EXTEND_LANDMARKS)
    provided_set = set(landmark_indices)
    if not trigger_set & provided_set:
        # ROI does not reach the top of the forehead; nothing to extend.
        print(f"[extend_forehead] No top-forehead landmarks {trigger_set} found in {provided_set}. Skipping extension.")
        return mask

    height, width = frame.shape
    extended_mask = mask.copy()

    # Get all current ROI pixels
    ys, xs = np.where(mask == 255)
    if len(ys) == 0:
        return extended_mask

    frame_f = frame.astype(np.float64)

    # Robust reference temperature of the skin from the base forehead ROI.
    # Median + MAD is far less sensitive to outliers (e.g. a few hair/edge
    # pixels already inside the base mask) than mean + std.
    base_temps = frame_f[ys, xs]
    skin_ref = float(np.median(base_temps))
    mad = float(np.median(np.abs(base_temps - skin_ref)))
    # Convert MAD to a std-equivalent (factor 1.4826) for a sensible spread.
    skin_spread = 1.4826 * mad if mad > 0 else float(np.std(base_temps))

    # Tolerance combines a percentage of the skin value and the natural spread
    # of the forehead skin, so columns are allowed to vary as much as the skin
    # itself does, but stop when they hit the much colder/warmer hairline.
    tolerance = max(abs(skin_ref) * tolerance_pct, 2.0 * skin_spread)

    # Number of consecutive deviating pixels tolerated before stopping a column.
    # This bridges thin warm artifacts (e.g. stray hair strands) without leaking
    # past the actual hairline.
    max_consecutive_outliers = 2

    # --- Vectorized per-column upward growth -------------------------------
    # Vertically smooth the frame once (3-row window) instead of recomputing a
    # local mean for every candidate pixel. Edges use the clamped (available)
    # window, matching the previous per-pixel behaviour.
    sm = frame_f.copy()
    if height >= 3:
        sm[1:-1, :] = (frame_f[:-2, :] + frame_f[1:-1, :] + frame_f[2:, :]) / 3.0
        sm[0, :] = (frame_f[0, :] + frame_f[1, :]) / 2.0
        sm[-1, :] = (frame_f[-1, :] + frame_f[-2, :]) / 2.0

    # Topmost masked pixel per column (vectorized): the start of each climb.
    # Columns absent from the base mask are skipped to keep the shape faithful.
    order = np.lexsort((ys, xs))  # sort by column, then row
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    first_idx = np.concatenate(([0], np.nonzero(np.diff(xs_sorted))[0] + 1))
    cols = xs_sorted[first_idx]                 # unique columns present
    start_y = ys_sorted[first_idx]              # topmost y per column
    n_cols = cols.size

    if n_cols and max_extend_px > 0:
        # Candidate pixel y for every (step, column): y = start_y - (step+1).
        steps = np.arange(1, max_extend_px + 1)[:, None]          # (S, 1)
        cand_y = start_y[None, :] - steps                         # (S, n_cols)
        valid = cand_y >= 0
        cand_y_clamped = np.clip(cand_y, 0, height - 1)

        # "Good" = looks like forehead skin (and still inside the image).
        vals = sm[cand_y_clamped, cols[None, :]]                  # (S, n_cols)
        good = (np.abs(vals - skin_ref) <= tolerance) & valid

        # Terminate a column at the first run of (max_consecutive_outliers + 1)
        # consecutive bad pixels (the hairline). A pixel is accepted iff it is
        # good and lies before that termination step.
        bad = ~good
        run = max_consecutive_outliers + 1
        triple = np.zeros_like(bad)
        if bad.shape[0] >= run:
            acc = bad[run - 1:].copy()
            for k in range(1, run):
                acc &= bad[run - 1 - k: bad.shape[0] - k]
            triple[run - 1:] = acc
        has_term = triple.any(axis=0)
        cutoff = np.where(has_term, triple.argmax(axis=0), max_extend_px)

        step_idx = np.arange(max_extend_px)[:, None]
        accepted = good & (step_idx < cutoff[None, :])

        ys_acc = cand_y[accepted]
        xs_acc = np.broadcast_to(cols[None, :], accepted.shape)[accepted]
        extended_mask[ys_acc, xs_acc] = 255

    # Light morphological closing to smooth the upper boundary and fill the small
    # one-pixel gaps left by the per-column growth, without changing the shape.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    extended_mask = cv2.morphologyEx(extended_mask, cv2.MORPH_CLOSE, kernel)

    return extended_mask


def apply_roi_masks_to_npy_frames(
    frame_stack: np.ndarray,
    roi_mask_log: List[Dict[str, np.ndarray]],
    roi_landmarks: Dict[str, List[int]] = None,
) -> List[Dict[str, Dict[str, object]]]:
    """
    Apply ROI masks to a precomputed numeric frame stack from a .npy file.
    
    Parameters:
        frame_stack: 3D numpy array of shape (n_frames, height, width) containing numeric frame data
        roi_mask_log: list of per-frame ROI masks where each entry is a dict {roi_name: mask_array}
        roi_landmarks: optional dict mapping ROI name to list of landmark indices (for forehead extension)
    
    Returns:
        roi_pixel_data: list (per frame) of dicts {roi_name: {"coords", "adc_values", "mean_value"}}
    """
    if frame_stack.ndim != 3:
        raise ValueError(f"Expected 3D frame stack, got shape {frame_stack.shape}")
    
    n_frames, height, width = frame_stack.shape
    
    if len(roi_mask_log) != n_frames:
        raise ValueError(f"Frame stack has {n_frames} frames but ROI mask log has {len(roi_mask_log)} entries")
    
    roi_pixel_data: List[Dict[str, Dict[str, object]]] = []
    
    for frame_idx, (frame, masks) in enumerate(zip(frame_stack, roi_mask_log)):
        frame_data: Dict[str, Dict[str, object]] = {}
        
        for roi_name, mask in masks.items():
            if mask.shape != frame.shape:
                raise ValueError(f"Mask shape {mask.shape} does not match frame shape {frame.shape} for ROI {roi_name}")
            
            # Apply forehead upward extension if applicable (forehead / left_forehead / right_forehead).
            # Each forehead-type ROI is expanded independently using only its own mask.
            if is_forehead_roi(roi_name) and roi_landmarks and roi_name in roi_landmarks:
                #print(f"[NPY] Extending forehead ROI '{roi_name}' upward for frame {frame_idx} with {len(roi_landmarks[roi_name])} landmarks")
                mask = extend_forehead_roi_upward(
                    frame=frame,
                    mask=mask,
                    landmark_indices=roi_landmarks[roi_name],
                )
            
            ys, xs = np.where(mask == 255)
            coords = list(zip(xs.tolist(), ys.tolist()))
            values = frame[ys, xs].astype(np.float64)
            mean_value = np.mean(values) if len(values) > 0 else 0.0
            
            frame_data[roi_name] = {
                "coords": coords,
                "adc_values": values.tolist(),
                "mean_value": mean_value,
            }
        
        roi_pixel_data.append(frame_data)
    
    return roi_pixel_data


def write_roi_tracked_video_from_npy_frames(
    frame_stack: np.ndarray,
    roi_mask_log: List[Dict[str, np.ndarray]],
    output_path: str,
    fps: float,
    start_frame: int = 0,
    end_frame: int = None,
) -> None:
    """Write a masked video from a precomputed numeric frame stack using ROI masks."""
    if frame_stack.ndim != 3:
        raise ValueError(f"Expected 3D frame stack, got shape {frame_stack.shape}")

    num_frames = frame_stack.shape[0]
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")

    if end_frame is None:
        end_frame = num_frames

    if end_frame < start_frame:
        raise ValueError("end_frame must be >= start_frame")

    if end_frame > num_frames:
        raise ValueError(f"end_frame {end_frame} exceeds available frames {num_frames}")

    frames_slice = frame_stack[start_frame:end_frame]
    if len(roi_mask_log) == num_frames:
        masks_slice = roi_mask_log[start_frame:end_frame]
    else:
        masks_slice = roi_mask_log

    if len(frames_slice) != len(masks_slice):
        raise ValueError(
            f"Mismatch between frames ({len(frames_slice)}) and masks ({len(masks_slice)}) in selected range"
        )

    if not len(frames_slice):
        return

    first_mask = next(iter(masks_slice[0].values()))
    height, width = first_mask.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame, masks in zip(frames_slice, masks_slice):
        vmin, vmax = np.percentile(frame, [1, 99])
        frame_clipped = np.clip(frame, vmin, vmax)
        frame_norm = cv2.normalize(frame_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        combined_mask = np.zeros(frame_norm.shape, dtype=np.uint8)
        for mask in masks.values():
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        frame_bgr = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_bgr, contours, -1, (0, 255, 255), 1)
        out.write(frame_bgr)

    out.release()
    print(f"Saved ROI tracked video (from npy frames) to: {output_path}")


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

    # Check if path_gray_frames is a .npy file (precomputed numeric frame stack from ASC)
    gray_frames_path_is_npy = os.path.isfile(configs.path_gray_frames) and configs.path_gray_frames.endswith('.npy')
    npy_frames_full = None
    npy_frames_slice = None
    if gray_frames_path_is_npy:
        print(f"[ROI Pipeline] Loading precomputed numeric frame stack from: {configs.path_gray_frames}")
        npy_frames_full = np.load(configs.path_gray_frames, allow_pickle=False)
        if npy_frames_full.ndim != 3:
            raise ValueError(f"Expected 3D frame stack from {configs.path_gray_frames}, got shape {npy_frames_full.shape}")
        num_frames = npy_frames_full.shape[0]
        if limit_to_timespan:
            if end_frame is None:
                end_frame = num_frames
            npy_frames_slice = npy_frames_full[start_frame:end_frame]
        else:
            npy_frames_slice = npy_frames_full
        del npy_frames_full
        print(f"[ROI Pipeline] Using frames {start_frame or 0} to {end_frame or num_frames} from NPY stack (shape: {npy_frames_slice.shape})")

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
            # Directory to persist the manual-selection overlay images (like the
            # other saved plots). Provided by the batch runner via env; falls back
            # to the run directory.
            manual_overlay_dir = (
                os.environ.get("MANUAL_ROI_SELECTION_SAVE_DIR")
                or os.path.dirname(configs.path_video_tracked_face)
                or configs.path_gray_frames
            )

            def _overlay_path_for(name: str):
                return (
                    os.path.join(manual_overlay_dir, f"manual_roi_selection_{name}.png")
                    if manual_overlay_dir else None
                )

            roi_mask_logs = []
            if gray_frames_path_is_npy:
                # Draw on the first frame of the (timespan-sliced) numeric stack.
                # The same mask is reused for every frame of the slice, matching
                # the length expected by apply_roi_masks_to_npy_frames.
                for i, roi_name in enumerate(configs.manual_roi_names):
                    roi_mask_logs.append(create_manual_roi_mask_log_from_npy_frames(
                        frame_stack=npy_frames_slice,
                        roi_name=roi_name,
                        display_title=f"Draw ROI '{roi_name}' (press ENTER to confirm)",
                        mode=configs.manual_roi_modes[i] if i < len(configs.manual_roi_modes) else configs.manual_roi_modes[0],
                        frame_index=0,
                        mask_load_path=configs.manual_roi_mask_load_paths[i] if i < len(configs.manual_roi_mask_load_paths) else None,
                        mask_save_path=configs.manual_roi_mask_save_paths[i] if i < len(configs.manual_roi_mask_save_paths) else None,
                        use_colormap=configs.manual_roi_selection_colormap,
                        selection_overlay_save_path=_overlay_path_for(roi_name),
                    ))
            else:
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
                            selection_overlay_save_path=_overlay_path_for(roi_name),
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
                            selection_overlay_save_path=_overlay_path_for(roi_name),
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
            # Determine flip flag from environment: batch runners set FLIP_UPSIDE_DOWN
            flip_env = os.environ.get("FLIP_UPSIDE_DOWN", "0").strip().lower()
            flip_flag = flip_env in ("1", "true", "yes")
            block_size = int(os.environ.get("TRACK_FACE_BLOCK_SIZE", "21"))
            adapt_center = float(os.environ.get("TRACK_FACE_ADAPT_CENTER", "0.1"))
            adapt_center_horizontal = float(os.environ.get("TRACK_FACE_ADAPT_CENTER_HORIZONTAL", "0.0"))
            # Horizontal-only widening of the tracker ROI (e.g. wider for the lepton
            # camera so the whole face is captured); 1.0 keeps the square ROI.
            width_scale = float(os.environ.get("TRACK_FACE_WIDTH_SCALE", "1.0"))
            use_rescaling_env = os.environ.get("TRACK_FACE_USE_RESCALING", "1").strip().lower()
            use_rescaling = use_rescaling_env in ("1", "true", "yes")

            if roi_detection_method == "tracker":
                roi_pixel_log = track_face(
                    in_path=configs.path_video,
                    out_path=configs.path_video_tracked_face,
                    fps=configs.fps,
                    block_size=block_size,
                    adapt_center=adapt_center,
                    adapt_center_horizontal=adapt_center_horizontal,
                    width_scale=width_scale,
                    use_rescaling=use_rescaling,
                    flip_upside_down=flip_flag,
                    start_frame=start_frame if limit_to_timespan else 0,
                    end_frame=end_frame if limit_to_timespan else None,
                    show_display=False
                )
            elif roi_detection_method == "dlib":
                print("Use Dlib-based landmark detection for ROI definition (instead of Facemesh).")
                roi_pixel_log = apply_dlib_to_thermal_video(
                    video_path=configs.path_video,
                    path_save_video_with_landmarks=configs.path_video_facemesh,
                    upsample=0,
                    resize_frame=False,
                    fps=configs.fps,
                    show_display=False,
                    start_frame=start_frame if limit_to_timespan else 0,
                    end_frame=end_frame if limit_to_timespan else None,
                    flip_upside_down=flip_flag,
                )
            else: # Standard method is facemesh-based
                # Build ROI landmarks to pass to facemesh: select the requested
                # ROIs (configs.used_rois) from the shared roi_landmarks_options
                # file (merged with any inline config definitions).
                if getattr(configs, 'used_rois', None):
                    roi_options = get_roi_landmarks_options(configs)
                    roi_landmarks_selected = select_roi_landmarks(configs.used_rois, roi_options)
                    if not roi_landmarks_selected:
                        raise ValueError(
                            "No valid ROIs selected from 'used_rois'. Ensure 'used_rois' names exist in 'roi_landmarks_options'."
                        )
                else:
                    raise ValueError(
                        "No ROI definitions found in config. Please provide 'used_rois' and a 'roi_landmarks_options' file/mapping."
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
                    flip_upside_down=flip_flag,
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
            # Handle .npy precomputed frame stack
            if gray_frames_path_is_npy:
                print(f"[ROI Pipeline] Extracting ROI data from NPY frame stack...")
                # Pass roi_landmarks to enable forehead upward extension
                roi_landmarks_for_extraction = None
                if not configs.manual_roi_enabled and getattr(configs, 'used_rois', None):
                    roi_options = get_roi_landmarks_options(configs)
                    roi_landmarks_for_extraction = select_roi_landmarks(
                        configs.used_rois, roi_options, strict=False
                    )
                    print(f"[ROI Pipeline] Built roi_landmarks_for_extraction: {list(roi_landmarks_for_extraction.keys())}")
                else:
                    print(f"[ROI Pipeline] Skipping roi_landmarks (manual_roi_enabled={configs.manual_roi_enabled}, used_rois={getattr(configs, 'used_rois', None)})")
                
                roi_pixel_data = apply_roi_masks_to_npy_frames(
                    frame_stack=npy_frames_slice,
                    roi_mask_log=roi_pixel_log,
                    roi_landmarks=roi_landmarks_for_extraction,
                )
            else:
                # Handle directory of ASC or gray files
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
                    # Build roi_landmarks for gray frame extraction to enable forehead upward extension
                    roi_landmarks_for_gray = None
                    if not configs.manual_roi_enabled and getattr(configs, 'used_rois', None):
                        roi_options = get_roi_landmarks_options(configs)
                        roi_landmarks_for_gray = select_roi_landmarks(
                            configs.used_rois, roi_options, strict=False
                        )
                        print(f"[ROI Pipeline] Built roi_landmarks_for_gray: {list(roi_landmarks_for_gray.keys())}")
                    else:
                        print(f"[ROI Pipeline] Skipping roi_landmarks for gray (manual_roi_enabled={configs.manual_roi_enabled}, used_rois={getattr(configs, 'used_rois', None)})")
                    
                    roi_pixel_data = apply_roi_masks_to_gray_frames(
                        gray_dir=configs.path_gray_frames,
                        roi_mask_log=roi_pixel_log,
                        output_npz=None,
                        start_frame=start_frame if limit_to_timespan else 0,
                        end_frame=end_frame if limit_to_timespan else None,
                        roi_landmarks=roi_landmarks_for_gray,
                    )
            os.makedirs(os.path.dirname(configs.path_roi_pixel_data), exist_ok=True)
            np.save(configs.path_roi_pixel_data, roi_pixel_data, allow_pickle=True)
        else:
            roi_pixel_data = np.load(configs.path_roi_pixel_data, allow_pickle=True)

    # Update roi_pixel_log with extended masks if forehead extension was applied
    # This ensures the video and PNG visualization show the extended ROI
    if (not extract_from_video_instead_of_gray_files and roi_pixel_data and 
        isinstance(roi_pixel_data, list) and len(roi_pixel_data) > 0):
        # Check if forehead extension was applied (roi_landmarks was passed)
        if (not gray_frames_path_is_npy and getattr(configs, 'used_rois', None) and 
            any(is_forehead_roi(rn) for rn in configs.used_rois)):
            print("[ROI Pipeline] Reconstructing masks from extended forehead coordinates...")
            # Reconstruct extended masks from roi_pixel_data coordinates
            for frame_idx, frame_data in enumerate(roi_pixel_data):
                if frame_idx < len(roi_pixel_log):
                    for roi_name in list(roi_pixel_log[frame_idx].keys()):
                        if is_forehead_roi(roi_name) and roi_name in frame_data:
                            if "coords" in frame_data[roi_name]:
                                coords = frame_data[roi_name]["coords"]
                                if coords:  # Only reconstruct if coords are not empty
                                    height, width = roi_pixel_log[frame_idx][roi_name].shape
                                    new_mask = np.zeros((height, width), dtype=np.uint8)
                                    for x, y in coords:
                                        if 0 <= y < height and 0 <= x < width:
                                            new_mask[y, x] = 255
                                    roi_pixel_log[frame_idx][roi_name] = new_mask
        elif gray_frames_path_is_npy and getattr(configs, 'used_rois', None) and any(is_forehead_roi(rn) for rn in configs.used_rois):
            print("[ROI Pipeline] Reconstructing masks from extended forehead coordinates (NPY)...")
            # Same logic for NPY frames
            for frame_idx, frame_data in enumerate(roi_pixel_data):
                if frame_idx < len(roi_pixel_log):
                    for roi_name in list(roi_pixel_log[frame_idx].keys()):
                        if is_forehead_roi(roi_name) and roi_name in frame_data:
                            if "coords" in frame_data[roi_name]:
                                coords = frame_data[roi_name]["coords"]
                                if coords:
                                    height, width = roi_pixel_log[frame_idx][roi_name].shape
                                    new_mask = np.zeros((height, width), dtype=np.uint8)
                                    for x, y in coords:
                                        if 0 <= y < height and 0 <= x < width:
                                            new_mask[y, x] = 255
                                    roi_pixel_log[frame_idx][roi_name] = new_mask

    roi_pixel_data_for_return = roi_pixel_data

    if configs.timespan_signal_to_look_at is not None:
        # Save ROI masked video for the selected timespan for verification
        start_time, end_time = configs.timespan_signal_to_look_at
        start_frame = int(start_time * configs.fps)
        end_frame = int(end_time * configs.fps)
        os.makedirs(os.path.dirname(configs.path_video_tracked_face), exist_ok=True)
        if gray_frames_path_is_npy:
            write_roi_tracked_video_from_npy_frames(
                frame_stack=npy_frames_slice,
                roi_mask_log=roi_pixel_log,
                output_path=configs.path_video_tracked_face,
                fps=configs.fps,
                start_frame=0,
                end_frame=None,
            )
        else:
            write_roi_tracked_video_from_gray_frames(
                gray_dir=configs.path_gray_frames,
                roi_mask_log=roi_pixel_log,
                output_path=configs.path_video_tracked_face,
                fps=configs.fps,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        roi_pixel_data_for_return = roi_pixel_data


        # Save a snapshot of the fifth frame (offset 4) AFTER the timespan start.
        # `roi_pixel_log` (and npy_frames_slice) are sliced to the timespan, so the
        # mask is indexed by the local offset; the underlying full gray/asc/video
        # frame must be read at the GLOBAL index start_frame + offset so the overlaid
        # mask matches the displayed frame.
        desired_offset = 4  # 0-based index -> 4 is the fifth frame after the start
        # Local index into the (timespan-sliced) roi_pixel_log / npy_frames_slice.
        roi_log_index = desired_offset
        if roi_log_index < 0:
            roi_log_index = 0
        if roi_log_index >= len(roi_pixel_log):
            roi_log_index = len(roi_pixel_log) - 1
        # Global frame index into the full gray/asc files and the original video.
        source_frame_index = (start_frame or 0) + roi_log_index

        roi_masks = roi_pixel_log[roi_log_index]
        output_dir = os.path.dirname(configs.path_video_tracked_face) or configs.path_gray_frames
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"roi_frame_{source_frame_index:06d}.png")

        # If extracting from video, read the selected frame of the video for visualization, else read the corresponding gray frame
        if extract_from_video_instead_of_gray_files:
            cap = cv2.VideoCapture(configs.path_video)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index)
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
            if gray_frames_path_is_npy:
                # npy_frames_slice is sliced to the timespan, so use the local offset
                # (roi_log_index), which matches roi_pixel_log indexing.
                idx_in_slice = max(0, min(roi_log_index, len(npy_frames_slice) - 1))
                frame = npy_frames_slice[idx_in_slice]
                vmin, vmax = np.percentile(frame, [1, 99])
                frame_clipped = np.clip(frame, vmin, vmax)
                display_frame = cv2.normalize(frame_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                files_in_dir = [
                    f for f in os.listdir(configs.path_gray_frames)
                    if os.path.isfile(os.path.join(configs.path_gray_frames, f))
                ]
                if not files_in_dir:
                    raise ValueError(f"No files found in {configs.path_gray_frames}")

                sample_ext = os.path.splitext(files_in_dir[0])[1].lower()
                if sample_ext == ".asc":
                    asc_paths = sorted(
                        [os.path.join(configs.path_gray_frames, f) for f in files_in_dir if f.lower().endswith(".asc")]
                    )
                    if source_frame_index >= len(asc_paths):
                        # clamp
                        frame_idx = len(asc_paths) - 1
                    else:
                        frame_idx = source_frame_index

                    frame = _parse_asc_file(asc_paths[frame_idx])
                    if configs.asc_unit == "kelvin":
                        frame = _asc_to_kelvin_x100(frame)

                    vmin, vmax = np.percentile(frame, [1, 99])
                    frame_clipped = np.clip(frame, vmin, vmax)
                    display_frame = cv2.normalize(frame_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                else:
                    # Read the corresponding gray frame for visualization
                    _, gray_files = sort_grayfiles_with_interpolations(configs.path_gray_frames)
                    if source_frame_index >= len(gray_files):
                        frame_idx = len(gray_files) - 1
                    else:
                        frame_idx = source_frame_index

                    frame = load_gray_frame(os.path.join(configs.path_gray_frames, gray_files[frame_idx]))
                    vmin, vmax = np.percentile(frame, [1, 99])
                    frame_clipped = np.clip(frame, vmin, vmax)
                    display_frame = cv2.normalize(frame_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            overlay = (
                cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
                if display_frame.ndim == 2
                else display_frame.copy()
            )
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
            print(
                f"Saved ROI snapshot (frame {source_frame_index}, offset {desired_offset} "
                f"after timespan start) to: {output_path}"
            )

    # Plot
    if configs.plot_steps and extract_from_video_instead_of_gray_files:
        plot_roi_mean_signals(roi_pixel_data=roi_pixel_data_for_return) 
    elif configs.plot_steps and not extract_from_video_instead_of_gray_files:
        plot_roi_temperature_means(roi_pixel_data=roi_pixel_data_for_return, fps=configs.fps)

    return roi_pixel_data_for_return, roi_pixel_log