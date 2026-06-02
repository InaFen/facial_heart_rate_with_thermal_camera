import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path
from typing import List, Dict
from src.approaches.new_approach.utils.utils_gray_files import sort_grayfiles_with_interpolations


def apply_facemesh_to_thermal_video(
    input_path: str,
    output_path: str,
    roi_landmarks: dict,
    fps: int,
    use_ema: bool,
    ema_alpha: float,
    show_display: bool = False,
    flip_upside_down: bool = True,
    start_frame: int = 0,
    end_frame: int = None,
) -> list[dict]:
    """
    Apply MediaPipe FaceMesh to a thermal video, draw ALL landmarks + ROIs,
    optionally display in real time, save annotated video, and collect ROI masks.
    If a frame has no detected face, a blank mask is stored to maintain alignment.
    Facemesh is trained on RGB images, so a perceptual colormap is applied to the thermal frames before processing.
    
    Parameters:
        input_path: Path to input thermal video
        output_path: Path to save annotated video (or None to skip saving)
        roi_landmarks: Dict mapping ROI names to landmark indices
        fps: Frames per second
        show_display: If True, display video during processing. Default: False (fastest)
        use_ema: If True, apply exponential moving average smoothing to landmarks. Default: True
        ema_alpha: EMA blending factor (0.0-1.0); lower = smoother but more lag. Default: 0.2
        flip_upside_down: If True, flip frames vertically for FaceMesh and flip masks/output back to match original orientation.
        start_frame: Start frame index to process (inclusive). Default: 0
        end_frame: End frame index to process (exclusive). Default: None (process to end)
    
    Returns:
        List of dictionaries containing ROI masks for each frame.
    """

    print("[IMPORTANT] Ensure that the input video is properly aligned with the original .gray frames (same order, not flipped) for correct ROI mapping later on.")
    if flip_upside_down:
        print("[Warning] Flipping frames vertically for FaceMesh; masks/output will be flipped back to match original orientation.")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open input video.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Optional video writer
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    else:
        out = None

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    roi_mask_log = []   # List of dicts storing ROI masks per frame
    smoothed_pts = None
    alpha = ema_alpha if use_ema else 1.0  # If not using EMA, alpha = 1.0 (always use current frame)

    if start_frame > 0:
        for _ in range(start_frame):
            ret, _ = cap.read()
            if not ret:
                break

    frame_idx = 0
    max_frames = None
    if end_frame is not None:
        max_frames = max(0, end_frame - start_frame)

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        thermal_original = frame.copy()

        if flip_upside_down:
            frame = cv2.flip(frame, 0)

        # Robust normalize 16-bit -> 8-bit: clip outliers then map to 0-255
        if frame.dtype != np.uint8:
            vmin, vmax = np.percentile(frame, [1, 99])
            frame_clipped = np.clip(frame, vmin, vmax)
            frame_norm = cv2.normalize(frame_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            frame_norm = frame

        # Apply perceptual colormap once; keep BGR for drawing, RGB for MediaPipe
        pseudo_bgr = cv2.applyColorMap(frame_norm, cv2.COLORMAP_TURBO)
        mesh_input = cv2.cvtColor(pseudo_bgr, cv2.COLOR_BGR2RGB) # convert to RGB for MediaPipe
        pseudo_rgb = pseudo_bgr

        results = face_mesh.process(mesh_input)

        frame_roi_masks = {}  # Store masks for all ROIs in this frame
        
        if results.multi_face_landmarks:
            face_lm = results.multi_face_landmarks[0]

            # Convert all 468 landmarks to pixel coordinates
            pts = np.array([
                [int(lm.x * width), int(lm.y * height)]
                for lm in face_lm.landmark
            ])

            pts = pts.astype(np.float32)

            if smoothed_pts is None:
                smoothed_pts = pts.copy()
            else:
                smoothed_pts = alpha * pts + (1 - alpha) * smoothed_pts

            pts_smooth = smoothed_pts.astype(np.int32)

            # Draw the ROI polygons & create ROI masks
            for roi_name, indices in roi_landmarks.items():
                roi_poly = pts_smooth[indices]  # polygon vertices

                # Draw polygon on output
                cv2.polylines(pseudo_rgb, [roi_poly], True, (255, 0, 0), 1)

                # Create ROI mask
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [roi_poly], 255)

                if flip_upside_down:
                    mask = cv2.flip(mask, 0)

                frame_roi_masks[roi_name] = mask
        else:
            # No face detected: create blank masks for each ROI
            smoothed_pts = None
            for roi_name in roi_landmarks.keys():

                blank_mask = np.zeros((height, width), dtype=np.uint8)
                if flip_upside_down:
                    blank_mask = cv2.flip(blank_mask, 0)

                frame_roi_masks[roi_name] = blank_mask

        # Append masks for this frame
        roi_mask_log.append(frame_roi_masks)

        if flip_upside_down:
            pseudo_rgb = cv2.flip(pseudo_rgb, 0)

        if show_display:
            # Show annotated output
            cv2.imshow("Thermal FaceMesh + Landmarks + ROIs", pseudo_rgb)

        if out is not None:
            out.write(pseudo_rgb)

        if show_display and cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1


    print("Collected ROI masks for", len(roi_mask_log), "frames.")
    print(f"Saved annotated video to: {output_path}" if output_path else "No annotated video saved.")
    return roi_mask_log


def apply_roi_masks_to_video(
    video_path: str,
    roi_mask_log: list,
    output_path: str,
    fps: float,
    show_display: bool = False,
    start_frame: int = 0,
    end_frame: int = None,
) -> list[dict]:
    """
    Apply ROI masks to the original video AND record:
      - all pixel values inside each ROI (per frame)
      - all pixel coordinates inside each ROI (per frame)

    Parameters:
        video_path: Path to input video
        roi_mask_log: List of per-frame ROI masks
        output_path: Path to save masked video
        show_display: If True, display video during processing. Default: False (fastest)

    Returns:
        roi_pixel_data : list of dicts, one entry per frame:
            {
                "roi_name": {
                    "coords": [(x1,y1), (x2,y2), ...],
                    "values": [v1, v2, ...],
                    "mean_value": float
                },
                ...
            }
    }
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if start_frame > 0:
        for _ in range(start_frame):
            ret, _ = cap.read()
            if not ret:
                break

    frame_idx = 0
    roi_pixel_data = []  

    max_frames = len(roi_mask_log)
    if end_frame is not None:
        max_frames = min(max_frames, max(0, end_frame - start_frame))

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(roi_mask_log):
            break

        # Convert 16-bit to 8-bit for display, but keep original frame for pixel extraction
        if frame.dtype != np.uint8:
            frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            frame_norm = frame.copy()

        roi_frame_data = {}   # stores data for all ROIs this frame
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        # Process each ROI
        for roi_name, mask in roi_mask_log[frame_idx].items():
            combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Extract pixel coordinates
            ys, xs = np.where(mask == 255)
            coords = list(zip(xs, ys))

            # Extract pixel values from original (grayscale) thermal frame
            # TODO it computes the "temperature" from the RGB to Grey frame, so one value is the weighted sum of R,G,B channels so not actual the temperature
            # TODO does this cause problems later on? Should not be important that not temperature but the greyscale might be a rounded value?
            if frame.ndim == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame.copy()

            values = gray_frame[ys, xs].tolist()

            roi_frame_data[roi_name] = {
                "coords": coords,
                "values": values,
                "mean_value": float(np.mean(values)) if len(values) > 0 else float('nan')
            }

        roi_pixel_data.append(roi_frame_data)

        # Apply combined mask for output video
        masked_frame = cv2.bitwise_and(frame_norm, frame_norm, mask=combined_mask)

        # Convert grayscale → BGR if needed
        if masked_frame.ndim == 2:
            masked_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_GRAY2BGR)
        else:
            masked_bgr = masked_frame

        out.write(masked_bgr)

        if show_display:
            cv2.imshow("ROI Masked Video", masked_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    out.release()
    if show_display:
        cv2.destroyAllWindows()
    print(f"Saved masked video to: {output_path}")

    return roi_pixel_data


def plot_roi_mean_signals(roi_pixel_data: list, fps: float = None, plot_in_secs: bool = False) -> None:
    """
    Plot the mean temperature signal for each ROI across all frames.

    Parameters:
        roi_pixel_data (list):
            Output of `apply_roi_masks_to_video()`, a list of dicts:
            roi_pixel_data[frame_idx][roi_name]["mean_value"]
    """

    if len(roi_pixel_data) == 0:
        print("Error: roi_pixel_data is empty.")
        return

    # Get list of ROI names from first frame
    roi_names = list(roi_pixel_data[0].keys())

    # Prepare time axis
    num_frames = len(roi_pixel_data)
    if fps is not None and plot_in_secs:
        x_axis_values = [frame / fps for frame in range(num_frames)]
    else:
        x_axis_values = list(range(num_frames))

    # Prepare data container
    roi_means = {roi: [] for roi in roi_names}

    # Extract mean values per frame
    for frame_data in roi_pixel_data:
        for roi in roi_names:
            val = frame_data[roi]["mean_value"]
            roi_means[roi].append(val)

    # Plot
    plt.figure(figsize=(12, 6))

    for roi in roi_names:
        plt.plot(x_axis_values, roi_means[roi], label=roi)

    if fps is not None and plot_in_secs:
        plt.xlabel("Time (seconds)", fontsize=12)
    else:
        plt.xlabel("Frame Index", fontsize=12)
    plt.ylabel("Mean Temperature", fontsize=12)
    plt.title("ROI Mean Temperature Over Time", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def load_gray_frame(filepath: str, width: int = 160, height: int = 122, drop_telemetry: bool = True) -> np.ndarray:
    """Load a .gray frame (Y16, little-endian) and optionally strip telemetry lines.
    
    Parameters:
        filepath: Path to the .gray file
        width: Width of the frame
        height: Height of the frame (default: 122: expects additional 2 lines of telemetry, so actual image data is 120)
        drop_telemetry: Whether to drop telemetry lines (default: True, drops last 2 lines if height > 120)

    Returns:
        np.ndarray: The loaded frame as a NumPy array
    """
    raw = np.fromfile(filepath, dtype=np.uint16)
    if raw.size != width * height:
        raise ValueError(f"Unexpected frame size in {filepath}: got {raw.size}, expected {width*height}")
    frame = raw.reshape(height, width)
    if drop_telemetry and height > 120:
        frame = frame[:-2]  # remove last 2 lines
    return frame


def _parse_asc_file(filepath: str) -> np.ndarray:
    text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    data_started = False
    data_rows: List[List[float]] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            data_started = line.lower() == "[data]"
            continue
        if not data_started:
            continue

        normalized = line.replace(",", ".")
        parts = re.findall(r"[-+]?\d*\.?\d+", normalized)
        if parts:
            data_rows.append([float(p) for p in parts])

    if not data_rows:
        raise ValueError(f"No data rows found in {filepath}")

    return np.array(data_rows, dtype=np.float32)


def _asc_to_kelvin_x100(data: np.ndarray, unit: str) -> np.ndarray:
    unit = unit.lower()
    if unit == "celsius":
        values = (data + 273.15) * 100.0
    elif unit == "kelvin":
        values = data * 100.0
    elif unit == "kelvin100":
        values = data
    elif unit == "raw":
        values = data
    else:
        raise ValueError(f"Unsupported unit: {unit}")
    return values.astype(np.float64)


def _resize_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    if mask.shape == (target_h, target_w):
        return mask
    return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def split_manual_roi_into_subrois(
    roi_pixel_log: List[Dict[str, np.ndarray]],
    roi_name: str,
    grid_rows: int,
    grid_cols: int,
) -> tuple[List[Dict[str, np.ndarray]], List[str]]:
    """Split a single manual ROI into a grid of sub-ROIs based on the bounding box of the original ROI mask.
    Parameters:
        roi_pixel_log: List of dictionaries containing ROI masks for each frame.
        roi_name: The name of the ROI to split (should match the key in the dictionaries).
        grid_rows: Number of rows in the sub-ROI grid.
        grid_cols: Number of columns in the sub-ROI grid.
        Returns:
            Tuple of (new_log, subroi_names) where:
            - new_log: List of dictionaries containing sub-ROI masks for each frame.
            - subroi_names: List of names for the sub-ROIs.
    """
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


def create_manual_roi_mask_log_from_gray_frames(
    gray_dir: str,
    roi_name: str = "manual_roi",
    display_title: str = "Draw ROI (press ENTER to confirm)",
    mode: str = "rectangle",
    frame_index: int = 0,
    mask_load_path: str = None,
    mask_save_path: str = None,
    use_colormap: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """Create a fixed ROI mask for all frames by manually drawing on the first gray frame.

    Parameters:
        gray_dir: Directory containing .gray frames
        roi_name: Name of the ROI (used as key in output log)
        display_title: Window title for ROI drawing interface
        mode: "rectangle", "polygon", "point", "point_surrounding" (point with 8 surrounding pixels) for ROI drawing method
        frame_index: Index of the frame to use for ROI drawing (default: 0)
        mask_load_path: Optional path to load an existing ROI mask (skips drawing if provided)
        mask_save_path: Optional path to save the drawn ROI mask for future reuse
        use_colormap: If True, apply a colormap to the frame for easier visualization during ROI drawing

    Returns:
        roi_mask_log: list of per-frame ROI masks (same mask reused for all frames)
    """
    # Sort gray files
    _, gray_files = sort_grayfiles_with_interpolations(gray_dir)
    if not gray_files:
        raise ValueError(f"No .gray files found in {gray_dir}")

    # Load mask if path provided
    if mask_load_path:
        loaded_mask = np.load(mask_load_path)
        if loaded_mask.ndim != 2:
            raise ValueError(f"Loaded mask must be 2D, got shape {loaded_mask.shape}")
        # Load first frame to verify mask dimensions match
        first_frame_check = load_gray_frame(os.path.join(gray_dir, gray_files[0]))
        if loaded_mask.shape != first_frame_check.shape:
            raise ValueError(f"Loaded mask shape {loaded_mask.shape} does not match frame shape {first_frame_check.shape}")
        mask = loaded_mask.astype(np.uint8)
        roi_mask_log = [{roi_name: mask} for _ in range(len(gray_files))]
        return roi_mask_log

    # Load first frame for ROI drawing
    frame_index = max(0, min(frame_index, len(gray_files) - 1))
    first_frame_path = os.path.join(gray_dir, gray_files[frame_index])
    first_frame = load_gray_frame(first_frame_path)

    vmin, vmax = np.percentile(first_frame, [1, 99])
    frame_clipped = np.clip(first_frame, vmin, vmax)
    frame_norm = cv2.normalize(frame_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if use_colormap:
        frame_display = cv2.applyColorMap(frame_norm, cv2.COLORMAP_TURBO)
    else:
        frame_display = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)

    # Get frame dimensions
    height, width = first_frame.shape
    mask = np.zeros(first_frame.shape, dtype=np.uint8)

    # User draws ROI on the first frame (options: rectangle, polygon, point, point_surrounding - point with 8 surrounding pixels)
    if mode == "rectangle":
        cv2.namedWindow(display_title, cv2.WINDOW_NORMAL)
        roi_rect = cv2.selectROI(display_title, frame_display, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(display_title)

        x, y, w, h = roi_rect
        if w <= 0 or h <= 0:
            raise ValueError("No ROI selected. Please draw a rectangle ROI and press ENTER.")

        # Draw filled rectangle directly - no coordinate transformation needed
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)
    elif mode == "polygon":
        points = []
        display = frame_display.copy()

        #instructions = "Left click: add point | Right click: undo | ENTER: finish | c: clear"

        def draw_overlay(img, pts):
            canvas = img.copy()
            if pts:
                cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (0, 255, 255), 1)
                for p in pts:
                    cv2.circle(canvas, p, 2, (0, 255, 255), -1)
            #cv2.putText(canvas, instructions, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 255), 1)
            return canvas

        def on_mouse(event, x, y, flags, param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and points:
                points.pop()

        cv2.namedWindow(display_title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(display_title, on_mouse)

        while True:
            canvas = draw_overlay(display, points)
            cv2.imshow(display_title, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == 13:  # Enter
                break
            if key == ord('c'):
                points = []

        cv2.destroyWindow(display_title)

        if len(points) < 3:
            raise ValueError("Polygon ROI needs at least 3 points. Please try again.")

        # Draw polygon directly - no coordinate transformation needed
        poly = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
    elif mode == "point":
        point = []
        display = frame_display.copy()

        def on_mouse(event, x, y, flags, param):
            nonlocal point
            if event == cv2.EVENT_LBUTTONDOWN:
                point = [(x, y)]

        cv2.namedWindow(display_title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(display_title, on_mouse)

        while True:
            canvas = display.copy()
            if point:
                cv2.circle(canvas, point[0], 2, (0, 255, 255), -1)
            cv2.imshow(display_title, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == 13:  # Enter
                break
            if key == ord('c'):
                point = []

        cv2.destroyWindow(display_title)

        if len(point) != 1:
            raise ValueError("Point ROI needs a single click. Please try again.")

        x, y = point[0]
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Point ROI is out of bounds. Please try again.")

        mask[y, x] = 255
    elif mode == "point_surrounding":
        point = []
        display = frame_display.copy()

        def on_mouse(event, x, y, flags, param):
            nonlocal point
            if event == cv2.EVENT_LBUTTONDOWN:
                point = [(x, y)]

        cv2.namedWindow(display_title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(display_title, on_mouse)

        while True:
            canvas = display.copy()
            if point:
                cv2.circle(canvas, point[0], 2, (0, 255, 255), -1)
            cv2.imshow(display_title, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == 13:  # Enter
                break
            if key == ord('c'):
                point = []

        cv2.destroyWindow(display_title)

        if len(point) != 1:
            raise ValueError("Point ROI needs a single click. Please try again.")

        x, y = point[0]
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Point ROI is out of bounds. Please try again.")

        offsets = [
            (0, 0, "center"),
            (-1, 0, "left"),
            (1, 0, "right"),
            (0, -1, "up"),
            (0, 1, "down"),
            (-1, -1, "up_left"),
            (1, -1, "up_right"),
            (-1, 1, "down_left"),
            (1, 1, "down_right"),
        ]
        masks = {}
        for dx, dy, suffix in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                sub_mask = np.zeros(first_frame.shape, dtype=np.uint8)
                sub_mask[ny, nx] = 255
                masks[f"{roi_name}_{suffix}"] = sub_mask
        if not masks:
            raise ValueError("No valid surrounding pixels found for the selected point.")
        mask = masks
    else:
        raise ValueError(f"Unknown manual ROI mode: {mode}")

    # Save mask if path provided
    if mask_save_path and isinstance(mask, np.ndarray):
        save_dir = os.path.dirname(mask_save_path)
        if save_dir:  # Only create directory if path contains one
            os.makedirs(save_dir, exist_ok=True)
        np.save(mask_save_path, mask)

    if mask_save_path and isinstance(mask, dict):
        save_dir = os.path.dirname(mask_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        base, ext = os.path.splitext(mask_save_path)
        ext = ext if ext else ".npy"
        for name, sub_mask in mask.items():
            np.save(f"{base}_{name}{ext}", sub_mask)

    if isinstance(mask, dict):
        roi_mask_log = [mask for _ in range(len(gray_files))]
    else:
        roi_mask_log = [{roi_name: mask} for _ in range(len(gray_files))]
    return roi_mask_log


def create_manual_roi_mask_log_from_asc_frames(
    asc_dir: str,
    roi_name: str = "manual_roi",
    display_title: str = "Draw ROI (press ENTER to confirm)",
    mode: str = "rectangle",
    frame_index: int = 0,
    mask_load_path: str = None,
    mask_save_path: str = None,
    use_colormap: bool = True,
    unit: str = "celsius",
) -> List[Dict[str, np.ndarray]]:
    """Create a fixed ROI mask for all .asc frames by manually drawing on the first ASC frame.

    Parameters:
        asc_dir: Directory containing .asc frames
        roi_name: Name of the ROI (used as key in output log)
        display_title: Window title for ROI drawing interface
        mode: "rectangle", "polygon", "point", "point_surrounding"
        frame_index: Index of the frame to use for ROI drawing (default: 0)
        mask_load_path: Optional path to load an existing ROI mask
        mask_save_path: Optional path to save the drawn ROI mask for future reuse
        use_colormap: If True, apply a colormap to the frame for easier visualization

    Returns:
        roi_mask_log: list of per-frame ROI masks (same mask reused for all frames)
    """
    asc_files = sorted([f for f in os.listdir(asc_dir) if f.lower().endswith(".asc")])
    if not asc_files:
        raise ValueError(f"No .asc files found in {asc_dir}")

    # Load mask if path provided
    if mask_load_path:
        loaded_mask = np.load(mask_load_path)
        if loaded_mask.ndim != 2:
            raise ValueError(f"Loaded mask must be 2D, got shape {loaded_mask.shape}")
        first_frame_check = _parse_asc_file(os.path.join(asc_dir, asc_files[0]))
        if loaded_mask.shape != first_frame_check.shape:
            raise ValueError(f"Loaded mask shape {loaded_mask.shape} does not match frame shape {first_frame_check.shape}")
        mask = loaded_mask.astype(np.uint8)
        roi_mask_log = [{roi_name: mask} for _ in range(len(asc_files))]
        return roi_mask_log

    # Load first frame for ROI drawing
    frame_index = max(0, min(frame_index, len(asc_files) - 1))
    first_frame_path = os.path.join(asc_dir, asc_files[frame_index])
    first_frame = _parse_asc_file(first_frame_path)
    first_frame = _asc_to_kelvin_x100(first_frame, unit)

    vmin, vmax = np.percentile(first_frame, [1, 99])
    frame_clipped = np.clip(first_frame, vmin, vmax)
    frame_norm = cv2.normalize(frame_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if use_colormap:
        frame_display = cv2.applyColorMap(frame_norm, cv2.COLORMAP_TURBO)
    else:
        frame_display = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)

    height, width = first_frame.shape
    mask = np.zeros(first_frame.shape, dtype=np.uint8)

    if mode == "rectangle":
        cv2.namedWindow(display_title, cv2.WINDOW_NORMAL)
        roi_rect = cv2.selectROI(display_title, frame_display, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(display_title)

        x, y, w, h = roi_rect
        if w <= 0 or h <= 0:
            raise ValueError("No ROI selected. Please draw a rectangle ROI and press ENTER.")

        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)
    elif mode == "polygon":
        points = []
        display = frame_display.copy()

        def draw_overlay(img, pts):
            canvas = img.copy()
            if pts:
                cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (0, 255, 255), 1)
                for p in pts:
                    cv2.circle(canvas, p, 2, (0, 255, 255), -1)
            return canvas

        def on_mouse(event, x, y, flags, param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and points:
                points.pop()

        cv2.namedWindow(display_title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(display_title, on_mouse)

        while True:
            canvas = draw_overlay(display, points)
            cv2.imshow(display_title, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == 13:  # Enter
                break
            if key == ord('c'):
                points = []

        cv2.destroyWindow(display_title)

        if len(points) < 3:
            raise ValueError("Polygon ROI needs at least 3 points. Please try again.")

        poly = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
    elif mode == "point":
        point = []
        display = frame_display.copy()

        def on_mouse(event, x, y, flags, param):
            nonlocal point
            if event == cv2.EVENT_LBUTTONDOWN:
                point = [(x, y)]

        cv2.namedWindow(display_title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(display_title, on_mouse)

        while True:
            canvas = display.copy()
            if point:
                cv2.circle(canvas, point[0], 2, (0, 255, 255), -1)
            cv2.imshow(display_title, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == 13:  # Enter
                break
            if key == ord('c'):
                point = []

        cv2.destroyWindow(display_title)

        if len(point) != 1:
            raise ValueError("Point ROI needs a single click. Please try again.")

        x, y = point[0]
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Point ROI is out of bounds. Please try again.")

        mask[y, x] = 255
    elif mode == "point_surrounding":
        point = []
        display = frame_display.copy()

        def on_mouse(event, x, y, flags, param):
            nonlocal point
            if event == cv2.EVENT_LBUTTONDOWN:
                point = [(x, y)]

        cv2.namedWindow(display_title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(display_title, on_mouse)

        while True:
            canvas = display.copy()
            if point:
                cv2.circle(canvas, point[0], 2, (0, 255, 255), -1)
            cv2.imshow(display_title, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == 13:  # Enter
                break
            if key == ord('c'):
                point = []

        cv2.destroyWindow(display_title)

        if len(point) != 1:
            raise ValueError("Point ROI needs a single click. Please try again.")

        x, y = point[0]
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Point ROI is out of bounds. Please try again.")

        offsets = [
            (0, 0, "center"),
            (-1, 0, "left"),
            (1, 0, "right"),
            (0, -1, "up"),
            (0, 1, "down"),
            (-1, -1, "up_left"),
            (1, -1, "up_right"),
            (-1, 1, "down_left"),
            (1, 1, "down_right"),
        ]
        masks = {}
        for dx, dy, suffix in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                sub_mask = np.zeros(first_frame.shape, dtype=np.uint8)
                sub_mask[ny, nx] = 255
                masks[f"{roi_name}_{suffix}"] = sub_mask
        if not masks:
            raise ValueError("No valid surrounding pixels found for the selected point.")
        mask = masks
    else:
        raise ValueError(f"Unknown manual ROI mode: {mode}")

    if mask_save_path and isinstance(mask, np.ndarray):
        save_dir = os.path.dirname(mask_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        np.save(mask_save_path, mask)

    if mask_save_path and isinstance(mask, dict):
        save_dir = os.path.dirname(mask_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        base, ext = os.path.splitext(mask_save_path)
        ext = ext if ext else ".npy"
        for name, sub_mask in mask.items():
            np.save(f"{base}_{name}{ext}", sub_mask)

    if isinstance(mask, dict):
        roi_mask_log = [mask for _ in range(len(asc_files))]
    else:
        roi_mask_log = [{roi_name: mask} for _ in range(len(asc_files))]
    return roi_mask_log


def apply_roi_masks_to_gray_frames(
    gray_dir: str,
    roi_mask_log: List[Dict[str, np.ndarray]],
    output_npz: str = None,
    start_frame: int = 0,
    end_frame: int = None,
    roi_landmarks: Dict[str, List[int]] = None,
) -> List[Dict[str, Dict[str, object]]]:
    """Apply ROI masks (from video detection) to original .gray frames to extract temperatures.

    Parameters:
        gray_dir: directory containing .gray frames
        roi_mask_log: list of per-frame ROI masks (from apply_facemesh_to_thermal_video)
        output_npz: optional path to save extracted data as npz
        start_frame: Start frame index to process (inclusive). Default: 0
        end_frame: End frame index to process (exclusive). Default: None (process to end)
        roi_landmarks: optional dict mapping ROI name to list of landmark indices (for forehead extension)

    Returns:
        roi_pixel_data: list (per frame) of dicts {roi_name: {"coords", "values", "mean_value"}}
    """
    # Import extension function here to avoid circular imports
    from src.approaches.new_approach.pipelines.get_roi_data import extend_forehead_roi_upward
    
    _, gray_files = sort_grayfiles_with_interpolations(gray_dir)
    if not gray_files:
        raise ValueError(f"No .gray files found in {gray_dir}")

    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")

    if end_frame is None:
        end_frame = len(gray_files)

    if end_frame < start_frame:
        raise ValueError("end_frame must be >= start_frame")

    if end_frame > len(gray_files):
        raise ValueError(f"end_frame {end_frame} exceeds available frames {len(gray_files)}")

    gray_files_slice = gray_files[start_frame:end_frame]

    if len(roi_mask_log) == len(gray_files):
        roi_masks_slice = roi_mask_log[start_frame:end_frame]
    else:
        roi_masks_slice = roi_mask_log

    if len(gray_files_slice) < len(roi_masks_slice):
        raise ValueError(f"Fewer .gray frames ({len(gray_files_slice)}) than masks ({len(roi_masks_slice)}) in selected range")
    elif len(gray_files_slice) > len(roi_masks_slice):
        raise ValueError(f"More .gray frames ({len(gray_files_slice)}) than masks ({len(roi_masks_slice)}) in selected range")

    roi_pixel_data: List[Dict[str, Dict[str, object]]] = []

    for idx, (fname, masks) in enumerate(zip(gray_files_slice, roi_masks_slice)):
        frame_path = os.path.join(gray_dir, fname)
        frame = load_gray_frame(frame_path)

        height, width = frame.shape
        frame_data: Dict[str, Dict[str, object]] = {}

        for roi_name, mask in masks.items():
            if mask.shape != frame.shape:
                raise ValueError(f"Mask shape {mask.shape} does not match frame shape {frame.shape} for ROI {roi_name}")

            # Apply forehead upward extension if applicable
            if roi_name == "forehead" and roi_landmarks and roi_name in roi_landmarks:
                mask = extend_forehead_roi_upward(
                    frame=frame,
                    mask=mask,
                    landmark_indices=roi_landmarks[roi_name],
                )

            ys, xs = np.where(mask == 255)
            coords = list(zip(xs.tolist(), ys.tolist()))
            values = frame[ys, xs].astype(np.float64)
            frame_data[roi_name] = {
                "coords": coords,
                "adc_values": values.tolist(),
                "mean_value": float(np.mean(values)) if values.size > 0 else float('nan'),
            }

        roi_pixel_data.append(frame_data)

        if (idx + 1) % 100 == 0 or idx == len(gray_files_slice) - 1:
            print(f"Processed {idx+1}/{len(gray_files_slice)} frames")

    if output_npz:
        os.makedirs(os.path.dirname(output_npz), exist_ok=True)
        np.savez(output_npz, roi_pixel_data=roi_pixel_data)
        print(f"Saved ROI pixel data to {output_npz}")

    return roi_pixel_data


def apply_roi_masks_to_asc_frames(
    asc_paths: List[str],
    roi_mask_log: List[Dict[str, np.ndarray]],
    unit: str = "celsius",
) -> List[Dict[str, Dict[str, object]]]:
    """Apply ROI masks to .asc frames to extract temperatures.

    Parameters:
        asc_paths: list of .asc file paths, one per frame
        roi_mask_log: list of per-frame ROI masks
        unit: unit of values in ASC (celsius|kelvin|kelvin100|raw)

    Returns:
        roi_pixel_data: list (per frame) of dicts {roi_name: {"coords", "values", "mean_value"}}
    """
    if not asc_paths:
        raise ValueError("asc_paths is empty")

    if len(asc_paths) != len(roi_mask_log):
        raise ValueError(f"Mismatch between asc files ({len(asc_paths)}) and masks ({len(roi_mask_log)})")

    roi_pixel_data: List[Dict[str, Dict[str, object]]] = []

    for idx, (path, masks) in enumerate(zip(asc_paths, roi_mask_log)):
        frame = _parse_asc_file(path)
        frame_values = _asc_to_kelvin_x100(frame, unit)

        frame_data: Dict[str, Dict[str, object]] = {}
        for roi_name, mask in masks.items():
            if mask.shape != frame_values.shape:
                mask = _resize_mask_to_shape(mask, frame_values.shape)
            if mask.shape != frame_values.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match frame shape {frame_values.shape} for ROI {roi_name}"
                )

            ys, xs = np.where(mask == 255)
            coords = list(zip(xs.tolist(), ys.tolist()))
            values = frame_values[ys, xs]
            frame_data[roi_name] = {
                "coords": coords,
                "adc_values": values.tolist(),
                "mean_value": float(np.mean(values)) if values.size > 0 else float("nan"),
            }

        roi_pixel_data.append(frame_data)

        if (idx + 1) % 100 == 0 or idx == len(asc_paths) - 1:
            print(f"Processed {idx+1}/{len(asc_paths)} frames")

    return roi_pixel_data


def write_roi_tracked_video_from_gray_frames(
    gray_dir: str,
    roi_mask_log: List[Dict[str, np.ndarray]],
    output_path: str,
    fps: float,
    start_frame: int = 0,
    end_frame: int = None,
) -> None:
    """Write a masked video from .gray frames using ROI masks.

    Parameters:
        gray_dir: directory containing .gray frames
        roi_mask_log: list of per-frame ROI masks
        output_path: path to save the masked video
        fps: frames per second
        start_frame: Start frame index to process (inclusive). Default: 0
        end_frame: End frame index to process (exclusive). Default: None (process to end)
    """
    _, gray_files = sort_grayfiles_with_interpolations(gray_dir)
    if not gray_files:
        raise ValueError(f"No .gray files found in {gray_dir}")

    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")

    if end_frame is None:
        end_frame = len(gray_files)

    if end_frame < start_frame:
        raise ValueError("end_frame must be >= start_frame")

    if end_frame > len(gray_files):
        raise ValueError(f"end_frame {end_frame} exceeds available frames {len(gray_files)}")

    gray_files_slice = gray_files[start_frame:end_frame]
    if len(roi_mask_log) == len(gray_files):
        masks_slice = roi_mask_log[start_frame:end_frame]
    else:
        masks_slice = roi_mask_log

    if len(gray_files_slice) != len(masks_slice):
        raise ValueError(
            f"Mismatch between gray frames ({len(gray_files_slice)}) and masks ({len(masks_slice)}) in selected range"
        )

    if not gray_files_slice:
        return

    first_mask = next(iter(masks_slice[0].values()))
    drop_telemetry = first_mask.shape[0] <= 120
    raw_height = first_mask.shape[0] + (2 if drop_telemetry else 0)
    raw_width = first_mask.shape[1]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (raw_width, first_mask.shape[0]))

    for fname, masks in zip(gray_files_slice, masks_slice):
        frame_path = os.path.join(gray_dir, fname)
        frame = load_gray_frame(
            frame_path,
            width=raw_width,
            height=raw_height,
            drop_telemetry=drop_telemetry,
        )

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
    print(f"Saved ROI tracked video (from gray frames) to: {output_path}")


def plot_roi_temperature_means(roi_pixel_data: List[Dict[str, Dict[str, object]]], fps: float = None) -> None:
    """Plot temperature mean values for each ROI across frames.

    Parameters:
        roi_pixel_data: output from apply_roi_masks_to_gray_frames
        fps: optional frames per second for x-axis in seconds; if None, x-axis is frame index
    """

    roi_names = list(roi_pixel_data[0].keys())
    num_frames = len(roi_pixel_data)
    t = np.arange(num_frames) if fps is None else np.arange(num_frames) / fps

    plt.figure(figsize=(12, 6))
    for roi in roi_names:
        means = [frame.get(roi, {}).get("mean_value", np.nan) for frame in roi_pixel_data]
        plt.plot(t, means, label=roi)

    plt.xlabel("Frames" if fps is None else "Time (s)")
    plt.ylabel("Mean Kelvin * 100")  # adc values are in 0.01 Kelvin
    plt.title("ROI Mean Temperature Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    gray_dir = "data/lepton/processed_lepton_data/29122025_interpolated/"
    output_npz = "data/saved_data/roi_pixel_data_from_gray.npz"

    path_pixels_log = "data/yang_et_al/roi_pixel_log.npy"
    roi_mask_log =np.load(path_pixels_log, allow_pickle=True)

    roi_pixel_data = apply_roi_masks_to_gray_frames(gray_dir=gray_dir, roi_mask_log=roi_mask_log, output_npz=output_npz)
    plot_roi_temporal_means(roi_pixel_data=roi_pixel_data, fps=8.77)