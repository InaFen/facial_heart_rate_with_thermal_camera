import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os
from typing import List, Dict
from src.approaches.new_approach.utils.utils_gray_files import sort_grayfiles_with_interpolations


def apply_facemesh_to_thermal_video(input_path: str, output_path: str, roi_landmarks: dict, fps: int, use_ema: bool, ema_alpha: float, show_display: bool = False):
    """
    Apply MediaPipe FaceMesh to a thermal video, draw ALL landmarks + ROIs,
    optionally display in real time, save annotated video, and collect ROI masks.
    If a frame has no detected face, a blank mask is stored to maintain alignment.
    
    Parameters:
        input_path: Path to input thermal video
        output_path: Path to save annotated video (or None to skip saving)
        roi_landmarks: Dict mapping ROI names to landmark indices
        fps: Frames per second
        show_display: If True, display video during processing. Default: False (fastest)
        use_ema: If True, apply exponential moving average smoothing to landmarks. Default: True
        ema_alpha: EMA blending factor (0.0-1.0); lower = smoother but more lag. Default: 0.2
    """

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        thermal_original = frame.copy()

        # Robust normalize 16-bit → 8-bit: clip outliers then map to 0-255
        if frame.dtype != np.uint8:
            vmin, vmax = np.percentile(frame, [1, 99])
            frame_clipped = np.clip(frame, vmin, vmax)
            frame_norm = cv2.normalize(frame_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            frame_norm = frame

        # Apply perceptual colormap once; keep BGR for drawing, RGB for MediaPipe
        pseudo_bgr = cv2.applyColorMap(frame_norm, cv2.COLORMAP_TURBO)
        mesh_input = cv2.cvtColor(pseudo_bgr, cv2.COLOR_BGR2RGB)
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


            pts_smooth = smoothed_pts.astype(np.int32)

            # Draw the ROI polygons & create ROI masks
            for roi_name, indices in roi_landmarks.items():
                roi_poly = pts_smooth[indices]  # polygon vertices

                # Draw polygon on output
                cv2.polylines(pseudo_rgb, [roi_poly], True, (255, 0, 0), 1)

                # Create ROI mask
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [roi_poly], 255)

                frame_roi_masks[roi_name] = mask
        else:
            # No face detected: create blank masks for each ROI
            smoothed_pts = None
            for roi_name in roi_landmarks.keys():

                frame_roi_masks[roi_name] = np.zeros((height, width), dtype=np.uint8)

        # Append masks for this frame
        roi_mask_log.append(frame_roi_masks)

        if show_display:
            # Show annotated output
            cv2.imshow("Thermal FaceMesh + Landmarks + ROIs", pseudo_rgb)

        if out is not None:
            out.write(pseudo_rgb)

        if show_display and cv2.waitKey(1) & 0xFF == ord('q'):
            break


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
    """Load a .gray frame (Y16, little-endian) and optionally strip telemetry lines."""
    raw = np.fromfile(filepath, dtype=np.uint16)
    if raw.size != width * height:
        raise ValueError(f"Unexpected frame size in {filepath}: got {raw.size}, expected {width*height}")
    frame = raw.reshape(height, width)
    if drop_telemetry and height > 120:
        frame = frame[:-2]  # remove last 2 lines
    return frame


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
    """Create a fixed ROI mask by manually drawing on the first gray frame.

    Returns:
        roi_mask_log: list of per-frame ROI masks (same mask reused for all frames)
    """
    _, gray_files = sort_grayfiles_with_interpolations(gray_dir)
    if not gray_files:
        raise ValueError(f"No .gray files found in {gray_dir}")

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

    frame_index = max(0, min(frame_index, len(gray_files) - 1))
    first_frame_path = os.path.join(gray_dir, gray_files[frame_index])
    first_frame = load_gray_frame(first_frame_path)
    
    # Flip the frame vertically to match video orientation
    # Gray frames are stored upside-down relative to the video
    print("[Warning]: Flipping frame vertically for ROI selection to match video orientation.")
    first_frame = cv2.flip(first_frame, 0)

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

        instructions = "Left click: add point | Right click: undo | ENTER: finish | c: clear"

        def draw_overlay(img, pts):
            canvas = img.copy()
            if pts:
                cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (0, 255, 255), 1)
                for p in pts:
                    cv2.circle(canvas, p, 2, (0, 255, 255), -1)
            cv2.putText(canvas, instructions, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 255), 1)
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
    else:
        raise ValueError(f"Unknown manual ROI mode: {mode}")

    # Mask is now in the flipped (video) orientation - no need to flip back
    # It will be applied correctly to the video

    if mask_save_path:
        save_dir = os.path.dirname(mask_save_path)
        if save_dir:  # Only create directory if path contains one
            os.makedirs(save_dir, exist_ok=True)
        np.save(mask_save_path, mask)

    roi_mask_log = [{roi_name: mask} for _ in range(len(gray_files))]
    return roi_mask_log


def apply_roi_masks_to_gray_frames(
    gray_dir: str,
    roi_mask_log: List[Dict[str, np.ndarray]],
    output_npz: str = None,
) -> List[Dict[str, Dict[str, object]]]:
    """Apply ROI masks (from video detection) to original .gray frames to extract temperatures.

    Parameters:
        gray_dir: directory containing .gray frames
        roi_mask_log: list of per-frame ROI masks (from apply_facemesh_to_thermal_video)
        output_npz: optional path to save extracted data as npz

    Returns:
        roi_pixel_data: list (per frame) of dicts {roi_name: {"coords", "values", "mean_value"}}
    """
    _, gray_files = sort_grayfiles_with_interpolations(gray_dir)
    if not gray_files:
        raise ValueError(f"No .gray files found in {gray_dir}")

    if len(gray_files) < len(roi_mask_log):
        raise ValueError(f"Fewer .gray frames ({len(gray_files)}) than masks ({len(roi_mask_log)})")
    elif len(gray_files) > len(roi_mask_log):
        raise ValueError(f"More .gray frames ({len(gray_files)}) than masks ({len(roi_mask_log)})")

    roi_pixel_data: List[Dict[str, Dict[str, object]]] = []

    for idx, (fname, masks) in enumerate(zip(gray_files, roi_mask_log)):
        frame_path = os.path.join(gray_dir, fname)
        frame = load_gray_frame(frame_path)

        height, width = frame.shape
        frame_data: Dict[str, Dict[str, object]] = {}

        for roi_name, mask in masks.items():
            if mask.shape != frame.shape:
                raise ValueError(f"Mask shape {mask.shape} does not match frame shape {frame.shape} for ROI {roi_name}")

            ys, xs = np.where(mask == 255)
            coords = list(zip(xs.tolist(), ys.tolist()))
            values = frame[ys, xs].astype(np.float64)
            frame_data[roi_name] = {
                "coords": coords,
                "adc_values": values.tolist(),
                "mean_kelvin_multiplied_100": float(np.mean(values)) if values.size > 0 else float('nan'),
            }

        roi_pixel_data.append(frame_data)

        if (idx + 1) % 100 == 0 or idx == len(gray_files) - 1:
            print(f"Processed {idx+1}/{len(gray_files)} frames")

    if output_npz:
        os.makedirs(os.path.dirname(output_npz), exist_ok=True)
        np.savez(output_npz, roi_pixel_data=roi_pixel_data)
        print(f"Saved ROI pixel data to {output_npz}")

    return roi_pixel_data


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
        means = [frame.get(roi, {}).get("mean_kelvin_multiplied_100", np.nan) for frame in roi_pixel_data]
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