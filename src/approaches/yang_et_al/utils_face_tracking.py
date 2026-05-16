import cv2
import numpy as np


# ============================================================
# HEAD TEMPLATE GENERATION
# ============================================================

def generate_head_templates() -> list:
    """
    Generate binary edge templates for scale-invariant
    Chamfer matching.
    """
    templates = []

    for scale in [0.75, 0.9, 1.0, 1.1, 1.25]:
        size = int(120 * scale)

        canvas = np.zeros((size, size), dtype=np.uint8)

        # head
        cv2.circle(canvas, (size // 2, int(size * 0.38)), int(size * 0.22), 255, -1)

        # shoulders
        cv2.ellipse(canvas, (size // 2, int(size * 0.72)), (int(size * 0.34), int(size * 0.18)), 0, 0, 180, 255, -1)

        edges = cv2.Canny(canvas, 50, 150)

        templates.append(edges)

    return templates


# ============================================================
# CHAMFER MATCHING
# ============================================================

def chamfer_match_candidates(
    gray: np.ndarray,
    templates: list,
    scales=None,
    top_k: int = 5,
):
    """
    Multi-scale Chamfer matching.
    Returns top candidate rectangles.
    """

    if scales is None:
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]

    frame_edges = cv2.Canny(gray, 50, 150)

    inv = cv2.bitwise_not(frame_edges)

    dist = cv2.distanceTransform(inv, cv2.DIST_L2,5  )

    H, W = gray.shape

    candidates = []

    for tmpl in templates:

        th, tw = tmpl.shape

        for s in scales:

            nh = int(th * s)
            nw = int(tw * s)

            if (
                nh < 20
                or nw < 20
                or nh >= H
                or nw >= W
            ):
                continue

            resized = cv2.resize(tmpl, (nw, nh), interpolation=cv2.INTER_NEAREST)

            kernel = (resized > 0).astype(np.float32)

            response = cv2.filter2D(dist, -1, kernel, anchor=(0, 0))

            valid_h = H - nh + 1
            valid_w = W - nw + 1

            cropped = response[:valid_h, :valid_w]

            min_val, _, min_loc, _ = cv2.minMaxLoc(cropped)

            x, y = min_loc

            candidates.append(((x, y, nw, nh), min_val))

    candidates.sort(key=lambda item: item[1])

    return candidates[:top_k]


# ============================================================
# HEMISPHERE FITTING
# ============================================================

def hemisphere_score(
    gray: np.ndarray,
    rect: tuple[int, int, int, int]
) -> float:
    """
    Thermal hemisphere correlation score.
    """

    x, y, w, h = rect

    patch = gray[y:y+h, x:x+w].astype(np.float32)

    if patch.size == 0:
        return -np.inf

    patch /= 255.0

    ys, xs = np.mgrid[0:h, 0:w]

    cx = w / 2
    cy = h / 2

    dx = (xs - cx) / (w / 2)
    dy = (ys - cy) / (h / 2)

    r2 = dx**2 + dy**2

    mask = r2 <= 1.0

    hemisphere = np.sqrt(np.maximum(0, 1 - r2))

    observed = patch[mask]
    expected = hemisphere[mask]

    observed -= observed.mean()
    expected -= expected.mean()

    denom = (np.linalg.norm(observed) * np.linalg.norm(expected) + 1e-6)

    score = np.sum(observed * expected) / denom

    return float(score)


# ============================================================
# STC TRACKER
# ============================================================

class STCTracker:
    """
    Spatial-Temporal Context tracker approximation.
    """

    def __init__(
        self,
        padding: float = 2.0,
        sigma: float = 2.0,
        alpha: float = 0.075,
    ):
        self.padding = padding
        self.sigma = sigma
        self.alpha = alpha

        self.pos = None
        self.size = None

        self.Hstcf = None
        self.window = None

    def _get_patch(
        self,
        image,
        center,
        size
    ):
        return cv2.getRectSubPix(image, (int(size[0]), int(size[1])), (float(center[0]), float(center[1])))

    def _create_context_region(
        self,
        target_size
    ):
        w = int(target_size[0] * (1 + self.padding))
        h = int(target_size[1] * (1 + self.padding))

        return (w, h)

    def _gaussian_confidence(
        self,
        size
    ):
        w, h = size

        xs = np.arange(w) - w / 2
        ys = np.arange(h) - h / 2

        X, Y = np.meshgrid(xs, ys)

        conf = np.exp(-(X**2 + Y**2) / (2 * self.sigma**2))

        return conf.astype(np.float32)

    def init(
        self,
        frame_gray,
        rect
    ):
        x, y, w, h = rect

        self.pos = (
            x + w // 2,
            y + h // 2
        )

        self.size = (w, h)

        context_size = self._create_context_region(self.size)

        context = self._get_patch(frame_gray, self.pos, context_size).astype(np.float32)

        self.window = cv2.createHanningWindow(context.shape[::-1], cv2.CV_32F)

        context = context * self.window

        conf = self._gaussian_confidence((context.shape[1], context.shape[0]))

        F = np.fft.fft2(context)
        G = np.fft.fft2(conf)

        self.Hstcf = G / (F + 1e-5)

    def update(
        self,
        frame_gray
    ):
        context_size = self._create_context_region(self.size)

        context = self._get_patch(frame_gray, self.pos, context_size).astype(np.float32)

        context = context * self.window

        F = np.fft.fft2(context)

        response = np.real(np.fft.ifft2(self.Hstcf * F))

        _, _, _, max_loc = cv2.minMaxLoc(response)

        dx = (
            max_loc[0]
            - response.shape[1] // 2
        )

        dy = (
            max_loc[1]
            - response.shape[0] // 2
        )

        cx = self.pos[0] + dx
        cy = self.pos[1] + dy

        self.pos = (
            int(cx),
            int(cy)
        )

        # -----------------------------------------
        # TEMPORAL UPDATE
        # -----------------------------------------

        new_context = self._get_patch(frame_gray, self.pos, context_size).astype(np.float32)

        new_context = new_context * self.window

        conf = self._gaussian_confidence((new_context.shape[1], new_context.shape[0]))

        F_new = np.fft.fft2(new_context)
        G_new = np.fft.fft2(conf)

        H_new = G_new / (F_new + 1e-5)

        self.Hstcf = ((1 - self.alpha) * self.Hstcf + self.alpha * H_new)

        x = int(
            self.pos[0]
            - self.size[0] // 2
        )

        y = int(
            self.pos[1]
            - self.size[1] // 2
        )

        return (
            x,
            y,
            self.size[0],
            self.size[1]
        )


# ============================================================
# ROI BLOCK EXTRACTION
# ============================================================

def extract_block_means(
    frame: np.ndarray,
    center: tuple[int, int],
    grid_size: int,
    block_size: int,
):
    """
    Extract M block-averaged thermal signals.
    """

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    cx, cy = center

    half_grid = grid_size // 2

    values = []

    for gy in range(-half_grid, half_grid + 1):

        for gx in range(-half_grid, half_grid + 1):

            x0 = int(
                cx + gx * block_size
                - block_size // 2
            )

            y0 = int(
                cy + gy * block_size
                - block_size // 2
            )

            x1 = x0 + block_size
            y1 = y0 + block_size

            x0 = max(0, x0)
            y0 = max(0, y0)

            x1 = min(gray.shape[1], x1)
            y1 = min(gray.shape[0], y1)

            block = gray[y0:y1, x0:x1]

            if block.size == 0:
                values.append(np.nan)
            else:
                values.append(
                    float(np.mean(block))
                )

    return np.array(values, dtype=np.float32)


# ============================================================
# MAIN PIPELINE
# ============================================================

def track_face(
    in_path: str,
    out_path: str,
    fps: float,
    grid_size: int = 5,
    block_size: int = 21,
    show_display: bool = True,
    flip_upside_down: bool = True,
    start_frame: int = 0,
    end_frame: int | None = None,
):
    """
    Full thermal head tracking pipeline.

    Returns:
        roi_pixel_log
    """
    print("[IMPORTANT] Ensure that the input video is properly aligned with the original .gray frames (same order, not flipped) for correct ROI mapping later on.")

    if flip_upside_down:
        print("[Warning] Flipping frames vertically for tracker; masks/output will be flipped back to match original orientation.")

    cap = cv2.VideoCapture(in_path)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    templates = generate_head_templates()

    tracker = STCTracker()

    frame_idx = 0

    centers = []
    roi_pixel_log = []

    initialized = False

    if start_frame > 0:
        for _ in range(start_frame):
            ret, _ = cap.read()
            if not ret:
                break

    max_frames = None
    if end_frame is not None:
        max_frames = max(0, end_frame - start_frame)

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ret, frame = cap.read()

        if not ret:
            break

        frame_display = frame.copy()
        frame_proc = cv2.flip(frame, 0) if flip_upside_down else frame

        gray_raw = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
        gray_raw_proc = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)

        # ----------------------------------------
        # THERMAL PREPROCESSING
        # ----------------------------------------

        gray = cv2.GaussianBlur(gray_raw_proc, (5, 5), 0)
        gray = cv2.equalizeHist(gray)

        # ----------------------------------------
        # INITIALIZATION
        # ----------------------------------------

        if not initialized:

            candidates = chamfer_match_candidates(gray, templates, top_k=5)

            if len(candidates) == 0:
                print("No head detected.")
                break

            best_rect = None
            best_score = -np.inf

            for rect, _ in candidates:

                score = hemisphere_score(gray, rect)

                if score > best_score:
                    best_score = score
                    best_rect = rect

            if best_rect is None:
                print("No valid hemisphere.")
                break

            tracker.init(gray, best_rect)

            initialized = True

        # ----------------------------------------
        # TRACKING
        # ----------------------------------------

        bbox = tracker.update(gray)

        x, y, w, h = bbox

        center_proc = (x + w // 2, y + h // 2)
        if flip_upside_down:
            center = (center_proc[0], (H - 1) - center_proc[1])
        else:
            center = center_proc

        roi_size = (grid_size * block_size)
        roi_half = roi_size // 2
        rx0 = max(0, int(center[0] - roi_half))
        ry0 = max(0, int(center[1] - roi_half))
        rx1 = min(gray_raw.shape[1], int(center[0] + roi_half))
        ry1 = min(gray_raw.shape[0], int(center[1] + roi_half))

        if rx1 > rx0 and ry1 > ry0:
            roi_patch = gray_raw[ry0:ry1, rx0:rx1].astype(np.float32)
            weights = roi_patch - float(np.min(roi_patch))
            weight_sum = float(np.sum(weights))
            if weight_sum > 0:
                ys, xs = np.mgrid[0:roi_patch.shape[0], 0:roi_patch.shape[1]]
                cx = float(np.sum(xs * weights) / weight_sum)
                cy = float(np.sum(ys * weights) / weight_sum)
                center = (int(rx0 + cx), int(ry0 + cy))

        center = (center[0], min(H - 1, center[1] - int(0.1 * roi_size)))

        centers.append(center)

        # ----------------------------------------
        # SIGNAL EXTRACTION
        # ----------------------------------------

        frame_roi_data = {}
        frame_roi_masks = {}
        half_grid = grid_size // 2

        for gy in range(-half_grid, half_grid + 1):
            for gx in range(-half_grid, half_grid + 1):
                roi_name = f"roi_r{gy + half_grid + 1}c{gx + half_grid + 1}"
                x0 = int(center[0] + gx * block_size - block_size // 2)
                y0 = int(center[1] + gy * block_size - block_size // 2)
                x1 = x0 + block_size
                y1 = y0 + block_size

                x0c = max(0, x0)
                y0c = max(0, y0)
                x1c = min(gray_raw.shape[1], x1)
                y1c = min(gray_raw.shape[0], y1)

                if x1c <= x0c or y1c <= y0c:
                    coords = []
                    values = []
                    mean_val = float("nan")
                else:
                    ys, xs = np.mgrid[y0c:y1c, x0c:x1c]
                    coords = list(zip(xs.ravel().tolist(), ys.ravel().tolist()))
                    values_array = gray_raw[y0c:y1c, x0c:x1c].astype(np.float32)
                    values = values_array.ravel().tolist()
                    mean_val = float(np.mean(values_array))

                frame_roi_data[roi_name] = {
                    "coords": coords,
                    "values": values,
                    "mean_value": mean_val,
                    "mean_kelvin_multiplied_100": mean_val,
                }

                mask = np.zeros(gray_raw.shape, dtype=np.uint8)
                if x1c > x0c and y1c > y0c:
                    mask[y0c:y1c, x0c:x1c] = 255
                frame_roi_masks[roi_name] = mask

        roi_pixel_log.append(frame_roi_masks)

        # ----------------------------------------
        # VISUALIZATION
        # ----------------------------------------

        rx = int(center[0] - roi_size // 2)

        ry = int(center[1] - roi_size // 2)

        cv2.rectangle(frame_display, (rx, ry), (rx + roi_size, ry + roi_size), (0, 255, 0), 2)

        cv2.circle(frame_display, center, 3, (0, 0, 255), -1)

        writer.write(frame_display)

        if show_display:

            cv2.imshow("Tracking", frame_display)

            key = cv2.waitKey(1)

            if key == ord("q"):
                break

        frame_idx += 1

    cap.release()

    writer.release()

    if show_display:
        cv2.destroyAllWindows()

    print("Output saved:", out_path)
    print("Frames processed:", len(roi_pixel_log))

    return roi_pixel_log


def filter_out_pixels_below_threshold(video_path, output_path="threshold_output.mp4", threshold_value=150):
    """
    Process a thermal video and keep only pixels above a threshold.

    Parameters:
        video_path (str): Path to input thermal video
        output_path (str): Path to save output video
        threshold_value (int): Pixel intensity threshold (0-255)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale (thermal intensity)
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Threshold: keep only pixels above the threshold
        mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)[1]

        # Optional: remove small noise
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Apply mask to frame
        face_only = cv2.bitwise_and(gray, gray, mask=mask)

        # Convert back to BGR for video writer
        frame_out = cv2.cvtColor(face_only, cv2.COLOR_GRAY2BGR)

        writer.write(frame_out)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Processed video saved to: {output_path}")
    return output_path


def filter_out_pixels_below_threshold_in_roi_pixel_data(roi_pixel_data, threshold_value=150, output_video_path=None, fps=None):
    """
    Filter ROI pixel values below a threshold and recompute means.

    Parameters:
        roi_pixel_data: list of dicts (per frame) with keys per ROI
        threshold_value: Pixel intensity threshold 

    Returns:
        filtered_roi_pixel_data: same structure with values filtered and mean_value updated
    """
    # Validate video settings
    if output_video_path is not None and fps is None:
        raise ValueError("fps must be provided when output_video_path is set")

    frame_width = 0
    frame_height = 0
    # Determine output frame size from ROI coordinates
    if output_video_path is not None:
        for frame_data in roi_pixel_data:
            for roi_info in frame_data.values():
                coords = roi_info.get("coords", [])
                for x, y in coords:
                    if x + 1 > frame_width:
                        frame_width = x + 1
                    if y + 1 > frame_height:
                        frame_height = y + 1

    writer = None
    # Initialize video writer if requested
    if output_video_path is not None and frame_width > 0 and frame_height > 0:
        writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )

    # Compute global min/max for normalization (across filtered values only)
    global_min = None
    global_max = None
    for frame_data in roi_pixel_data:
        for roi_info in frame_data.values():
            values = roi_info.get("adc_values", [])
            for value in values:
                if value < threshold_value:
                    continue
                if global_min is None or value < global_min:
                    global_min = value
                if global_max is None or value > global_max:
                    global_max = value

    filtered_roi_pixel_data = []
    roi_has_values = {}

    # Filter values per ROI and optionally render kept pixels
    for frame_data in roi_pixel_data:
        filtered_frame = {}
        if writer is not None:
            output_frame = np.zeros((frame_height, frame_width), dtype=np.uint8)
        for roi_name, roi_info in frame_data.items():
            values = roi_info.get("adc_values", [])
            coords = roi_info.get("coords", [])

            filtered_values = []
            filtered_coords = []
            for coord, value in zip(coords, values):
                if value >= threshold_value:
                    filtered_coords.append(coord)
                    filtered_values.append(value)

            if filtered_values:
                mean_val = float(np.mean(filtered_values))
                roi_has_values[roi_name] = True
            else:
                mean_val = float("nan")
                roi_has_values.setdefault(roi_name, False)

            filtered_frame[roi_name] = {
                "coords": filtered_coords,
                "adc_values": filtered_values,
                "mean_kelvin_multiplied_100": mean_val,
            }

            if writer is not None:
                for (x, y), value in zip(filtered_coords, filtered_values):
                    if 0 <= x < frame_width and 0 <= y < frame_height:
                        if global_min is not None and global_max is not None and global_max > global_min:
                            norm_val = (value - global_min) / (global_max - global_min)
                            output_frame[y, x] = np.uint8(np.clip(norm_val * 255.0, 0, 255))
                        else:
                            output_frame[y, x] = 0

        filtered_roi_pixel_data.append(filtered_frame)
        if writer is not None:
            output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)
            writer.write(output_frame_bgr)

    # Finalize video output
    if writer is not None:
        writer.release()

    # Drop ROIs that never had any values across all frames
    if roi_has_values:
        roi_to_keep = {name for name, has_values in roi_has_values.items() if has_values}
        filtered_roi_pixel_data = [
            {name: info for name, info in frame.items() if name in roi_to_keep}
            for frame in filtered_roi_pixel_data
        ]

    return filtered_roi_pixel_data
