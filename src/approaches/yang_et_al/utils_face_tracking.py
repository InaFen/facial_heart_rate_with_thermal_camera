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
    grid_size: int = 5,
    block_size: int = 16,
    show_display: bool = True,
):
    """
    Full thermal head tracking pipeline.

    Returns:
        centers
        Y (N x M temporal signal matrix)
    """

    cap = cv2.VideoCapture(in_path)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    templates = generate_head_templates()

    tracker = STCTracker()

    frame_idx = 0

    centers = []

    signals = []

    initialized = False

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ----------------------------------------
        # THERMAL PREPROCESSING
        # ----------------------------------------

        gray = cv2.GaussianBlur(gray, (5, 5), 0)

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

        center = (x + w // 2, y + h // 2)

        centers.append(center)

        # ----------------------------------------
        # SIGNAL EXTRACTION
        # ----------------------------------------

        block_means = extract_block_means(gray, center, grid_size, block_size)

        signals.append(block_means)

        # ----------------------------------------
        # VISUALIZATION
        # ----------------------------------------

        roi_size = (grid_size * block_size)

        rx = int(center[0] - roi_size // 2)

        ry = int(center[1] - roi_size // 2)

        cv2.rectangle(frame, (rx, ry), (rx + roi_size, ry + roi_size), (0, 255, 0), 2)

        cv2.circle(frame, center, 3, (0, 0, 255), -1)

        writer.write(frame)

        if show_display:

            cv2.imshow("Tracking", frame)

            key = cv2.waitKey(1)

            if key == ord("q"):
                break

        frame_idx += 1

    cap.release()

    writer.release()

    if show_display:
        cv2.destroyAllWindows()

    if len(signals) > 0:
        Y = np.vstack(signals)
    else:
        Y = np.zeros((0, grid_size * grid_size), dtype=np.float32)

    print("Output saved:", out_path)

    print("Signal matrix shape:", Y.shape)

    return centers, Y


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
