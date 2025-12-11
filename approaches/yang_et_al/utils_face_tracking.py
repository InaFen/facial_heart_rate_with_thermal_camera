import cv2
import numpy as np

def generate_head_templates() -> list:
    """
    Generate head templates for Chamfer matching.
    Returns a list of binary edge images representing head templates at different scales.
    """
    templates = []
    
    # base circle head
    for scale in [1.0, 0.85, 1.15]:
        size = int(120 * scale)
        canvas = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(canvas, (size//2, size//2), int(size*0.45), 255, -1)
        
        # shoulders (semi-ellipse)
        cv2.ellipse(canvas,
                    (size//2, int(size*0.75)),
                    (int(size*0.55), int(size*0.25)),
                    0, 0, 180, 255, -1)
        
        # get edges
        edges = cv2.Canny(canvas, 50, 150)
        templates.append(edges)
    
    return templates

def chamfer_match(gray, templates, scales=[1.0, 0.9, 1.1]):
    """
    Applies Chamfer matching to detect head in the given grayscale image.
    Returns the best matching rectangle and its score.  

    :param gray: Grayscale input image
    :param templates: List of head templates (edge images)
    :param scales: List of scales to apply to templates
    :return: (best_rect, best_score)
    """
    # 1. Frame edges
    frame_edges = cv2.Canny(gray, 50, 150)
    inv = cv2.bitwise_not(frame_edges)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

    best_score = float("inf")
    best_rect = None

    H, W = gray.shape

    for tmpl in templates:
        th, tw = tmpl.shape

        for s in scales:
            nh, nw = int(th * s), int(tw * s)

            if nh < 20 or nw < 20 or nh >= H or nw >= W:
                continue

            resized = cv2.resize(tmpl, (nw, nh), interpolation=cv2.INTER_NEAREST)
            kernel = (resized > 0).astype(np.float32)

            resp = cv2.filter2D(dist, -1, kernel, anchor=(0,0))
            valid_h = H - nh + 1
            valid_w = W - nw + 1

            cropped = resp[:valid_h, :valid_w]
            min_val, _, min_loc, _ = cv2.minMaxLoc(cropped)

            if min_val < best_score:
                best_score = min_val
                x, y = min_loc
                best_rect = (x, y, nw, nh)

    return best_rect, best_score

def make_tracker():
    """
    Create and return a CSRT tracker if available, otherwise fallback to KCF or MOSSE.
    """
    # preference order
    for name in ["CSRT", "KCF", "MOSSE"]:
        try:
            return getattr(cv2, f"Tracker{name}_create")()
        except:
            try:
                return getattr(cv2.legacy, f"Tracker{name}_create")()
            except:
                pass
    return None


def track_face(in_path: str, out_path: str) -> None:
    """
    Track face in the input video and save the output with tracked bounding box.

    :param in_path: Path to input video
    :param out_path: Path to output video
    :return: None
    """
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print("Error: cannot open video.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    templates = generate_head_templates()
    tracker = None
    frame_idx = 0
    roi = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -----------------------------------------------------
        # First frame: detect via Chamfer
        # -----------------------------------------------------
        if frame_idx == 0:
            print("Running Chamfer detection...")
            rect, score = chamfer_match(gray, templates)

            if rect is None:
                print("No head detected.")
                break

            x, y, w, h = rect
            roi = (x, y, w, h)

            tracker = make_tracker()
            tracker.init(frame, roi)

            print("Initial ROI:", roi)

        # -----------------------------------------------------
        # Subsequent frames: track
        # -----------------------------------------------------
        else:
            ok, bbox = tracker.update(frame)
            if ok:
                x, y, w, h = map(int, bbox)
                roi = (x, y, w, h)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, "Tracking", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.putText(frame, "LOST TRACKING", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print("Done. Output saved as:", out_path)



def filter_out_pixels_below_threshold(video_path, output_path="threshold_output.mp4",
                                    threshold_value=150):
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
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

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
