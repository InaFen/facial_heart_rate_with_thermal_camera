import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt


def apply_facemesh_to_thermal_video(input_path: str, output_path: str, roi_landmarks: dict):
    """
    Apply MediaPipe FaceMesh to a thermal video, draw ALL landmarks + ROIs,
    display in real time, optionally save annotated video, and collect ROI masks.
    If a frame has no detected face, a blank mask is stored to maintain alignment.
    """

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open input video.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        thermal_original = frame.copy()

        # Normalize if 16-bit
        if frame.dtype != np.uint8:
            frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame_norm = frame_norm.astype(np.uint8)
        else:
            frame_norm = frame

        # Convert to pseudo-RGB for visualization
        pseudo_rgb = cv2.applyColorMap(frame_norm, cv2.COLORMAP_JET)
        mesh_input = cv2.cvtColor(pseudo_rgb, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(mesh_input)

        frame_roi_masks = {}  # Store masks for all ROIs in this frame

        if results.multi_face_landmarks:
            face_lm = results.multi_face_landmarks[0]

            # Convert all 468 landmarks to pixel coordinates
            pts = np.array([
                [int(lm.x * width), int(lm.y * height)]
                for lm in face_lm.landmark
            ])

            # Draw ALL 468 landmarks 
            for (x, y) in pts:
                cv2.circle(pseudo_rgb, (x, y), 1, (0, 255, 0), -1)

            # Draw the ROI polygons & create ROI masks
            for roi_name, indices in roi_landmarks.items():
                roi_poly = pts[indices]  # polygon vertices

                # Draw polygon on output
                cv2.polylines(pseudo_rgb, [roi_poly], True, (255, 0, 0), 2)

                # Create ROI mask
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [roi_poly], 255)

                frame_roi_masks[roi_name] = mask
        else:
            # No face detected: create blank masks for each ROI
            
            for roi_name in roi_landmarks.keys():
                frame_roi_masks[roi_name] = np.zeros((height, width), dtype=np.uint8)

        # Append masks for this frame
        roi_mask_log.append(frame_roi_masks)

        # Show annotated output
        cv2.imshow("Thermal FaceMesh + Landmarks + ROIs", pseudo_rgb)

        if out is not None:
            out.write(pseudo_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    face_mesh.close()
    cv2.destroyAllWindows()

    print("Collected ROI masks for", len(roi_mask_log), "frames.")
    return roi_mask_log


def apply_roi_masks_to_video(video_path: str, roi_mask_log: list, output_path: str) -> list[dict]:
    """
    Apply ROI masks to the original video AND record:
      - all pixel values inside each ROI (per frame)
      - all pixel coordinates inside each ROI (per frame)

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
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    roi_pixel_data = []   # <-- NEW: store all ROI pixel/coord info here

    while True:
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

        cv2.imshow("ROI Masked Video", masked_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved masked video to: {output_path}")

    return roi_pixel_data


def plot_roi_mean_signals(roi_pixel_data: list):
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
    frames = list(range(num_frames))

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
        plt.plot(frames, roi_means[roi], label=roi)

    plt.xlabel("Frame Index", fontsize=12)
    plt.ylabel("Mean Temperature", fontsize=12)
    plt.title("ROI Mean Temperature Over Time", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
