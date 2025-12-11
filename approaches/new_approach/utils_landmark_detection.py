import cv2
import numpy as np
import mediapipe as mp

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

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    face_mesh.close()
    cv2.destroyAllWindows()

    print("Collected ROI masks for", len(roi_mask_log), "frames.")
    return roi_mask_log


def apply_roi_masks_to_video(video_path: str, roi_mask_log: list, output_path: str):
    """
    Apply per-frame ROI masks to the original video and save the result.

    Parameters:
        video_path (str): Path to the original thermal video
        roi_mask_log (list): List of dicts containing ROI masks per frame
        output_path (str): Path to save the output video
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

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(roi_mask_log):
            break

        # If frame is 16-bit, normalize to 0-255 for visualization
        if frame.dtype != np.uint8:
            frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame_norm = frame_norm.astype(np.uint8)
        else:
            frame_norm = frame.copy()

        # Create a combined mask if there are multiple ROIs
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        for mask in roi_mask_log[frame_idx].values():
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Apply the mask to the original frame
        masked_frame = cv2.bitwise_and(frame_norm, frame_norm, mask=combined_mask)

        # Ensure the frame we write is BGR (3-channel). The input `frame_norm`
        # may already be color (3-channel) or grayscale (1-channel). Only
        # convert if it's a single-channel image.
        if masked_frame.ndim == 2 or masked_frame.shape[2] == 1:
            masked_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_GRAY2BGR)
        else:
            masked_bgr = masked_frame

        # Optional: overlay the mask in color for visibility
        # color_overlay = np.zeros_like(masked_bgr)
        # color_overlay[:, :, 2] = combined_mask  # red overlay
        # masked_bgr = cv2.addWeighted(masked_bgr, 1.0, color_overlay, 0.5, 0)

        out.write(masked_bgr)

        # Optional display
        cv2.imshow("ROI Masked Video", masked_bgr)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved masked video to: {output_path}")
