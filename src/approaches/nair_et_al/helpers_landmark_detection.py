import cv2
import dlib
import numpy as np
import mediapipe as mp
from imutils import face_utils, resize as imutils_resize


def draw_landmarks_with_numbers(frame, landmarks, circle_color=(255, 0, 0),
                                text_color=(0, 255, 0), font_scale=0.4, thickness=1):
    """
    Draw all landmarks and their indices on the frame.
    """
    frame_copy = frame.copy()
    for idx, (x, y) in enumerate(landmarks):
        cv2.circle(frame_copy, (x, y), 2, circle_color, -1)
        cv2.putText(frame_copy, str(idx), (x + 2, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    return frame_copy

def annotate_video_with_landmarks(input_video_path: str, output_video_path: str,
                                  upsample: int = 1, resize_frame: bool = True):
    """
    Process a video and generate a new video with all landmarks drawn and numbered.
    
    Parameters:
        input_video_path: str
            Path to the input video.
        output_video_path: str
            Path to save the annotated output video.
        upsample: int
            Number of times to upsample image for face detection (dlib).
        resize_frame: bool
            Resize frames to width=300 for faster processing.
    """
    # Load face detector and landmark predictor
    detector = dlib.simple_object_detector("external_library/thermal-facial-landmarks-detection/models/dlib_face_detector.svm")
    predictor = dlib.shape_predictor("external_library/thermal-facial-landmarks-detection/models/dlib_landmark_predictor.dat")

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    writer = None

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        if resize_frame:
            frame = imutils_resize(frame, width=300)

        frame_annotated = frame.copy()
        frame_gray = cv2.cvtColor(frame_annotated, cv2.COLOR_BGRA2GRAY)

        rects = detector(frame_gray, upsample_num_times=upsample)

        for rect in rects:
            shape = predictor(frame_gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Draw landmarks with numbers
            frame_annotated = draw_landmarks_with_numbers(frame_annotated, shape)

        # Initialize video writer once
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps,
                                     (frame_annotated.shape[1], frame_annotated.shape[0]), True)

        writer.write(frame_annotated)
        cv2.imshow("Annotated Video", frame_annotated)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Annotated video saved to: {output_video_path}")


def apply_dlib_to_thermal_video(
    video_path: str,
    path_save_video_with_landmarks: str,
    upsample: int,
    resize_frame: bool,
    fps: float | None = None,
    show_display: bool = False,
    start_frame: int = 0,
    end_frame: int | None = None,
    flip_upside_down: bool = True,
) -> list[dict]:
    """
    Detects facial landmarks, divides face into 5 subregions, visualizes them,
    and returns ROI masks per frame (same format as apply_facemesh_to_thermal_video).
    """
    print("[IMPORTANT] Ensure that the input video is properly aligned with the original .gray frames (same order, not flipped) for correct ROI mapping later on.")

    if flip_upside_down:
        print("[Warning] Flipping frames vertically for Dlib; masks/output will be flipped back to match original orientation.")
            
    # Load detector and predictor
    detector = dlib.simple_object_detector("external/thermal-facial-landmarks-detection/models/dlib_face_detector.svm")
    predictor = dlib.shape_predictor("external/thermal-facial-landmarks-detection/models/dlib_landmark_predictor.dat")

    vs = cv2.VideoCapture(video_path)
    input_fps = fps 
    delay = int(1000 / input_fps)

    writer = None
    (W, H) = (None, None)
    frame_counter = 0

    roi_mask_log = []
    roi_names = ["forehead", "left_cheek", "right_cheek", "nose", "chin"]

    if start_frame > 0:
        for _ in range(start_frame):
            grabbed, _ = vs.read()
            if not grabbed:
                break

    frame_idx = 0
    max_frames = None
    if end_frame is not None:
        max_frames = max(0, end_frame - start_frame)

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        grabbed, frame = vs.read()
        if not grabbed:
            break

        frame_counter += 1

        if resize_frame:
            frame = imutils_resize(frame, width=300)

        frame_display = frame.copy()
        if flip_upside_down:
            frame = cv2.flip(frame, 0)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        frame_gray_original = cv2.cvtColor(frame_display, cv2.COLOR_BGRA2GRAY)

        rects = detector(frame_gray, upsample_num_times=upsample)

        frame_roi_masks = {}

        for rect in rects:
            shape = predictor(frame_gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Draw landmarks
            if flip_upside_down:
                h = frame_display.shape[0]
                for (sx, sy) in shape:
                    cv2.circle(frame_display, (sx, h - 1 - sy), 2, (255, 0, 0), -1)
            else:
                for (sx, sy) in shape:
                    cv2.circle(frame_display, (sx, sy), 2, (255, 0, 0), -1)

            # Define subregions using landmarks
            # These indices assume a standard 68-point dlib predictor
            # TODO points seem one point off 27 in picture is 26 in truth
            left_cheek_pts = shape[[1,2,3,4,5,48,28]].astype(np.int32) 
            right_cheek_pts = shape[[15,14,13,13,50,28]].astype(np.int32)  
            nose_pts = shape[27:36]  
            chin_pts = shape[[6,7,8,9,10,11,51]].astype(np.int32) 

            # Forehead points: The landmarks only include eyebrows
            # Eyebrow points
            left = shape[17]
            right = shape[26]

            # Height above eyebrows
            forehead_height = 30  # pixels, adjust as needed

            # Top points diagonally inward (so top width is smaller than bottom width)
            top_left_x = left[0] + int(0.2 * (right[0] - left[0]))   # shift right 20% of width
            top_right_x = right[0] - int(0.2 * (right[0] - left[0])) # shift left 20% of width
            top_y = max(0, np.min(shape[17:27,1]) - forehead_height)

            # Create trapezoid polygon
            forehead_pts = np.array([
                [left[0], left[1]],        # bottom-left
                [right[0], right[1]],      # bottom-right
                [top_right_x, top_y],      # top-right
                [top_left_x, top_y]        # top-left
            ], dtype=np.int32)

            subregions = [forehead_pts, left_cheek_pts, right_cheek_pts, nose_pts, chin_pts]

            for i, pts in enumerate(subregions):
                # Create convex hull and mask
                mask = np.zeros(frame_gray.shape, dtype=np.uint8)
                hull = cv2.convexHull(pts)
                cv2.drawContours(mask, [hull], -1, 255, -1)  # fill the polygon
                if flip_upside_down:
                    # Flip mask back to match original orientation for correct coordinate mapping
                    mask = cv2.flip(mask, 0)
                frame_roi_masks[roi_names[i]] = mask

                # Draw subregion overlay
                overlay_color = [(0,255,255), (255,0,255), (0,255,0), (255,255,0), (0,0,255)]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame_display, contours, -1, overlay_color[i], 2)

        if not frame_roi_masks:
            frame_roi_masks = {
                name: np.zeros(frame_gray.shape, dtype=np.uint8)
                for name in roi_names
            }

        roi_mask_log.append(frame_roi_masks)

        if W is None or H is None:
            (H, W) = frame_gray.shape[:2]

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path_save_video_with_landmarks, fourcc, input_fps,
                                     (frame_display.shape[1], frame_display.shape[0]), True)

        output_frame = cv2.flip(frame_display, 0) if flip_upside_down else frame_display
        writer.write(output_frame)
        if show_display:
            cv2.imshow("Frame", output_frame)
            display_delay = 1
            key = cv2.waitKey(display_delay) & 0xFF
            if key == ord("q"):
                break

        frame_idx += 1

    if show_display:
        cv2.destroyAllWindows()
    vs.release()
    writer.release()

    print("Collected ROI masks for", len(roi_mask_log), "frames")
    return roi_mask_log





