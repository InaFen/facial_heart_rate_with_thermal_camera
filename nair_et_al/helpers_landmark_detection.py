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


def get_facial_dlib_landmarks_regions_and_save_their_mean(video_path: str, path_save_video_with_landmarks: str, upsample: int , resize_frame: bool) -> np.ndarray:
    """
    Detects facial landmarks, divides face into 5 subregions, visualizes them, 
    and calculates mean values for each subregion per frame.
    """
    # Load detector and predictor
    detector = dlib.simple_object_detector("external_library/thermal-facial-landmarks-detection/models/dlib_face_detector.svm")
    predictor = dlib.shape_predictor("external_library/thermal-facial-landmarks-detection/models/dlib_landmark_predictor.dat")

    vs = cv2.VideoCapture(video_path)
    input_fps = vs.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / input_fps)

    writer = None
    (W, H) = (None, None)
    frame_counter = 0

    all_means = []  # will store mean values for all frames (5 x num_frames)

    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break
        frame_counter += 1

        if resize_frame:
            frame = imutils_resize(frame, width=300)

        frame_thermal = frame.copy()
        frame_gray = cv2.cvtColor(frame_thermal, cv2.COLOR_BGRA2GRAY)

        rects = detector(frame_gray, upsample_num_times=upsample)

        frame_means = np.zeros(5)  # forehead, left cheek, right cheek, nose, chin

        for rect in rects:
            shape = predictor(frame_gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Draw landmarks
            for (sx, sy) in shape:
                cv2.circle(frame_thermal, (sx, sy), 2, (255, 0, 0), -1)

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
                mean_val = cv2.mean(frame_gray, mask=mask)[0]
                frame_means[i] = mean_val

                # Draw subregion overlay
                overlay_color = [(0,255,255), (255,0,255), (0,255,0), (255,255,0), (0,0,255)]
                cv2.drawContours(frame_thermal, [hull], -1, overlay_color[i], 2)

        all_means.append(frame_means)

        if W is None or H is None:
            (H, W) = frame_gray.shape[:2]

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(path_save_video_with_landmarks, fourcc, input_fps,
                                     (frame_thermal.shape[1], frame_thermal.shape[0]), True)

        writer.write(frame_thermal)
        cv2.imshow("Frame", frame_thermal)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.release()
    writer.release()

    all_means = np.array(all_means).T  # 5 x num_frames
    print("Shape of mean values matrix:", all_means.shape)
    return all_means



def apply_facemesh_to_thermal_video(input_path: str, output_path: str=None):
    """
    Apply MediaPipe FaceMesh to a thermal video, display the result in real-time,
    and optionally save the annotated output video.

    Args:
        input_path: Path to the thermal input video.
        output_path: Path to save the annotated output video (optional).
                     If None, video is not saved.
    """

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open input video.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

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

    print("Processing video... Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Normalize IF 16-bit thermal
        if frame.dtype != np.uint8:
            frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame_norm = frame_norm.astype(np.uint8)
        else:
            frame_norm = frame

        # Pseudo-RGB for MediaPipe
        pseudo_rgb = cv2.applyColorMap(frame_norm, cv2.COLORMAP_JET)
        rgb_for_mesh = cv2.cvtColor(pseudo_rgb, cv2.COLOR_BGR2RGB)

        # Run FaceMesh
        results = face_mesh.process(rgb_for_mesh)

        # Draw landmarks
        if results.multi_face_landmarks:
            for face_lm in results.multi_face_landmarks:
                for lm in face_lm.landmark:
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    cv2.circle(pseudo_rgb, (x, y), 1, (0, 255, 0), -1)

        # Show video live
        cv2.imshow("Thermal FaceMesh", pseudo_rgb)

        # Write frame if saving enabled
        if out is not None:
            out.write(pseudo_rgb)

        # Quit on Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    face_mesh.close()
    cv2.destroyAllWindows()
