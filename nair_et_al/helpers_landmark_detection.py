import cv2
import dlib
import numpy as np
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
