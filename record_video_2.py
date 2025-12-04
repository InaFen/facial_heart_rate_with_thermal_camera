import cv2

import os

def record_video_thermal():
    #TODO record with highest possible resolution
    cap = cv2.VideoCapture("/dev/video2")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  

    print(f"Recording video with resolution: {width}x{height} at {fps} FPS")
    output_dir = 'recorded_videos'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'thermal_video.avi')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)

        cv2.imshow('Recording', frame)

        # Press 'q' to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

record_video_thermal()