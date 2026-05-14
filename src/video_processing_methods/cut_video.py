import cv2
import os

def extract_video_in_specific_timespan(start_seconds, end_seconds, input_path, output_path, fps):

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate start and end frames
    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps)

    # Ensure frame indices are within bounds
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(0, min(end_frame, total_frames - 1))

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read and write frames in the specified range
    for frame_num in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Release resources
    cap.release()
    out.release()

input_video_path = "data/recorded_videos/thermal_video_from_y16_frames/03022026_3.mp4"
output_video_path = "data/temp/03022026_3_thermal_cut.mp4"
start_time = 180.0  # Start time in seconds
end_time = 210.0    # End time in seconds

fps_yuvu = 11.323650645773151
fps_mjpg = 18.564911881141693
fps_thermal = 8.767704384613774

fps = fps_thermal

extract_video_in_specific_timespan(start_time, end_time, input_video_path, output_video_path, fps)