import cv2
import os
import time

def record_and_save_thermal_frames(device_path, output_folder="frames", duration=5, warmup_frames=30):
    """
    Record thermal frames and save each frame as an image named by its timestamp.

    Args:
        device_path (str): Path to the thermal camera (e.g., /dev/videoX or V4L2 by-id path)
        output_folder (str): Folder to save frames
        duration (float): Recording duration in seconds
        warmup_frames (int): Number of frames to discard for camera warmup
    """
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(device_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open thermal camera at {device_path}")
    
    # Camera warmup
    print(f"Warming up camera ({warmup_frames} frames)...")
    for _ in range(warmup_frames):
        cap.read()

    start_time = time.time()
    frame_count = 0
    timestamps = []

    print(f"Recording for {duration} seconds...")

    while time.time() - start_time < duration and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed")
            continue

        # Timestamp relative to start
        ts = time.time() - start_time
        timestamps.append(ts)

        # Save frame with timestamp as filename
        filename = os.path.join(output_folder, f"frame_{ts:.3f}.png")
        cv2.imwrite(filename, frame)

        frame_count += 1

    cap.release()
    print(f"Recording finished. {frame_count} frames saved to '{output_folder}', which equals overall {frame_count/duration:.2f} FPS.")
    return timestamps

if __name__ == "__main__":
    device_path = "/dev/v4l/by-id/usb-GroupGets_PureThermal__fw:v1.3.0__80030034-5101-3133-3332-373300000000-video-index0"  # Change as needed
    output_folder = "thermal_frames"
    duration = 5  

    record_and_save_thermal_frames(device_path, output_folder, duration)