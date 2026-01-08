import cv2
import os
import time
import glob


def record_and_save_thermal_frames(device_path, output_folder="frames", duration=5, warmup_frames=30):
    """
    Record thermal frames and save each frame as an image named by its timestamp.

    Args:
        device_path (str): Path to the thermal camera (e.g., /dev/videoX or V4L2 by-id path)
        output_folder (str): Folder to save frames
        duration (float): Recording duration in seconds
        warmup_frames (int): Number of frames to discard for camera warmup
    """
    folder = os.path.join(output_folder,time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    
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
        unix_ts = time.time()
        timestamps.append(ts)

        # Save frame with unix timestamp and seconds since start as filename
        filename = os.path.join(folder, f"frame_{unix_ts:.3f}_{ts:.3f}.png")
        cv2.imwrite(filename, frame)

        frame_count += 1

    cap.release()
    print(f"Recording finished. {frame_count} frames saved to '{folder}', which equals overall {frame_count/duration:.2f} FPS.")
    return timestamps, folder


def analyze_frame_timing_dual(frame_folder):
    """
    Analyze timestamped frames and compute time differences and FPS.

    Frame format: frame_<unix_time>_<time_since_start>.png

    Args:
        frame_folder (str): Folder containing frames

    Returns:
        time_diffs_unix (list of float): Differences between consecutive frames using Unix time
        time_diffs_rel (list of float): Differences using time since start
        fps_unix (list of float): Instantaneous FPS using Unix time
        fps_rel (list of float): Instantaneous FPS using time since start
        unix_times (list of float): Extracted Unix timestamps
        rel_times (list of float): Extracted relative times since start
    """

    frame_files = glob.glob(os.path.join(frame_folder, "*.png"))
    if not frame_files:
        raise RuntimeError("No frames found in folder.")

    # Function to extract both unix time and relative time
    def extract_times(f):
        base = os.path.basename(f).replace(".png","")
        parts = base.split("_")
        unix_time = float(parts[1])
        rel_time = float(parts[2])
        return unix_time, rel_time

    # Sort files by relative time
    frame_files.sort(key=lambda f: extract_times(f)[1])

    # Extract times
    unix_times = []
    rel_times = []
    for f in frame_files:
        unix_time, rel_time = extract_times(f)
        unix_times.append(unix_time)
        rel_times.append(rel_time)

    # Compute time differences
    time_diffs_unix = [round(t2 - t1, 3) for t1, t2 in zip(unix_times[:-1], unix_times[1:])]
    time_diffs_rel = [round(t2 - t1, 3) for t1, t2 in zip(rel_times[:-1], rel_times[1:])]

    # Compute instantaneous FPS
    fps_unix = [round(1/dt, 3) if dt > 0 else 0 for dt in time_diffs_unix]
    fps_rel = [round(1/dt, 3) if dt > 0 else 0 for dt in time_diffs_rel]

    return time_diffs_unix, time_diffs_rel, fps_unix, fps_rel, unix_times, rel_times



if __name__ == "__main__":
    device_path = "/dev/v4l/by-id/usb-GroupGets_PureThermal__fw:v1.3.0__80030034-5101-3133-3332-373300000000-video-index0"
    output_folder = "thermal_frames"
    duration = 5  

    timestamps, folder = record_and_save_thermal_frames(device_path, output_folder, duration)

    td_unix, td_rel, fps_unix, fps_rel, unix_times, rel_times = analyze_frame_timing_dual(folder)

    print(f"First 5 Unix timestamps diffs:", td_unix[:5])
    print(f"First 5 Relative timestamps diffs:", td_rel[:5])
    print("Unique time diffs (Unix):", list(set(td_unix)))
    print("Unique time diffs (relative):", list(set(td_rel)))
    print("Unique FPS (Unix):", list(set(fps_unix)))
    print("Unique FPS (relative):", list(set(fps_rel)))