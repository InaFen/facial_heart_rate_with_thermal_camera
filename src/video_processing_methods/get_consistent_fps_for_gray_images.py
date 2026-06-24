import os
from collections import Counter
import numpy as np
import shutil

def load_gray_frame(file_path) -> np.ndarray:
    """
    Load a .gray frame (160x122 Y16 format)
    Assumes the last 2 rows are telemetry and strips them off. 
    Parameters:
        file_path: path to the .gray file
    Returns:
        numpy array of shape (120, 160) with 16-bit values"""
    
    # print("[Warning] The frame to be loaded is expected to be in 160x122 Y16 format.")
    with open(file_path, 'rb') as f:
        frame_data = np.frombuffer(f.read(), dtype=np.uint16).reshape((122, 160))
    # Strip telemetry lines (last 2 rows) to get 160x120 image
    return frame_data[:120, :]

def save_gray_frame(frame, file_path) -> None:
    """
    Save a frame as .gray file with telemetry lines (zeros).
    Parameters:
        frame: numpy array of shape (120, 160) with 16-bit values
        file_path: path to save the .gray file
    Returns:
        None
    """
    # print("[Warning] The frame to be saved is expected to be in 160x122 Y16 format.")
    # Add 2 rows of zeros for telemetry to make it 160x122
    frame_with_telemetry = np.vstack([frame, np.zeros((2, 160), dtype=np.uint16)])
    if os.path.exists(file_path):
        print(f"[Warning] {file_path} already exists and will NOT be overwritten.")
        return None
    with open(file_path, 'wb') as f:
        f.write(frame_with_telemetry.tobytes())

def interpolate_frames(path_frame_1, path_frame_2, expected_time_difference, output_dir: str):
    """
    Interpolate frames between two .gray frames based on expected time difference.
    Parameters:
        path_frame_1: path to the first .gray frame
        path_frame_2: path to the second .gray frame
        expected_time_difference: expected time difference in ms between frames
        output_dir: directory to save interpolated frames
    Returns:
        None
    """
    timestamp_1 = float(os.path.basename(path_frame_1).split('.')[0])
    timestamp_2 = float(os.path.basename(path_frame_2).split('.')[0])
    time_difference = timestamp_2 - timestamp_1
    
    multiple = time_difference // expected_time_difference
    num_frames_to_interpolate = int(multiple - 1)

    print(f"Interpolating frames between {timestamp_1} and {timestamp_2} with time difference {time_difference} ms")
    print(f"Expected time difference: {expected_time_difference} ms, Number of frames to interpolate: {num_frames_to_interpolate}")
    
    # Load the two frames
    frame_1 = load_gray_frame(path_frame_1)
    frame_2 = load_gray_frame(path_frame_2)
    
    # Create interpolated frames
    for i in range(1, num_frames_to_interpolate + 1):
        # Linear interpolation weight
        alpha = i / (num_frames_to_interpolate + 1)
        
        # Blend the two frames
        interpolated_frame = ((1 - alpha) * frame_1 + alpha * frame_2).astype(np.uint16)
        
        # Calculate interpolated timestamp
        interpolated_timestamp = int(timestamp_1 + i * expected_time_difference)
        
        # Save interpolated frame
        output_path = os.path.join(output_dir, f"{interpolated_timestamp}_interpolated.gray")
        save_gray_frame(interpolated_frame, output_path)
        print(f"  Created interpolated frame: {interpolated_timestamp}_interpolated.gray")
    
    return None

def check_and_interpolate_frame_timing(path_gray_images: str, expected_time_differences: list[int], output_dir: str) -> None:
    """Copy raw .gray frames and interpolate missing frames based on expected timing.

    - Copies all source frames to `output_dir` (skips if already present)
    - Detects gaps larger than expected_time_differences (e.g., 114/115 ms)
    - If a gap is an exact multiple (or multiple+1) of an expected interval, it linearly interpolates
      the missing frames and writes them with `_interpolated` suffix.

    Args:
        path_gray_images: Directory containing source .gray files.
        expected_time_differences: List of expected inter-frame times in ms (e.g., [114, 115]).
        output_dir: Destination directory to store original and interpolated frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .gray files timestamps in the directory
    frames_timestamps = sorted([f.split('.')[0] for f in os.listdir(path_gray_images) if f.endswith('.gray')])

    # Save first frame to output directory as loop starts from second frame
    src_path = os.path.join(path_gray_images, f"{frames_timestamps[0]}.gray")
    dst_path = os.path.join(output_dir, f"{frames_timestamps[0]}.gray")
    if not os.path.exists(dst_path):
        shutil.copyfile(src_path, dst_path)

    # Calculate time differences between consecutive frames
    time_diffs = []
    for i in range(1, len(frames_timestamps)):
        # Save frame to output directory
        src_path = os.path.join(path_gray_images, f"{frames_timestamps[i]}.gray")
        dst_path = os.path.join(output_dir, f"{frames_timestamps[i]}.gray")
        if not os.path.exists(dst_path):
            shutil.copyfile(src_path, dst_path)

        # Calculate time difference to previous frame
        timestamp_1 = float(frames_timestamps[i-1])
        timestamp_2 = float(frames_timestamps[i])
        time_difference = timestamp_2 - timestamp_1
        time_diffs.append(time_difference)

        if time_difference not in expected_time_differences:
            # Time difference btw. two frames is smaller than expected
            if time_difference < min(expected_time_differences):
                raise ValueError(f"Time difference {time_difference} is smaller than expected minimum {min(expected_time_differences)}")
            # Time difference btw. two frames is larger than expected
            elif time_difference > max(expected_time_differences):
                should_interpolate = False
                for expected_time_difference in expected_time_differences:
                    rest = time_difference % expected_time_difference
                    # Interpolate
                    if rest == 0 or rest == 1:
                        should_interpolate = True
                        interpolate_frames(path_frame_1=os.path.join(path_gray_images, f"{frames_timestamps[i-1]}.gray"),
                                         path_frame_2=os.path.join(path_gray_images, f"{frames_timestamps[i]}.gray"),
                                         expected_time_difference=expected_time_difference,
                                         output_dir=output_dir)
                if not should_interpolate:
                    raise ValueError(f"Time difference {time_difference} is larger than expected and not a valid multiple")

    return None

def get_fps_of_frames_in_directory(path_gray_images: str) -> float:
    """
    Calculate the frames per second (FPS) of .gray frames in a directory based on timestamps.
    Parameters:
        path_gray_images: directory containing .gray files
    """
    # Get all .gray files timestamps sorted in the directory
    frames_timestamps_interpolated_removed = [f.replace('_interpolated', '') for f in os.listdir(path_gray_images) if f.endswith('.gray')]
    frames_timestamps_redundant_zeros_removed = [ts.lstrip('0') if ts.lstrip('0') != '' else '0' for ts in frames_timestamps_interpolated_removed]
    frames_timestamps = sorted([f.split('.')[0] for f in frames_timestamps_redundant_zeros_removed])
    
    # Calculate time differences between consecutive frames
    time_diffs = []
    for i in range(1, len(frames_timestamps)):
        timestamp_1 = float(frames_timestamps[i-1])
        timestamp_2 = float(frames_timestamps[i])
        time_difference = timestamp_2 - timestamp_1
        time_diffs.append(time_difference)

        if time_difference < 114 or time_difference > 115:
            print(f"{frames_timestamps[i-1]}, {frames_timestamps[i]}: Time difference {time_difference} ms")

    if not time_diffs:
        print("No time differences found. Cannot calculate FPS.")
        return 0.0

    avg_time_diff_ms = sum(time_diffs) / len(time_diffs)
    fps = 1000.0 / avg_time_diff_ms  # Convert ms to seconds for FPS

    time_diff_counts = Counter(time_diffs)
    print(dict(time_diff_counts))

    print(f"Average time difference: {avg_time_diff_ms} ms, Estimated FPS: {fps}")
    return fps
    
    
if __name__ == "__main__":
    path_original_gray_images = "/data3/ina/Wasserhahn/05032026_Wasserhahn"
    path_frames_interpolated = "/data3/ina/Wasserhahn/05032026_Wasserhahn_interpolated"

    expected_timedifference = [114, 115] # in ms, an inaccuracy of 1 ms can be expected

    check_and_interpolate_frame_timing(path_original_gray_images, expected_timedifference, path_frames_interpolated)
    get_fps_of_frames_in_directory(path_frames_interpolated)
