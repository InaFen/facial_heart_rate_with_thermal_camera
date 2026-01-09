
import os

def sort_grayfiles_with_interpolations(path_gray_images: str):
    """Takes a directory with .gray files named with timestamps and returns a sorted list of filenames.
    Parameters:
        path_gray_images: directory containing .gray files
    Returns:
        sorted_frames: list of sorted timestamps as integers
        sorted_frames_original_name: list of sorted original filenames
    """

    frames_original_name = [f for f in os.listdir(path_gray_images) if f.endswith('.gray')]
    idx_frames = list(range(len(frames_original_name)))

    # Remove prefixes to allow correct sorting
    frames_timestamps_interpolated_removed = [f.replace('_interpolated', '') for f in os.listdir(path_gray_images) if f.endswith('.gray')]
    frames_timestamps_redundant_zeros_removed = [ts.lstrip('0') if ts.lstrip('0') != '' else '0' for ts in frames_timestamps_interpolated_removed]
    
    # Sort frames and indexes based on timestamps
    sorted_frames = sorted([int(f.split('.')[0]) for f in frames_timestamps_redundant_zeros_removed])
    # Sort indexes based on correct order (takes "cleaned" timestamps and sorts the indexes accordingly to how the timestamps would be sorted)
    idx_frames = sorted(idx_frames,key=lambda i: int(frames_timestamps_redundant_zeros_removed[i].split('.')[0]))

    # Get sorted original filenames
    sorted_frames_original_name = [frames_original_name[idx] for idx in idx_frames]
    
    return sorted_frames, sorted_frames_original_name
    
