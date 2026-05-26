import numpy as np

def read_lepton35_gray(filename, width=160, height=120):
    """
    Reads a FLIR Lepton 3.5 .gray file and returns a 3D numpy array of shape (num_frames, height, width)
    
    Parameters:
        filename (str): Path to the .gray file
        width (int): Frame width (default 160)
        height (int): Frame height (default 120)
    
    Returns:
        frames (np.ndarray): Array of shape (num_frames, height, width), dtype=np.uint16
    """
    # Read the full file as uint16
    raw = np.fromfile(filename, dtype=np.uint16)
    total_pixels = len(raw)
    
    # Number of pixels per frame
    frame_pixels = width * height
    
    # Calculate number of full frames in the file
    num_frames = total_pixels // frame_pixels
    if num_frames == 0:
        raise ValueError("File is too small to contain even one full frame.")
    
    # Trim any extra padding/partial frame at the end
    raw = raw[:num_frames * frame_pixels]
    
    # Reshape into (num_frames, height, width)
    frames = raw.reshape((num_frames, height, width))
    
    return frames

