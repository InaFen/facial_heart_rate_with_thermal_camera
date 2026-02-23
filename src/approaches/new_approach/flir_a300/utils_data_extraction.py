from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
import cv2


def extract_frame_number_from_csv(file_path: Path) -> int:
    """Extract frame number from CSV filename."""
    match = re.search(r"_(\d+)\.csv$", file_path.name)
    if not match:
        raise ValueError(f"Could not extract frame number from filename: {file_path.name}")
    return int(match.group(1))


def define_roi_pixel_from_first_frame(csv_file_path: str | Path, fps: float, roi_width: int | None = 10, roi_height: int | None = 10, output_video_dir: str | Path | None = None) -> dict:
    """
    Read a thermal CSV file and allow manual ROI selection on the first frame.
    If roi_width and roi_height are specified, the ROI will have fixed dimensions.
    Otherwise, the user can select any size ROI.
    After selection, creates a video showing the ROI on all frames.
    
    Args:
        csv_file_path: Path to the CSV file containing temperature data
        fps: Frames per second for the output video
        roi_width: Optional fixed width of the ROI in pixels. If None, user can select any width.
        roi_height: Optional fixed height of the ROI in pixels. If None, user can select any height.
        output_video_dir: Directory to save the output video. If None, saves to the same directory as the CSV file.
        
    Returns:
        A dictionary containing:
        - 'roi_pixels': List of (x, y) pixel coordinates within the selected ROI
        - 'roi_bounds': Tuple of (x_min, x_max, y_min, y_max) of the ROI
        - 'video_path': Path to the saved video file
    """
    csv_file_path = Path(csv_file_path)
    
    # Read the CSV file
    temperature_data = read_thermal_csv(csv_file_path)
    
    # Visualize and get ROI selection (closes window after selection)
    roi_bounds, roi_pixels = visualize_and_select_roi(temperature_data, roi_width, roi_height)
    
    # Read all frames from the directory
    frame_dir = csv_file_path.parent
    all_frames = read_all_thermal_frames(frame_dir)
    
    # Set output directory
    if output_video_dir is None:
        output_video_dir = frame_dir
    else:
        output_video_dir = Path(output_video_dir)
        output_video_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and save video with ROI overlay
    video_path = create_roi_video(all_frames, roi_bounds, output_video_dir, csv_file_path.stem, fps)
    
    return {
        'roi_pixels': roi_pixels,
        'roi_bounds': roi_bounds,
        'video_path': video_path
    }


def read_thermal_csv(csv_file_path: str | Path) -> np.ndarray:
    """
    Read a thermal CSV file and return a 2D numpy array of temperature values.
    
    The CSV file format is expected to have:
    - Header lines with metadata (Filename, Units, Time, FrameNumber, Preset)
    - Data rows containing comma-separated temperature values
    - Each row represents a row of pixels in the thermal image
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        2D numpy array of temperature values
    """
    csv_file_path = Path(csv_file_path)
    
    temperature_rows = []
    
    with open(csv_file_path, 'r') as f:
        reader = csv.reader(f)
        skip_header = True
        
        for row in reader:
            # Skip header rows (those that don't start with a number)
            if skip_header:
                try:
                    # Try to convert first element to float
                    float(row[0])
                    skip_header = False
                except (ValueError, IndexError):
                    continue
            
            if not skip_header:
                # Convert row to floats
                try:
                    temp_row = [float(val) for val in row]
                    temperature_rows.append(temp_row)
                except ValueError:
                    # Skip rows that can't be converted to floats
                    continue
    
    # Convert to numpy array
    temperature_data = np.array(temperature_rows)
    
    return temperature_data


def visualize_and_select_roi(temperature_data: np.ndarray, roi_width: int | None = None, roi_height: int | None = None) -> tuple:
    """
    Visualize the thermal image and allow manual ROI selection.
    If roi_width and roi_height are provided, enforces fixed dimensions via click.
    Otherwise, allows free selection via click-and-drag rectangle.
    Window closes automatically after one ROI selection.
    
    Args:
        temperature_data: 2D numpy array of temperature values
        roi_width: Optional fixed width of the ROI in pixels
        roi_height: Optional fixed height of the ROI in pixels
        
    Returns:
        Tuple of (roi_bounds, roi_pixels) where:
        - roi_bounds: Tuple of (x_min, x_max, y_min, y_max)
        - roi_pixels: List of (x, y) pixel coordinates within the ROI
    """
    # Normalize temperature data for visualization
    temp_min = np.min(temperature_data)
    temp_max = np.max(temperature_data)
    temp_normalized = (temperature_data - temp_min) / (temp_max - temp_min)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Display thermal image
    im = ax.imshow(temp_normalized, cmap='hot', origin='upper', aspect='auto')
    
    # Determine title based on whether dimensions are fixed
    if roi_width is not None and roi_height is not None:
        title = f'Thermal Image - Select ROI Position\n(Click to set top-left corner of {roi_width}x{roi_height} ROI)'
    else:
        title = 'Thermal Image - Select ROI with Rectangle\n(Click and drag to select)'
    
    ax.set_title(title)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'Temperature (normalized from {temp_min:.2f}°C to {temp_max:.2f}°C)')
    
    # Variable to store the selection
    roi_bounds = {'coords': None}
    rect_patch = {'obj': None}
    
    if roi_width is not None and roi_height is not None:
        # Fixed dimensions mode - use click selection
        def on_click(event):
            """Callback for mouse click to set ROI position."""
            if event.xdata is None or event.ydata is None:
                return
            
            x_click = int(event.xdata)
            y_click = int(event.ydata)
            
            # Calculate bounds with fixed dimensions
            x_min = max(0, x_click)
            y_min = max(0, y_click)
            x_max = min(temperature_data.shape[1] - 1, x_click + roi_width - 1)
            y_max = min(temperature_data.shape[0] - 1, y_click + roi_height - 1)
            
            # Adjust if ROI goes beyond boundaries
            if x_max - x_min + 1 < roi_width:
                x_min = max(0, x_max - roi_width + 1)
            if y_max - y_min + 1 < roi_height:
                y_min = max(0, y_max - roi_height + 1)
            
            roi_bounds['coords'] = (x_min, x_max, y_min, y_max)
            
            # Remove previous rectangle if exists
            if rect_patch['obj'] is not None:
                rect_patch['obj'].remove()
            
            # Draw rectangle on the image to show selection
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             linewidth=2, edgecolor='cyan', facecolor='none')
            rect_patch['obj'] = rect
            ax.add_patch(rect)
            fig.canvas.draw_idle()
            
            print(f"ROI selected: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}] (size: {x_max - x_min + 1}x{y_max - y_min + 1})")
            
            # Close the window after selection
            plt.close(fig)
        
        # Connect mouse click event
        fig.canvas.mpl_connect('button_press_event', on_click)
    else:
        # Free selection mode - use rectangle selector
        def on_select(eclick, erelease):
            """Callback for rectangle selection."""
            x_min = int(min(eclick.xdata, erelease.xdata))
            x_max = int(max(eclick.xdata, erelease.xdata))
            y_min = int(min(eclick.ydata, erelease.ydata))
            y_max = int(max(eclick.ydata, erelease.ydata))
            
            roi_bounds['coords'] = (x_min, x_max, y_min, y_max)
            
            # Draw rectangle on the image to show selection
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             linewidth=2, edgecolor='cyan', facecolor='none')
            ax.add_patch(rect)
            fig.canvas.draw_idle()
            
            print(f"ROI selected: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}] (size: {x_max - x_min + 1}x{y_max - y_min + 1})")
            
            # Close the window after selection
            plt.close(fig)
        
        # Create rectangle selector
        rect_selector = RectangleSelector(
            ax,
            on_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True
        )
        
        rect_selector.set_active(True)
    
    # Show the plot
    plt.show()
    
    # Get the selected coordinates
    if roi_bounds['coords'] is None:
        raise ValueError("No ROI was selected. Please select a region of interest.")
    
    x_min, x_max, y_min, y_max = roi_bounds['coords']
    
    # Generate list of pixels within the ROI
    roi_pixels = []
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            # Check if within image bounds
            if 0 <= y < temperature_data.shape[0] and 0 <= x < temperature_data.shape[1]:
                roi_pixels.append((x, y))
    
    return (x_min, x_max, y_min, y_max), roi_pixels


def read_all_thermal_frames(frame_dir: str | Path) -> list:
    """
    Read all thermal CSV files from a directory and return list of temperature data arrays.
    
    Args:
        frame_dir: Directory containing thermal CSV files
        
    Returns:
        List of 2D numpy arrays, one for each frame
    """
    frame_dir = Path(frame_dir)
    csv_files = sorted(frame_dir.glob('Rec-temp-*.csv'), key=extract_frame_number_from_csv)
    
    frames = []
    for csv_file in csv_files:
        try:
            frame_data = read_thermal_csv(csv_file)
            if frame_data.size > 0:  # Only add non-empty frames
                frames.append(frame_data)
        except Exception as e:
            print(f"Warning: Could not read frame from {csv_file.name}: {e}")
            continue
    
    if not frames:
        raise ValueError(f"No valid thermal frames found in {frame_dir}")
    
    print(f"Loaded {len(frames)} thermal frames")
    return frames


def create_roi_video(frames: list, roi_bounds: tuple, output_dir: str | Path, base_name: str, fps: float) -> Path:
    """
    Create a video showing ROI rectangle overlay on all thermal frames.
    
    Args:
        frames: List of 2D numpy arrays (thermal frames)
        roi_bounds: Tuple of (x_min, x_max, y_min, y_max)
        output_dir: Directory to save the video
        base_name: Base name for the output video file
        
    Returns:
        Path to the saved video file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not frames:
        raise ValueError("No frames provided to create video")
    
    x_min, x_max, y_min, y_max = roi_bounds
    
    # Get frame dimensions
    height, width = frames[0].shape
    
    # Define video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    output_path = output_dir / f"{base_name}_roi_visualization.mp4"
    
    # Create video writer
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Normalize frames for visualization
    global_min = min(np.min(frame) for frame in frames)
    global_max = max(np.max(frame) for frame in frames)
    
    print(f"Creating video: {output_path}")
    print(f"Frame size: {width}x{height}, FPS: {fps}, Total frames: {len(frames)}")
    
    for frame_idx, frame in enumerate(frames):
        # Normalize frame to 0-255
        frame_normalized = ((frame - global_min) / (global_max - global_min) * 255).astype(np.uint8)
        
        # Convert grayscale to BGR for color frame
        frame_bgr = cv2.cvtColor(frame_normalized, cv2.COLOR_GRAY2BGR)
        
        # Draw ROI rectangle in cyan
        cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
        
        # Add frame number text
        cv2.putText(frame_bgr, f"Frame: {frame_idx + 1}/{len(frames)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Write frame to video
        out.write(frame_bgr)
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  Processed {frame_idx + 1}/{len(frames)} frames")
    
    # Release the video writer
    out.release()
    
    print(f"Video saved successfully: {output_path}")
    return output_path
