import os
import numpy as np
import cv2
from typing import List, Dict
import matplotlib.pyplot as plt
from utils_gray_files import sort_grayfiles_with_interpolations


def load_gray_frame(filepath: str, width: int = 160, height: int = 122, drop_telemetry: bool = True) -> np.ndarray:
    """Load a .gray frame (Y16, little-endian) and optionally strip telemetry lines."""
    raw = np.fromfile(filepath, dtype=np.uint16)
    if raw.size != width * height:
        raise ValueError(f"Unexpected frame size in {filepath}: got {raw.size}, expected {width*height}")
    frame = raw.reshape(height, width)
    if drop_telemetry and height > 120:
        frame = frame[:-2]  # remove last 2 lines
    return frame


def apply_roi_masks_to_gray_frames(
    gray_dir: str,
    roi_mask_log: List[Dict[str, np.ndarray]],
    output_npz: str = None,
) -> List[Dict[str, Dict[str, object]]]:
    """Apply ROI masks (from video detection) to original .gray frames to extract temperatures.

    Parameters:
        gray_dir: directory containing .gray frames (sorted lexicographically)
        roi_mask_log: list of per-frame ROI masks (from apply_facemesh_to_thermal_video)
        output_npz: optional path to save extracted data as npz

    Returns:
        roi_pixel_data: list (per frame) of dicts {roi_name: {"coords", "values", "mean_value"}}
    """
    _, gray_files = sort_grayfiles_with_interpolations(gray_dir)
    if not gray_files:
        raise ValueError(f"No .gray files found in {gray_dir}")

    if len(gray_files) < len(roi_mask_log):
        raise ValueError(f"Fewer .gray frames ({len(gray_files)}) than masks ({len(roi_mask_log)})")
    elif len(gray_files) > len(roi_mask_log):
        raise ValueError(f"More .gray frames ({len(gray_files)}) than masks ({len(roi_mask_log)})")

    roi_pixel_data: List[Dict[str, Dict[str, object]]] = []

    for idx, (fname, masks) in enumerate(zip(gray_files, roi_mask_log)):
        frame_path = os.path.join(gray_dir, fname)
        frame = load_gray_frame(frame_path)

        height, width = frame.shape
        frame_data: Dict[str, Dict[str, object]] = {}

        for roi_name, mask in masks.items():
            if mask.shape != frame.shape:
                raise ValueError(f"Mask shape {mask.shape} does not match frame shape {frame.shape} for ROI {roi_name}")

            ys, xs = np.where(mask == 255)
            coords = list(zip(xs.tolist(), ys.tolist()))
            values = frame[ys, xs].astype(np.float64)
            # TODO next step: convert raw values to temperature
            frame_data[roi_name] = {
                "coords": coords,
                "values": values.tolist(),
                "mean_value": float(np.mean(values)) if values.size > 0 else float('nan'),
            }

        roi_pixel_data.append(frame_data)

        if (idx + 1) % 100 == 0 or idx == len(gray_files) - 1:
            print(f"Processed {idx+1}/{len(gray_files)} frames")

    if output_npz:
        os.makedirs(os.path.dirname(output_npz), exist_ok=True)
        np.savez(output_npz, roi_pixel_data=roi_pixel_data)
        print(f"Saved ROI pixel data to {output_npz}")

    return roi_pixel_data


def plot_roi_temporal_means(roi_pixel_data: List[Dict[str, Dict[str, object]]], fps: float = None) -> None:
    """Plot temporal mean values for each ROI across frames.

    Parameters:
        roi_pixel_data: output from apply_roi_masks_to_gray_frames
        fps: optional frames per second for x-axis in seconds; if None, x-axis is frame index
    """
    if not roi_pixel_data:
        print("No ROI data to plot.")
        return

    roi_names = list(roi_pixel_data[0].keys())
    num_frames = len(roi_pixel_data)
    t = np.arange(num_frames) if fps is None else np.arange(num_frames) / fps

    plt.figure(figsize=(12, 6))
    for roi in roi_names:
        means = [frame.get(roi, {}).get("mean_value", np.nan) for frame in roi_pixel_data]
        plt.plot(t, means, label=roi)

    plt.xlabel("Frames" if fps is None else "Time (s)")
    plt.ylabel("Mean Temperature (raw units)")
    plt.title("ROI Mean Temperature Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage: update paths and provide roi_mask_log from detection stage
    gray_dir = "data/lepton/processed_lepton_data/29122025_interpolated/"
    output_npz = "data/saved_data/roi_pixel_data_from_gray.npz"

    path_pixels_log = "data/yang_et_al/roi_pixel_log.npy"
    roi_mask_log =np.load(path_pixels_log, allow_pickle=True)

    roi_pixel_data = apply_roi_masks_to_gray_frames(gray_dir=gray_dir, roi_mask_log=roi_mask_log, output_npz=output_npz)
    plot_roi_temporal_means(roi_pixel_data=roi_pixel_data, fps=8.77)