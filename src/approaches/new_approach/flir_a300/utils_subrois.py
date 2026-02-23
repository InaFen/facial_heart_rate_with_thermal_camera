from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2

from utils_data_extraction import extract_frame_number_from_csv, read_thermal_csv


def roi_pixels_from_bounds(roi_bounds: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    x_min, x_max, y_min, y_max = roi_bounds
    roi_pixels = []
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            roi_pixels.append((x, y))
    return roi_pixels


def generate_subroi_bounds(
    roi_bounds: tuple[int, int, int, int],
    subroi_width: int | None = None,
    subroi_height: int | None = None,
    step_x: int | None = None,
    step_y: int | None = None,
) -> list[tuple[int, int, int, int]]:
    x_min, x_max, y_min, y_max = roi_bounds
    roi_width = x_max - x_min + 1
    roi_height = y_max - y_min + 1

    if subroi_width is None:
        subroi_width = max(3, roi_width // 2)
    if subroi_height is None:
        subroi_height = max(3, roi_height // 2)

    subroi_width = min(subroi_width, roi_width)
    subroi_height = min(subroi_height, roi_height)

    if step_x is None:
        step_x = max(1, subroi_width // 2)
    if step_y is None:
        step_y = max(1, subroi_height // 2)

    bounds_list = []
    for y in range(y_min, y_max - subroi_height + 2, step_y):
        for x in range(x_min, x_max - subroi_width + 2, step_x):
            bounds_list.append((x, x + subroi_width - 1, y, y + subroi_height - 1))

    if not bounds_list:
        bounds_list.append((x_min, x_max, y_min, y_max))
    return bounds_list


def generate_subroi_bounds_multi(
    roi_bounds: tuple[int, int, int, int],
    sizes: list[tuple[int, int]] | None = None,
    size_fractions: list[float] | None = None,
    step_x: int | None = None,
    step_y: int | None = None,
) -> list[tuple[int, int, int, int]]:
    x_min, x_max, y_min, y_max = roi_bounds
    roi_width = x_max - x_min + 1
    roi_height = y_max - y_min + 1

    if sizes is None:
        if size_fractions is None:
            size_fractions = [0.3, 0.5, 0.7]
        sizes = []
        for frac in size_fractions:
            width = max(3, int(round(roi_width * frac)))
            height = max(3, int(round(roi_height * frac)))
            sizes.append((min(width, roi_width), min(height, roi_height)))

    unique_sizes = sorted(set(sizes))
    all_bounds = []
    for width, height in unique_sizes:
        all_bounds.extend(
            generate_subroi_bounds(
                roi_bounds,
                subroi_width=width,
                subroi_height=height,
                step_x=step_x,
                step_y=step_y,
            )
        )
    return all_bounds


def compute_mean_temperatures_for_roi_bounds(
    frame_dir: str | Path,
    roi_bounds_list: list[tuple[int, int, int, int]],
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> tuple[list[int], list[list[float]]]:
    frame_dir = Path(frame_dir)
    csv_files = sorted(frame_dir.glob("Rec-temp-*.csv"), key=extract_frame_number_from_csv)

    if not csv_files:
        raise ValueError(f"No thermal frames found in {frame_dir}")

    frame_numbers: list[int] = []
    mean_temperatures_list = [[] for _ in roi_bounds_list]

    for csv_file in csv_files:
        try:
            frame_num = extract_frame_number_from_csv(csv_file)

            if start_frame is not None and frame_num < start_frame:
                continue
            if end_frame is not None and frame_num >= end_frame:
                continue

            frame_data = read_thermal_csv(csv_file)
            frame_numbers.append(frame_num)

            for idx, (x_min, x_max, y_min, y_max) in enumerate(roi_bounds_list):
                roi_region = frame_data[y_min:y_max + 1, x_min:x_max + 1]
                mean_temperatures_list[idx].append(float(np.mean(roi_region)))

        except Exception as e:
            print(f"Warning: Could not process frame from {csv_file.name}: {e}")
            continue

    if not frame_numbers:
        raise ValueError("No valid frames processed for sub-ROI selection")

    return frame_numbers, mean_temperatures_list


def select_subroi_by_frequency_power(
    mean_temperatures_list: list[list[float]],
    fps: float,
    target_frequency: float = 1.2,
) -> tuple[int, list[float]]:
    powers = []
    for mean_temps in mean_temperatures_list:
        if len(mean_temps) < 2:
            powers.append(0.0)
            continue
        signal = np.asarray(mean_temps, dtype=float)
        signal = signal - np.mean(signal)
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)
        fft_mag = np.abs(np.fft.rfft(signal))
        idx = int(np.argmin(np.abs(freqs - target_frequency)))
        powers.append(float(fft_mag[idx]))

    best_index = int(np.argmax(powers)) if powers else 0
    return best_index, powers


def save_subroi_candidate_previews(
    frame_dir: str | Path,
    roi_bounds_list: list[tuple[int, int, int, int]],
    output_dir: str | Path,
    base_name: str,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> Path:
    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(frame_dir.glob("Rec-temp-*.csv"), key=extract_frame_number_from_csv)
    if not csv_files:
        raise ValueError(f"No thermal frames found in {frame_dir}")

    selected_csv = None
    for csv_file in csv_files:
        frame_num = extract_frame_number_from_csv(csv_file)
        if start_frame is not None and frame_num < start_frame:
            continue
        if end_frame is not None and frame_num >= end_frame:
            continue
        selected_csv = csv_file
        break

    if selected_csv is None:
        raise ValueError("No frame found in the requested range for sub-ROI previews")

    frame_data = read_thermal_csv(selected_csv)
    if frame_data.size == 0:
        raise ValueError("Selected frame is empty for sub-ROI previews")

    frame_min = float(np.min(frame_data))
    frame_max = float(np.max(frame_data))
    if frame_max == frame_min:
        frame_normalized = np.zeros_like(frame_data, dtype=np.uint8)
    else:
        frame_normalized = ((frame_data - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)

    frame_bgr = cv2.cvtColor(frame_normalized, cv2.COLOR_GRAY2BGR)

    for idx, (x_min, x_max, y_min, y_max) in enumerate(roi_bounds_list):
        preview = frame_bgr.copy()
        cv2.rectangle(preview, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
        filename = f"{base_name}_subroi_{idx:04d}_x{x_min}-{x_max}_y{y_min}-{y_max}.png"
        cv2.imwrite(str(output_dir / filename), preview)

    return output_dir
