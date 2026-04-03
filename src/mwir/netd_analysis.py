"""
NETD analysis helpers for MWIR ASCII thermal frames.

The usual workflow for a uniform black-body target is:
1. Load multiple thermal frames from `.asc` files.
2. Select a uniform ROI.
3. Compute the NETD according to the first method: https://optris.com/knowledge-library/netd-thermal-sensitivity/
The infrared camera observes a stable, uniform blackbody at a fixed temperature. 
At least 100 consecutive frames are recorded at the desired integration time and 
frame rate. For each pixel element, the temporal standard deviation of its temperature
 signal is computed. The NETD is then taken as the average of those pixel-wise 
 standard deviations. This procedure is equivalent to plotting a histogram of 
 the pixel values over time and taking its RMS width.

That temporal standard deviation is reported here as NETD in °C.
Multiply by 1000 to get mK.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

try:
	from .data_extraction import extract_thermal_data
except ImportError:  # pragma: no cover - allows running the file directly
	from data_extraction import extract_thermal_data


def load_thermal_frames(filepaths: Sequence[str | Path]) -> np.ndarray:
	"""Load multiple MWIR ASC files into a 3D array with shape (n_frames, 512, 640)."""
	frames = [extract_thermal_data(str(filepath)) for filepath in filepaths]
	if not frames:
		raise ValueError("No ASC files were provided.")
	return np.stack(frames, axis=0)


def discover_asc_files(directory: str | Path, recursive: bool = False) -> list[Path]:
	"""Return ASC files from a directory, sorted by name."""
	directory = Path(directory)
	if not directory.exists():
		raise FileNotFoundError(f"Directory does not exist: {directory}")

	pattern = "**/*.asc" if recursive else "*.asc"
	files = sorted(directory.glob(pattern))
	if not files:
		raise ValueError(f"No ASC files found in: {directory}")
	return files


def _normalize_roi(roi: tuple[int, int, int, int] | None, shape: tuple[int, int]) -> tuple[slice, slice]:
	"""Convert an ROI tuple `(x_min, x_max, y_min, y_max)` into array slices."""
	if roi is None:
		return slice(None), slice(None)

	x_min, x_max, y_min, y_max = roi
	height, width = shape

	if not (0 <= x_min <= x_max < width and 0 <= y_min <= y_max < height):
		raise ValueError(
			f"Invalid ROI {roi} for frame shape {shape}. Expected x within [0, {width - 1}] and y within [0, {height - 1}]."
		)

	return slice(y_min, y_max + 1), slice(x_min, x_max + 1)


def calculate_netd_from_pixelwise_std(
	thermal_frames: np.ndarray,
	roi: tuple[int, int, int, int] | None = None,
	ddof: int = 1,
) -> dict[str, float | int | tuple[int, int]]:
	"""
	Calculate NETD from a stack of thermal frames.

	The NETD definition used here is the average of the per-pixel temporal
	standard deviations across the selected image region.

	Args:
		thermal_frames: Array with shape (n_frames, 512, 640).
		roi: Optional `(x_min, x_max, y_min, y_max)` ROI in pixel coordinates.
		ddof: Delta degrees of freedom for the standard deviation.

	Returns:
		Dictionary with NETD in °C and mK, plus summary statistics.
	"""
	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected a 3D array (n_frames, height, width), got shape {thermal_frames.shape}")

	y_slice, x_slice = _normalize_roi(roi, thermal_frames.shape[1:])
	roi_frames = thermal_frames[:, y_slice, x_slice]
	pixelwise_std = roi_frames.std(axis=0, ddof=ddof)
	netd_value = float(pixelwise_std.mean())
	std_of_std = float(pixelwise_std.std(ddof=ddof))
	mean_temperature = float(roi_frames.mean())

	return {
		"netd_c": netd_value,
		"netd_mk": float(netd_value * 1000.0),
		"mean_temperature_c": mean_temperature,
		"frame_mean_std_c": std_of_std,
		"frame_count": int(thermal_frames.shape[0]),
		"roi_shape": tuple(int(v) for v in roi_frames.shape[1:]),
	}


def _format_histogram_summary(values: np.ndarray, frame_count: int, bins: int) -> str:
	"""Format summary statistics for histogram annotation."""
	values = np.asarray(values, dtype=np.float32)
	return (
		f"Frames: {frame_count}    "
		f"Bins: {bins}    "
		f"Mean: {values.mean():.3f} °C    "
		f"Std: {values.std(ddof=1):.3f} °C    "
		f"Min: {values.min():.3f} °C    "
		f"Max: {values.max():.3f} °C"
	)


def plot_all_pixels_histogram(
	thermal_frames: np.ndarray,
	bins: int = 500,
	title: str = "Histogram of All Pixel Values Across All Frames",
) -> None:
	"""Plot a histogram using all pixel temperatures from all frames."""
	all_values = np.asarray(thermal_frames, dtype=np.float32).ravel()
	frame_count = int(np.asarray(thermal_frames).shape[0])
	summary_text = _format_histogram_summary(all_values, frame_count, bins)

	fig, ax = plt.subplots(figsize=(12, 6))
	ax.hist(all_values, bins=bins, color="steelblue", edgecolor="black", alpha=0.85)
	ax.set_title(title)
	ax.set_xlabel("Temperature (°C)")
	ax.set_ylabel("Count")
	ax.grid(alpha=0.3)
	fig.subplots_adjust(bottom=0.22)
	fig.text(0.5, 0.03, summary_text, ha="center", va="bottom", fontsize=10)
	plt.show()


def format_netd_result(result: dict[str, float | int | tuple[int, int]]) -> str:
	"""Format a NETD result dictionary for human-readable output."""
	return (
		f"NETD: {result['netd_c']:.6f} °C ({result['netd_mk']:.2f} mK)\n"
		f"Mean temperature: {result['mean_temperature_c']:.6f} °C\n"
		f"Frames: {result['frame_count']}\n"
		f"ROI shape: {result['roi_shape']}"
	)


def main() -> None:
	path = Path("data/infratec_mwir/Black_body_Ascii")
	#path = Path("data/infratec_mwir/Julian_31_03_1_Ascii")
	recursive = False
	roi: tuple[int, int, int, int] | None = None
	plot_histogram = True

	if path.is_dir():
		filepaths = discover_asc_files(path, recursive=recursive)
	else:
		filepaths = [path]

	thermal_frames = load_thermal_frames(filepaths)
	result = calculate_netd_from_pixelwise_std(thermal_frames, roi=roi)
	print(format_netd_result(result))

	if plot_histogram:
		plot_all_pixels_histogram(thermal_frames)


if __name__ == "__main__":
	main()
