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


def calculate_spatial_noise_from_frame_std(
	thermal_frames: np.ndarray,
	roi: tuple[int, int, int, int] | None = None,
	ddof: int = 1,
) -> dict[str, float | int | tuple[int, int]]:
	"""
	Calculate spatial noise as the average of per-frame spatial standard deviations.

	For each frame, compute the standard deviation across all pixels (or ROI pixels),
	then average those per-frame standard deviations over time.
	"""
	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected a 3D array (n_frames, height, width), got shape {thermal_frames.shape}")

	y_slice, x_slice = _normalize_roi(roi, thermal_frames.shape[1:])
	roi_frames = thermal_frames[:, y_slice, x_slice]

	frame_spatial_std = roi_frames.std(axis=(1, 2), ddof=ddof)
	spatial_noise = float(frame_spatial_std.mean())

	return {
		"spatial_noise_c": spatial_noise,
		"spatial_noise_mk": float(spatial_noise * 1000.0),
		"frame_std_mean_c": spatial_noise,
		"frame_std_std_c": float(frame_spatial_std.std(ddof=ddof)),
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


def plot_single_frame_heatmap(
	thermal_frames: np.ndarray,
	frame_index: int = 0,
	title: str = "Thermal Heatmap (Single Frame)",
	cmap: str = "inferno",
) -> None:
	"""Plot one thermal frame as a heatmap."""
	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected a 3D array (n_frames, height, width), got shape {thermal_frames.shape}")

	frame_count = thermal_frames.shape[0]
	if not (0 <= frame_index < frame_count):
		raise ValueError(f"frame_index must be in [0, {frame_count - 1}], got {frame_index}")

	frame = thermal_frames[frame_index]
	fig, ax = plt.subplots(figsize=(12, 8))
	im = ax.imshow(frame, cmap=cmap, origin="upper", aspect="auto")
	ax.set_title(f"{title} (index={frame_index})")
	ax.set_xlabel("X Pixel")
	ax.set_ylabel("Y Pixel")
	fig.colorbar(im, ax=ax, label="Temperature (°C)")
	plt.tight_layout()
	plt.show()


def format_netd_result(result: dict[str, float | int | tuple[int, int]]) -> str:
	"""Format a NETD result dictionary for human-readable output."""
	return (
		f"NETD (temporal noise): {result['netd_c']:.6f} °C ({result['netd_mk']:.2f} mK)\n"
		f"Mean temperature: {result['mean_temperature_c']:.6f} °C\n"
	)


def format_spatial_noise_result(result: dict[str, float | int | tuple[int, int]]) -> str:
	"""Format a spatial-noise result dictionary for human-readable output."""
	return (
		f"Spatial noise (Average over all frames): {result['spatial_noise_c']:.6f} °C ({result['spatial_noise_mk']:.2f} mK)\n"
		f"Per-frame std mean: {result['frame_std_mean_c']:.6f} °C\n"
		f"Per-frame std std: {result['frame_std_std_c']:.6f} °C\n"
		f"Frames: {result['frame_count']}\n"
		f"ROI shape: {result['roi_shape']}"
	)


def main() -> None:
	path = Path("data/infratec_mwir/Black_body_Ascii")
	#path = Path("data/infratec_mwir/Julian_31_03_1_Ascii")
	recursive = False
	roi: tuple[int, int, int, int] | None = None # If None, whole image is used 
	plot_histogram = True
	plot_one_frame_heatmap = True
	heatmap_frame_index = 0

	if path.is_dir():
		filepaths = discover_asc_files(path, recursive=recursive)
	else:
		filepaths = [path]

	thermal_frames = load_thermal_frames(filepaths)
	result = calculate_netd_from_pixelwise_std(thermal_frames, roi=roi)
	print(format_netd_result(result))

	spatial_noise_result = calculate_spatial_noise_from_frame_std(thermal_frames, roi=roi)
	print(format_spatial_noise_result(spatial_noise_result))

	if plot_histogram:
		# Check whether noise is Gaussian distributed by plotting a histogram of all pixel values across all frames
		plot_all_pixels_histogram(thermal_frames)

	if plot_one_frame_heatmap:
		plot_single_frame_heatmap(thermal_frames, frame_index=heatmap_frame_index)


if __name__ == "__main__":
	main()
