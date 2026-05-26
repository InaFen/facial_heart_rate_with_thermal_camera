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
from scipy import stats, signal

try:
	from .data_extraction import extract_thermal_data
except ImportError:  # pragma: no cover - allows running the file directly
	from data_extraction import extract_thermal_data


def load_thermal_frames(filepaths: Sequence[str | Path]) -> np.ndarray:
	"""Load multiple MWIR ASC files into a 3D array with shape (n_frames, 512, 640)."""
	frames = [extract_thermal_data(str(filepath)) for filepath in filepaths]
	if not frames:
		raise ValueError("No ASC files were provided.")
	print("Frames loaded")
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

	# +1 because numpy slicing is exclusive at the end index
	return slice(y_min, y_max + 1), slice(x_min, x_max + 1)


def average_consecutive_frames(thermal_frames: np.ndarray, window_size: int) -> np.ndarray:
	"""Average consecutive frames in non-overlapping windows.
	Pixels are averaged over time, but not spatially, only temporally."""
	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected a 3D array (n_frames, height, width), got shape {thermal_frames.shape}")
	if window_size < 1:
		raise ValueError(f"window_size must be >= 1, got {window_size}")
	if window_size == 1:
		return thermal_frames

	frame_count = thermal_frames.shape[0]
	usable_count = (frame_count // window_size) * window_size
	if usable_count == 0:
		raise ValueError(
			f"window_size={window_size} is larger than available frame count ({frame_count})."
		)

	trimmed = thermal_frames[:usable_count]
	averaged = trimmed.reshape(-1, window_size, *thermal_frames.shape[1:]).mean(axis=1)
	print(f"Averaged {frame_count} frames temporally into {averaged.shape[0]} frames using window_size={window_size}.")
	return averaged


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
	roi_frames = thermal_frames[:, y_slice, x_slice] # Get frames in ROI
	pixelwise_std = roi_frames.std(axis=0, ddof=ddof) # Std dev over time for each pixel in ROI
	netd_value = float(pixelwise_std.mean()) # Average of per-pixel temporal standard deviations
	
	mean_temperature = float(roi_frames.mean())

	return {
		"netd_c": netd_value,
		"netd_mk": float(netd_value * 1000.0),
		"mean_temperature_c": mean_temperature,
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
	roi_frames = thermal_frames[:, y_slice, x_slice] # Apply ROI to all frames (if None use whole image)

	frame_spatial_std = roi_frames.std(axis=(1, 2), ddof=ddof) # Std dev across pixels for each frame
	spatial_noise = float(frame_spatial_std.mean())

	return {
		"spatial_noise_c": spatial_noise,
		"spatial_noise_mk": float(spatial_noise * 1000.0),
		"frame_std_std_c": float(frame_spatial_std.std(ddof=ddof)),
		"frame_count": int(thermal_frames.shape[0]),
		"roi_shape": tuple(int(v) for v in roi_frames.shape[1:]),
	}


def calculate_neighbor_temporal_correlation(
	thermal_frames: np.ndarray,
	roi: tuple[int, int, int, int] | None = None,
	ddof: int = 1,
) -> dict[str, object]:
	"""
	Calculate whether temporal noise in neighboring pixels is correlated (Pearson correlation).

	For each pixel, compute its temporal residuals by subtracting the per-pixel
	mean over time. Then compute the Pearson correlation with the four direct
	neighbors (up, down, left, right) and average those correlations per pixel.
	"""
	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected a 3D array (n_frames, height, width), got shape {thermal_frames.shape}")

	y_slice, x_slice = _normalize_roi(roi, thermal_frames.shape[1:])
	roi_frames = thermal_frames[:, y_slice, x_slice] # Apply ROI to all frames (if None use whole image)

	height, width = roi_frames.shape[1:]
	corr_sum = np.zeros((height, width), dtype=np.float32)
	corr_count = np.zeros((height, width), dtype=np.int32)

	def _pairwise_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
		"""
		Pairwise Pearson correlation computed over time
		"""
		a_centered = a - a.mean(axis=0)
		b_centered = b - b.mean(axis=0)
		cov = np.mean(a_centered * b_centered, axis=0) # covariance (how much vary together)
		std_a = a_centered.std(axis=0, ddof=ddof)
		std_b = b_centered.std(axis=0, ddof=ddof)
		denom = std_a * std_b # variability of each pixel's noise (how much vary individually)
		corr = np.zeros_like(cov, dtype=np.float32)
		np.divide(cov, denom, out=corr, where=denom > 0)
		return corr

	if height > 1:
		vertical_corr = _pairwise_correlation(roi_frames[:, :-1, :], roi_frames[:, 1:, :])
		corr_sum[:-1, :] += vertical_corr
		corr_sum[1:, :] += vertical_corr
		corr_count[:-1, :] += 1
		corr_count[1:, :] += 1

	if width > 1:
		horizontal_corr = _pairwise_correlation(roi_frames[:, :, :-1], roi_frames[:, :, 1:])
		corr_sum[:, :-1] += horizontal_corr
		corr_sum[:, 1:] += horizontal_corr
		corr_count[:, :-1] += 1
		corr_count[:, 1:] += 1

	neighbor_corr_map = np.full((height, width), np.nan, dtype=np.float32)
	valid = corr_count > 0
	neighbor_corr_map[valid] = corr_sum[valid] / corr_count[valid]

	valid_values = neighbor_corr_map[np.isfinite(neighbor_corr_map)]
	mean_corr = float(valid_values.mean()) if valid_values.size else float("nan")
	min_corr = float(valid_values.min()) if valid_values.size else float("nan")
	max_corr = float(valid_values.max()) if valid_values.size else float("nan")

	return {
		"neighbor_corr_map": neighbor_corr_map,
		"mean_neighbor_corr": mean_corr,
		"min_neighbor_corr": min_corr,
		"max_neighbor_corr": max_corr,
		"valid_pixel_count": int(valid_values.size),
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


def plot_all_pixels_qq(
	thermal_frames: np.ndarray,
	title: str = "Q-Q Plot of All Pixel Values",
) -> None:
	"""Plot a Q-Q plot to assess normality of pixel values across all frames."""
	all_values = np.asarray(thermal_frames, dtype=np.float32).ravel()
	fig, ax = plt.subplots(figsize=(6, 6))
	stats.probplot(all_values, dist="norm", plot=ax)
	ax.set_title(title)
	plt.tight_layout()
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


def plot_neighbor_correlation_heatmap(
	neighbor_corr_map: np.ndarray,
	title: str = "Spatial Correlation of Temporal Noise (How similar are neighboring pixels' noise patterns over time?)",
) -> None:
	"""Plot a heatmap of average correlation with neighboring pixels."""
	neighbor_corr_map = np.asarray(neighbor_corr_map, dtype=np.float32)
	valid_values = neighbor_corr_map[np.isfinite(neighbor_corr_map)]
	if valid_values.size:
		vmin = float(valid_values.min())
		vmax = float(valid_values.max())
		if np.isclose(vmin, vmax):
			padding = 0.01 if vmin == 0 else abs(vmin) * 0.01
			vmin -= padding
			vmax += padding
	else:
		vmin, vmax = -1.0, 1.0

	fig, ax = plt.subplots(figsize=(8, 6))
	im = ax.imshow(neighbor_corr_map, cmap="coolwarm", vmin=vmin, vmax=vmax, origin="upper", aspect="auto")
	ax.set_title(title)
	ax.set_xlabel("Pixel column (j)")
	ax.set_ylabel("Pixel row (i)")
	fig.colorbar(im, ax=ax, label=f"Average correlation with neighbors ({vmin:.3f} to {vmax:.3f})")
	plt.tight_layout()
	plt.show()


def plot_roi_mean_signal_fft(
	thermal_frames: np.ndarray,
	frame_rate_hz: float,
	roi: tuple[int, int, int, int] | None = None,
	title: str = "FFT of ROI Mean Temperature Signal",
) -> None:
	"""Plot FFT magnitude of the temporal mean temperature signal in an ROI."""
	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected a 3D array (n_frames, height, width), got shape {thermal_frames.shape}")
	if thermal_frames.shape[0] < 2:
		raise ValueError("At least 2 frames are required for FFT plotting.")
	if frame_rate_hz <= 0:
		raise ValueError(f"frame_rate_hz must be > 0, got {frame_rate_hz}")

	y_slice, x_slice = _normalize_roi(roi, thermal_frames.shape[1:])
	roi_frames = thermal_frames[:, y_slice, x_slice]
	roi_mean_signal = roi_frames.mean(axis=(1, 2))
	roi_mean_signal_centered = roi_mean_signal - roi_mean_signal.mean()

	frequency_axis_hz = np.fft.rfftfreq(roi_mean_signal_centered.size, d=1.0 / frame_rate_hz)
	fft_magnitude = np.abs(np.fft.rfft(roi_mean_signal_centered))

	fig, ax = plt.subplots(figsize=(10, 4))
	ax.plot(frequency_axis_hz, fft_magnitude, color="darkorange", linewidth=2)
	ax.set_title(title)
	ax.set_xlabel("Frequency (Hz)")
	ax.set_ylabel("Magnitude")
	ax.grid(alpha=0.3)
	plt.tight_layout()
	plt.show()


def plot_pixelwise_mean_psd(
	thermal_frames: np.ndarray,
	frame_rate_hz: float,
	title: str = "Mean PSD Across Pixels (Welch)",
) -> None:
	"""Plot mean PSD across all pixels to assess noise type.
	Takes every pixel’s time series (temperature vs. frame).
	Centers each series (subtracts its mean).
	Computes a PSD per pixel using Welch’s method.
	Averages the PSDs across all pixels to get one mean PSD curve.
	Plots that mean PSD vs frequency.
	"""
	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected a 3D array (n_frames, height, width), got shape {thermal_frames.shape}")
	if thermal_frames.shape[0] < 2:
		raise ValueError("At least 2 frames are required for PSD plotting.")
	if frame_rate_hz <= 0:
		raise ValueError(f"frame_rate_hz must be > 0, got {frame_rate_hz}")

	frame_count = thermal_frames.shape[0]
	nperseg = min(256, frame_count)

	series = thermal_frames.reshape(frame_count, -1)
	series = series - series.mean(axis=0, keepdims=True)

	freqs, psd = signal.welch(series, fs=frame_rate_hz, nperseg=nperseg, axis=0)
	mean_psd = psd.mean(axis=1)

	fig, ax = plt.subplots(figsize=(10, 4))
	ax.plot(freqs, mean_psd, color="slateblue", linewidth=2)
	ax.set_title(title)
	ax.set_xlabel("Frequency (Hz)")
	ax.set_ylabel("Power")
	ax.grid(alpha=0.3)
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
		f"Per-frame std std (so std of spatial noise): {result['frame_std_std_c']:.6f} °C\n"
		f"Frames: {result['frame_count']}\n"
		f"ROI shape: {result['roi_shape']}"
	)


def format_neighbor_correlation_result(result: dict[str, object]) -> str:
	"""Format a neighbor-correlation result dictionary for human-readable output."""
	return (
		f"Mean neighbor correlation: {result['mean_neighbor_corr']:.6f}\n"
		f"Min neighbor correlation: {result['min_neighbor_corr']:.6f}\n"
		f"Max neighbor correlation: {result['max_neighbor_corr']:.6f}\n"
		f"Valid pixels: {result['valid_pixel_count']}\n"
		f"Frames: {result['frame_count']}\n"
		f"ROI shape: {result['roi_shape']}"
	)


def main() -> None:
	"""Run a NETD/noise analysis pipeline on a folder of ASC frames."""
	#path = Path("data/infratec_mwir/Black_body_Ascii")
	path = Path("/mnt/wwn-0x50000398325009f6-part1/Masterarbeit/10_secs_NETD_Ascii")
    
	recursive = False
	roi: tuple[int, int, int, int] | None = None # If None, whole image is used 
	plot_histogram = True
	plot_qq = True
	plot_one_frame_heatmap = True
	plot_neighbor_correlation_heatmap_enabled = True
	plot_roi_fft = True
	plot_psd = True
	heatmap_frame_index = 0
	frame_rate_hz = 100.0
	frame_average_window = 10
	effective_frame_rate_hz = frame_rate_hz / frame_average_window

	if path.is_dir():
		filepaths = discover_asc_files(path, recursive=recursive)
	else:
		filepaths = [path]

	# Get images
	thermal_frames = load_thermal_frames(filepaths)
	#thermal_frames_averaged = average_consecutive_frames(thermal_frames, window_size=frame_average_window)

	# Calculate NETD (use raw frames to avoid underestimating temporal noise)
	result = calculate_netd_from_pixelwise_std(thermal_frames, roi=roi)
	print(format_netd_result(result))

	# Calculate spatial noise (how much differs pixel values in one frame)
	spatial_noise_result = calculate_spatial_noise_from_frame_std(thermal_frames, roi=roi)
	print(format_spatial_noise_result(spatial_noise_result))

	# Calculate how much temporal noise is correlated between neighboring pixels
	neighbor_correlation_result = calculate_neighbor_temporal_correlation(thermal_frames, roi=roi)
	print(format_neighbor_correlation_result(neighbor_correlation_result))

	if plot_histogram:
		# Check whether noise is Gaussian distributed by plotting a histogram of all pixel values across all frames
		# Disadvantage of using histogram for this: Bin size can heavily affect visual impression
		plot_all_pixels_histogram(thermal_frames)

	if plot_qq:
		plot_all_pixels_qq(thermal_frames)

	if plot_one_frame_heatmap:
		plot_single_frame_heatmap(thermal_frames, frame_index=heatmap_frame_index)

	if plot_neighbor_correlation_heatmap_enabled:
		plot_neighbor_correlation_heatmap(neighbor_correlation_result["neighbor_corr_map"])

	if plot_roi_fft:
		plot_roi_mean_signal_fft(thermal_frames, frame_rate_hz=effective_frame_rate_hz, roi=roi)

	if plot_psd:
		plot_pixelwise_mean_psd(thermal_frames, frame_rate_hz=frame_rate_hz)


if __name__ == "__main__":
	main()


# TODO add what kind of noise anaylsis
# TODO in which frequencies dominant for bandpass filter
# TODO apply to different cameras
