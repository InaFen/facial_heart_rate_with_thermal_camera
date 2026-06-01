"""
NETD analysis helpers for MWIR ASCII thermal frames and Lepton .gray sequences.

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

try:
	from src.approaches.new_approach.utils.utils_gray_files import sort_grayfiles_with_interpolations
	from src.video_processing_methods.gray_file_methods import read_lepton35_gray
except ImportError:  # pragma: no cover - allows running the file directly
	import sys
	project_root = Path(__file__).resolve().parents[3]
	sys.path.insert(0, str(project_root))
	from src.approaches.new_approach.utils.utils_gray_files import sort_grayfiles_with_interpolations
	from src.video_processing_methods.gray_file_methods import read_lepton35_gray


def load_thermal_frames(filepaths: Sequence[str | Path]) -> np.ndarray:
	"""Load MWIR ASC files and stack them into a 3D array."""
	frames = [extract_thermal_data(str(filepath)) for filepath in filepaths]
	if not frames:
		raise ValueError("No ASC files were provided.")
	print("Frames loaded")
	return np.stack(frames, axis=0)


def extract_thermal_data_vario(filepath: str | Path) -> np.ndarray:
	"""Load an Infratec Vario ASC file into a 2D array."""
	path = Path(filepath)
	text = path.read_text(encoding="latin-1", errors="replace")
	lines = text.splitlines()

	data_start = None
	for index, line in enumerate(lines):
		if line.strip() == "[Data]":
			data_start = index + 1
			break

	if data_start is None:
		raise ValueError(f"Missing [Data] section in {path}")

	data_lines = [line for line in lines[data_start:] if line.strip()]
	frame_height = 480
	frame_width = 640
	if len(data_lines) < frame_height:
		raise ValueError(
			f"Not enough data rows in {path}: expected {frame_height}, got {len(data_lines)}"
		)

	frame = np.empty((frame_height, frame_width), dtype=np.float32)
	for row in range(frame_height):
		line = data_lines[row].replace(",", ".")
		values = np.fromstring(line, sep="\t", dtype=np.float32)

		if values.size != frame_width:
			values = np.fromstring(line, sep=";", dtype=np.float32)

		if values.size != frame_width:
			values = np.fromstring(line, sep=" ", dtype=np.float32)

		if values.size != frame_width:
			raise ValueError(
				f"Row {row} in {path} has {values.size} columns, expected {frame_width}"
			)

		frame[row] = values

	return frame


def load_thermal_frames_asc(filepaths: Sequence[str | Path], camera: str) -> np.ndarray:
	"""Load ASC files for MWIR or Vario cameras and stack them into a 3D array."""
	if camera == "mwir":
		frames = [extract_thermal_data(str(filepath)) for filepath in filepaths]
	elif camera == "vario":
		frames = [extract_thermal_data_vario(filepath) for filepath in filepaths]
	else:
		raise ValueError(f"Unsupported camera for ASC loading: {camera}")

	if not frames:
		raise ValueError("No ASC files were provided.")
	print("Frames loaded")
	return np.stack(frames, axis=0)


def discover_gray_files(directory: str | Path) -> list[Path]:
	"""Return .gray files from a directory, sorted by timestamp (handles _interpolated)."""
	directory = Path(directory)
	if not directory.exists():
		raise FileNotFoundError(f"Directory does not exist: {directory}")

	_, sorted_frames_original_name = sort_grayfiles_with_interpolations(str(directory))
	if not sorted_frames_original_name:
		raise ValueError(f"No .gray files found in: {directory}")
	return [directory / name for name in sorted_frames_original_name]


def load_gray_frames(filepaths: Sequence[str | Path]) -> np.ndarray:
	"""Load Lepton .gray files and stack them into a 3D array."""
	frames: list[np.ndarray] = []
	for filepath in filepaths:
		gray_frames = read_lepton35_gray(str(filepath))
		if gray_frames.ndim == 3:
			for frame in gray_frames:
				frames.append(frame)
		else:
			frames.append(gray_frames)
	if not frames:
		raise ValueError("No .gray frames were loaded.")
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
	bins: int = 250,
	title: str = "Histogram of All Pixel Values Across All Frames",
	save_path: str | Path | None = None,
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
	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(str(save_path), dpi=200)
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
	save_path: str | Path | None = None,
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
	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(str(save_path), dpi=200)
	plt.show()


def plot_neighbor_correlation_heatmap(
	neighbor_corr_map: np.ndarray,
	title: str = "Spatial Correlation of Temporal Noise (How similar are neighboring pixels' noise patterns over time?)",
	save_path: str | Path | None = None,
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
	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(str(save_path), dpi=200)
	plt.show()


def plot_roi_mean_signal_fft(
	thermal_frames: np.ndarray,
	frame_rate_hz: float,
	roi: tuple[int, int, int, int] | None = None,
	title: str = "FFT of ROI Mean Temperature Signal",
	save_path: str | Path | None = None,
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
	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(str(save_path), dpi=200)
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

	freqs, mean_psd = get_pixelwise_mean_psd(thermal_frames, frame_rate_hz)

	fig, ax = plt.subplots(figsize=(10, 4))
	ax.plot(freqs, mean_psd, color="slateblue", linewidth=2)
	ax.set_title(title)
	ax.set_xlabel("Frequency (Hz)")
	ax.set_ylabel("Power")
	ax.grid(alpha=0.3)
	plt.tight_layout()
	plt.show()


def get_pixelwise_mean_psd(
	thermal_frames: np.ndarray,
	frame_rate_hz: float,
	nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Compute the mean PSD across pixels (Welch) and return frequency axis and mean PSD.

	Returns (freqs, mean_psd).
	"""
	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected a 3D array (n_frames, height, width), got shape {thermal_frames.shape}")
	if thermal_frames.shape[0] < 2:
		raise ValueError("At least 2 frames are required for PSD computation.")
	if frame_rate_hz <= 0:
		raise ValueError(f"frame_rate_hz must be > 0, got {frame_rate_hz}")

	frame_count = thermal_frames.shape[0]
	if nperseg is None:
		nperseg = min(256, frame_count)

	series = thermal_frames.reshape(frame_count, -1)
	series = series - series.mean(axis=0, keepdims=True)

	freqs, psd = signal.welch(series, fs=frame_rate_hz, nperseg=nperseg, axis=0)
	mean_psd = psd.mean(axis=1)
	return freqs, mean_psd


def plot_compare_pixelwise_mean_psd(
	paths: Sequence[str | Path],
	cameras: Sequence[str],
	frame_rates_hz: Sequence[float],
	labels: Sequence[str] | None = None,
	roi: tuple[int, int, int, int] | None = None,
) -> None:
	"""
	Compute and plot mean PSDs for multiple recordings on a single plot.

	- `paths`: sequence of directories or single-file paths containing frames (ASC or .gray)
	- `cameras`: sequence of camera types matching `paths` ("mwir", "vario", "lepton")
	- `frame_rates_hz`: sequence of frame rates for each recording
	- `labels`: optional legend labels; if None, camera names are used
	- `roi`: optional ROI applied when loading frames
	"""
	if len(paths) != len(cameras) or len(paths) != len(frame_rates_hz):
		raise ValueError("paths, cameras and frame_rates_hz must have the same length")

	if labels is None:
		labels = [str(c) for c in cameras]

	# Load all frame stacks and determine a common nperseg (min frame count)
	stacks: list[np.ndarray] = []
	frame_counts: list[int] = []
	for p, cam in zip(paths, cameras):
		p = Path(p)
		if p.is_dir():
			if cam == "lepton":
				filepaths = discover_gray_files(p)
				frames = load_gray_frames(filepaths)
				frames = frames / 100.0 - 273.15
			else:
				filepaths = discover_asc_files(p)
				frames = load_thermal_frames_asc(filepaths, camera=cam)
		else:
			# single file
			if cam == "lepton":
				frames = load_gray_frames([p])
				frames = frames / 100.0 - 273.15
			else:
				frames = load_thermal_frames_asc([p], camera=cam)

		if roi is not None:
			y_slice, x_slice = _normalize_roi(roi, frames.shape[1:])
			frames = frames[:, y_slice, x_slice]

		stacks.append(np.asarray(frames, dtype=np.float32))
		frame_counts.append(frames.shape[0])

	if not stacks:
		raise ValueError("No recordings were loaded for comparison.")

	common_nperseg = min(256, min(frame_counts))

	# Compute PSDs
	freqs_list: list[np.ndarray] = []
	mean_psds: list[np.ndarray] = []
	for frames, fs in zip(stacks, frame_rates_hz):
		freqs, mean_psd = get_pixelwise_mean_psd(frames, frame_rate_hz=fs, nperseg=common_nperseg)
		freqs_list.append(freqs)
		mean_psds.append(mean_psd)

	# For plotting, use the frequency axis of the first entry
	freqs_plot = freqs_list[0]

	fig, ax = plt.subplots(figsize=(10, 4))
	for mean_psd, label in zip(mean_psds, labels):
		ax.plot(freqs_plot, mean_psd, linewidth=2, label=label)

	ax.set_title("Comparison of Mean PSDs Across Recordings")
	ax.set_xlabel("Frequency (Hz)")
	ax.set_ylabel("Power")
	# Set y-axis max to the maximum amplitude across all PSDs
	max_amp = max(float(p.max()) for p in mean_psds)
	ax.set_ylim(0, max_amp * 1.05)
	ax.grid(alpha=0.3)
	ax.legend()
	plt.tight_layout()
	plt.show()


def reduce_spatial_noise_gaussian(
	thermal_frames: np.ndarray,
	kernel_size: int,
	sigma: float | None = None,
) -> np.ndarray:
	"""
	Reduce spatial noise in thermal frames using a Gaussian blur kernel.

	Args:
		thermal_frames: Array with shape (n_frames, height, width).
		kernel_size: Size of the Gaussian kernel (odd integer > 0).
		sigma: Standard deviation of the Gaussian kernel. If None, OpenCV computes it.

	Returns:
		Smoothed frames with the same shape as input.
	"""
	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected a 3D array (n_frames, height, width), got shape {thermal_frames.shape}")
	if kernel_size < 1 or kernel_size % 2 == 0:
		raise ValueError(f"kernel_size must be an odd positive integer, got {kernel_size}")

	smoothed_frames = np.zeros_like(thermal_frames)
	for i in range(thermal_frames.shape[0]):
		smoothed_frames[i] = signal.gaussian_filter(thermal_frames[i], sigma=kernel_size / 2.0)

	return smoothed_frames


def find_optimal_kernel_size(
	thermal_frames: np.ndarray,
	kernel_sizes: list[int] | None = None,
	roi: tuple[int, int, int, int] | None = None,
	ddof: int = 1,
) -> dict[str, object]:
	"""
	Find the optimal Gaussian kernel size for reducing spatial noise.

	Tests different kernel sizes and calculates spatial noise for each smoothed version.
	Returns the kernel size that minimizes spatial noise along with full results.

	Args:
		thermal_frames: Array with shape (n_frames, height, width).
		kernel_sizes: List of kernel sizes to try (must be odd integers). 
		             If None, defaults to [3, 5, 7, 9, 11, 15].
		roi: Optional (x_min, x_max, y_min, y_max) ROI in pixel coordinates.
		ddof: Delta degrees of freedom for standard deviation calculation.

	Returns:
		Dictionary with:
		- 'optimal_kernel_size': The kernel size with lowest spatial noise.
		- 'optimal_spatial_noise_c': Spatial noise value for optimal kernel.
		- 'optimal_spatial_noise_mk': Spatial noise in mK.
		- 'results_by_kernel': Dict mapping kernel_size -> spatial noise result dict.
	"""
	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected a 3D array (n_frames, height, width), got shape {thermal_frames.shape}")

	if kernel_sizes is None:
		kernel_sizes = [3, 5, 7, 9, 11, 15]

	results_by_kernel = {}
	spatial_noise_values = {}

	for kernel_size in kernel_sizes:
		if kernel_size < 1 or kernel_size % 2 == 0:
			print(f"  Skipping invalid kernel_size {kernel_size} (must be odd positive integer)")
			continue

		# Apply Gaussian smoothing
		smoothed = reduce_spatial_noise_gaussian(thermal_frames, kernel_size=kernel_size)

		# Calculate spatial noise for smoothed frames
		spatial_result = calculate_spatial_noise_from_frame_std(smoothed, roi=roi, ddof=ddof)
		results_by_kernel[kernel_size] = spatial_result
		spatial_noise_values[kernel_size] = spatial_result["spatial_noise_c"]

		print(f"  Kernel size {kernel_size:2d}: spatial noise = {spatial_result['spatial_noise_c']:.6f} °C ({spatial_result['spatial_noise_mk']:.2f} mK)")

	if not spatial_noise_values:
		raise ValueError("No valid kernel sizes were tested.")

	# Find optimal kernel size
	optimal_kernel_size = min(spatial_noise_values, key=spatial_noise_values.get)
	optimal_result = results_by_kernel[optimal_kernel_size]

	return {
		"optimal_kernel_size": optimal_kernel_size,
		"optimal_spatial_noise_c": optimal_result["spatial_noise_c"],
		"optimal_spatial_noise_mk": optimal_result["spatial_noise_mk"],
		"results_by_kernel": results_by_kernel,
	}


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
	"""Run full pipeline on three hard-coded recordings and produce combined PSD plot."""

	# Hard-coded list of three recordings to analyze (edit these paths as needed)
	compare_paths = [
		Path("/data3/ina/NETD/5_secs_NETD_MWIR"),
		Path("/data3/ina/NETD/NETD_500frames_Vario"),
		Path("/data3/ina/NETD/NETD_500frames_Lepton"),
	]
	compare_cameras = ["mwir", "vario", "lepton"]
	compare_frame_rates = [100, 25.0, 8.76]

	# Common settings
	recursive = False
	roi: tuple[int, int, int, int] | None = None
	plot_histogram = True
	plot_qq = False
	plot_one_frame_heatmap = True
	plot_neighbor_correlation_heatmap_enabled = True
	plot_roi_fft = True
	plot_psd = True
	heatmap_frame_index = 0

	# Storage for PSD plotting labels
	labels: list[str] = []

	# Run full pipeline for each recording
	for path, camera, frame_rate in zip(compare_paths, compare_cameras, compare_frame_rates):
		print(f"\n=== Processing recording: {path} (camera={camera}, frame_rate={frame_rate}) ===")

		if not path.exists():
			print(f"  Warning: path does not exist: {path} -- skipping")
			continue

		# Load frames
		if camera == "lepton":
			filepaths = discover_gray_files(path) if path.is_dir() else [path]
			thermal_frames = load_gray_frames(filepaths)
			thermal_frames = thermal_frames / 100.0 - 273.15 # transform from Kelvin*100 to Celsius
		else:
			filepaths = discover_asc_files(path, recursive=recursive) if path.is_dir() else [path]
			thermal_frames = load_thermal_frames_asc(filepaths, camera=camera)

		# Calculate NETD (temporal noise)
		result = calculate_netd_from_pixelwise_std(thermal_frames, roi=roi)
		print(format_netd_result(result))

		# Spatial noise before smoothing
		spatial_noise_result = calculate_spatial_noise_from_frame_std(thermal_frames, roi=roi)
		print(format_spatial_noise_result(spatial_noise_result))

		# Find optimal spatial smoothing kernel
		print("  Finding optimal Gaussian kernel size for spatial noise reduction...")
		kernel_optimization = find_optimal_kernel_size(thermal_frames, roi=roi)
		opt_k = kernel_optimization["optimal_kernel_size"]
		print(f"  Optimal kernel size: {opt_k} -> {kernel_optimization['optimal_spatial_noise_c']:.6f} °C")

		# Spatial noise after applying optimal smoothing
		smoothed = reduce_spatial_noise_gaussian(thermal_frames, kernel_size=opt_k)
		spatial_noise_after = calculate_spatial_noise_from_frame_std(smoothed, roi=roi)
		print("  After smoothing:")
		print(format_spatial_noise_result(spatial_noise_after))

		# Per-recording plots (saved with camera name)
		results_dir = Path("saved_results/analysis_plots")
		if plot_histogram:
			hist_path = results_dir / f"{camera}_{path.name}_histogram.png"
			plot_all_pixels_histogram(thermal_frames, save_path=hist_path)
		if plot_one_frame_heatmap:
			heatmap_path = results_dir / f"{camera}_{path.name}_heatmap.png"
			plot_single_frame_heatmap(thermal_frames, frame_index=heatmap_frame_index, save_path=heatmap_path)
		if plot_neighbor_correlation_heatmap_enabled:
			neighbor_path = results_dir / f"{camera}_{path.name}_neighbor_corr.png"
			plot_neighbor_correlation_heatmap(neighbor_correlation_result["neighbor_corr_map"], save_path=neighbor_path)
		if plot_roi_fft:
			fft_path = results_dir / f"{camera}_{path.name}_roi_fft.png"
			plot_roi_mean_signal_fft(thermal_frames, frame_rate_hz=frame_rate, roi=roi, save_path=fft_path)

 

		# Collect label for combined PSD plot
		labels.append(f"{camera} ({path.name})")

	# After processing all recordings, plot combined PSDs if requested and at least one recording processed
	if plot_psd and len(labels) >= 1:
		print("\nGenerating combined PSD comparison plot...")
		# Use the same lists for the comparison plot function
		valid_paths = [p for p in compare_paths if p.exists()]
		valid_cameras = [c for p, c in zip(compare_paths, compare_cameras) if p.exists()]
		valid_frame_rates = [fr for p, fr in zip(compare_paths, compare_frame_rates) if p.exists()]
		plot_compare_pixelwise_mean_psd(valid_paths, valid_cameras, valid_frame_rates, labels=labels, roi=roi)


if __name__ == "__main__":
	main()

