"""Utilities for fast MWIR thermal signal extraction from many ASC files.

This module provides a low-memory workflow:
1. Convert many `.asc` files into a disk-backed `.npy` cache (streaming).
2. Load the cache with memory mapping (`mmap_mode='r'`).
3. Extract a thermal signal over time and plot it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
import sys

import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import butter, filtfilt

try:
	from .data_extraction import extract_thermal_data
except ImportError:  # pragma: no cover - allows running the file directly
	from data_extraction import extract_thermal_data

try:
	from ..ground_truth_methods.offset_detection_ground_truth import read_opensignals_txt
except ImportError:  # pragma: no cover - allows running the file directly
	project_root = Path(__file__).resolve().parents[2]
	if str(project_root) not in sys.path:
		sys.path.insert(0, str(project_root))
	from src.ground_truth_methods.offset_detection_ground_truth import read_opensignals_txt


def discover_asc_files(directory: str | Path, recursive: bool = False) -> list[Path]:
	"""Return sorted ASC files from a directory."""
	directory = Path(directory)
	if not directory.exists():
		raise FileNotFoundError(f"Directory does not exist: {directory}")
	if not directory.is_dir():
		raise NotADirectoryError(f"Expected a directory, got: {directory}")

	pattern = "**/*.asc" if recursive else "*.asc"
	files = sorted(directory.glob(pattern))
	if not files:
		raise ValueError(f"No ASC files found in: {directory}")
	return files


def build_thermal_cache_npy(
	asc_directory: str | Path,
	cache_path: str | Path,
	recursive: bool = False,
	overwrite: bool = False,
	progress_every: int = 100,
) -> tuple[Path, tuple[int, int, int]]:
	"""Build a `.npy` cache from ASC files using low RAM (streaming write)."""
	cache_path = Path(cache_path)
	if cache_path.exists() and not overwrite:
		raise FileExistsError(
			f"Cache already exists: {cache_path}. Set overwrite=True to rebuild."
		)

	files = discover_asc_files(asc_directory, recursive=recursive)
	first_frame = extract_thermal_data(str(files[0])).astype(np.float32)
	height, width = first_frame.shape
	frame_count = len(files)
	shape = (frame_count, height, width)

	cache_path.parent.mkdir(parents=True, exist_ok=True)
	cache = np.lib.format.open_memmap(
		cache_path,
		mode="w+",
		dtype=np.float32,
		shape=shape,
	)

	cache[0] = first_frame
	for index, file_path in enumerate(files[1:], start=1):
		cache[index] = extract_thermal_data(str(file_path)).astype(np.float32)
		if progress_every > 0 and (index + 1) % progress_every == 0:
			print(f"Cached {index + 1}/{frame_count} frames...")

	cache.flush()
	print(f"Cache written: {cache_path} with shape {shape}")
	return cache_path, shape


def load_thermal_cache_npy(cache_path: str | Path, mmap_mode: Literal["r", "r+"] = "r") -> np.ndarray:
	"""Load cached thermal frames quickly using NumPy memory mapping."""
	cache_path = Path(cache_path)
	if not cache_path.exists():
		raise FileNotFoundError(f"Cache file does not exist: {cache_path}")
	return np.load(cache_path, mmap_mode=mmap_mode)


def _normalize_roi(roi: tuple[int, int, int, int] | None, shape: tuple[int, int]) -> tuple[slice, slice]:
	"""Convert `(x_min, x_max, y_min, y_max)` ROI into NumPy slices."""
	if roi is None:
		return slice(None), slice(None)

	x_min, x_max, y_min, y_max = roi
	height, width = shape
	if not (0 <= x_min <= x_max < width and 0 <= y_min <= y_max < height):
		raise ValueError(
			f"Invalid ROI {roi} for shape {shape}. Expected x in [0,{width - 1}] and y in [0,{height - 1}]."
		)

	return slice(y_min, y_max + 1), slice(x_min, x_max + 1)


def extract_thermal_signal_over_time(
	thermal_frames: np.ndarray,
	roi: tuple[int, int, int, int] | None = None,
	reducer: Literal["mean", "median"] = "mean",
) -> np.ndarray:
	"""Extract one thermal signal value per frame from a full frame stack."""
	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected shape (n_frames, height, width), got {thermal_frames.shape}")

	y_slice, x_slice = _normalize_roi(roi, thermal_frames.shape[1:])
	roi_frames = thermal_frames[:, y_slice, x_slice]

	if reducer == "mean":
		return roi_frames.mean(axis=(1, 2))
	if reducer == "median":
		return np.median(roi_frames, axis=(1, 2))
	raise ValueError(f"Unsupported reducer: {reducer}")


def plot_thermal_signal_over_time(
	signal: np.ndarray,
	fps: float | None = None,
	title: str = "Thermal Signal Over Time",
) -> None:
	"""Plot thermal signal over frame index or time in seconds."""
	signal = np.asarray(signal, dtype=np.float32)
	if signal.ndim != 1:
		raise ValueError(f"Expected 1D signal, got shape {signal.shape}")

	if fps is not None and fps > 0:
		x = np.arange(signal.size, dtype=np.float32) / float(fps)
		x_label = "Time (s)"
	else:
		x = np.arange(signal.size, dtype=np.int32)
		x_label = "Frame Index"

	plt.figure(figsize=(12, 5))
	plt.plot(x, signal, linewidth=1.5)
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel("Temperature (°C)")
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.show()


def load_ground_truth_segment_from_npz(
	npz_path: str | Path,
	offset_seconds: float,
	recording_length_seconds: float,
	ground_truth_fps: float,
) -> tuple[np.ndarray, np.ndarray, str]:
	"""Load one 1D ground-truth signal from NPZ and return the requested time segment."""
	npz_path = Path(npz_path)
	if not npz_path.exists():
		raise FileNotFoundError(f"Ground truth NPZ does not exist: {npz_path}")
	if ground_truth_fps <= 0:
		raise ValueError(f"ground_truth_fps must be > 0, got {ground_truth_fps}")
	if recording_length_seconds <= 0:
		raise ValueError(f"recording_length_seconds must be > 0, got {recording_length_seconds}")

	with np.load(npz_path) as data:
		if not data.files:
			raise ValueError(f"No arrays found in NPZ file: {npz_path}")

		lower_to_original = {key.lower(): key for key in data.files}
		preferred_names = ["ecg", "ppg", "signal", "ground_truth", "heart_rate", "hr"]

		selected_key = None
		for preferred in preferred_names:
			if preferred in lower_to_original:
				selected_key = lower_to_original[preferred]
				break

		if selected_key is None:
			for key in data.files:
				candidate = np.asarray(data[key]).squeeze()
				if candidate.ndim == 1 and np.issubdtype(candidate.dtype, np.number):
					selected_key = key
					break

		if selected_key is None:
			raise ValueError(
				f"Could not find a 1D numeric signal in NPZ file: {npz_path}. Keys: {data.files}"
			)

		signal = np.asarray(data[selected_key], dtype=np.float32).squeeze()

	start_idx = max(0, int(round(offset_seconds * ground_truth_fps)))
	segment_len = int(round(recording_length_seconds * ground_truth_fps))
	end_idx = min(signal.size, start_idx + segment_len)
	if end_idx <= start_idx:
		raise ValueError(
			"Requested ground-truth segment is empty. "
			f"offset={offset_seconds}s, length={recording_length_seconds}s, fps={ground_truth_fps}, signal_size={signal.size}"
		)

	segment = signal[start_idx:end_idx]
	segment_time = offset_seconds + np.arange(segment.size, dtype=np.float32) / float(ground_truth_fps)
	return segment_time, segment, selected_key


def load_button_segment_from_opensignals_txt(
	txt_path: str | Path,
	offset_seconds: float,
	recording_length_seconds: float,
) -> tuple[np.ndarray, np.ndarray, float]:
	"""Load button signal from OpenSignals TXT and return requested time segment."""
	txt_path = Path(txt_path)
	if not txt_path.exists():
		raise FileNotFoundError(f"OpenSignals TXT does not exist: {txt_path}")
	if recording_length_seconds <= 0:
		raise ValueError(f"recording_length_seconds must be > 0, got {recording_length_seconds}")

	_, button_data, sampling_rate = read_opensignals_txt(str(txt_path))
	if sampling_rate <= 0:
		raise ValueError(f"Invalid sampling rate from OpenSignals TXT: {sampling_rate}")

	start_idx = max(0, int(round(offset_seconds * sampling_rate)))
	segment_len = int(round(recording_length_seconds * sampling_rate))
	end_idx = min(button_data.size, start_idx + segment_len)
	if end_idx <= start_idx:
		raise ValueError(
			"Requested button segment is empty. "
			f"offset={offset_seconds}s, length={recording_length_seconds}s, fs={sampling_rate}, signal_size={button_data.size}"
		)

	button_segment = np.asarray(button_data[start_idx:end_idx], dtype=np.float32)
	button_time = offset_seconds + np.arange(button_segment.size, dtype=np.float32) / float(sampling_rate)
	return button_time, button_segment, float(sampling_rate)


def plot_thermal_and_ground_truth_linked(
	thermal_signal: np.ndarray,
	thermal_fps: float,
	ground_truth_time: np.ndarray,
	ground_truth_signal: np.ndarray,
	button_time: np.ndarray,
	button_signal: np.ndarray,
	offset_seconds: float,
	recording_length_seconds: float,
	thermal_start_seconds: float,
	ground_truth_label: str,
	thermal_title: str = "Thermal Signal",
) -> None:
	"""Plot thermal, ground-truth, and button signals in linked subplots (shared x-axis)."""
	thermal_signal = np.asarray(thermal_signal, dtype=np.float32)
	ground_truth_time = np.asarray(ground_truth_time, dtype=np.float32)
	ground_truth_signal = np.asarray(ground_truth_signal, dtype=np.float32)
	button_time = np.asarray(button_time, dtype=np.float32)
	button_signal = np.asarray(button_signal, dtype=np.float32)

	if thermal_signal.ndim != 1:
		raise ValueError(f"Expected 1D thermal signal, got shape {thermal_signal.shape}")
	if ground_truth_time.ndim != 1 or ground_truth_signal.ndim != 1:
		raise ValueError("Ground-truth time and signal must be 1D arrays")
	if ground_truth_time.size != ground_truth_signal.size:
		raise ValueError("Ground-truth time and signal lengths must match")
	if button_time.ndim != 1 or button_signal.ndim != 1:
		raise ValueError("Button time and signal must be 1D arrays")
	if button_time.size != button_signal.size:
		raise ValueError("Button time and signal lengths must match")
	if thermal_fps <= 0:
		raise ValueError(f"thermal_fps must be > 0, got {thermal_fps}")

	thermal_time = thermal_start_seconds + np.arange(thermal_signal.size, dtype=np.float32) / float(thermal_fps)

	fig, (ax_top, ax_mid, ax_bottom) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
	ax_top.plot(thermal_time, thermal_signal, color="tab:red", linewidth=1.2)
	ax_top.set_title(thermal_title)
	ax_top.set_ylabel("Temperature (°C)")
	ax_top.grid(alpha=0.3)

	ax_mid.plot(ground_truth_time, ground_truth_signal, color="tab:blue", linewidth=1.0)
	ax_mid.set_title(f"Ground Truth ({ground_truth_label})")
	ax_mid.set_ylabel("Amplitude")
	ax_mid.grid(alpha=0.3)

	ax_bottom.step(button_time, button_signal, where="post", color="tab:green", linewidth=1.2)
	ax_bottom.set_title("Button Pressed (OpenSignals DI)")
	ax_bottom.set_xlabel("Time (s)")
	ax_bottom.set_ylabel("Button")
	ax_bottom.set_ylim(-0.1, 1.1)
	ax_bottom.grid(alpha=0.3)

	x_start = offset_seconds
	x_end = offset_seconds + recording_length_seconds
	ax_bottom.set_xlim(x_start, x_end)

	fig.tight_layout()
	plt.show()


def apply_bandpass_filter(
	signal: np.ndarray,
	fps: float,
	lowcut_hz: float,
	highcut_hz: float,
	order: int = 4,
) -> np.ndarray:
	"""Apply a zero-phase Butterworth bandpass filter to a 1D signal."""
	signal = np.asarray(signal, dtype=np.float32)
	if signal.ndim != 1:
		raise ValueError(f"Expected 1D signal, got shape {signal.shape}")
	if fps <= 0:
		raise ValueError(f"fps must be > 0, got {fps}")
	if lowcut_hz <= 0 or highcut_hz <= 0:
		raise ValueError(f"lowcut_hz and highcut_hz must be > 0, got {lowcut_hz}, {highcut_hz}")
	if lowcut_hz >= highcut_hz:
		raise ValueError(f"Expected lowcut_hz < highcut_hz, got {lowcut_hz} >= {highcut_hz}")
	if order < 1:
		raise ValueError(f"order must be >= 1, got {order}")

	nyquist = 0.5 * float(fps)
	if highcut_hz >= nyquist:
		raise ValueError(
			f"highcut_hz must be < Nyquist ({nyquist:.3f} Hz) for fps={fps}, got {highcut_hz}"
		)

	b, a = butter(order, [lowcut_hz / nyquist, highcut_hz / nyquist], btype="band")
	return filtfilt(b, a, signal).astype(np.float32)


def compute_fft_magnitude(signal: np.ndarray, sampling_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
	"""Compute one-sided FFT magnitude for a 1D signal."""
	signal = np.asarray(signal, dtype=np.float32)
	if signal.ndim != 1:
		raise ValueError(f"Expected 1D signal, got shape {signal.shape}")
	if sampling_rate_hz <= 0:
		raise ValueError(f"sampling_rate_hz must be > 0, got {sampling_rate_hz}")
	if signal.size < 2:
		raise ValueError("Need at least 2 samples to compute FFT")

	signal_centered = signal - float(signal.mean())
	freqs = np.fft.rfftfreq(signal_centered.size, d=1.0 / float(sampling_rate_hz))
	magnitude = np.abs(np.fft.rfft(signal_centered)) / signal_centered.size
	return freqs.astype(np.float32), magnitude.astype(np.float32)


def plot_fft_thermal_and_ground_truth(
	thermal_signal: np.ndarray,
	thermal_fps: float,
	ground_truth_signal: np.ndarray,
	ground_truth_fps: float,
	ground_truth_label: str,
	max_frequency_hz: float | None = None,
) -> None:
	"""Plot FFT magnitude spectra for thermal and ground-truth signals."""
	thermal_freqs, thermal_mag = compute_fft_magnitude(thermal_signal, thermal_fps)
	gt_freqs, gt_mag = compute_fft_magnitude(ground_truth_signal, ground_truth_fps)

	if max_frequency_hz is not None and max_frequency_hz > 0:
		thermal_mask = thermal_freqs <= max_frequency_hz
		gt_mask = gt_freqs <= max_frequency_hz
	else:
		thermal_mask = np.ones_like(thermal_freqs, dtype=bool)
		gt_mask = np.ones_like(gt_freqs, dtype=bool)

	fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
	ax_top.plot(thermal_freqs[thermal_mask], thermal_mag[thermal_mask], color="tab:red", linewidth=1.2)
	ax_top.set_title("FFT Magnitude - Thermal Signal")
	ax_top.set_ylabel("Magnitude")
	ax_top.grid(alpha=0.3)

	ax_bottom.plot(gt_freqs[gt_mask], gt_mag[gt_mask], color="tab:blue", linewidth=1.0)
	ax_bottom.set_title(f"FFT Magnitude - Ground Truth ({ground_truth_label})")
	ax_bottom.set_xlabel("Frequency (Hz)")
	ax_bottom.set_ylabel("Magnitude")
	ax_bottom.grid(alpha=0.3)

	fig.tight_layout()
	plt.show()

def create_video_from_thermal_frames(
	thermal_frames: np.ndarray,
	output_path: str | Path,
	fps: float = 100.0,
	normalization: Literal["global", "per-frame"] = "global",
	colormap: Literal["inferno", "turbo", "hot"] = "inferno",
) -> Path:
	"""Create an MP4 video from thermal frames stored in a NumPy array."""
	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected shape (n_frames, height, width), got {thermal_frames.shape}")
	if fps <= 0:
		raise ValueError(f"fps must be > 0, got {fps}")

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	colormap_code = {
		"inferno": cv2.COLORMAP_INFERNO,
		"turbo": cv2.COLORMAP_TURBO,
		"hot": cv2.COLORMAP_HOT,
	}[colormap]

	frame_count, height, width = thermal_frames.shape
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
	if not writer.isOpened():
		raise RuntimeError(f"Could not open video writer for: {output_path}")

	if normalization == "global":
		global_min = float(np.nanmin(thermal_frames))
		global_max = float(np.nanmax(thermal_frames))
		denominator = max(global_max - global_min, 1e-8)

	for index in range(frame_count):
		frame = thermal_frames[index]

		if normalization == "global":
			normalized = (frame - global_min) / denominator
		else:
			frame_min = float(np.nanmin(frame))
			frame_max = float(np.nanmax(frame))
			denominator = max(frame_max - frame_min, 1e-8)
			normalized = (frame - frame_min) / denominator

		gray_u8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
		colored = cv2.applyColorMap(gray_u8, colormap_code)
		writer.write(colored)

	writer.release()
	print(f"Video written: {output_path} ({frame_count} frames at {fps} FPS)")
	return output_path


def main() -> None:
	asc_directory = Path("/mnt/wwn-0x50000398325009f6-part1/Julian_31_03_1_Ascii/")
	cache_path = Path("/mnt/wwn-0x50000398325009f6-part1/Julian_31_03_1_Ascii.npy")
	ground_truth_path = Path("data/ground_truth/opensignals/extracted_npy_files/31032026/Julian_31_03_1/opensignals_0007800F7623_2026-03-31_16-24-55.npz")
	ground_truth_txt_path = Path("data/ground_truth/opensignals/original_signal/31032026/Julian_31_03_1/opensignals_0007800F7623_2026-03-31_16-24-55.txt")
	offset_ground_truth_seconds = 110.1
	recording_length_seconds = 60.0
	ground_truth__fps = 1000.0
	max_fft_frequency_hz = 5.0

	recursive = False
	overwrite_cache = False
	build_cache_if_missing = False
	create_video = False
	fps: float | None = 100.0
	video_output_path = Path("/mnt/wwn-0x50000398325009f6-part1/Julian_31_03_1_Ascii.mp4")
	plot_bandpass_filtered_signal = True
	bandpass_lowcut_hz = 0.5
	bandpass_highcut_hz = 3.0
	bandpass_order = 4

	roi: tuple[int, int, int, int] | None = None
	reducer: Literal["mean", "median"] = "mean"

	if build_cache_if_missing and not cache_path.exists():
		build_thermal_cache_npy(
			asc_directory=asc_directory,
			cache_path=cache_path,
			recursive=recursive,
			overwrite=overwrite_cache,
			progress_every=100,
		)

	frames = load_thermal_cache_npy(cache_path, mmap_mode="r")
	print(f"Loaded cached frames with shape: {frames.shape}")

	if create_video:
		create_video_from_thermal_frames(
			frames,
			output_path=video_output_path,
			fps=fps,
			normalization="global",
			colormap="inferno",
		)

	signal = extract_thermal_signal_over_time(frames, roi=roi, reducer=reducer)
	if fps is None or fps <= 0:
		raise ValueError("`fps` must be > 0 for linked thermal/ground-truth plotting.")

	ground_truth_time, ground_truth_signal, selected_ground_truth_key = load_ground_truth_segment_from_npz(
		npz_path=ground_truth_path,
		offset_seconds=offset_ground_truth_seconds,
		recording_length_seconds=recording_length_seconds,
		ground_truth_fps=ground_truth__fps,
	)

	button_time, button_signal, _ = load_button_segment_from_opensignals_txt(
		txt_path=ground_truth_txt_path,
		offset_seconds=offset_ground_truth_seconds,
		recording_length_seconds=recording_length_seconds,
	)

	plot_thermal_and_ground_truth_linked(
		thermal_signal=signal,
		thermal_fps=fps,
		ground_truth_time=ground_truth_time,
		ground_truth_signal=ground_truth_signal,
		button_time=button_time,
		button_signal=button_signal,
		offset_seconds=offset_ground_truth_seconds,
		recording_length_seconds=recording_length_seconds,
		thermal_start_seconds=offset_ground_truth_seconds,
		ground_truth_label=selected_ground_truth_key,
		thermal_title="Thermal Signal (Raw)",
	)

	if plot_bandpass_filtered_signal:
		filtered_signal = apply_bandpass_filter(
			signal=signal,
			fps=fps,
			lowcut_hz=bandpass_lowcut_hz,
			highcut_hz=bandpass_highcut_hz,
			order=bandpass_order,
		)
		plot_thermal_and_ground_truth_linked(
			thermal_signal=filtered_signal,
			thermal_fps=fps,
			ground_truth_time=ground_truth_time,
			ground_truth_signal=ground_truth_signal,
			button_time=button_time,
			button_signal=button_signal,
			offset_seconds=offset_ground_truth_seconds,
			recording_length_seconds=recording_length_seconds,
			thermal_start_seconds=offset_ground_truth_seconds,
			ground_truth_label=selected_ground_truth_key,
			thermal_title=(
				"Thermal Signal (Bandpass "
				f"{bandpass_lowcut_hz:.2f}-{bandpass_highcut_hz:.2f} Hz, order={bandpass_order})"
			),
		)

	plot_fft_thermal_and_ground_truth(
		thermal_signal=signal,
		thermal_fps=fps,
		ground_truth_signal=ground_truth_signal,
		ground_truth_fps=ground_truth__fps,
		ground_truth_label=selected_ground_truth_key,
		max_frequency_hz=max_fft_frequency_hz,
	)


if __name__ == "__main__":
	main()


#TODO print ground truth correctly - look trhough code
#TODO compare before and after sports (after sports ground truth is currently not correct)