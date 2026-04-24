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

project_root = "/home/arbeit/Dokumente/master_thesis/"
sys.path.insert(0, project_root)

try:
	import mediapipe as mp
except ImportError:  # pragma: no cover - optional dependency
	mp = None

try:
	from .data_extraction import extract_thermal_data
except ImportError:  # pragma: no cover - allows running the file directly
	from data_extraction import extract_thermal_data

try:
	from .netd_analysis import average_consecutive_frames
except ImportError:  # pragma: no cover - allows running the file directly
	from netd_analysis import average_consecutive_frames

try:
	from .experiments_config import load_experiment_from_yaml
except ImportError:  # pragma: no cover - allows running the file directly
	from experiments_config import load_experiment_from_yaml

try:
	from ..ground_truth_methods.offset_detection_ground_truth import read_opensignals_txt, read_opensignals_txt_converted, offset_and_saving_pipeline
except ImportError:  # pragma: no cover - allows running the file directly
	project_root = Path(__file__).resolve().parents[2]
	if str(project_root) not in sys.path:
		sys.path.insert(0, str(project_root))
	from src.ground_truth_methods.offset_detection_ground_truth import read_opensignals_txt, read_opensignals_txt_converted, offset_and_saving_pipeline


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


def _get_colormap_code(colormap: Literal["inferno", "turbo", "hot"]) -> int:
	"""Map colormap name to OpenCV colormap code."""
	return {
		"inferno": cv2.COLORMAP_INFERNO,
		"turbo": cv2.COLORMAP_TURBO,
		"hot": cv2.COLORMAP_HOT,
	}[colormap]


def _frame_to_detection_image(
	frame: np.ndarray,
	global_min: float,
	global_max: float,
	normalization: Literal["global", "per-frame"],
	colormap: Literal["inferno", "turbo", "hot"],
) -> np.ndarray:
	"""Convert one thermal frame to a color image suitable for MediaPipe detection."""
	if normalization == "global":
		vmin = global_min
		vmax = global_max
	else:
		vmin = float(np.nanmin(frame))
		vmax = float(np.nanmax(frame))

	denominator = max(vmax - vmin, 1e-8)
	normalized = (frame - vmin) / denominator
	gray_u8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
	return cv2.applyColorMap(gray_u8, _get_colormap_code(colormap))


def detect_face_rois_over_time(
	thermal_frames: np.ndarray,
	normalization: Literal["global", "per-frame"] = "global",
	colormap: Literal["inferno", "turbo", "hot"] = "inferno",
	use_ema: bool = True,
	ema_alpha: float = 0.2,
	flip_upside_down: bool = False,
	min_detection_confidence: float = 0.5,
	min_tracking_confidence: float = 0.5,
	padding_ratio: float = 0.05,
) -> list[tuple[int, int, int, int] | None]:
	"""Detect a face on every frame and return a per-frame bounding-box ROI.

	The ROI is derived from all FaceMesh landmarks in the frame and padded slightly
	to make the extracted thermal signal more stable.
	"""
	if mp is None:
		raise ImportError("mediapipe is required for face ROI detection but is not installed.")

	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected shape (n_frames, height, width), got {thermal_frames.shape}")
	if not (0.0 <= padding_ratio < 0.5):
		raise ValueError(f"padding_ratio must be in [0.0, 0.5), got {padding_ratio}")
	if not (0.0 < ema_alpha <= 1.0):
		raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")

	frame_count, height, width = thermal_frames.shape
	global_min = float(np.nanmin(thermal_frames))
	global_max = float(np.nanmax(thermal_frames))
	face_rois: list[tuple[int, int, int, int] | None] = []
	smoothed_pts: np.ndarray | None = None
	alpha = ema_alpha if use_ema else 1.0

	with mp.solutions.face_mesh.FaceMesh(
		max_num_faces=1,
		refine_landmarks=True,
		min_detection_confidence=min_detection_confidence,
		min_tracking_confidence=min_tracking_confidence,
	) as face_mesh:
		for index in range(frame_count):
			frame = thermal_frames[index]
			frame_for_mesh = np.flipud(frame) if flip_upside_down else frame
			detection_bgr = _frame_to_detection_image(frame_for_mesh, global_min, global_max, normalization, colormap)
			detection_rgb = cv2.cvtColor(detection_bgr, cv2.COLOR_BGR2RGB)
			results = face_mesh.process(detection_rgb)

			if not results.multi_face_landmarks:
				smoothed_pts = None
				face_rois.append(None)
				continue

			landmarks = results.multi_face_landmarks[0].landmark
			points = np.array(
				[[int(lm.x * width), int(lm.y * height)] for lm in landmarks],
				dtype=np.float32,
			)
			points[:, 0] = np.clip(points[:, 0], 0, width - 1)
			points[:, 1] = np.clip(points[:, 1], 0, height - 1)

			if smoothed_pts is None:
				smoothed_pts = points.copy()
			else:
				smoothed_pts = alpha * points + (1.0 - alpha) * smoothed_pts

			points_int = smoothed_pts.astype(np.int32)
			if flip_upside_down:
				points_int[:, 1] = (height - 1) - points_int[:, 1]

			points_int[:, 0] = np.clip(points_int[:, 0], 0, width - 1)
			points_int[:, 1] = np.clip(points_int[:, 1], 0, height - 1)

			x_min = int(points_int[:, 0].min())
			x_max = int(points_int[:, 0].max())
			y_min = int(points_int[:, 1].min())
			y_max = int(points_int[:, 1].max())

			if x_max <= x_min or y_max <= y_min:
				face_rois.append(None)
				continue

			pad_x = max(1, int(round((x_max - x_min) * padding_ratio)))
			pad_y = max(1, int(round((y_max - y_min) * padding_ratio)))
			x_min = max(0, x_min - pad_x)
			x_max = min(width - 1, x_max + pad_x)
			y_min = max(0, y_min - pad_y)
			y_max = min(height - 1, y_max + pad_y)

			face_rois.append((x_min, x_max, y_min, y_max))

	return face_rois


def detect_landmark_roi_masks_over_time(
	thermal_frames: np.ndarray,
	roi_landmarks: dict[str, list[int]],
	normalization: Literal["global", "per-frame"] = "global",
	colormap: Literal["inferno", "turbo", "hot"] = "inferno",
	use_ema: bool = True,
	ema_alpha: float = 0.2,
	flip_upside_down: bool = False,
	start_frame: int = 0,
	end_frame: int | None = None,
	min_detection_confidence: float = 0.5,
	min_tracking_confidence: float = 0.5,
) -> list[dict[str, np.ndarray]]:
	"""Detect FaceMesh per frame and build ROI masks from landmark index groups."""
	if mp is None:
		raise ImportError("mediapipe is required for landmark ROI detection but is not installed.")

	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected shape (n_frames, height, width), got {thermal_frames.shape}")
	if not roi_landmarks:
		raise ValueError("`roi_landmarks` must not be empty.")
	if not (0.0 < ema_alpha <= 1.0):
		raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
	if start_frame < 0:
		raise ValueError(f"start_frame must be >= 0, got {start_frame}")

	frame_count, height, width = thermal_frames.shape
	if end_frame is None:
		end_frame = frame_count
	end_frame = min(end_frame, frame_count)
	if end_frame < start_frame:
		raise ValueError(f"end_frame ({end_frame}) must be >= start_frame ({start_frame})")

	global_min = float(np.nanmin(thermal_frames))
	global_max = float(np.nanmax(thermal_frames))
	roi_names = list(roi_landmarks.keys())
	roi_mask_log: list[dict[str, np.ndarray]] = []
	for _ in range(frame_count):
		roi_mask_log.append({name: np.zeros((height, width), dtype=np.uint8) for name in roi_names})

	smoothed_pts: np.ndarray | None = None
	alpha = ema_alpha if use_ema else 1.0

	with mp.solutions.face_mesh.FaceMesh(
		max_num_faces=1,
		refine_landmarks=True,
		min_detection_confidence=min_detection_confidence,
		min_tracking_confidence=min_tracking_confidence,
	) as face_mesh:
		for frame_idx in range(start_frame, end_frame):
			frame = thermal_frames[frame_idx]
			frame_for_mesh = np.flipud(frame) if flip_upside_down else frame
			detection_bgr = _frame_to_detection_image(frame_for_mesh, global_min, global_max, normalization, colormap)
			detection_rgb = cv2.cvtColor(detection_bgr, cv2.COLOR_BGR2RGB)
			results = face_mesh.process(detection_rgb)

			if not results.multi_face_landmarks:
				smoothed_pts = None
				continue

			landmarks = results.multi_face_landmarks[0].landmark
			points = np.array(
				[[int(lm.x * width), int(lm.y * height)] for lm in landmarks],
				dtype=np.float32,
			)
			points[:, 0] = np.clip(points[:, 0], 0, width - 1)
			points[:, 1] = np.clip(points[:, 1], 0, height - 1)

			if smoothed_pts is None:
				smoothed_pts = points.copy()
			else:
				smoothed_pts = alpha * points + (1.0 - alpha) * smoothed_pts

			points_int = smoothed_pts.astype(np.int32)
			if flip_upside_down:
				points_int[:, 1] = (height - 1) - points_int[:, 1]

			points_int[:, 0] = np.clip(points_int[:, 0], 0, width - 1)
			points_int[:, 1] = np.clip(points_int[:, 1], 0, height - 1)

			for roi_name, indices in roi_landmarks.items():
				if not indices:
					continue

				index_array = np.asarray(indices, dtype=np.int32)
				if np.any(index_array < 0) or np.any(index_array >= points_int.shape[0]):
					raise ValueError(
						f"ROI '{roi_name}' contains invalid landmark index. "
						f"Valid range is [0, {points_int.shape[0] - 1}]"
					)

				roi_poly = points_int[index_array]
				mask = np.zeros((height, width), dtype=np.uint8)

				if roi_poly.shape[0] >= 3:
					cv2.fillPoly(mask, [roi_poly], 255)
				elif roi_poly.shape[0] == 2:
					cv2.line(mask, tuple(roi_poly[0]), tuple(roi_poly[1]), 255, thickness=1)
				else:
					x, y = roi_poly[0]
					mask[int(y), int(x)] = 255

				roi_mask_log[frame_idx][roi_name] = mask

	return roi_mask_log


def _fill_missing_signal(
	signal: np.ndarray,
	fill_missing: Literal["nan", "interpolate", "previous"],
	error_message: str,
) -> np.ndarray:
	"""Fill missing values in a 1D signal represented by NaNs."""
	signal = np.asarray(signal, dtype=np.float32)
	if np.all(np.isnan(signal)):
		raise ValueError(error_message)

	if fill_missing == "nan":
		return signal

	valid_indices = np.flatnonzero(~np.isnan(signal))
	if valid_indices.size == 0:
		raise ValueError(error_message)

	if fill_missing == "previous":
		filled = signal.copy()
		last_value = np.nan
		for index, value in enumerate(filled):
			if np.isnan(value):
				filled[index] = last_value
			else:
				last_value = value

		if np.isnan(filled[0]):
			first_valid = valid_indices[0]
			filled[:first_valid] = signal[first_valid]
		return filled

	if fill_missing != "interpolate":
		raise ValueError(f"Unsupported fill_missing mode: {fill_missing}")

	filled = signal.copy()
	missing = np.isnan(filled)
	if missing.any():
		filled[missing] = np.interp(np.flatnonzero(missing), valid_indices, filled[valid_indices])
	return filled


def extract_signal_from_roi_mask_log(
	thermal_frames: np.ndarray,
	roi_mask_log: list[dict[str, np.ndarray]],
	roi_name: str,
	reducer: Literal["mean", "median"] = "mean",
	fill_missing: Literal["nan", "interpolate", "previous"] = "interpolate",
) -> np.ndarray:
	"""Extract one signal from one named ROI in a per-frame ROI-mask log."""
	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected shape (n_frames, height, width), got {thermal_frames.shape}")
	if len(roi_mask_log) != thermal_frames.shape[0]:
		raise ValueError("`roi_mask_log` must have the same length as `thermal_frames`.")

	values: list[float] = []
	for frame, frame_masks in zip(thermal_frames, roi_mask_log):
		mask = frame_masks.get(roi_name)
		if mask is None or mask.shape != frame.shape:
			values.append(float("nan"))
			continue

		ys, xs = np.where(mask == 255)
		if ys.size == 0:
			values.append(float("nan"))
			continue

		roi_values = frame[ys, xs]
		if reducer == "mean":
			values.append(float(np.nanmean(roi_values)))
		elif reducer == "median":
			values.append(float(np.nanmedian(roi_values)))
		else:
			raise ValueError(f"Unsupported reducer: {reducer}")

	signal = np.asarray(values, dtype=np.float32)
	return _fill_missing_signal(
		signal,
		fill_missing=fill_missing,
		error_message=f"No valid landmark ROI values were found for ROI '{roi_name}'.",
	)


def extract_signal_from_roi_sequence(
	thermal_frames: np.ndarray,
	roi_sequence: list[tuple[int, int, int, int] | None],
	reducer: Literal["mean", "median"] = "mean",
	fill_missing: Literal["nan", "interpolate", "previous"] = "interpolate",
) -> np.ndarray:
	"""Extract one thermal value per frame from a per-frame ROI sequence."""
	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected shape (n_frames, height, width), got {thermal_frames.shape}")
	if len(thermal_frames) != len(roi_sequence):
		raise ValueError("`roi_sequence` must have the same length as `thermal_frames`.")

	values: list[float] = []
	for frame, roi in zip(thermal_frames, roi_sequence):
		if roi is None:
			values.append(float("nan"))
			continue

		y_slice, x_slice = _normalize_roi(roi, frame.shape)
		roi_frame = frame[y_slice, x_slice]
		if roi_frame.size == 0:
			values.append(float("nan"))
			continue

		if reducer == "mean":
			values.append(float(np.nanmean(roi_frame)))
		elif reducer == "median":
			values.append(float(np.nanmedian(roi_frame)))
		else:
			raise ValueError(f"Unsupported reducer: {reducer}")

	signal = np.asarray(values, dtype=np.float32)
	return _fill_missing_signal(
		signal,
		fill_missing=fill_missing,
		error_message="No valid face ROIs were detected in any frame.",
	)


def roi_mask_log_to_bboxes(
	roi_mask_log: list[dict[str, np.ndarray]],
	roi_name: str,
) -> list[tuple[int, int, int, int] | None]:
	"""Convert per-frame ROI masks to per-frame bounding boxes for visualization."""
	bboxes: list[tuple[int, int, int, int] | None] = []
	for frame_masks in roi_mask_log:
		mask = frame_masks.get(roi_name)
		if mask is None:
			bboxes.append(None)
			continue

		ys, xs = np.where(mask == 255)
		if ys.size == 0:
			bboxes.append(None)
			continue

		bboxes.append((int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())))

	return bboxes


def extract_landmark_roi_signal_over_time(
	thermal_frames: np.ndarray,
	roi_landmarks: dict[str, list[int]],
	roi_name: str,
	normalization: Literal["global", "per-frame"] = "global",
	colormap: Literal["inferno", "turbo", "hot"] = "inferno",
	reducer: Literal["mean", "median"] = "mean",
	fill_missing: Literal["nan", "interpolate", "previous"] = "interpolate",
	use_ema: bool = True,
	ema_alpha: float = 0.2,
	flip_upside_down: bool = False,
	start_frame: int = 0,
	end_frame: int | None = None,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
	"""Extract thermal signal from a landmark-defined ROI tracked per frame with FaceMesh."""
	roi_mask_log = detect_landmark_roi_masks_over_time(
		thermal_frames=thermal_frames,
		roi_landmarks=roi_landmarks,
		normalization=normalization,
		colormap=colormap,
		use_ema=use_ema,
		ema_alpha=ema_alpha,
		flip_upside_down=flip_upside_down,
		start_frame=start_frame,
		end_frame=end_frame,
	)
	signal = extract_signal_from_roi_mask_log(
		thermal_frames=thermal_frames,
		roi_mask_log=roi_mask_log,
		roi_name=roi_name,
		reducer=reducer,
		fill_missing=fill_missing,
	)
	return signal, roi_mask_log


def extract_face_signal_over_time(
	thermal_frames: np.ndarray,
	normalization: Literal["global", "per-frame"] = "global",
	colormap: Literal["inferno", "turbo", "hot"] = "inferno",
	reducer: Literal["mean", "median"] = "mean",
	fill_missing: Literal["nan", "interpolate", "previous"] = "interpolate",
	padding_ratio: float = 0.05,
	use_ema: bool = True,
	ema_alpha: float = 0.2,
	flip_upside_down: bool = False,
) -> tuple[np.ndarray, list[tuple[int, int, int, int] | None]]:
	"""Detect the face in every frame and return the face-based thermal signal."""
	face_rois = detect_face_rois_over_time(
		thermal_frames,
		normalization=normalization,
		colormap=colormap,
		use_ema=use_ema,
		ema_alpha=ema_alpha,
		flip_upside_down=flip_upside_down,
		padding_ratio=padding_ratio,
	)
	signal = extract_signal_from_roi_sequence(
		thermal_frames,
		face_rois,
		reducer=reducer,
		fill_missing=fill_missing,
	)
	return signal, face_rois


def select_roi_on_first_frame(
	thermal_frames: np.ndarray,
	normalization: Literal["global", "per-frame"] = "global",
	colormap: Literal["inferno", "turbo", "hot"] = "inferno",
	window_name: str = "Select ROI (press ENTER to confirm)",
) -> tuple[int, int, int, int]:
	"""Interactively select ROI on the first frame and return `(x_min, x_max, y_min, y_max)`."""
	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected shape (n_frames, height, width), got {thermal_frames.shape}")

	frame = thermal_frames[0]
	if normalization == "global":
		vmin = float(np.nanmin(thermal_frames))
		vmax = float(np.nanmax(thermal_frames))
	else:
		vmin = float(np.nanmin(frame))
		vmax = float(np.nanmax(frame))

	denominator = max(vmax - vmin, 1e-8)
	normalized = (frame - vmin) / denominator
	gray_u8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
	colored = cv2.applyColorMap(gray_u8, _get_colormap_code(colormap))

	roi = cv2.selectROI(window_name, colored, showCrosshair=True, fromCenter=False)
	cv2.destroyWindow(window_name)

	x, y, w, h = [int(v) for v in roi]
	if w <= 0 or h <= 0:
		raise ValueError("ROI selection cancelled or empty. Please select a valid ROI.")

	return x, x + w - 1, y, y + h - 1


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


def plot_averaged_vs_non_averaged_signals(
	non_averaged_signal: np.ndarray,
	non_averaged_fps: float,
	averaged_signal: np.ndarray,
	averaged_fps: float,
	title: str = "Thermal Signal: Non-Averaged vs Averaged",
) -> None:
	"""Plot non-averaged and averaged thermal signals together for comparison."""
	non_averaged_signal = np.asarray(non_averaged_signal, dtype=np.float32)
	averaged_signal = np.asarray(averaged_signal, dtype=np.float32)

	if non_averaged_signal.ndim != 1:
		raise ValueError(f"Expected 1D non_averaged_signal, got shape {non_averaged_signal.shape}")
	if averaged_signal.ndim != 1:
		raise ValueError(f"Expected 1D averaged_signal, got shape {averaged_signal.shape}")
	if non_averaged_fps <= 0:
		raise ValueError(f"non_averaged_fps must be > 0, got {non_averaged_fps}")
	if averaged_fps <= 0:
		raise ValueError(f"averaged_fps must be > 0, got {averaged_fps}")

	t_non_averaged = np.arange(non_averaged_signal.size, dtype=np.float32) / float(non_averaged_fps)
	t_averaged = np.arange(averaged_signal.size, dtype=np.float32) / float(averaged_fps)

	plt.figure(figsize=(12, 5))
	plt.plot(t_non_averaged, non_averaged_signal, linewidth=1.0, alpha=0.7, label="Non-averaged")
	plt.plot(t_averaged, averaged_signal, linewidth=1.6, alpha=0.95, label="Averaged")
	plt.title(title)
	plt.xlabel("Time (s)")
	plt.ylabel("Temperature (°C)")
	plt.grid(alpha=0.3)
	plt.legend()
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

def load_button_segment_from_opensignals_txt_converted(
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

	_, button_data, sampling_rate = read_opensignals_txt_converted(str(txt_path))
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
	roi: tuple[int, int, int, int] | None = None,
	roi_per_frame: list[tuple[int, int, int, int] | None] | None = None,
	roi_mask_per_frame: list[np.ndarray | None] | None = None,
	roi_color_bgr: tuple[int, int, int] = (0, 255, 255),
	roi_thickness: int = 2,
) -> Path:
	"""Create an MP4 video from thermal frames stored in a NumPy array."""
	thermal_frames = np.asarray(thermal_frames, dtype=np.float32)
	if thermal_frames.ndim != 3:
		raise ValueError(f"Expected shape (n_frames, height, width), got {thermal_frames.shape}")
	if fps <= 0:
		raise ValueError(f"fps must be > 0, got {fps}")
	if roi is not None:
		_normalize_roi(roi, thermal_frames.shape[1:])
	if roi_per_frame is not None and len(roi_per_frame) != thermal_frames.shape[0]:
		raise ValueError("`roi_per_frame` must have the same length as `thermal_frames`.")
	if roi_mask_per_frame is not None and len(roi_mask_per_frame) != thermal_frames.shape[0]:
		raise ValueError("`roi_mask_per_frame` must have the same length as `thermal_frames`.")

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	colormap_code = _get_colormap_code(colormap)

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

		frame_mask = None if roi_mask_per_frame is None else roi_mask_per_frame[index]
		if frame_mask is not None:
			if frame_mask.shape != (height, width):
				raise ValueError(
					f"ROI mask at frame {index} has shape {frame_mask.shape}, expected {(height, width)}"
				)
			mask_u8 = (frame_mask > 0).astype(np.uint8) * 255
			contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			if contours:
				cv2.drawContours(colored, contours, -1, roi_color_bgr, roi_thickness)

		frame_roi = roi_per_frame[index] if roi_per_frame is not None else roi
		if frame_mask is None and frame_roi is not None:
			x_min, x_max, y_min, y_max = frame_roi
			cv2.rectangle(colored, (x_min, y_min), (x_max, y_max), roi_color_bgr, roi_thickness)
		writer.write(colored)

	writer.release()
	print(f"Video written: {output_path} ({frame_count} frames at {fps} FPS)")
	return output_path


def main() -> None:
	experiments_yaml_path = Path("src/cameras_pipelines/mwir/experiments_mwir.yaml")
	selected_person = "paulo"
	selected_region: Literal["temple", "face"] = "temple"
	selected_phase: Literal["before", "after"] = "before"

	selected_experiment = load_experiment_from_yaml(
		yaml_path=experiments_yaml_path,
		person=selected_person,
		region=selected_region,
		phase=selected_phase,
	)
	print(
		"Using experiment: "
		f"person={selected_person}, region={selected_region}, phase={selected_phase}"
	)

	recording_length_seconds = 60.0
	ground_truth__fps = 1000.0
	max_fft_frequency_hz = 5.0

	recursive = False
	overwrite_cache = False
	build_cache_if_missing = True
	create_video = True
	fps: float | None = 100.0
	apply_average_consecutive_frames_first = False
	frame_average_window_size = 10
	plot_averaged_vs_non_averaged_comparison = False
	plot_bandpass_filtered_signal = True
	bandpass_lowcut_hz = 0.5
	bandpass_highcut_hz = 3.0
	bandpass_order = 4

	select_roi_on_first_frame_enabled = True

	use_mediapipe_landmark_roi_detection = False
	use_mediapipe_face_roi_detection = False
	face_roi_padding_ratio = 0.05
	use_landmark_ema = False
	landmark_ema_alpha = 0.2
	landmark_flip_upside_down = False
	roi_landmarks: dict[str, list[int]] = {
		"whole_face": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
		"forehead": [10, 338, 297, 332, 284, 251, 389, 368, 300, 293, 334, 296, 336, 9, 107, 66, 105, 63, 71, 162, 21, 54, 103, 67, 109]
	}
	selected_landmark_roi_name = "forehead"

	roi: tuple[int, int, int, int] | None = None
	face_rois: list[tuple[int, int, int, int] | None] | None = None
	selected_roi_masks_per_frame: list[np.ndarray | None] | None = None
	non_averaged_signal_for_comparison: np.ndarray | None = None
	reducer: Literal["mean", "median"] = "mean"

	npz_path = selected_experiment.ground_truth_path.parent
	npz_filename = selected_experiment.ground_truth_path.name
	if not selected_experiment.ground_truth_path.exists():
		offset_and_saving_pipeline(
			signal_path=selected_experiment.ground_truth_txt_path,
			npz_path=npz_path,
			npz_filename=npz_filename,
		)

	ground_truth_time, ground_truth_signal, selected_ground_truth_key = load_ground_truth_segment_from_npz(
		npz_path=selected_experiment.ground_truth_path,
		offset_seconds=selected_experiment.offset_ground_truth_seconds,
		recording_length_seconds=recording_length_seconds,
		ground_truth_fps=ground_truth__fps,
	)

	if str(selected_experiment.ground_truth_txt_path).lower().endswith("_converted.txt"):
		button_time, button_signal, _ = load_button_segment_from_opensignals_txt_converted(
			txt_path=selected_experiment.ground_truth_txt_path,
			offset_seconds=selected_experiment.offset_ground_truth_seconds,
			recording_length_seconds=recording_length_seconds,
		)
	else:
		button_time, button_signal, _ = load_button_segment_from_opensignals_txt(
			txt_path=selected_experiment.ground_truth_txt_path,
			offset_seconds=selected_experiment.offset_ground_truth_seconds,
			recording_length_seconds=recording_length_seconds,
		)

	if build_cache_if_missing and not selected_experiment.cache_path.exists():
		build_thermal_cache_npy(
			asc_directory=selected_experiment.asc_directory,
			cache_path=selected_experiment.cache_path,
			recursive=recursive,
			overwrite=overwrite_cache,
			progress_every=100,
		)

	original_frames = load_thermal_cache_npy(selected_experiment.cache_path, mmap_mode="r")
	frames = original_frames
	print(f"Loaded cached frames with shape: {frames.shape}")

	if apply_average_consecutive_frames_first:
		frames = average_consecutive_frames(frames, window_size=frame_average_window_size)
		print(f"Using temporally averaged frames with shape: {frames.shape}")

	thermal_fps = fps
	if thermal_fps is not None and apply_average_consecutive_frames_first:
		thermal_fps = thermal_fps / float(frame_average_window_size)
		if thermal_fps <= 0:
			raise ValueError(
				"Effective thermal FPS became <= 0 after averaging. "
				"Check `fps` and `frame_average_window_size`."
			)
		print(f"Effective thermal FPS after averaging: {thermal_fps:.6f}")

	if select_roi_on_first_frame_enabled:
		roi = select_roi_on_first_frame(
			frames,
			normalization="global",
			colormap="inferno",
		)
		print(f"Selected ROI (x_min, x_max, y_min, y_max): {roi}")

	if use_mediapipe_landmark_roi_detection:
		signal, roi_mask_log = extract_landmark_roi_signal_over_time(
			thermal_frames=frames,
			roi_landmarks=roi_landmarks,
			roi_name=selected_landmark_roi_name,
			normalization="global",
			colormap="inferno",
			reducer=reducer,
			fill_missing="interpolate",
			use_ema=use_landmark_ema,
			ema_alpha=landmark_ema_alpha,
			flip_upside_down=landmark_flip_upside_down,
		)
		selected_roi_masks_per_frame = [
			frame_masks.get(selected_landmark_roi_name)
			for frame_masks in roi_mask_log
		]
		face_rois = roi_mask_log_to_bboxes(roi_mask_log, selected_landmark_roi_name)
		roi = None
		if apply_average_consecutive_frames_first and plot_averaged_vs_non_averaged_comparison:
			non_averaged_signal_for_comparison, _ = extract_landmark_roi_signal_over_time(
				thermal_frames=original_frames,
				roi_landmarks=roi_landmarks,
				roi_name=selected_landmark_roi_name,
				normalization="global",
				colormap="inferno",
				reducer=reducer,
				fill_missing="interpolate",
				use_ema=use_landmark_ema,
				ema_alpha=landmark_ema_alpha,
				flip_upside_down=landmark_flip_upside_down,
			)
	elif use_mediapipe_face_roi_detection:
		signal, face_rois = extract_face_signal_over_time(
			frames,
			normalization="global",
			colormap="inferno",
			reducer=reducer,
			fill_missing="interpolate",
			padding_ratio=face_roi_padding_ratio,
			use_ema=use_landmark_ema,
			ema_alpha=landmark_ema_alpha,
			flip_upside_down=landmark_flip_upside_down,
		)
		roi = None
		if apply_average_consecutive_frames_first and plot_averaged_vs_non_averaged_comparison:
			non_averaged_signal_for_comparison, _ = extract_face_signal_over_time(
				original_frames,
				normalization="global",
				colormap="inferno",
				reducer=reducer,
				fill_missing="interpolate",
				padding_ratio=face_roi_padding_ratio,
				use_ema=use_landmark_ema,
				ema_alpha=landmark_ema_alpha,
				flip_upside_down=landmark_flip_upside_down,
			)
	else:
		signal = extract_thermal_signal_over_time(frames, roi=roi, reducer=reducer)
		if apply_average_consecutive_frames_first and plot_averaged_vs_non_averaged_comparison:
			non_averaged_signal_for_comparison = extract_thermal_signal_over_time(original_frames, roi=roi, reducer=reducer)

	if (
		apply_average_consecutive_frames_first
		and plot_averaged_vs_non_averaged_comparison
		and non_averaged_signal_for_comparison is not None
		and fps is not None
		and thermal_fps is not None
	):
		plot_averaged_vs_non_averaged_signals(
			non_averaged_signal=non_averaged_signal_for_comparison,
			non_averaged_fps=float(fps),
			averaged_signal=signal,
			averaged_fps=float(thermal_fps),
			title="Thermal Signal Comparison: Non-Averaged vs Averaged",
		)

	if create_video:
		create_video_from_thermal_frames(
			frames,
			output_path=selected_experiment.video_output_path,
			fps=thermal_fps,
			normalization="global",
			colormap="inferno",
			roi=roi,
			roi_per_frame=face_rois,
			roi_mask_per_frame=selected_roi_masks_per_frame,
		)
	if thermal_fps is None or thermal_fps <= 0:
		raise ValueError("`fps` must be > 0 for linked thermal/ground-truth plotting.")

	plot_thermal_and_ground_truth_linked(
		thermal_signal=signal,
		thermal_fps=thermal_fps,
		ground_truth_time=ground_truth_time,
		ground_truth_signal=ground_truth_signal,
		button_time=button_time,
		button_signal=button_signal,
		offset_seconds=selected_experiment.offset_ground_truth_seconds,
		recording_length_seconds=recording_length_seconds,
		thermal_start_seconds=selected_experiment.offset_ground_truth_seconds,
		ground_truth_label=selected_ground_truth_key,
		thermal_title="Thermal Signal (Raw)",
	)

	plot_fft_thermal_and_ground_truth(
		thermal_signal=signal,
		thermal_fps=thermal_fps,
		ground_truth_signal=ground_truth_signal,
		ground_truth_fps=ground_truth__fps,
		ground_truth_label=selected_ground_truth_key,
		max_frequency_hz=max_fft_frequency_hz,
	)

	if plot_bandpass_filtered_signal:
		filtered_signal = apply_bandpass_filter(
			signal=signal,
			fps=thermal_fps,
			lowcut_hz=bandpass_lowcut_hz,
			highcut_hz=bandpass_highcut_hz,
			order=bandpass_order,
		)
		plot_thermal_and_ground_truth_linked(
			thermal_signal=filtered_signal,
			thermal_fps=thermal_fps,
			ground_truth_time=ground_truth_time,
			ground_truth_signal=ground_truth_signal,
			button_time=button_time,
			button_signal=button_signal,
			offset_seconds=selected_experiment.offset_ground_truth_seconds,
			recording_length_seconds=recording_length_seconds,
			thermal_start_seconds=selected_experiment.offset_ground_truth_seconds,
			ground_truth_label=selected_ground_truth_key,
			thermal_title=(
				"Thermal Signal (Bandpass "
				f"{bandpass_lowcut_hz:.2f}-{bandpass_highcut_hz:.2f} Hz, order={bandpass_order})"
			),
		)

		plot_fft_thermal_and_ground_truth(
			thermal_signal=filtered_signal,
			thermal_fps=thermal_fps,
			ground_truth_signal=ground_truth_signal,
			ground_truth_fps=ground_truth__fps,
			ground_truth_label=selected_ground_truth_key,
			max_frequency_hz=max_fft_frequency_hz,
		)

	


if __name__ == "__main__":
	main()
