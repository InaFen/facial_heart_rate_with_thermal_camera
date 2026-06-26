#!/usr/bin/env python3
"""Unified batch runner for Yang et al. and Nair et al. pipelines.

This script reads an experiments CSV, keeps only rows with `area == face`,
prepares a per-row temporary configuration, converts `.npy` recordings
into the frame/video layout expected by the pipeline, and launches the
main script once per row. The pipeline can be selected via --pipeline argument.

Usage:
	python run_batch.py --pipeline yang [--camera lepton] [--dry-run] [--limit 5]
	python run_batch.py --pipeline nair [--camera vario] [--dry-run]
"""

from __future__ import annotations

import csv
from datetime import datetime
import json
import os
import re
import subprocess
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Pipeline configurations
PIPELINE_CONFIG = {
	"yang": {
		"template_config": PROJECT_ROOT / "src" / "approaches" / "yang_et_al" / "configs_yang_et_al.yaml",
		"main_script": PROJECT_ROOT / "src" / "approaches" / "yang_et_al" / "main_yang_et_al.py",
		"config_env_var": "YANG_CONFIG_PATH",
		"plots_env_var": "YANG_PLOTS_SAVE_DIR",
		"output_subdir": "yang_et_al",
		"generated_config_name": "configs_yang_et_al.generated.yaml",
	},
	"nair": {
		"template_config": PROJECT_ROOT / "src" / "approaches" / "nair_et_al" / "configs_nair_et_al.yaml",
		"main_script": PROJECT_ROOT / "src" / "approaches" / "nair_et_al" / "main_nair_et_al.py",
		"config_env_var": "NAIR_CONFIG_PATH",
		"plots_env_var": "NAIR_PLOTS_SAVE_DIR",
		"output_subdir": "nair_et_al",
		"generated_config_name": "configs_nair_et_al.generated.yaml",
	},
	"new": {
		"template_config": PROJECT_ROOT / "src" / "approaches" / "new_approach" / "configs_new_approach.yaml",
		"main_script": PROJECT_ROOT / "src" / "approaches" / "new_approach" / "main_new_approach.py",
		"config_env_var": "NEW_APPROACH_CONFIG_PATH",
		"plots_env_var": "NEW_APPROACH_PLOTS_SAVE_DIR",
		"output_subdir": "new_approach",
		"generated_config_name": "configs_new_approach.generated.yaml",
	},
}

TRACK_FACE_SETTINGS_BY_CAMERA = {
	"lepton": {"block_size": 21, "adapt_center": 0.1, "adapt_center_horizontal": -0.35, "use_rescaling": False, "width_scale": 2.5},
	"vario": {"block_size": 30, "adapt_center": 0.1, "adapt_center_horizontal": 0.0, "use_rescaling": True, "width_scale": 1.0},
	"mwir": {"block_size": 21, "adapt_center": 0.1, "adapt_center_horizontal": 0.0, "use_rescaling": True, "width_scale": 1.0},
}

# Yang pipeline: per-camera temperature threshold for the tracker ROI (only keep
# pixels above this value). Used when tracker-based ROI detection is active (the
# roi name starts with "tracker"). Passed to main_yang_et_al.py via
# YANG_THRESHOLD_VALUE; cameras not listed fall back to the in-script default.
YANG_TRACKER_THRESHOLD_BY_CAMERA = {
	"lepton": 30500, # in Kelvin * 100 (since the Lepton recordings are in centi-Kelvin)
	"vario": 30,
	"mwir": 30,
}

# Yang pipeline (tracker + camera vario only): per-individual overrides keyed by a
# substring of the recording filename (case-insensitive).
#   - ina:    the tracker ROI sits too high -> a larger adapt_center pushes the
#             ROI down on the y axis (vario default adapt_center is 0.1).
#   - julian: temperature threshold 30.5
#   - paulo:  temperature threshold 29.5
YANG_VARIO_INDIVIDUAL_OVERRIDES = {
	"ina": {"adapt_center": 0.3},
	"julian": {"threshold": 30.0},
	"paulo": {"threshold": 29.5},
}

# new_approach: per-camera spatial Gaussian + temporal bandpass settings. Passed
# to main_new_approach.py via NEW_APPROACH_* env vars (override the in-script
# defaults). Tune per camera (resolution / fps) as needed.
NEW_APPROACH_FILTER_SETTINGS_BY_CAMERA = {
	"lepton": {"gaussian_kernel_size": 3, "lowcut_bandpass": 0.85, "highcut_bandpass": 4.3, "order_bandpass": 2},
	"vario": {"gaussian_kernel_size": 5, "lowcut_bandpass": 0.92, "highcut_bandpass": 10.0, "order_bandpass": 2},
	"mwir": {"gaussian_kernel_size": 9, "lowcut_bandpass": 0.86, "highcut_bandpass": 10.0, "order_bandpass": 2},
}

# new_approach: per-ROI sub-ROI grid size (rows; columns are set equal to rows).
# Larger ROIs (whole forehead) are split into more sub-ROIs than smaller ones
# (single-side forehead, glabella). ROI names not listed fall back to the
# in-script default. Passed via NEW_APPROACH_SUBROI_GRID_ROWS_BY_ROI (JSON).
NEW_APPROACH_SUBROI_GRID_ROWS_BY_ROI = {
	"forehead":  6,#[3,4,6,8],
	"left_forehead": 4,
	"right_forehead": 4,
	"glabella": 3,
	"manual_rectangle": 2,
	"manual_rectangle_front_view": 8,
	"manual_rectangle_temp": 1,
	"manual_rectangle_forehead": 2,
}


# Manual ROI masks are cached here and shared across pipelines, so the SAME video's
# ROI is drawn only once instead of being re-drawn for every pipeline. The cache key
# is (camera, recording, roi_name, mode) and deliberately excludes the pipeline: the
# drawn mask depends only on the recording's frames, and frame flip/geometry are set
# per-camera (identical across pipelines). The first run for a recording draws and
# saves the mask; later runs (other pipelines / re-runs) load it and skip drawing.
MANUAL_ROI_MASK_CACHE_DIR = PROJECT_ROOT / "saved_results" / "manual_roi_masks"

# Shared cache paths whose manual ROI has already been (re)drawn during THIS batch
# process. Used so an "overwrite" run still asks the user to redraw each video's ROI
# only ONCE and then reuses that fresh mask across the remaining pipelines.
_MANUAL_ROI_MASKS_REDRAWN_THIS_RUN: set[str] = set()


def sanitize_name(value: str) -> str:
	return re.sub(r"[^A-Za-z0-9_.-]", "_", value.strip())


def shared_manual_roi_mask_path(
	camera: str | None,
	recording_path: Path,
	roi_name: str,
	mode: str,
) -> Path:
	"""Return the pipeline-independent cache path for a recording's manual ROI mask.

	Keyed by camera / recording / roi_name / drawing mode (never the pipeline), so
	yang, nair and new all share -- and therefore reuse -- a single drawn mask for
	the same video.
	"""
	camera_dir = sanitize_name(camera or "unknown_camera")
	file_name = (
		f"{sanitize_name(recording_path.stem)}__{sanitize_name(roi_name)}"
		f"__{sanitize_name(mode)}.npy"
	)
	return MANUAL_ROI_MASK_CACHE_DIR / camera_dir / file_name


def write_batch_report(
	*,
	combinations: list[tuple[str, str | None, dict[str, Any]]],
	skipped_combinations: list[tuple[str, str | None, dict[str, Any]]],
	successful_runs: list[str],
	failures: list[tuple[str, str, str]],
	total_rows: int,
	dry_run: bool,
) -> Path:
	"""Persist a human-readable end-of-run batch report.

	The report mirrors the final terminal summary and includes:
	- all successful run identifiers,
	- all failed runs with captured error text/tracebacks,
	- skipped combinations due to pipeline-specific roi_name routing.
	"""
	reports_dir = PROJECT_ROOT / "saved_results" / "batch_reports"
	reports_dir.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	report_path = reports_dir / f"run_batch_report_{timestamp}.txt"

	with open(report_path, "w", encoding="utf-8") as handle:
		handle.write("Run Batch Report\n")
		handle.write("=" * 80 + "\n")
		handle.write(f"Timestamp: {timestamp}\n")
		handle.write(f"Dry run: {dry_run}\n")
		handle.write(f"Combinations considered: {len(combinations) + len(skipped_combinations)}\n")
		handle.write(f"Combinations run: {len(combinations)}\n")
		handle.write(f"Combinations skipped: {len(skipped_combinations)}\n")
		handle.write(f"Total rows processed: {total_rows}\n")
		handle.write(f"Successful runs: {len(successful_runs)}\n")
		handle.write(f"Failed runs: {len(failures)}\n")

		handle.write("\nSuccessful runs\n")
		handle.write("-" * 80 + "\n")
		if successful_runs:
			for run_name in successful_runs:
				handle.write(f"- {run_name}\n")
		else:
			handle.write("(none)\n")

		handle.write("\nSkipped combinations\n")
		handle.write("-" * 80 + "\n")
		if skipped_combinations:
			for pipeline, camera, roi_config in skipped_combinations:
				required = required_pipeline_for_roi_name(roi_config.get("roi_name"))
				handle.write(
					f"- pipeline={pipeline} camera={camera} roi_name={roi_config.get('roi_name')} "
					f"used_rois={roi_config.get('used_rois')} (requires pipeline={required})\n"
				)
		else:
			handle.write("(none)\n")

		handle.write("\nFailures with error details\n")
		handle.write("-" * 80 + "\n")
		if failures:
			for run_name, summary, detail in failures:
				handle.write(f"\n[{run_name}] {summary}\n")
				handle.write(detail.strip() if detail.strip() else "(no error output captured)")
				handle.write("\n")
		else:
			handle.write("(none)\n")

	return report_path


def parse_timespan(value: str) -> list[float]:
	parts = [part.strip() for part in str(value).split(",") if part.strip()]
	if len(parts) != 2:
		raise ValueError(f"Invalid timeframe_use value {value!r}; expected 'start,end'.")
	return [float(parts[0]), float(parts[1])]


def resolve_existing_video_path(recording_path: Path) -> Path:
	folder = recording_path.parent
	stem = recording_path.stem

	candidates = []
	thermal_suffix = "_thermal_data_all_frames"
	if stem.endswith(thermal_suffix):
		candidates.append(folder / f"{stem.removesuffix(thermal_suffix)}_video_visualization.mp4")
	candidates.extend(
		[
			folder / f"{stem}_video_visualization.mp4",
			folder / f"{recording_path.stem}.mp4",
		]
	)

	for candidate in candidates:
		if candidate.exists():
			print(f"Found existing video for {recording_path.name}: {candidate}")
			return candidate

	expected = candidates[0] if candidates else folder / f"{stem}_video_visualization.mp4"
	raise FileNotFoundError(
		f"Could not find a prebuilt video next to {recording_path}. Expected something like {expected.name}."
	)


def prepare_row_inputs(recording_path: Path, run_dir: Path) -> Path:
	print(f"Using recording: {recording_path.name}")
	existing_video = resolve_existing_video_path(recording_path)
	print(f"Using existing video: {existing_video}")
	return existing_video


def build_row_config(
	template_config: dict[str, Any],
	row: dict[str, str],
	run_dir: Path,
	fps: float,
	roi_name: str,
	camera: str | None = None,
	used_rois: list[str] | None = None,
	manual_roi_enabled: bool = False,
	manual_roi_mode: str = "polygon",
	manual_roi_split_enabled: bool = False,
	manual_roi_subroi_grid: list[int] | None = None,
	manual_roi_selection_colormap: bool = True,
	reuse_manual_roi_masks: bool = True,
	overwrite_manual_roi_masks: bool = False,
) -> Path:
	recording_path = Path(row["recording_path"])
	base_path = PROJECT_ROOT
	ground_truth_path = base_path / row["ground_truth_path"]
	timespan = parse_timespan(row["timeframe_use"])
	offset_ground_truth = float(row["offset_ground_truth"])

	print(f"Preparing inputs for recording: {recording_path}")
	existing_video = prepare_row_inputs(recording_path, run_dir)

	config = deepcopy(template_config)
	config.setdefault("paths", {})
	config.setdefault("processing", {})
	config.setdefault("timing", {})

	config["paths"]["gray_frames"] = str(recording_path)
	config["paths"]["video"] = str(existing_video)
	config["paths"]["video_tracked_face"] = str(run_dir / "video_tracked_face.mp4")
	config["paths"]["video_thresholded_pixel"] = str(run_dir / "video_thresholded_pixel.mp4")
	config["paths"]["video_facemesh"] = str(run_dir / "video_facemesh.mp4")
	config["paths"]["pixels_log"] = str(run_dir / "roi_pixel_log.npy")
	config["paths"]["video_thermal_with_roi_masks"] = str(run_dir / "video_with_roi_masks.mp4")
	config["paths"]["roi_pixel_data"] = str(run_dir / "roi_pixel_data.npy")
	config["paths"]["ground_truth"] = str(ground_truth_path)

	config["processing"]["fps"] = fps
	config["processing"]["roi_name"] = roi_name
	config["processing"]["used_rois"] = used_rois

	config["timing"]["time_first_frame_signal"] = offset_ground_truth
	config["timing"]["time_first_frame_ecg"] = 0.0
	config["timing"]["timespan_signal_to_look_at"] = timespan

	config["processing"]["plot_steps"] = False

	# Manual ROI selection (enabled when the roi_name contains "manual"). The single
	# manual ROI is named after roi_name and is optionally split into a grid of
	# sub-ROIs.
	if manual_roi_enabled:
		config["processing"]["manual_roi_enabled"] = True
		config["processing"]["manual_roi_split_enabled"] = bool(manual_roi_split_enabled)
		config["processing"]["manual_roi_names"] = [roi_name]
		config["processing"]["manual_roi_modes"] = [manual_roi_mode]
		config["processing"]["manual_roi_subroi_grid"] = list(manual_roi_subroi_grid or [3, 3])
		config["processing"]["manual_roi_selection_colormap"] = bool(manual_roi_selection_colormap)
		if reuse_manual_roi_masks:
			# Share one drawn mask across all pipelines for this recording: save it to
			# a pipeline-independent cache and, whenever it already exists (drawn by an
			# earlier pipeline/run), load it so the interactive drawing is skipped.
			shared_mask_path = shared_manual_roi_mask_path(camera, recording_path, roi_name, manual_roi_mode)
			shared_mask_path.parent.mkdir(parents=True, exist_ok=True)
			config["processing"]["manual_roi_mask_save_paths"] = [str(shared_mask_path)]
			shared_mask_key = str(shared_mask_path)
			# "overwrite" forces a fresh user-drawn ROI even when a cached mask already
			# exists on disk -- but only ONCE per video this run: after the first
			# pipeline redraws (overwriting the cache), the remaining pipelines reuse it
			# via _MANUAL_ROI_MASKS_REDRAWN_THIS_RUN.
			force_redraw = (
				overwrite_manual_roi_masks
				and shared_mask_key not in _MANUAL_ROI_MASKS_REDRAWN_THIS_RUN
			)
			if force_redraw:
				config["processing"]["manual_roi_mask_load_paths"] = [None]
				_MANUAL_ROI_MASKS_REDRAWN_THIS_RUN.add(shared_mask_key)
				print(f"Overwrite: will redraw manual ROI and overwrite cache at: {shared_mask_path}")
			elif shared_mask_path.exists():
				config["processing"]["manual_roi_mask_load_paths"] = [shared_mask_key]
				print(f"Reusing cached manual ROI mask (no re-draw): {shared_mask_path}")
			else:
				config["processing"]["manual_roi_mask_load_paths"] = [None]
				print(f"Will draw manual ROI once and cache it at: {shared_mask_path}")
		else:
			# Per-run mask (previous behaviour): draw the ROI fresh for every run.
			config["processing"]["manual_roi_mask_save_paths"] = [str(run_dir / f"manual_roi_mask_{sanitize_name(roi_name)}.npy")]
			config["processing"]["manual_roi_mask_load_paths"] = [None]
	else:
		config["processing"]["manual_roi_enabled"] = False

	config_path = run_dir / "configs_generated.yaml"
	print(
		"Row settings:\n"
		f"  recording={recording_path}\n"
		f"  video={existing_video}\n"
		f"  ground_truth={ground_truth_path}\n"
		f"  timespan={timespan}\n"
		f"  offset_ground_truth={offset_ground_truth}\n"
		f"  fps={fps}\n"
		f"  roi_name={roi_name}"
	)
	with open(config_path, "w", encoding="utf-8") as handle:
		yaml.safe_dump(config, handle, sort_keys=False)
	print(f"Wrote generated config to: {config_path}")
	return config_path


def iter_face_rows(csv_path: Path, area: str = "face") -> list[dict[str, str]]:
	with open(csv_path, newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle))
	target = area.strip().lower()
	return [row for row in rows if row.get("area", "").strip().lower() == target]


# The individuals recorded across every camera CSV (the `person` column uses these
# lowercase names). Used by the optional per-individual batch filter.
KNOWN_INDIVIDUALS = ("ina", "julian", "paulo")


def normalize_individuals_filter(
	individuals: tuple[str, ...] | list[str] | str | None,
) -> set[str] | None:
	"""Return a lowercase set of individual names to keep, or ``None`` for all.

	Accepts a single name, an iterable of names, or ``None``/empty (meaning no
	filtering). Unknown names are kept as-is (so custom names still work) but a
	warning is printed when a name matches none of ``KNOWN_INDIVIDUALS``.
	"""
	if individuals is None:
		return None
	if isinstance(individuals, str):
		individuals = [individuals]
	wanted = {
		str(name).strip().lower()
		for name in individuals
		if name is not None and str(name).strip()
	}
	if not wanted:
		return None
	unknown = wanted - set(KNOWN_INDIVIDUALS)
	if unknown:
		print(
			f"Warning: individuals filter contains name(s) not in {KNOWN_INDIVIDUALS}: "
			f"{sorted(unknown)}. They only match if a CSV 'person'/recording name contains them."
		)
	return wanted


def filter_rows_by_individual(
	rows: list[dict[str, str]],
	individuals: set[str] | None,
) -> list[dict[str, str]]:
	"""Keep only rows whose individual is in ``individuals`` (``None``/empty keeps all).

	Each requested name is matched (case-insensitively) against the row's ``person``
	column and, as a fallback, its recording filename stem -- so "paulo" selects both
	the mwir/lepton recordings (person='paulo') and the vario ones whose file is named
	'P1_thermal_data_all_frames' but whose person column is still 'paulo'.
	"""
	if not individuals:
		return rows
	kept: list[dict[str, str]] = []
	for row in rows:
		person = str(row.get("person", "")).strip().lower()
		stem = Path(str(row.get("recording_path", ""))).stem.lower()
		if any(want == person or want in person or want in stem for want in individuals):
			kept.append(row)
	return kept


def run_one_row(
	main_script: Path,
	config_path: Path,
	*,
	cwd: Path,
	plots_save_dir: Path | None = None,
	camera: str | None = None,
	pipeline: str | None = None,
	roi_name: str | None = None,
	individual: str | None = None,
	manual_roi_enabled: bool = False,
	manual_roi_backend: str = "browser",
	config_env_var: str = "CONFIG_PATH",
	plots_env_var: str = "PLOTS_SAVE_DIR",
) -> tuple[int, str]:
	env = os.environ.copy()
	env[config_env_var] = str(config_path)
	if plots_save_dir is not None:
		env[plots_env_var] = str(plots_save_dir)
	if camera:
		env["CAMERA_NAME"] = str(camera)
	# Manual ROI: choose the interactive backend (browser on headless servers) and
	# tell the pipeline to save the selection overlay into the plots directory.
	if manual_roi_enabled:
		env["MANUAL_ROI_BACKEND"] = str(manual_roi_backend)
		if plots_save_dir is not None:
			env["MANUAL_ROI_SELECTION_SAVE_DIR"] = str(plots_save_dir)
	# new_approach: camera-dependent filter settings + ROI-dependent sub-ROI grid.
	if pipeline == "new":
		filter_settings = NEW_APPROACH_FILTER_SETTINGS_BY_CAMERA.get(camera or "", {})
		if filter_settings:
			env["NEW_APPROACH_GAUSSIAN_KERNEL_SIZE"] = str(filter_settings["gaussian_kernel_size"])
			env["NEW_APPROACH_LOWCUT_BANDPASS"] = str(filter_settings["lowcut_bandpass"])
			env["NEW_APPROACH_HIGHCUT_BANDPASS"] = str(filter_settings["highcut_bandpass"])
			env["NEW_APPROACH_ORDER_BANDPASS"] = str(filter_settings["order_bandpass"])
		env["NEW_APPROACH_SUBROI_GRID_ROWS_BY_ROI"] = json.dumps(NEW_APPROACH_SUBROI_GRID_ROWS_BY_ROI)
	# yang: use tracker ROI detection when the roi name starts with "tracker"
	# (else MediaPipe facemesh), and pick the camera-dependent threshold value.
	use_tracker = (
		pipeline == "yang"
		and bool(roi_name)
		and str(roi_name).strip().lower().startswith("tracker")
	)
	# Per-individual overrides (yang + vario + tracker only), keyed by a substring
	# of the recording filename: ina -> push the ROI down (adapt_center),
	# julian/paulo -> per-individual temperature threshold.
	individual_override: dict[str, float] = {}
	if pipeline == "yang" and camera == "vario" and use_tracker and individual:
		individual_lower = str(individual).lower()
		for key, override in YANG_VARIO_INDIVIDUAL_OVERRIDES.items():
			if key in individual_lower:
				individual_override = override
				break
	if pipeline == "yang":
		env["YANG_APPLY_TRACKER_LANDMARK_DETECTION"] = "1" if use_tracker else "0"
		threshold = YANG_TRACKER_THRESHOLD_BY_CAMERA.get(camera or "")
		# Per-individual temperature threshold override (e.g. julian 30.5, paulo 29.5).
		if "threshold" in individual_override:
			threshold = individual_override["threshold"]
		if threshold is not None:
			env["YANG_THRESHOLD_VALUE"] = str(threshold)
	# nair: use dlib ROI detection when the roi name contains "dlib" (else MediaPipe
	# facemesh). Read by main_nair_et_al.py via NAIR_APPLY_DLIB_LANDMARK_DETECTION.
	if pipeline == "nair":
		use_dlib = bool(roi_name) and "dlib" in str(roi_name).strip().lower()
		env["NAIR_APPLY_DLIB_LANDMARK_DETECTION"] = "1" if use_dlib else "0"
	track_face_settings = TRACK_FACE_SETTINGS_BY_CAMERA.get(camera or "", {})
	if track_face_settings:
		# Per-individual vertical ROI shift override (e.g. ina: ROI too high -> down).
		adapt_center = individual_override.get("adapt_center", track_face_settings["adapt_center"])
		env["TRACK_FACE_BLOCK_SIZE"] = str(track_face_settings["block_size"])
		env["TRACK_FACE_ADAPT_CENTER"] = str(adapt_center)
		env["TRACK_FACE_ADAPT_CENTER_HORIZONTAL"] = str(track_face_settings["adapt_center_horizontal"])
		env["TRACK_FACE_USE_RESCALING"] = "1" if track_face_settings["use_rescaling"] else "0"
		env["TRACK_FACE_WIDTH_SCALE"] = str(track_face_settings.get("width_scale", 1.0))
	# FLIP_UPSIDE_DOWN: True for lepton camera
	if camera == "lepton":
		env["FLIP_UPSIDE_DOWN"] = "1"
	else:
		env["FLIP_UPSIDE_DOWN"] = "0"
	# Capture stderr so the traceback can be re-printed in the final summary,
	# while stdout still streams live. The captured stderr is echoed right after
	# the run so warnings/tracebacks remain visible during the batch.
	completed = subprocess.run(
		[sys.executable, str(main_script)],
		cwd=str(cwd),
		env=env,
		stderr=subprocess.PIPE,
		text=True,
	)
	error_output = completed.stderr or ""
	if error_output:
		print(error_output, file=sys.stderr, end="" if error_output.endswith("\n") else "\n")
	return completed.returncode, error_output


def get_fps_for_camera(camera: str) -> float:
	fps_map = {
		"mwir": 100.0,
		"vario": 25.0,
		"lepton": 8.76,
	}
	if camera not in fps_map:
		raise ValueError(f"Unknown camera type {camera!r}. Supported: {', '.join(fps_map.keys())}")
	return fps_map[camera]


def required_pipeline_for_roi_name(roi_name: str | None) -> str | None:
	"""Return the pipeline a roi_name is restricted to, or None if unrestricted.

	Some ROI detection methods are pipeline-specific:
	  - "tracker" in roi_name -> only the yang pipeline
	  - "dlib"    in roi_name -> only the nair pipeline
	Any other roi_name may run with any pipeline.
	"""
	name = str(roi_name or "").lower()
	if "tracker" in name:
		return "yang"
	if "dlib" in name:
		return "nair"
	return None


def is_combination_allowed(pipeline: str, roi_name: str | None) -> bool:
	"""Whether a (pipeline, roi_name) combination should run.

	"tracker" roi_names run only with yang and "dlib" roi_names only with nair;
	all other roi_names are allowed for every pipeline.
	"""
	required = required_pipeline_for_roi_name(roi_name)
	return required is None or required == pipeline


def run_combination(
	pipeline: str,
	camera: str | None,
	roi_name: str | None,
	used_rois: list[str] | None,
	*,
	individuals: set[str] | None = None,
	limit_rows: int | None,
	dry_run: bool,
	save_plots: bool,
	manual_roi_mode: str,
	manual_roi_split_enabled: bool,
	manual_roi_subroi_grid: list[int],
	manual_roi_selection_colormap: bool,
	manual_roi_backend: str,
	reuse_manual_roi_masks: bool = True,
	overwrite_manual_roi_masks: bool = False,
) -> tuple[list[tuple[str, str, str]], list[str], int]:
	"""Run a single (pipeline, camera, roi_name/used_rois) combination.

	Returns a tuple of (failures, successful_runs, rows_processed).
	``failures`` is a list of (run_name, summary, detail) tuples so the caller
	can build one global summary across all combinations.
	"""
	combo_label = f"{pipeline}/{camera}/{roi_name}"

	# Get pipeline configuration
	if pipeline not in PIPELINE_CONFIG:
		print(f"Error: Unknown pipeline {pipeline!r}. Supported: {', '.join(PIPELINE_CONFIG.keys())}")
		return ([(combo_label, f"unknown pipeline {pipeline!r}", "")], [], 0)

	cfg = PIPELINE_CONFIG[pipeline]
	template_config_path = cfg["template_config"]
	main_script_path = cfg["main_script"]
	config_env_var = cfg["config_env_var"]
	plots_env_var = cfg["plots_env_var"]
	output_subdir = cfg["output_subdir"]

	# Set defaults based on pipeline if not provided
	if camera is None:
		camera = "lepton" if pipeline in {"yang", "new"} else "vario"
	if roi_name is None:
		roi_name = "tracker_landmark_threshold_30_c" if pipeline in {"yang", "new"} else "dlib"
	combo_label = f"{pipeline}/{camera}/{roi_name}"

	fps = get_fps_for_camera(camera)

	# Manual ROI detection is selected purely by the roi_name containing "manual"
	# (else the tracker/mediapipe/dlib detection is used, as before).
	manual_roi_enabled = "manual" in str(roi_name).lower()
	# A "front view" manual ROI (roi_name contains "front_view", e.g.
	# "manual_rectangle_front_view") is drawn on the FRONT-view recordings
	# (area == "face") -- i.e. exactly the files for which manual detection is
	# normally NOT applied. Every other manual ROI targets the temple recordings
	# (the manual drawing targets the temple area); non-manual detection processes
	# the face recordings as before.
	manual_roi_front_view = manual_roi_enabled and "front_view" in str(roi_name).lower()
	if manual_roi_front_view:
		area_filter = "face"
	elif manual_roi_enabled:
		area_filter = "temple"
	else:
		area_filter = "face"
	# Resolve CSV and config paths
	csv_path = PROJECT_ROOT / "experiments" / "experiments_info" / f"{camera}_experiments.csv"
	# Group runs by roi_name under the camera dir so multiple ROI configs for the
	# same camera/pipeline don't overwrite each other.
	output_root = (
		PROJECT_ROOT / "saved_results" / output_subdir / "batch_runs" / camera / sanitize_name(roi_name)
	)

	# Validate paths
	if not csv_path.exists():
		print(f"Error: CSV not found: {csv_path}")
		return ([(combo_label, f"CSV not found: {csv_path}", "")], [], 0)
	if not template_config_path.exists():
		print(f"Error: Template config not found: {template_config_path}")
		return ([(combo_label, f"template config not found: {template_config_path}", "")], [], 0)
	if not main_script_path.exists():
		print(f"Error: Main script not found: {main_script_path}")
		return ([(combo_label, f"main script not found: {main_script_path}", "")], [], 0)

	# Load template config
	with open(template_config_path, "r", encoding="utf-8") as handle:
		template_config = yaml.safe_load(handle)

	# Get the rows for the selected area from CSV (temple for manual ROI, else face)
	face_rows = iter_face_rows(csv_path, area=area_filter)
	# Optional per-individual filter (e.g. only "paulo"); None keeps all individuals.
	face_rows = filter_rows_by_individual(face_rows, individuals)
	if limit_rows is not None:
		face_rows = face_rows[:limit_rows]

	print(f"\n{'='*60}")
	print(f"Combination Configuration")
	print(f"{'='*60}")
	print(f"Pipeline:        {pipeline}")
	print(f"Camera:          {camera}")
	print(f"FPS:             {fps}")
	print(f"ROI Name:        {roi_name}")
	print(f"Used ROIs:       {used_rois}")
	print(f"Individuals:     {sorted(individuals) if individuals else 'all'}")
	print(f"Manual ROI:      {manual_roi_enabled}"
	      + (f" (mode={manual_roi_mode}, split={manual_roi_split_enabled}, grid={manual_roi_subroi_grid}, backend={manual_roi_backend})" if manual_roi_enabled else ""))
	print(f"Area Filter:     {area_filter}")
	print(f"CSV:             {csv_path}")
	print(f"Output Root:     {output_root}")
	print(f"Face Rows:       {len(face_rows)}")
	print(f"Dry Run:         {dry_run}")
	print(f"Save Plots:      {save_plots}")
	print(f"{'='*60}\n")

	# Create output root
	output_root.mkdir(parents=True, exist_ok=True)

	if dry_run:
		print(f"DRY RUN: Would process {len(face_rows)} rows for {combo_label}\n")
		for idx, row in enumerate(face_rows, 1):
			print(
				f"  {idx}. {row['person'].strip()} | {row['before_after_sport'].strip()} | "
				f"{Path(row['recording_path']).stem}"
			)
		return ([], [], 0)

	# Process each row
	failures: list[tuple[str, str, str]] = []
	successful_runs: list[str] = []
	for index, row in enumerate(face_rows, start=1):
		run_name = sanitize_name(
			f"{index:02d}_{camera}_{Path(row['recording_path']).stem}_{row['person'].strip()}_{row['before_after_sport'].strip()}_{row['area'].strip()}"
		)
		run_dir = output_root / run_name
		run_dir.mkdir(parents=True, exist_ok=True)

		try:
			config_path = build_row_config(
				template_config,
				row,
				run_dir,
				fps,
				roi_name,
				camera=camera,
				used_rois=(None if manual_roi_enabled else used_rois),
				manual_roi_enabled=manual_roi_enabled,
				manual_roi_mode=manual_roi_mode,
				manual_roi_split_enabled=manual_roi_split_enabled,
				manual_roi_subroi_grid=manual_roi_subroi_grid,
				manual_roi_selection_colormap=manual_roi_selection_colormap,
				reuse_manual_roi_masks=reuse_manual_roi_masks,
				overwrite_manual_roi_masks=overwrite_manual_roi_masks,
			)
			print(f"[{index}/{len(face_rows)}] Running {run_name}")
			plots_save_dir = (run_dir / "plots") if save_plots else None
			return_code, error_output = run_one_row(
				main_script_path,
				config_path,
				cwd=PROJECT_ROOT,
				plots_save_dir=plots_save_dir,
				camera=camera,
				pipeline=pipeline,
				roi_name=roi_name,
				individual=Path(row["recording_path"]).stem,
				manual_roi_enabled=manual_roi_enabled,
				manual_roi_backend=manual_roi_backend,
				config_env_var=config_env_var,
				plots_env_var=plots_env_var,
			)
			if return_code != 0:
				failures.append(
					(f"{combo_label} :: {run_name}", f"exited with code {return_code}", error_output.strip())
				)
			else:
				successful_runs.append(f"{combo_label} :: {run_name}")
		except Exception as exc:
			failures.append((f"{combo_label} :: {run_name}", str(exc), traceback.format_exc()))

	if failures:
		print(f"\n{combo_label}: finished with {len(failures)} failure(s).")
	else:
		print(f"\n{combo_label}: finished successfully ({len(face_rows)} rows processed).")
	return (failures, successful_runs, len(face_rows))


def main() -> int:
	# ===== CONFIGURATION (edit these variables) =====
	# The batch runs for every combination of pipeline x camera x ROI config below.
	# Put a single entry in a list to run just one (previous behaviour).
	pipelines = ["own", "nair", "yang"]              # any of "yang", "nair", "new"
	cameras = ["lepton", "vario", "mwir"]#, "vario", "lepton"]               # any of "lepton", "vario", "mwir" (None = pipeline default)
	# Restrict the batch to specific individual(s): any of "ina", "julian", "paulo".
	# None (or []) processes every individual. Matched case-insensitively against each
	# CSV row's `person` column, e.g. individuals = ["paulo"] or ["ina", "julian"].
	individuals = ["ina", "julian", "paulo"]  # None or list of individual names to include (matched against CSV `person` column)
	# Each ROI config pairs a roi_name with its used_rois list; the batch runs each.
	roi_configs = [
		#{"roi_name": "tracker", "used_rois": ["forehead"]},
		#{"roi_name": "dlib", "used_rois": ["forehead"]},
		#{"roi_name": "forehead", "used_rois": ["forehead"]},
		#{"roi_name": "glabella", "used_rois": ["glabella"]},
		#{"roi_name": "left_forehead", "used_rois": ["left_forehead"]},
		#{"roi_name": "right_forehead", "used_rois": ["right_forehead"]},
		#{"roi_name": "whole_face", "used_rois": ["whole_face"]},
		#{"roi_name": "manual_rectangle", "used_rois": ["forehead"]}, # side view
		{"roi_name": "manual_rectangle_forehead", "used_rois": ["forehead"]}, # side view
		#{"roi_name": "manual_rectangle_front_view", "used_rois": ["forehead"]},
	]
	limit_rows = None  # e.g., 5 (None = process all rows)
	dry_run = False  # True to print what would be done, False to execute
	save_plots = True  # True to save plots, False otherwise
	# Manual ROI (used only when a roi_name contains "manual"). On a headless server
	# the interactive ROI drawing opens in the browser (forwarded localhost port).
	manual_roi_mode = "rectangle"            # rectangle | polygon | point | point_surrounding
	manual_roi_split_enabled = False        # split the single manual ROI into a grid of sub-ROIs
	manual_roi_subroi_grid = [3, 3]        # [rows, cols] used when split is enabled
	manual_roi_selection_colormap = True   # apply a colormap to the drawing frame
	manual_roi_backend = "browser"         # browser | cv2 | auto
	# Draw each video's manual ROI only ONCE and reuse it across all pipelines (and
	# across re-runs): the mask is cached per (camera, recording, roi_name, mode) in
	# saved_results/manual_roi_masks. Set False to draw a fresh ROI for every run.
	reuse_manual_roi_masks = True
	# Force the user to redraw the manual ROI even when a cached mask already exists
	# (the fresh drawing overwrites the cache). Still redrawn only ONCE per video per
	# run, then reused across the remaining pipelines. False = reuse existing caches.
	overwrite_manual_roi_masks = False
	# ===== END CONFIGURATION =====

	# A redraw triggered by overwrite_manual_roi_masks should happen only once per
	# video for this run; reset the per-run tracker before processing combinations.
	_MANUAL_ROI_MASKS_REDRAWN_THIS_RUN.clear()

	# Allow single values instead of lists for convenience.
	if isinstance(pipelines, str):
		pipelines = [pipelines]
	if isinstance(cameras, str) or cameras is None:
		cameras = [cameras]
	if isinstance(roi_configs, dict):
		roi_configs = [roi_configs]
	# Normalize the optional per-individual filter to a lowercase set (None = all).
	individuals_filter = normalize_individuals_filter(individuals)

	# Cartesian product: pipeline x camera x ROI config.
	all_combinations = [
		(pipeline, camera, roi_config)
		for pipeline in pipelines
		for camera in cameras
		for roi_config in roi_configs
	]

	# Some ROI detection methods are pipeline-specific: a "tracker" roi_name runs
	# only the yang pipeline and a "dlib" roi_name only the nair pipeline. Skip the
	# combinations whose pipeline doesn't match the roi_name's required pipeline.
	combinations = []
	skipped_combinations = []
	for combo in all_combinations:
		combo_pipeline, _combo_camera, combo_roi_config = combo
		if is_combination_allowed(combo_pipeline, combo_roi_config.get("roi_name")):
			combinations.append(combo)
		else:
			skipped_combinations.append(combo)

	print(f"\n{'#'*60}")
	print(f"Batch Runner: {len(combinations)} combination(s) to run")
	if individuals_filter:
		print(f"Restricted to individual(s): {sorted(individuals_filter)}")
	print(f"{'#'*60}")
	for idx, (pipeline, camera, roi_config) in enumerate(combinations, 1):
		print(
			f"  {idx}. pipeline={pipeline} camera={camera} "
			f"roi_name={roi_config.get('roi_name')} used_rois={roi_config.get('used_rois')}"
		)
	if skipped_combinations:
		print(f"\nSkipped {len(skipped_combinations)} combination(s) (ROI detection is pipeline-specific):")
		for (pipeline, camera, roi_config) in skipped_combinations:
			required = required_pipeline_for_roi_name(roi_config.get("roi_name"))
			print(
				f"  - pipeline={pipeline} camera={camera} "
				f"roi_name={roi_config.get('roi_name')}: only the {required!r} pipeline runs this roi_name"
			)
	print(f"{'#'*60}")

	all_failures: list[tuple[str, str, str]] = []
	all_successful_runs: list[str] = []
	total_rows = 0
	for idx, (pipeline, camera, roi_config) in enumerate(combinations, 1):
		print(f"\n{'#'*60}")
		print(
			f"# Combination {idx}/{len(combinations)}: pipeline={pipeline} "
			f"camera={camera} roi_name={roi_config.get('roi_name')}"
		)
		print(f"{'#'*60}")
		failures, successful_runs, rows = run_combination(
			pipeline,
			camera,
			roi_config.get("roi_name"),
			roi_config.get("used_rois"),
			individuals=individuals_filter,
			limit_rows=limit_rows,
			dry_run=dry_run,
			save_plots=save_plots,
			manual_roi_mode=manual_roi_mode,
			manual_roi_split_enabled=manual_roi_split_enabled,
			manual_roi_subroi_grid=manual_roi_subroi_grid,
			manual_roi_selection_colormap=manual_roi_selection_colormap,
			manual_roi_backend=manual_roi_backend,
			reuse_manual_roi_masks=reuse_manual_roi_masks,
			overwrite_manual_roi_masks=overwrite_manual_roi_masks,
		)
		all_failures.extend(failures)
		all_successful_runs.extend(successful_runs)
		total_rows += rows

	# Global summary across all combinations.
	report_path = write_batch_report(
		combinations=combinations,
		skipped_combinations=skipped_combinations,
		successful_runs=all_successful_runs,
		failures=all_failures,
		total_rows=total_rows,
		dry_run=dry_run,
	)
	print(f"Saved batch report to: {report_path}")

	if all_failures:
		print(f"\nBatch finished with {len(all_failures)} failure(s) across {len(combinations)} combination(s):")
		for run_name, summary, _detail in all_failures:
			print(f"  - {run_name}: {summary}")
		# Re-print the full error/traceback for each failed run.
		print(f"\n{'='*60}")
		print("Error details for failed runs")
		print(f"{'='*60}")
		for run_name, summary, detail in all_failures:
			print(f"\n----- {run_name}: {summary} -----")
			print(detail if detail.strip() else "(no error output captured)")
		return 1

	if dry_run:
		print(f"\nDry run complete for {len(combinations)} combination(s).")
		return 0

	print(
		f"\nBatch finished successfully "
		f"({len(combinations)} combination(s), {total_rows} rows processed)."
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
