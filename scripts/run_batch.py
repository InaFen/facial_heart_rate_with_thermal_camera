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
	"lepton": {"gaussian_kernel_size": 7, "lowcut_bandpass": 0.01, "highcut_bandpass": 0.7, "order_bandpass": 4},
	"vario": {"gaussian_kernel_size": 7, "lowcut_bandpass": 0.01, "highcut_bandpass": 0.7, "order_bandpass": 4},
	"mwir": {"gaussian_kernel_size": 9, "lowcut_bandpass": 0.01, "highcut_bandpass": 0.7, "order_bandpass": 4},
}

# new_approach: per-ROI sub-ROI grid size (rows; columns are set equal to rows).
# Larger ROIs (whole forehead) are split into more sub-ROIs than smaller ones
# (single-side forehead, glabella). ROI names not listed fall back to the
# in-script default. Passed via NEW_APPROACH_SUBROI_GRID_ROWS_BY_ROI (JSON).
NEW_APPROACH_SUBROI_GRID_ROWS_BY_ROI = {
	"forehead": 8,
	"left_forehead": 5,
	"right_forehead": 5,
	"glabella": 3,
}


def sanitize_name(value: str) -> str:
	return re.sub(r"[^A-Za-z0-9_.-]", "_", value.strip())


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
	# sub-ROIs. The drawn mask is persisted in the run dir.
	if manual_roi_enabled:
		config["processing"]["manual_roi_enabled"] = True
		config["processing"]["manual_roi_split_enabled"] = bool(manual_roi_split_enabled)
		config["processing"]["manual_roi_names"] = [roi_name]
		config["processing"]["manual_roi_modes"] = [manual_roi_mode]
		config["processing"]["manual_roi_subroi_grid"] = list(manual_roi_subroi_grid or [3, 3])
		config["processing"]["manual_roi_selection_colormap"] = bool(manual_roi_selection_colormap)
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


def main() -> int:
	# ===== CONFIGURATION (edit these variables) =====
	pipeline = "new"  # "yang", "nair", or "new"
	camera = "mwir"  # "lepton", "vario", or "mwir" (None = use pipeline default)
	roi_name = "forehead"  # e.g., "tracker_landmark_threshold_30_c" (None = use pipeline default)
	limit_rows = None  # e.g., 5 (None = process all rows)
	dry_run = False  # True to print what would be done, False to execute
	save_plots = True  # True to save plots, False otherwise
	used_rois = ["forehead"]
	# Manual ROI (used only when roi_name contains "manual"). On a headless server
	# the interactive ROI drawing opens in the browser (forwarded localhost port).
	manual_roi_mode = "rectangle"            # rectangle | polygon | point | point_surrounding
	manual_roi_split_enabled = False        # split the single manual ROI into a grid of sub-ROIs
	manual_roi_subroi_grid = [3, 3]        # [rows, cols] used when split is enabled
	manual_roi_selection_colormap = True   # apply a colormap to the drawing frame
	manual_roi_backend = "browser"         # browser | cv2 | auto
	# ===== END CONFIGURATION =====

	# Get pipeline configuration
	if pipeline not in PIPELINE_CONFIG:
		print(f"Error: Unknown pipeline {pipeline!r}. Supported: {', '.join(PIPELINE_CONFIG.keys())}")
		return 1

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

	fps = get_fps_for_camera(camera)

	# Manual ROI detection is selected purely by the roi_name containing "manual"
	# (else the tracker/mediapipe/dlib detection is used, as before).
	manual_roi_enabled = "manual" in str(roi_name).lower()
	# When manual ROI is used, process the temple recordings (the manual drawing
	# targets the temple area); otherwise process the face recordings as before.
	area_filter = "temple" if manual_roi_enabled else "face"
	# Resolve CSV and config paths
	csv_path = PROJECT_ROOT / "experiments" / "experiments_info" / f"{camera}_experiments.csv"
	output_root = PROJECT_ROOT / "saved_results" / output_subdir / "batch_runs" / camera

	# Validate paths
	if not csv_path.exists():
		print(f"Error: CSV not found: {csv_path}")
		return 1
	if not template_config_path.exists():
		print(f"Error: Template config not found: {template_config_path}")
		return 1
	if not main_script_path.exists():
		print(f"Error: Main script not found: {main_script_path}")
		return 1

	# Load template config
	with open(template_config_path, "r", encoding="utf-8") as handle:
		template_config = yaml.safe_load(handle)

	# Get the rows for the selected area from CSV (temple for manual ROI, else face)
	face_rows = iter_face_rows(csv_path, area=area_filter)
	if limit_rows is not None:
		face_rows = face_rows[:limit_rows]

	print(f"\n{'='*60}")
	print(f"Batch Runner Configuration")
	print(f"{'='*60}")
	print(f"Pipeline:        {pipeline}")
	print(f"Camera:          {camera}")
	print(f"FPS:             {fps}")
	print(f"ROI Name:        {roi_name}")
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
		print(f"DRY RUN: Would process {len(face_rows)} face rows\n")
		for idx, row in enumerate(face_rows, 1):
			print(
				f"  {idx}. {row['person'].strip()} | {row['before_after_sport'].strip()} | "
				f"{Path(row['recording_path']).stem}"
			)
		return 0

	# Process each row
	failures: list[tuple[str, str, str]] = []
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
					(run_name, f"exited with code {return_code}", error_output.strip())
				)
		except Exception as exc:
			failures.append((run_name, str(exc), traceback.format_exc()))

	if failures:
		print(f"\nBatch finished with {len(failures)} failure(s):")
		for run_name, summary, _detail in failures:
			print(f"  - {run_name}: {summary}")
		# Re-print the full error/traceback for each failed run.
		print(f"\n{'='*60}")
		print("Error details for failed runs")
		print(f"{'='*60}")
		for run_name, summary, detail in failures:
			print(f"\n----- {run_name}: {summary} -----")
			print(detail if detail.strip() else "(no error output captured)")
		return 1

	print(f"\nBatch finished successfully ({len(face_rows)} rows processed).")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
