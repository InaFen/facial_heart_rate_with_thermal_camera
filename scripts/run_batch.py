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
import os
import re
import subprocess
import sys
from copy import deepcopy
import shutil
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
) -> Path:
    recording_path = Path(row["recording_path"])
    base_path = PROJECT_ROOT
    ground_truth_path = base_path / row["ground_truth_path"]
    timespan = parse_timespan(row["timeframe_use"])
    offset_ground_truth = float(row["offset_ground_truth"])

    print(f"Preparing inputs for recording: {recording_path}")
    existing_video = prepare_row_inputs(recording_path, run_dir)

    # Prepare per-run copy (or fallback) of the .npy with camera prefix if requested
    if camera:
        new_recording_name = f"{camera}_{recording_path.name}"
    else:
        new_recording_name = recording_path.name
    new_recording_path = run_dir / new_recording_name
    if not new_recording_path.exists():
        try:
            os.symlink(recording_path, new_recording_path)
            print(f"Created symlink to recording: {new_recording_path} -> {recording_path}")
        except Exception as exc:
            try:
                shutil.copy2(recording_path, new_recording_path)
                print(f"Symlink failed; copied recording to: {new_recording_path}")
            except Exception as exc2:
                print(f"Warning: failed to link or copy recording to run dir: {exc2}. Using original path.")
                new_recording_path = recording_path

    config = deepcopy(template_config)
    config.setdefault("paths", {})
    config.setdefault("processing", {})
    config.setdefault("timing", {})

    config["paths"]["gray_frames"] = str(new_recording_path)
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

    config["timing"]["time_first_frame_signal"] = offset_ground_truth
    config["timing"]["time_first_frame_ecg"] = 0.0
    config["timing"]["timespan_signal_to_look_at"] = timespan

    config["processing"]["plot_steps"] = False

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


def iter_face_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("area", "").strip().lower() == "face"]


def run_one_row(
    main_script: Path,
    config_path: Path,
    *,
    cwd: Path,
    plots_save_dir: Path | None = None,
    camera: str | None = None,
    config_env_var: str = "CONFIG_PATH",
    plots_env_var: str = "PLOTS_SAVE_DIR",
) -> int:
    env = os.environ.copy()
    env[config_env_var] = str(config_path)
    if plots_save_dir is not None:
        env[plots_env_var] = str(plots_save_dir)
    # FLIP_UPSIDE_DOWN: True for lepton camera
    if camera == "lepton":
        env["FLIP_UPSIDE_DOWN"] = "1"
    else:
        env["FLIP_UPSIDE_DOWN"] = "0"
    completed = subprocess.run([sys.executable, str(main_script)], cwd=str(cwd), env=env)
    return completed.returncode


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
    pipeline = "yang"  # "yang" or "nair"
    camera = "lepton"  # "lepton", "vario", or "mwir" (None = use pipeline default)
    roi_name = "tracker_landmark_threshold_30_c"  # e.g., "tracker_landmark_threshold_30_c" (None = use pipeline default)
    limit_rows = None  # e.g., 5 (None = process all rows)
    dry_run = False  # True to print what would be done, False to execute
    save_plots = True  # True to save plots, False otherwise
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
        camera = "lepton" if pipeline == "yang" else "vario"
    if roi_name is None:
        roi_name = "tracker_landmark_threshold_30_c" if pipeline == "yang" else "dlib"

    fps = get_fps_for_camera(camera)

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

    # Get face rows from CSV
    face_rows = iter_face_rows(csv_path)
    if limit_rows is not None:
        face_rows = face_rows[:limit_rows]

    print(f"\n{'='*60}")
    print(f"Batch Runner Configuration")
    print(f"{'='*60}")
    print(f"Pipeline:        {pipeline}")
    print(f"Camera:          {camera}")
    print(f"FPS:             {fps}")
    print(f"ROI Name:        {roi_name}")
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
    failures: list[str] = []
    for index, row in enumerate(face_rows, start=1):
        run_name = sanitize_name(
            f"{index:02d}_{camera}_{Path(row['recording_path']).stem}_{row['person'].strip()}_{row['before_after_sport'].strip()}_{row['area'].strip()}"
        )
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            config_path = build_row_config(template_config, row, run_dir, fps, roi_name, camera=camera)
            print(f"[{index}/{len(face_rows)}] Running {run_name}")
            plots_save_dir = (run_dir / "plots") if save_plots else None
            return_code = run_one_row(
                main_script_path,
                config_path,
                cwd=PROJECT_ROOT,
                plots_save_dir=plots_save_dir,
                camera=camera,
                config_env_var=config_env_var,
                plots_env_var=plots_env_var,
            )
            if return_code != 0:
                failures.append(f"{run_name}: exited with code {return_code}")
        except Exception as exc:
            failures.append(f"{run_name}: {exc}")

    if failures:
        print(f"\nBatch finished with {len(failures)} failure(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"\nBatch finished successfully ({len(face_rows)} rows processed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
