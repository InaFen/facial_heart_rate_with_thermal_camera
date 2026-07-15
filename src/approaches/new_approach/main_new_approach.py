from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.approaches.new_approach.classes.user_input import Config
from src.approaches.new_approach.pipelines.get_roi_data import get_roi_data_pipeline
from src.approaches.new_approach.utils.utils_gray_files import sort_grayfiles_with_interpolations
from src.approaches.new_approach.utils.utils_landmark_detection import (
    _asc_to_kelvin_x100,
    _parse_asc_file,
    load_gray_frame,
)
from src.approaches.new_approach.utils_pixels import compute_subroi_means, write_subrois_video, plot_subroi_means, bandpass_subroi_means, plot_subroi_psds, select_best_subroi, save_subroi_overlay_with_highlight, compute_subroi_location_info
from src.approaches.new_approach.utils_water_experiment import (
    compute_high_temp_bands,
    save_water_timestamps,
    shade_high_temp_bands,
)
from src.approaches.yang_et_al.utils_ground_truth import (
	load_ground_truth_signal,
	get_ground_truth_segment
)

# Values to adapt--------------------------------
DEFAULT_CONFIGS_PATH = "src/approaches/new_approach/configs_new_approach.yaml"
CONFIG_PATH_ENV_VAR = "NEW_APPROACH_CONFIG_PATH"
PLOTS_SAVE_DIR_ENV_VAR = "NEW_APPROACH_PLOTS_SAVE_DIR"
DEFAULT_OUTPUT_ROOT = Path("saved_results/new_approach")

GAUSSIAN_KERNEL_SIZE = 7
GAUSSIAN_SIGMA = None

# Sub-ROI grid: the ROI is divided into SUBROI_GRID_ROWS x SUBROI_GRID_COLS areas before temproal denoising
# to make sure that the amount of elements per frame is constant over frames, without having to interpolate pixels which might be wrongful as ROI size changes over frames
# (n = rows * cols sub-ROIs)
SUBROI_GRID_ROWS = 2 # in pixels
SUBROI_GRID_COLS = SUBROI_GRID_ROWS

LOWCUT_BANDPASS = 0.85 # corresponds to 55 bpm, in Hz (0.92/60)
HIGHCUT_BANDPASS = 4.3 # according to individual camera possibilities
ORDER_BANDPASS = 2

HR_BAND = (0.92, 3.0) # 55 bpm to 180 bpm, in Hz (0.92/60 to 3.0/60)

# Sub-ROI selection. By default the best sub-ROI is chosen SOLELY by the HR-band power
# ratio (SQI = power inside the HR band / total power), i.e. purely by signal strength
# in the heart-rate band. Set CONSIDER_PEAK_CLARITY_IN_SELECTION = True to ALSO factor
# in a peak-clarity score (how clean and regular the peaks are), so a signal with "many
# small peaks on a broad peak" is not chosen just because its power happens to sit in
# the HR band; the two criteria are then normalised across sub-ROIs and combined with
# the weights below.
CONSIDER_PEAK_CLARITY_IN_SELECTION = True

# Used ONLY when CONSIDER_PEAK_CLARITY_IN_SELECTION is True. SUBROI_WEIGHT_* weight the
# (normalised) HR-band power and peak-clarity criteria; SUBROI_MIN_PEAK_CLARITY (in
# [0, 1], e.g. 0.3) additionally drops sub-ROIs with very noisy peaks before ranking
# (None = no hard gate). When the flag is False these are ignored and the selection is
# SQI-only (weight_band_power=1, weight_peak_clarity=0, no clarity gate).
SUBROI_WEIGHT_BAND_POWER = 0.5
SUBROI_WEIGHT_PEAK_CLARITY = 0.5
SUBROI_MIN_PEAK_CLARITY = None

# Resolve the selection weights/gate actually passed to select_best_subroi from the
# flag above: combined SQI + peak-clarity scoring when enabled, otherwise SQI-only.
if CONSIDER_PEAK_CLARITY_IN_SELECTION:
    _SELECT_WEIGHT_BAND_POWER = SUBROI_WEIGHT_BAND_POWER
    _SELECT_WEIGHT_PEAK_CLARITY = SUBROI_WEIGHT_PEAK_CLARITY
    _SELECT_MIN_PEAK_CLARITY = SUBROI_MIN_PEAK_CLARITY
else:
    _SELECT_WEIGHT_BAND_POWER = 1.0
    _SELECT_WEIGHT_PEAK_CLARITY = 0.0
    _SELECT_MIN_PEAK_CLARITY = None

# Steps that should be added for water experiment 
WATER_EXPERIMENT = False
# Water experiment: raw-signal temperature threshold (in the recording's native unit,
# centi-Kelvin for the thermal cameras, e.g. 29150 == 291.50 K). When WATER_EXPERIMENT is
# True, every frame whose SELECTED sub-ROI RAW mean signal is >= this value has its
# timestamp saved and is highlighted with a slight blue background on
# selected_subroi_signal_no_unit.png.
WATER_EXPERIMENT_RAW_THRESHOLD = 29175

# --- run_batch overrides ---------------------------------------------------
# When launched via scripts/run_batch.py, the spatial Gaussian + temporal
# bandpass settings are provided per camera (NEW_APPROACH_FILTER_SETTINGS_BY_CAMERA),
# and the sub-ROI grid size per ROI. These env vars override the defaults above
# (standalone runs keep the defaults). The resolved values + their source are logged
# below so it is always visible which settings a run actually used.
def _resolve_filter_override(env_var, default, cast):
    """Return ``(value, source)`` from ``env_var`` if set (cast), else the default."""
    raw = os.environ.get(env_var)
    if raw is None or str(raw).strip() == "":
        return default, "in-script default"
    try:
        return cast(raw), f"run_batch env ({env_var})"
    except (ValueError, TypeError):
        print(f"[new_approach] Could not parse {env_var}={raw!r}; using default {default}.")
        return default, "in-script default (env parse failed)"


GAUSSIAN_KERNEL_SIZE, _src_gaussian = _resolve_filter_override(
    "NEW_APPROACH_GAUSSIAN_KERNEL_SIZE", GAUSSIAN_KERNEL_SIZE, int
)
LOWCUT_BANDPASS, _src_lowcut = _resolve_filter_override(
    "NEW_APPROACH_LOWCUT_BANDPASS", LOWCUT_BANDPASS, float
)
HIGHCUT_BANDPASS, _src_highcut = _resolve_filter_override(
    "NEW_APPROACH_HIGHCUT_BANDPASS", HIGHCUT_BANDPASS, float
)
ORDER_BANDPASS, _src_order = _resolve_filter_override(
    "NEW_APPROACH_ORDER_BANDPASS", ORDER_BANDPASS, int
)
print(
    "[new_approach] Filter settings resolved (run_batch env override vs in-script default):\n"
    f"    GAUSSIAN_KERNEL_SIZE = {GAUSSIAN_KERNEL_SIZE}  [{_src_gaussian}]\n"
    f"    LOWCUT_BANDPASS      = {LOWCUT_BANDPASS} Hz  [{_src_lowcut}]\n"
    f"    HIGHCUT_BANDPASS     = {HIGHCUT_BANDPASS} Hz  [{_src_highcut}]\n"
    f"    ORDER_BANDPASS       = {ORDER_BANDPASS}  [{_src_order}]"
)

# Default per-ROI sub-ROI grid sizes (rows; cols = rows). Each ROI maps to EITHER
# a single grid size (one int) OR a list of grid sizes to try, e.g. [3, 4, 6, 8].
# When several sizes are given, the full sub-ROI pipeline is run once per grid size
# and the best signal across ALL grid sizes is chosen by SQI (HR-band power ratio).
# ROI names not listed fall back to the scalar SUBROI_GRID_ROWS default. Overridable
# via NEW_APPROACH_SUBROI_GRID_ROWS_BY_ROI (JSON mapping {roi_name: rows} OR
# {roi_name: [rows, ...]}).
SUBROI_GRID_ROWS_BY_ROI = {
    "forehead": [3, 4, 6, 8],
    "left_forehead": 5,
    "right_forehead": 5,
    "glabella": 3,
    "nostrils": 1
}
_grid_by_roi_env = os.environ.get("NEW_APPROACH_SUBROI_GRID_ROWS_BY_ROI")
if _grid_by_roi_env:
    try:
        # Keep raw values (int OR list); they are normalised per ROI in
        # resolve_subroi_grid_rows_list, so both scalars and lists round-trip.
        SUBROI_GRID_ROWS_BY_ROI.update(
            {str(k).strip().lower(): v for k, v in json.loads(_grid_by_roi_env).items()}
        )
    except (ValueError, TypeError) as exc:
        print(f"[new_approach] Could not parse NEW_APPROACH_SUBROI_GRID_ROWS_BY_ROI: {exc}")


def _normalize_grid_sizes(value) -> list[int]:
    """Coerce a grid-size spec (a single int or a list of ints) into a sorted
    list of unique positive grid sizes."""
    raw = list(value) if isinstance(value, (list, tuple, set)) else [value]
    sizes = []
    for v in raw:
        try:
            iv = int(v)
        except (ValueError, TypeError):
            continue
        if iv >= 1:
            sizes.append(iv)
    return sorted(set(sizes))


def resolve_subroi_grid_rows_list(roi_name: str) -> list[int]:
    """Return the list of sub-ROI grid row counts to try for an ROI.

    Accepts both scalar (single grid) and list (multiple grids) entries in
    SUBROI_GRID_ROWS_BY_ROI; ROI names not listed fall back to the scalar
    SUBROI_GRID_ROWS default. When more than one size is returned, the pipeline
    is run once per grid size and the best signal across all of them is chosen
    by SQI (HR-band power ratio).
    """
    sizes = _normalize_grid_sizes(
        SUBROI_GRID_ROWS_BY_ROI.get(str(roi_name).strip().lower(), SUBROI_GRID_ROWS)
    )
    return sizes or _normalize_grid_sizes(SUBROI_GRID_ROWS)


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("_")



# Get configs -----------------------------------
config_path = os.environ.get(CONFIG_PATH_ENV_VAR) or os.environ.get("CONFIG_PATH") or DEFAULT_CONFIGS_PATH
configs = Config.from_yaml(config_path)
configs.plot_steps = False

plots_save_dir = os.environ.get(PLOTS_SAVE_DIR_ENV_VAR)
output_root = Path(plots_save_dir) if plots_save_dir else DEFAULT_OUTPUT_ROOT

output_root.mkdir(parents=True, exist_ok=True)
fps = float(configs.fps)

# Run ROI pipeline -------------------------------------------
print("Expects one big ROI (will be automatically split in sub-ROIs later)")
roi_pixel_data, _ = get_roi_data_pipeline(configs=configs)
if not roi_pixel_data:
    raise ValueError("No ROI pixel data was returned by the ROI pipeline.")

roi_names = list(roi_pixel_data[0].keys())
if not roi_names:
    raise ValueError("No ROI names were found in the first frame.")

# --- Load the source frames once, in the same order/range as roi_pixel_data,
#     so the sub-ROI overlay video can be drawn onto the actual thermal frames
#     (mirrors the whole-ROI video written by the ROI pipeline). -------------
if configs.timespan_signal_to_look_at is not None:
    _start_t, _end_t = configs.timespan_signal_to_look_at
    src_start_frame = int(_start_t * fps)
    src_end_frame = int(_end_t * fps)
else:
    src_start_frame = 0
    src_end_frame = None


def process_subroi_grid(
    grid_rows: int,
    *,
    roi_name: str,
    grid_dir: Path,
    raw_frame_signals: list[np.ndarray],
    smoothed_frames: list[np.ndarray],
    coords_per_frame: list[np.ndarray],
    time_axis: np.ndarray,
    n_frames: int,
    source_frames: list[np.ndarray],
    fps: float,
) -> dict:
    """Run the full sub-ROI pipeline for ONE grid size and return its result.

    Splits the ROI into ``grid_rows x grid_rows`` sub-ROIs, computes/plots the
    raw, Gaussian-smoothed and band-passed sub-ROI mean signals, writes the
    sub-ROI overlay video, and selects the best sub-ROI (combined HR-band SQI +
    peak-clarity score). All artifacts are written into ``grid_dir``. The
    returned dict carries the selected signal plus its SQI so the caller can
    pick the best grid size across several candidates (cross-grid winner is
    chosen by SQI, which - unlike the within-grid 'combined' score - is
    comparable across grid sizes).
    """
    SUBROI_GRID_ROWS = int(grid_rows)
    SUBROI_GRID_COLS = SUBROI_GRID_ROWS
    n_cells = SUBROI_GRID_ROWS * SUBROI_GRID_COLS

    # Compute subROI means and plot before and after smoothing ------------------------------------------------
    # For each stage (raw and Gaussian-smoothed): divide the ROI into a
    # SUBROI_GRID_ROWS x SUBROI_GRID_COLS grid using each frame's own
    # bounding box and compute every sub-ROI's mean signal per frame.
    # Frames where a sub-ROI has no pixels stay NaN (shown as gaps, no
    # interpolation).
    stages = [
        ("01_raw", raw_frame_signals, f"{roi_name} - Raw"),
        (
            "02_gaussian_smoothed",
            smoothed_frames,
            f"{roi_name} - Gaussian smoothed (kernel={GAUSSIAN_KERNEL_SIZE})",
        ),
    ]

    cmap = plt.get_cmap("tab10")
    # Keep the RAW-stage sub-ROI means around for the water experiment: the loop variable
    # subroi_means below is overwritten by the later Gaussian-smoothed stage, but the water
    # experiment needs the unprocessed (raw) temperature signal of the selected sub-ROI.
    raw_subroi_means = None
    for prefix, frame_values_list, label in stages:
        # Whole-ROI mean signal over time (mean of all pixels per frame).
        mean_signal = np.array(
            [np.nanmean(v) if np.asarray(v).size else np.nan for v in frame_values_list],
            dtype=float,
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(time_axis, mean_signal, color="black", linewidth=2.0, label="Mean raw signal")
        #ax.set_title(f"{label} mean signal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
        fig.savefig(grid_dir / f"{prefix}_mean_signal_of_whole_unprocessed_raw_roi.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Compute subROI means ----------------------------------------------------------------
        subroi_means = compute_subroi_means(frame_values_list, coords_per_frame, len(frame_values_list), n_cells, SUBROI_GRID_ROWS, SUBROI_GRID_COLS)
        # Remember the raw (unprocessed) sub-ROI means for the water experiment below.
        if prefix == "01_raw":
            raw_subroi_means = subroi_means

        # Sub-ROI mean signals, each sub-ROI in its own subplot
        plot_subroi_means(
            n_cells=n_cells,
            subroi_means=subroi_means,
            time_axis=time_axis,
            label=label,
            roi_dir=grid_dir,
            prefix=prefix,
            SUBROI_GRID_ROWS=SUBROI_GRID_ROWS,
            SUBROI_GRID_COLS=SUBROI_GRID_COLS)

    # Sub-ROI overlay video-------------------------------------------------
    # draw the sub-ROI grid onto the (already loaded) source frames.
    # Distinct BGR colors per sub-ROI cell (matching the tab10 colormap used in plots)
    subroi_cmap = plt.get_cmap("tab10")
    cell_colors_bgr = [
        tuple(int(round(c * 255)) for c in subroi_cmap(cell % 10)[:3][::-1]) for cell in range(n_cells)
    ]
    if source_frames:
        n_video = min(len(source_frames), n_frames)
        frame_height, frame_width = source_frames[0].shape[:2]
        video_path = str(grid_dir / "subrois_overlay.mp4")
        first_frame_png_path = str(grid_dir / "subrois_overlay_first_frame.png")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

        write_subrois_video(
            writer = writer,
            n_video=n_video,
            source_frames=source_frames,
            coords_per_frame=coords_per_frame,
            video_path=video_path,
            SUBROI_GRID_ROWS=SUBROI_GRID_ROWS,
            SUBROI_GRID_COLS=SUBROI_GRID_COLS,
            n_cells=n_cells,
            cell_colors_bgr=cell_colors_bgr,
            frame_height=frame_height,
            frame_width=frame_width,
            first_frame_png_path=first_frame_png_path)

    # PSDs of sub-ROIs before and after temporal denoising -----------
    plot_subroi_psds(
        subroi_means = subroi_means,
        fs = fps,
        fmin= 0,
        fmax= None,
        nperseg= None,
        roi_dir= grid_dir,
        prefix= "before_bandpass"
    )

    # Apply bandpass ----------------------------------------------------------------------
    filtered_subroi_means = bandpass_subroi_means(
        subroi_means,
        fs=fps,
        lowcut=LOWCUT_BANDPASS,
        highcut=HIGHCUT_BANDPASS,
        order=ORDER_BANDPASS,
        max_gap_frames=3,      # interpolate gaps of ≤3 frames # TODO double check
        max_missing_fraction=0.15, # TODO double check
    )

    # Sub-ROI mean signals after the temporal bandpass, one sub-ROI per subplot
    # (same layout as 01_raw_subrois / 02_gaussian_smoothed_subrois).
    plot_subroi_means(
        n_cells=n_cells,
        subroi_means=filtered_subroi_means,
        time_axis=time_axis,
        label=f"{roi_name} - Bandpassed ({LOWCUT_BANDPASS}-{HIGHCUT_BANDPASS} Hz, order={ORDER_BANDPASS})",
        roi_dir=grid_dir,
        prefix="03_bandpassed",
        SUBROI_GRID_ROWS=SUBROI_GRID_ROWS,
        SUBROI_GRID_COLS=SUBROI_GRID_COLS,
    )

    plot_subroi_psds(
        subroi_means = filtered_subroi_means,
        fs = fps,
        fmin= 0,
        fmax= None,
        nperseg= None,
        roi_dir= grid_dir,
        prefix= "after_bandpass"
    )

    # Choose sub-ROI for final heart rate detection. By default this is based SOLELY on
    # the HR-band power ratio (SQI = signal strength in the HR band); when
    # CONSIDER_PEAK_CLARITY_IN_SELECTION is True a peak-clarity score is additionally
    # factored in so a signal with many small/ragged peaks is not selected just because
    # its power sits in the HR band. ---------------
    best_idx, subroi_scores = select_best_subroi(
        subroi_signals=filtered_subroi_means,
        fs=fps,
        hr_band=HR_BAND,
        weight_band_power=_SELECT_WEIGHT_BAND_POWER,
        weight_peak_clarity=_SELECT_WEIGHT_PEAK_CLARITY,
        min_peak_clarity=_SELECT_MIN_PEAK_CLARITY,
    )

    # Plot selected sub-ROI
    best_signal = filtered_subroi_means[:, best_idx]
    t = np.arange(len(best_signal)) / fps
    fig_selected = plt.figure(figsize=(10, 4))
    plt.plot(t, best_signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.title(
        f"Selected Sub-ROI {best_idx} (grid {SUBROI_GRID_ROWS}x{SUBROI_GRID_COLS}) "
        f"(combined={subroi_scores['combined'][best_idx]:.3f}, "
        f"SQI={subroi_scores['band_power_ratio'][best_idx]:.3f}, "
        f"peak clarity={subroi_scores['peak_clarity'][best_idx]:.3f})"
    )
    plt.tight_layout()
    fig_selected.savefig(grid_dir / "selected_subroi_signal.png", dpi=200, bbox_inches="tight")
    plt.close(fig_selected)

    # Plot selected sub-ROI
    best_signal = filtered_subroi_means[:, best_idx]
    t = np.arange(len(best_signal)) / fps
    fig_selected = plt.figure(figsize=(10, 4))
    plt.plot(t, best_signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (arbitrary unit)")
    plt.tight_layout()
    fig_selected.savefig(grid_dir / "selected_subroi_signal_no_unit.png", dpi=200, bbox_inches="tight")
    plt.close(fig_selected)

    # Water experiment ------------------------------------------------------------
    # From the RAW signal of the selected sub-ROI, record every timestamp whose value is
    # >= WATER_EXPERIMENT_RAW_THRESHOLD and save a highlighted copy of
    # selected_subroi_signal_no_unit.png with those timestamps shaded a slight blue.
    # The payload is also returned so the caller can write the single reusable
    # timestamps file (shared with the Yang/Nair pipelines) from the winning grid.
    water_payload = None
    if WATER_EXPERIMENT:
        raw_selected_signal = (
            np.asarray(raw_subroi_means[:, best_idx], dtype=float)
            if raw_subroi_means is not None
            else np.full(len(best_signal), np.nan)
        )
        # NaN-safe threshold mask (frames without a value never count as high).
        high_temp_mask = np.isfinite(raw_selected_signal) & (
            raw_selected_signal >= WATER_EXPERIMENT_RAW_THRESHOLD
        )
        high_temp_frame_indices = np.nonzero(high_temp_mask)[0]
        high_temp_timestamps = (high_temp_frame_indices / fps).astype(float)
        # Pre-merged [start_s, end_s] spans (independent of fps) so the Yang/Nair
        # pipelines can shade the exact same periods without re-deriving anything.
        high_temp_bands = compute_high_temp_bands(high_temp_frame_indices, fps)

        # Save the high-temperature timestamps (seconds from segment start) together with
        # the merged bands, frame indices and the raw values that triggered them.
        water_payload = {
            "raw_threshold": float(WATER_EXPERIMENT_RAW_THRESHOLD),
            "selected_subroi_index": int(best_idx),
            "grid": f"{SUBROI_GRID_ROWS}x{SUBROI_GRID_COLS}",
            "fps": float(fps),
            "num_frames_total": int(raw_selected_signal.shape[0]),
            "num_frames_high": int(high_temp_frame_indices.size),
            "high_temperature_frame_indices": high_temp_frame_indices.tolist(),
            "high_temperature_timestamps_s": high_temp_timestamps.tolist(),
            "high_temperature_bands_s": high_temp_bands,
            "high_temperature_raw_values": [
                float(raw_selected_signal[i]) for i in high_temp_frame_indices
            ],
        }
        with open(
            grid_dir / "water_experiment_high_temp_timestamps.json", "w", encoding="utf-8"
        ) as jf:
            json.dump(water_payload, jf, ensure_ascii=False, indent=2)
        print(
            f"  [grid {SUBROI_GRID_ROWS}x{SUBROI_GRID_COLS}] water experiment: "
            f"{high_temp_frame_indices.size}/{raw_selected_signal.shape[0]} frame(s) "
            f">= {WATER_EXPERIMENT_RAW_THRESHOLD} (raw signal of selected sub-ROI)"
        )

        # Save a highlighted copy of selected_subroi_signal_no_unit.png with a slight blue
        # background over every high-temperature time span (merged consecutive frames).
        fig_selected = plt.figure(figsize=(10, 4))
        ax_selected = plt.gca()
        ax_selected.plot(t, best_signal)
        ax_selected.set_xlabel("Time (s)")
        ax_selected.set_ylabel("Amplitude (arbitrary unit)")
        shade_high_temp_bands(ax_selected, high_temp_bands)
        fig_selected.tight_layout()
        fig_selected.savefig(
            grid_dir / "selected_subroi_signal_no_unit_highlighted.png", dpi=200, bbox_inches="tight"
        )
        plt.close(fig_selected)

    # Mark which sub-ROI was selected ----------------------------------------
    # 1) Save a face overlay (sub-ROI grid) with the chosen sub-ROI highlighted,
    #    so it is visually clear which sub-ROI produced the final signal.
    # 2) Save the sub-ROI's location id (index/row/col + normalized center) so it
    #    can be compared across persons/recordings (the normalized center is
    #    comparable even across different grid sizes).
    best_subroi_info = compute_subroi_location_info(
        best_idx, SUBROI_GRID_ROWS, SUBROI_GRID_COLS, roi_name=roi_name
    )
    best_subroi_info["sqi_hr_band_power_ratio"] = float(subroi_scores["band_power_ratio"][best_idx])
    best_subroi_info["peak_clarity"] = float(subroi_scores["peak_clarity"][best_idx])
    best_subroi_info["combined_selection_score"] = float(subroi_scores["combined"][best_idx])
    with open(grid_dir / "selected_subroi_location.json", "w", encoding="utf-8") as jf:
        json.dump(best_subroi_info, jf, ensure_ascii=False, indent=2)
    print(f"  [grid {SUBROI_GRID_ROWS}x{SUBROI_GRID_COLS}] selected sub-ROI location: {best_subroi_info}")

    if source_frames:
        save_subroi_overlay_with_highlight(
            source_frame=source_frames[0],
            coords=coords_per_frame[0],
            grid_rows=SUBROI_GRID_ROWS,
            grid_cols=SUBROI_GRID_COLS,
            n_cells=n_cells,
            cell_colors_bgr=cell_colors_bgr,
            output_path=grid_dir / "selected_subroi_overlay.png",
            highlight_cell=best_idx,
        )

    return {
        "grid_rows": SUBROI_GRID_ROWS,
        "grid_cols": SUBROI_GRID_COLS,
        "n_cells": n_cells,
        "best_idx": int(best_idx),
        "best_signal": np.asarray(best_signal, dtype=float),
        "best_subroi_info": best_subroi_info,
        "subroi_scores": subroi_scores,
        "sqi": float(subroi_scores["band_power_ratio"][best_idx]),
        "combined": float(subroi_scores["combined"][best_idx]),
        "peak_clarity": float(subroi_scores["peak_clarity"][best_idx]),
        "grid_dir": grid_dir,
        "water_payload": water_payload,
    }


# --- Process every selected ROI --------------------------------------------
# best_signal / best_subroi_info hold the result of the last processed ROI; the
# latter is saved into the result metadata so it can be compared across
# persons/recordings.
best_signal = None
best_subroi_info = None
for roi_name in roi_names:
    print(f"Processing ROI: {roi_name}")

    # Per-ROI sub-ROI grid sizes: larger ROIs (whole forehead) are split into more
    # sub-ROIs than smaller ones (single-side forehead, glabella). Each ROI may map
    # to SEVERAL grid sizes (e.g. forehead -> [3, 4, 6, 8]); the full sub-ROI
    # pipeline is then run once per grid size and the best signal across ALL grid
    # sizes is picked by SQI further below. Driven by SUBROI_GRID_ROWS_BY_ROI
    # (overridable via the run_batch env var).
    grid_rows_list = resolve_subroi_grid_rows_list(roi_name)
    print(f"  sub-ROI grid sizes to try: {', '.join(f'{g}x{g}' for g in grid_rows_list)}")

    roi_dir = output_root / safe_name(roi_name)
    roi_dir.mkdir(parents=True, exist_ok=True)

    # Extract per-frame coordinates and raw pixel values for current ROI.
    #  The raw pixels are kept as-is (ragged, variable length per frame); frames with no pixels become empty (0, 2) coord arrays 
    #  and show up as gaps (NaN) in the plots. No padding/interpolation is applied at this point.

    coords_per_frame: list[np.ndarray] = []
    raw_frame_signals: list[np.ndarray] = []
    for frame_data in roi_pixel_data:
        roi_data = frame_data.get(roi_name)
        if roi_data is None:
            continue
        coords = np.asarray(roi_data["coords"], dtype=int)
        if coords.size == 0:
            coords = coords.reshape(0, 2)
        elif coords.ndim != 2 or coords.shape[1] != 2:
            coords = coords.reshape(-1, 2)
        coords_per_frame.append(coords)
        raw_frame_signals.append(np.asarray(roi_data["adc_values"], dtype=float))

    if not raw_frame_signals:
        raise ValueError(f"No pixel data found for ROI '{roi_name}'.")

    n_frames = len(raw_frame_signals)
    time_axis_full = np.arange(n_frames)
    time_axis = time_axis_full / fps

    
    # Spatial Gaussian noise reduction inside each frame's ROI ------------------------------------------------
    # (ROI lengths of frames are not alligned to each other as not needed, only inside each frame seperately).
    if GAUSSIAN_KERNEL_SIZE < 1 or GAUSSIAN_KERNEL_SIZE % 2 == 0:
        raise ValueError(f"GAUSSIAN_KERNEL_SIZE must be a positive odd integer, got {GAUSSIAN_KERNEL_SIZE}")
    sigma_used = float(GAUSSIAN_SIGMA) if GAUSSIAN_SIGMA is not None else GAUSSIAN_KERNEL_SIZE / 2.0

    smoothed_frames: list[np.ndarray] = []
    for values, coords in zip(raw_frame_signals, coords_per_frame):
        values = np.asarray(values, dtype=float)
        # If there are no pixels for this frame, create an empty array and skip the smoothing to avoid errors.
        if coords.shape[0] == 0 or values.size == 0:
            smoothed_frames.append(np.zeros((0,), dtype=float))
            continue
        if values.size != coords.shape[0]:
            raise ValueError(
                f"Pixel count mismatch for ROI smoothing: coords has {coords.shape[0]} pixels "
                f"but values has {values.size}."
            )

        # Shift coordinates so the ROI starts at (0, 0) (relative coordinates instead of absolute), 
        # allowing a much smaller image (more computationally efficient)
        # for smoothing without changing the result.
        xs = coords[:, 0]
        ys = coords[:, 1]
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        rel_xs = xs - x_min
        rel_ys = ys - y_min
        width = x_max - x_min + 1
        height = y_max - y_min + 1

        # Create a blank image and mask for the bounding box, place the pixel values at the relative coordinates, and apply Gaussian smoothing.
        image = np.zeros((height, width), dtype=float)
        mask = np.zeros((height, width), dtype=float)
        # Only place finite values to avoid issues with NaNs/Infs during smoothing; the mask ensures that only the actual ROI pixels are considered in the normalization step after smoothing.
        finite = np.isfinite(values)
        image[rel_ys[finite], rel_xs[finite]] = np.nan_to_num(values[finite], nan=0.0)
        mask[rel_ys[finite], rel_xs[finite]] = 1.0

        # Apply Gaussian filter
        filtered_image = gaussian_filter(image, sigma=sigma_used)
        filtered_mask = gaussian_filter(mask, sigma=sigma_used)

        # Normalize by the smoothed mask to compute Gaussian-weighted averages using only valid pixels (so no nan pixels (zeros) have an influence)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.divide(
                filtered_image,
                filtered_mask,
                out=np.zeros_like(filtered_image),
                where=filtered_mask > 0,
            )

        # Store soley the smoothed pixel values (not coordinates, relative or absoulte, are stored)
        smoothed_frames.append(normalized[rel_ys, rel_xs])

    # Sub-ROI overlay source frames (grid-independent: load ONCE here and reuse
    # for every grid size below, instead of re-reading them per grid).
    source_frames: list[np.ndarray] = []
    gray_path = configs.path_gray_frames
    if os.path.isfile(gray_path) and gray_path.endswith(".npy"):
        _stack = np.load(gray_path, allow_pickle=False)
        _end = src_end_frame if src_end_frame is not None else _stack.shape[0]
        source_frames = [_stack[i] for i in range(src_start_frame, min(_end, _stack.shape[0]))]
    elif os.path.isdir(gray_path):
        _files = [f for f in os.listdir(gray_path) if os.path.isfile(os.path.join(gray_path, f))]
        _ext = os.path.splitext(_files[0])[1].lower() if _files else ""
        if _ext == ".asc":
            _asc_paths = sorted(os.path.join(gray_path, f) for f in _files if f.lower().endswith(".asc"))
            _end = src_end_frame if src_end_frame is not None else len(_asc_paths)
            for _p in _asc_paths[src_start_frame:_end]:
                _fr = _parse_asc_file(_p)
                if configs.asc_unit == "kelvin":
                    _fr = _asc_to_kelvin_x100(_fr)
                source_frames.append(_fr)
        else:
            _, _gray_files = sort_grayfiles_with_interpolations(gray_path)
            _end = src_end_frame if src_end_frame is not None else len(_gray_files)
            for _fname in _gray_files[src_start_frame:_end]:
                source_frames.append(load_gray_frame(os.path.join(gray_path, _fname)))
    if not source_frames:
        print(f"[sub-ROI video] No source frames could be loaded from {gray_path}; overlay video will be skipped.")

    # Run the full sub-ROI pipeline once per grid size. Per-grid plots/videos go
    # into a grid_<r>x<c> subfolder so successive grids do not overwrite each
    # other; within each grid the best sub-ROI is chosen by the combined SQI +
    # peak-clarity score (see select_best_subroi / process_subroi_grid).
    grid_results = []
    for grid_rows in grid_rows_list:
        grid_dir = roi_dir / f"grid_{grid_rows}x{grid_rows}"
        grid_dir.mkdir(parents=True, exist_ok=True)
        print(f"  sub-ROI grid: {grid_rows}x{grid_rows} ({grid_rows * grid_rows} sub-ROIs) -> {grid_dir.name}")
        grid_results.append(
            process_subroi_grid(
                grid_rows,
                roi_name=roi_name,
                grid_dir=grid_dir,
                raw_frame_signals=raw_frame_signals,
                smoothed_frames=smoothed_frames,
                coords_per_frame=coords_per_frame,
                time_axis=time_axis,
                n_frames=n_frames,
                source_frames=source_frames,
                fps=fps,
            )
        )

    # Cross-grid selection: the per-grid 'combined' score is min-max normalised
    # WITHIN each grid's sub-ROIs, so it is NOT comparable across grids. The SQI
    # (HR-band power ratio) is an absolute fraction of in-band power and IS
    # comparable across grid sizes -> use it to pick the overall best grid/signal.
    best_grid = max(
        grid_results,
        key=lambda r: r["sqi"] if np.isfinite(r["sqi"]) else -np.inf,
    )
    best_signal = best_grid["best_signal"]
    best_subroi_info = dict(best_grid["best_subroi_info"])
    best_subroi_info["grid_sizes_tried"] = list(grid_rows_list)
    best_subroi_info["cross_grid_selection_metric"] = "sqi_hr_band_power_ratio"
    _grid_summary = ", ".join(
        f"{r['grid_rows']}x{r['grid_rows']} (SQI={r['sqi']:.3f}, clarity={r['peak_clarity']:.3f})"
        for r in grid_results
    )
    print(
        f"[{roi_name}] Best grid across {len(grid_results)} size(s): "
        f"{best_grid['grid_rows']}x{best_grid['grid_cols']} by SQI={best_grid['sqi']:.3f} "
        f"(sub-ROI {best_grid['best_idx']}). Candidates: {_grid_summary}"
    )

    # Water experiment: write the single reusable timestamps file from the WINNING grid
    # so the Yang/Nair pipelines highlight the same periods on their component plots.
    if WATER_EXPERIMENT and best_grid.get("water_payload"):
        _water_payload = dict(best_grid["water_payload"])
        _water_payload["roi_name"] = roi_name
        _shared_water_path = save_water_timestamps(_water_payload)
        print(f"[{roi_name}] Saved reusable water-experiment timestamps to: {_shared_water_path}")

    # Mirror the winning grid's selection artifacts into the ROI-level folder so
    # the top-level ROI directory always shows the final chosen sub-ROI.
    for _fname in (
        "selected_subroi_signal.png",
        "selected_subroi_location.json",
        "selected_subroi_overlay.png",
    ):
        _src_artifact = best_grid["grid_dir"] / _fname
        if _src_artifact.is_file():
            shutil.copy2(_src_artifact, roi_dir / _fname)


print("Own pipeline done.")

# Ground truth steps---------------------------------------------------------
if configs.timespan_signal_to_look_at is not None:
	start_time, end_time = configs.timespan_signal_to_look_at
	signal_offset = float(configs.time_first_frame_signal or 0.0)
	ecg_offset = float(configs.time_first_frame_ecg or 0.0)
	ground_truth_signal, ground_truth_fs = load_ground_truth_signal(
		configs.path_ground_truth,
		flip=False,
	)
	ground_truth_segment, ground_truth_time = get_ground_truth_segment(
		ground_truth_signal,
		ground_truth_fs,
		start_time,
		end_time,
		signal_offset=signal_offset,
		ecg_offset=ecg_offset,
	)


# Save results at end of pipeline
try:
    # Directory for saved results inside project, grouped per camera
    # (e.g. saved_results/nair_et_al/results/vario). The camera name is provided
    # by run_batch via the CAMERA_NAME env var; falls back to "unknown".
    camera_name = os.environ.get("CAMERA_NAME", "unknown") or "unknown"
    save_root = Path(project_root) / "saved_results" / "new_approach" / "results" / camera_name
    save_root.mkdir(parents=True, exist_ok=True)

    # Build descriptive filename: signalName_roiName_timespan
    signal_path_cfg = getattr(configs, "path_gray_frames", "") or ""
    signal_name = Path(signal_path_cfg).stem if signal_path_cfg else "signal"
    roi_name_cfg = getattr(configs, "roi_name", "roi") or "roi"
    timespan_cfg = getattr(configs, "timespan_signal_to_look_at", None)
    if timespan_cfg:
        try:
            start_s, end_s = timespan_cfg
            timespan_str = f"{float(start_s):.2f}-{float(end_s):.2f}s"
        except Exception:
            timespan_str = str(timespan_cfg)
    else:
        timespan_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_name = f"{signal_name}_{roi_name_cfg}_{timespan_str}"
    safe_base = re.sub(r"[^A-Za-z0-9_-]", "_", base_name)
    save_file = save_root / f"{safe_base}.npz"

    metadata = {
        "pipeline": "nair_et_al",
        "path_signal": getattr(configs, "path_gray_frames", ""),
        "path_ground_truth": getattr(configs, "path_ground_truth", ""),
        "timespan_signal_to_look_at": getattr(configs, "timespan_signal_to_look_at", None),
        "roi_name": getattr(configs, "roi_name", ""),
        "heart_rate_estimate_bpm": None,
        "fps": float(configs.fps),
        "fps_ground_truth": float(ground_truth_fs) if "ground_truth_fs" in globals() else None,
        "best_subroi_location": best_subroi_info,
    }

    if "ground_truth_segment" in globals():
        np.savez_compressed(
            save_file,
            predicted_signal=np.asarray(best_signal[:], dtype=float),
            ground_truth_segment=np.asarray(ground_truth_segment, dtype=float),
            heart_rate_estimate=None,
            metadata=json.dumps(metadata),
        )
    else:
        np.savez_compressed(
            save_file,
            predicted_signal=np.asarray(best_signal[:], dtype=float),
            heart_rate_estimate=None,
            metadata=json.dumps(metadata),
        )

    try:
        meta_path = save_file.with_suffix('.json')
        with open(meta_path, 'w', encoding='utf-8') as jf:
            json.dump(metadata, jf, ensure_ascii=False, indent=2)
        print(f"Saved pipeline results to: {save_file}")
        print(f"Saved metadata JSON to: {meta_path}")
    except Exception as e:
        print(f"Warning: failed to save metadata JSON: {e}")
except Exception as e:
    print(f"Warning: failed to save results: {e}")

