from __future__ import annotations

import datetime
import json
import os
import re
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
from src.approaches.new_approach.utils_pixels import compute_subroi_means, write_subrois_video, plot_subroi_means, bandpass_subroi_means, plot_subroi_psds, select_best_subroi_by_hr_band_power
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
SUBROI_GRID_ROWS = 8 # in pixels
SUBROI_GRID_COLS = SUBROI_GRID_ROWS

LOWCUT_BANDPASS = 0.92 # corresponds to 55 bpm, in Hz (0.92/60)
HIGHCUT_BANDPASS = 0.7 # according to individual camera possibilities
ORDER_BANDPASS = 4

HR_BAND = (0.92, 3.0) # 55 bpm to 180 bpm, in Hz (0.92/60 to 3.0/60)

# --- run_batch overrides ---------------------------------------------------
# When launched via scripts/run_batch.py, the spatial Gaussian + temporal
# bandpass settings are provided per camera, and the sub-ROI grid size per ROI.
# These env vars override the defaults above (standalone runs keep the defaults).
GAUSSIAN_KERNEL_SIZE = int(os.environ.get("NEW_APPROACH_GAUSSIAN_KERNEL_SIZE", GAUSSIAN_KERNEL_SIZE))
LOWCUT_BANDPASS = float(os.environ.get("NEW_APPROACH_LOWCUT_BANDPASS", LOWCUT_BANDPASS))
HIGHCUT_BANDPASS = float(os.environ.get("NEW_APPROACH_HIGHCUT_BANDPASS", HIGHCUT_BANDPASS))
ORDER_BANDPASS = int(os.environ.get("NEW_APPROACH_ORDER_BANDPASS", ORDER_BANDPASS))

# Default per-ROI sub-ROI grid (rows; cols = rows). Overridable via the
# NEW_APPROACH_SUBROI_GRID_ROWS_BY_ROI env var (JSON mapping {roi_name: rows}).
SUBROI_GRID_ROWS_BY_ROI = {
    "forehead": 8,
    "left_forehead": 5,
    "right_forehead": 5,
    "glabella": 3,
}
_grid_by_roi_env = os.environ.get("NEW_APPROACH_SUBROI_GRID_ROWS_BY_ROI")
if _grid_by_roi_env:
    try:
        SUBROI_GRID_ROWS_BY_ROI.update(
            {str(k).strip().lower(): int(v) for k, v in json.loads(_grid_by_roi_env).items()}
        )
    except (ValueError, TypeError) as exc:
        print(f"[new_approach] Could not parse NEW_APPROACH_SUBROI_GRID_ROWS_BY_ROI: {exc}")


def resolve_subroi_grid_rows(roi_name: str) -> int:
    """Return the sub-ROI grid row count for an ROI (falls back to the default)."""
    return int(SUBROI_GRID_ROWS_BY_ROI.get(str(roi_name).strip().lower(), SUBROI_GRID_ROWS))


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


# --- Process every selected ROI --------------------------------------------
for roi_name in roi_names:
    print(f"Processing ROI: {roi_name}")

    # Per-ROI sub-ROI grid: larger ROIs (whole forehead) are split into more
    # sub-ROIs than smaller ones (single-side forehead, glabella). Driven by
    # SUBROI_GRID_ROWS_BY_ROI (overridable via run_batch env var).
    SUBROI_GRID_ROWS = resolve_subroi_grid_rows(roi_name)
    SUBROI_GRID_COLS = SUBROI_GRID_ROWS
    n_cells = SUBROI_GRID_ROWS * SUBROI_GRID_COLS
    print(f"  sub-ROI grid: {SUBROI_GRID_ROWS}x{SUBROI_GRID_COLS} ({n_cells} sub-ROIs)")

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
    for prefix, frame_values_list, label in stages:
        # Whole-ROI mean signal over time (mean of all pixels per frame).
        mean_signal = np.array(
            [np.nanmean(v) if np.asarray(v).size else np.nan for v in frame_values_list],
            dtype=float,
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(time_axis, mean_signal, color="black", linewidth=2.0, label="Mean signal")
        ax.set_title(f"{label} mean signal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temperature")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
        fig.savefig(roi_dir / f"{prefix}_mean_signal.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Compute subROI means ----------------------------------------------------------------
        subroi_means = compute_subroi_means(frame_values_list, coords_per_frame, len(frame_values_list), n_cells, SUBROI_GRID_ROWS, SUBROI_GRID_COLS)


        # Sub-ROI mean signals, each sub-ROI in its own subplot
        plot_subroi_means(
            n_cells=n_cells,
            subroi_means=subroi_means,
            time_axis=time_axis,
            label=label,
            roi_dir=roi_dir,
            prefix=prefix,
            SUBROI_GRID_ROWS=SUBROI_GRID_ROWS,
            SUBROI_GRID_COLS=SUBROI_GRID_COLS)
        

    # Sub-ROI overlay video-------------------------------------------------
    # draw the sub-ROI grid onto the source frames (one video per ROI)
    # Get whole data (rather than only ROI data) for video creation
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

    # Distinct BGR colors per sub-ROI cell (matching the tab10 colormap used in plots)
    subroi_cmap = plt.get_cmap("tab10")
    cell_colors_bgr = [
        tuple(int(round(c * 255)) for c in subroi_cmap(cell % 10)[:3][::-1]) for cell in range(n_cells)
    ]   
    if source_frames:
        n_video = min(len(source_frames), n_frames)
        frame_height, frame_width = source_frames[0].shape[:2]
        video_path = str(roi_dir / "subrois_overlay.mp4")
        first_frame_png_path = str(roi_dir / "subrois_overlay_first_frame.png")
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
        roi_dir= roi_dir,
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


    plot_subroi_psds(
        subroi_means = filtered_subroi_means,
        fs = fps,
        fmin= 0,
        fmax= None,
        nperseg= None,
        roi_dir= roi_dir,
        prefix= "after_bandpass"
    )

    # Choose sub-ROI for final heart rate detection based on power in HR frequency band/ total power ---------------
    best_idx, ratios = select_best_subroi_by_hr_band_power(
        subroi_signals = filtered_subroi_means,
        fs = fps,
        hr_band = HR_BAND)
    
    # Plot selected sub-ROI
    # Get selected signal
    best_signal = filtered_subroi_means[:, best_idx]

    t = np.arange(len(best_signal)) / fps

    plt.figure(figsize=(10, 4))
    plt.plot(t, best_signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.title(f"Selected Sub-ROI {best_idx} (SQI={ratios[best_idx]:.3f})")
    plt.tight_layout()
    fig.savefig(roi_dir / f"selected_subroi_signal.png", dpi=200, bbox_inches="tight")
    plt.close()


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

