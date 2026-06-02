from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt, iirnotch, welch

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
from src.approaches.yang_et_al.utils_post_processing import apply_butterworth_bandpass_filter


DEFAULT_CONFIGS_PATH = "src/approaches/new_approach/configs_new_approach.yaml"
CONFIG_PATH_ENV_VAR = "NEW_APPROACH_CONFIG_PATH"
PLOTS_SAVE_DIR_ENV_VAR = "NEW_APPROACH_PLOTS_SAVE_DIR"
DEFAULT_OUTPUT_ROOT = Path("saved_results/new_approach")

FILTER_TYPE = "bandpass"  # "none", "highpass", "bandpass", "highpass and notch"
GAUSSIAN_KERNEL_SIZE = 7
GAUSSIAN_SIGMA = None
HIGHPASS_CUTOFF_HZ = 0.4
BANDPASS_LOW_HZ = 0.4
BANDPASS_HIGH_HZ = 4.0
FILTER_ORDER = 5
NOTCH_FREQUENCY_HZ = 1.0
NOTCH_QUALITY_FACTOR = 30.0

# Sub-ROI grid: the ROI is divided into SUBROI_GRID_ROWS x SUBROI_GRID_COLS areas
# (n = rows * cols sub-ROIs). For each sub-ROI the mean signal over time and its
# PSD are plotted together in one figure (per processing stage).
SUBROI_GRID_ROWS = 3
SUBROI_GRID_COLS = 3


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("_")



# --- Resolve config and output locations -----------------------------------
config_path = os.environ.get(CONFIG_PATH_ENV_VAR) or os.environ.get("CONFIG_PATH") or DEFAULT_CONFIGS_PATH
configs = Config.from_yaml(config_path)
configs.plot_steps = False

plots_save_dir = os.environ.get(PLOTS_SAVE_DIR_ENV_VAR)
output_root = Path(plots_save_dir) if plots_save_dir else DEFAULT_OUTPUT_ROOT

filter_type_normalized = FILTER_TYPE.lower().strip()
safe_filter = safe_name(FILTER_TYPE)

# --- Run ROI extraction pipeline -------------------------------------------
print("Expects one big ROI")
roi_pixel_data, _ = get_roi_data_pipeline(configs=configs)
if not roi_pixel_data:
    raise ValueError("No ROI pixel data was returned by the ROI pipeline.")

roi_names = list(roi_pixel_data[0].keys())
if not roi_names:
    raise ValueError("No ROI names were found in the first frame.")

selected_rois = [roi_name for roi_name in roi_names if not configs.used_rois or roi_name in configs.used_rois]
if not selected_rois:
    raise ValueError(
        f"None of the configured ROIs {configs.used_rois!r} are available. Available ROIs: {roi_names!r}"
    )

output_root.mkdir(parents=True, exist_ok=True)
fps = float(configs.fps)
n_cells = SUBROI_GRID_ROWS * SUBROI_GRID_COLS

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

# Distinct BGR colors per sub-ROI cell (matching the tab10 colormap used in plots).
subroi_cmap = plt.get_cmap("tab10")
cell_colors_bgr = [
    tuple(int(round(c * 255)) for c in subroi_cmap(cell % 10)[:3][::-1]) for cell in range(n_cells)
]

# --- Process every selected ROI --------------------------------------------
for roi_name in selected_rois:
    print(f"Processing ROI: {roi_name}")

    roi_dir = output_root / safe_name(roi_name)
    roi_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Extract per-frame coordinates and raw pixel values for this ROI.
    #    Frames with no pixels are kept as empty (0, 2) coord arrays so the
    #    time axis stays consistent; they become NaN rows that are
    #    interpolated over time later (fixes the (0,)-shape ValueError).
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2) Pad ragged per-frame raw signals into a (time, pixels) matrix and
    #    interpolate NaNs along time (per pixel column).
    # ------------------------------------------------------------------
    max_len_raw = max(signal.size for signal in raw_frame_signals)
    raw_padded = np.full((n_frames, max_len_raw), np.nan, dtype=float)
    for index, signal_values in enumerate(raw_frame_signals):
        raw_padded[index, : signal_values.size] = signal_values

    finite_global = np.isfinite(raw_padded)
    global_mean = float(np.nanmean(raw_padded[finite_global])) if finite_global.any() else 0.0
    if not np.isfinite(global_mean):
        global_mean = 0.0
    raw_signals = raw_padded.copy()
    for pixel_idx in range(raw_signals.shape[1]):
        series = raw_signals[:, pixel_idx]
        finite = np.isfinite(series)
        if finite.sum() == 0:
            raw_signals[:, pixel_idx] = global_mean
        elif finite.sum() == 1:
            raw_signals[:, pixel_idx] = series[finite][0]
        else:
            raw_signals[~finite, pixel_idx] = np.interp(
                time_axis_full[~finite], time_axis_full[finite], series[finite]
            )

    # ------------------------------------------------------------------
    # 3) Spatial Gaussian noise reduction inside each frame's ROI.
    #    Empty frames produce an empty result (-> NaN row after padding).
    # ------------------------------------------------------------------
    if GAUSSIAN_KERNEL_SIZE < 1 or GAUSSIAN_KERNEL_SIZE % 2 == 0:
        raise ValueError(f"GAUSSIAN_KERNEL_SIZE must be a positive odd integer, got {GAUSSIAN_KERNEL_SIZE}")
    sigma_used = float(GAUSSIAN_SIGMA) if GAUSSIAN_SIGMA is not None else GAUSSIAN_KERNEL_SIZE / 2.0

    smoothed_frames: list[np.ndarray] = []
    for values, coords in zip(raw_frame_signals, coords_per_frame):
        values = np.asarray(values, dtype=float)
        if coords.shape[0] == 0 or values.size == 0:
            smoothed_frames.append(np.zeros((0,), dtype=float))
            continue
        if values.size != coords.shape[0]:
            raise ValueError(
                f"Pixel count mismatch for ROI smoothing: coords has {coords.shape[0]} pixels "
                f"but values has {values.size}."
            )

        xs = coords[:, 0]
        ys = coords[:, 1]
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        rel_xs = xs - x_min
        rel_ys = ys - y_min
        width = x_max - x_min + 1
        height = y_max - y_min + 1

        image = np.zeros((height, width), dtype=float)
        mask = np.zeros((height, width), dtype=float)
        finite = np.isfinite(values)
        image[rel_ys[finite], rel_xs[finite]] = np.nan_to_num(values[finite], nan=0.0)
        mask[rel_ys[finite], rel_xs[finite]] = 1.0

        filtered_image = gaussian_filter(image, sigma=sigma_used)
        filtered_mask = gaussian_filter(mask, sigma=sigma_used)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.divide(
                filtered_image,
                filtered_mask,
                out=np.zeros_like(filtered_image),
                where=filtered_mask > 0,
            )
        smoothed_frames.append(normalized[rel_ys, rel_xs])

    max_len_smoothed = max((signal.size for signal in smoothed_frames), default=0)
    max_len_smoothed = max(max_len_smoothed, 1)
    smoothed_padded = np.full((n_frames, max_len_smoothed), np.nan, dtype=float)
    for index, signal_values in enumerate(smoothed_frames):
        if signal_values.size:
            smoothed_padded[index, : signal_values.size] = signal_values

    finite_global_s = np.isfinite(smoothed_padded)
    global_mean_s = float(np.nanmean(smoothed_padded[finite_global_s])) if finite_global_s.any() else 0.0
    if not np.isfinite(global_mean_s):
        global_mean_s = 0.0
    smoothed_signals = smoothed_padded.copy()
    for pixel_idx in range(smoothed_signals.shape[1]):
        series = smoothed_signals[:, pixel_idx]
        finite = np.isfinite(series)
        if finite.sum() == 0:
            smoothed_signals[:, pixel_idx] = global_mean_s
        elif finite.sum() == 1:
            smoothed_signals[:, pixel_idx] = series[finite][0]
        else:
            smoothed_signals[~finite, pixel_idx] = np.interp(
                time_axis_full[~finite], time_axis_full[finite], series[finite]
            )

    # ------------------------------------------------------------------
    # 4) Temporal filtering of the smoothed signals.
    # ------------------------------------------------------------------
    nyquist = 0.5 * fps
    if filter_type_normalized == "none":
        filtered_signals = np.asarray(smoothed_signals, dtype=float)
    elif filter_type_normalized == "bandpass":
        filtered_signals = apply_butterworth_bandpass_filter(
            np.asarray(smoothed_signals, dtype=float),
            configs.bandpass_lowcut,
            configs.bandpass_highcut,
            fps,
            configs.bandpass_order,
        )
    else:
        if fps <= 0:
            raise ValueError(f"Sampling frequency must be positive. Got fs={fps}.")
        if not (0 < HIGHPASS_CUTOFF_HZ < nyquist):
            raise ValueError(
                f"Highpass cutoff must satisfy 0 < cutoff < Nyquist. "
                f"Got cutoff={HIGHPASS_CUTOFF_HZ}, Nyquist={nyquist}."
            )
        b_hp, a_hp = butter(FILTER_ORDER, HIGHPASS_CUTOFF_HZ / nyquist, btype="high")
        filtered_signals = filtfilt(b_hp, a_hp, np.asarray(smoothed_signals, dtype=float), axis=0)

        if filter_type_normalized == "highpass and notch":
            if not (0 < NOTCH_FREQUENCY_HZ < nyquist):
                raise ValueError(
                    f"Notch frequency must satisfy 0 < frequency < Nyquist. "
                    f"Got frequency={NOTCH_FREQUENCY_HZ}, Nyquist={nyquist}."
                )
            b_n, a_n = iirnotch(NOTCH_FREQUENCY_HZ / nyquist, NOTCH_QUALITY_FACTOR)
            filtered_signals = filtfilt(b_n, a_n, filtered_signals, axis=0)
        elif filter_type_normalized != "highpass":
            raise ValueError(f"Unsupported FILTER_TYPE: {FILTER_TYPE}")

    # ------------------------------------------------------------------
    # 5) Plot every processing stage:
    #    a) all pixel signals + mean over time
    #    b) pixelwise PSD + mean PSD
    #    c) sub-ROI mean signals over time AND their PSDs (one figure)
    # ------------------------------------------------------------------
    stages = [
        ("01_raw", raw_signals, f"{roi_name} - Raw"),
        (
            "02_gaussian_smoothed",
            smoothed_signals,
            f"{roi_name} - Gaussian smoothed (kernel={GAUSSIAN_KERNEL_SIZE})",
        ),
        (f"03_{safe_filter}", filtered_signals, f"{roi_name} - Filtered ({FILTER_TYPE})"),
    ]

    for prefix, signals, label in stages:
        signals = np.asarray(signals, dtype=float)
        mean_signal = np.nanmean(signals, axis=1)

        # (a) Pixel signals with mean over time
        fig, ax = plt.subplots(figsize=(12, 5))
        for pixel_idx in range(signals.shape[1]):
            ax.plot(time_axis, signals[:, pixel_idx], color="tab:blue", alpha=0.15, linewidth=0.8)
        ax.plot(time_axis, mean_signal, color="black", linewidth=2.5, label="Mean signal")
        ax.set_title(f"{label} mean signal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temperature")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
        fig.savefig(roi_dir / f"{prefix}_mean_signal.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # (b) Pixelwise PSD with mean PSD
        if signals.shape[0] >= 2:
            centered = signals - np.nanmean(signals, axis=0, keepdims=True)
            centered = np.nan_to_num(centered, nan=0.0)
            nperseg = min(256, centered.shape[0])
            freqs, psd = welch(centered, fs=fps, nperseg=nperseg, axis=0)
            mean_psd = psd.mean(axis=1)

            fig, ax = plt.subplots(figsize=(12, 5))
            for pixel_idx in range(psd.shape[1]):
                ax.plot(freqs, psd[:, pixel_idx], color="tab:blue", alpha=0.12, linewidth=0.8)
            ax.plot(freqs, mean_psd, color="black", linewidth=2.5, label="Mean PSD")
            ax.set_title(f"{label} PSD")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right")
            fig.savefig(roi_dir / f"{prefix}_psd.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

        # (c) Sub-ROI analysis: divide the ROI into a grid of sub-ROIs and
        #     compute the per-sub-ROI mean signal for every frame. Column j of
        #     row f in `signals` corresponds to coords_per_frame[f][j], so we
        #     can align values with coordinates per frame.
        subroi_means = np.full((n_frames, n_cells), np.nan, dtype=float)
        for frame_idx in range(n_frames):
            coords = coords_per_frame[frame_idx]
            n_pixels = coords.shape[0]
            if n_pixels == 0:
                continue
            values = signals[frame_idx, :n_pixels]

            xs = coords[:, 0]
            ys = coords[:, 1]
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            x_span = max(x_max - x_min + 1, 1)
            y_span = max(y_max - y_min + 1, 1)

            col_idx = ((xs - x_min) * SUBROI_GRID_COLS) // x_span
            row_idx = ((ys - y_min) * SUBROI_GRID_ROWS) // y_span
            col_idx = np.clip(col_idx, 0, SUBROI_GRID_COLS - 1)
            row_idx = np.clip(row_idx, 0, SUBROI_GRID_ROWS - 1)
            cell_idx = row_idx * SUBROI_GRID_COLS + col_idx

            finite = np.isfinite(values)
            sums = np.zeros(n_cells, dtype=float)
            counts = np.zeros(n_cells, dtype=float)
            np.add.at(sums, cell_idx, np.where(finite, np.nan_to_num(values, nan=0.0), 0.0))
            np.add.at(counts, cell_idx, finite.astype(float))
            with np.errstate(divide="ignore", invalid="ignore"):
                cell_means = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
            subroi_means[frame_idx, :] = cell_means

        # Interpolate NaNs (empty sub-ROIs / frames) along time for plotting.
        subroi_plot = subroi_means.copy()
        for cell in range(n_cells):
            series = subroi_plot[:, cell]
            finite = np.isfinite(series)
            if finite.sum() == 0:
                subroi_plot[:, cell] = 0.0
            elif finite.sum() == 1:
                subroi_plot[:, cell] = series[finite][0]
            else:
                subroi_plot[~finite, cell] = np.interp(
                    time_axis_full[~finite], time_axis_full[finite], series[finite]
                )

        fig, (ax_time, ax_psd) = plt.subplots(2, 1, figsize=(12, 9))
        cmap = plt.get_cmap("tab10")
        for cell in range(n_cells):
            row = cell // SUBROI_GRID_COLS
            col = cell % SUBROI_GRID_COLS
            color = cmap(cell % 10)
            ax_time.plot(
                time_axis,
                subroi_plot[:, cell],
                color=color,
                linewidth=1.2,
                label=f"sub-ROI r{row}c{col}",
            )

        if subroi_plot.shape[0] >= 2:
            centered_sub = subroi_plot - np.nanmean(subroi_plot, axis=0, keepdims=True)
            centered_sub = np.nan_to_num(centered_sub, nan=0.0)
            nperseg_sub = min(256, centered_sub.shape[0])
            freqs_sub, psd_sub = welch(centered_sub, fs=fps, nperseg=nperseg_sub, axis=0)
            for cell in range(n_cells):
                ax_psd.plot(freqs_sub, psd_sub[:, cell], color=cmap(cell % 10), linewidth=1.2)

        ax_time.set_title(f"{label} sub-ROI mean signals ({SUBROI_GRID_ROWS}x{SUBROI_GRID_COLS} grid)")
        ax_time.set_xlabel("Time (s)")
        ax_time.set_ylabel("Temperature")
        ax_time.grid(alpha=0.25)
        ax_time.legend(loc="upper right", fontsize=8, ncol=2)

        ax_psd.set_title(f"{label} sub-ROI PSD ({SUBROI_GRID_ROWS}x{SUBROI_GRID_COLS} grid)")
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_ylabel("Power")
        ax_psd.grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(roi_dir / f"{prefix}_subrois.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 6) Sub-ROI overlay video: draw the sub-ROI grid onto the source frames
    #    (one video per ROI), similar to the whole-ROI video produced by the
    #    ROI pipeline. The spatial sub-ROI layout is identical across stages,
    #    so a single video per ROI is written. Each sub-ROI is filled with a
    #    distinct translucent color and outlined, matching the plot colors.
    # ------------------------------------------------------------------
    if source_frames:
        n_video = min(len(source_frames), n_frames)
        frame_height, frame_width = source_frames[0].shape[:2]
        video_path = str(roi_dir / "subrois_overlay.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

        for frame_idx in range(n_video):
            frame = source_frames[frame_idx]
            vmin, vmax = np.percentile(frame, [1, 99])
            frame_clipped = np.clip(frame, vmin, vmax)
            frame_norm = cv2.normalize(frame_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)

            coords = coords_per_frame[frame_idx]
            if coords.shape[0] > 0:
                xs = coords[:, 0]
                ys = coords[:, 1]
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                x_span = max(x_max - x_min + 1, 1)
                y_span = max(y_max - y_min + 1, 1)

                col_idx = np.clip(((xs - x_min) * SUBROI_GRID_COLS) // x_span, 0, SUBROI_GRID_COLS - 1)
                row_idx = np.clip(((ys - y_min) * SUBROI_GRID_ROWS) // y_span, 0, SUBROI_GRID_ROWS - 1)
                cell_idx = row_idx * SUBROI_GRID_COLS + col_idx

                # Keep only in-bounds pixels (coords are absolute image coordinates).
                in_bounds = (ys >= 0) & (ys < frame_height) & (xs >= 0) & (xs < frame_width)

                # Filled translucent color per sub-ROI.
                overlay = frame_bgr.copy()
                for cell in range(n_cells):
                    sel = (cell_idx == cell) & in_bounds
                    if not np.any(sel):
                        continue
                    overlay[ys[sel], xs[sel]] = cell_colors_bgr[cell]
                frame_bgr = cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0)

                # Outline each sub-ROI so the grid boundaries are visible.
                for cell in range(n_cells):
                    sel = (cell_idx == cell) & in_bounds
                    if not np.any(sel):
                        continue
                    cell_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                    cell_mask[ys[sel], xs[sel]] = 255
                    contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame_bgr, contours, -1, cell_colors_bgr[cell], 1)

            writer.write(frame_bgr)

        writer.release()
        print(f"Saved sub-ROI overlay video to: {video_path}")

print("Done.")

