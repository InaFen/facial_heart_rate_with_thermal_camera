"""Metric pipeline for comparing predicted signals with ground truth.

Functions:
- resample_signal: resample a 1D signal from one sampling rate to another using linear interpolation.
- compute_regression_metrics: RMSE, MAE, R2
- compute_ks_pvalue: two-sample KS p-value between distributions of signals
- bland_altman: compute mean difference and limits of agreement and optionally plot
- evaluate_predictions: high-level function that takes ground truth (signal, fs) and
  predicted signal (signal, fs) and returns metrics and optionally shows/saves Bland-Altman plot.

Assumptions:
- Ground truth format follows `load_ground_truth_signal` (ndarray, fs).
- If sampling rates differ, predicted signal is resampled to ground-truth sampling rate by linear interp.
- Both signals are aligned at t=0. If you need time offsets, resample externally before calling.
"""

from typing import Tuple, Dict, Optional
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import matplotlib

# Use the non-interactive "Agg" backend: this module only saves figures to disk
# (never displays them), so it must not try to open an X window. Running under
# tmux/SSH the interactive backend connects to the forwarded X display
# (e.g. localhost:10); when that SSH session detaches the X connection breaks with
# "X connection to localhost:10 broken (explicit kill or shutdown)" and kills the
# run. Forcing Agg removes the X dependency entirely. Set before importing pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

try:
    from src.approaches.yang_et_al.utils_ground_truth import (
        load_ground_truth_signal,
        get_ground_truth_segment,
        detect_peaks_in_signal,
        mean_hr_from_peak_times,
    )
except ModuleNotFoundError:  # pragma: no cover - allows running the file directly
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from src.approaches.yang_et_al.utils_ground_truth import (
        load_ground_truth_signal,
        get_ground_truth_segment,
        detect_peaks_in_signal,
        mean_hr_from_peak_times,
    )

# Global defaults (previously set inside __main__); make available to all functions
PEAK_PROMINENCE_SIGNAL = 0.0#0.5 # absolute floor for the predicted-signal peak prominence
#   effective prominence = max(PEAK_PROMINENCE_SIGNAL,
#                              PEAK_PROMINENCE_SIGNAL_STD_FRACTION * std(signal)).
PEAK_PROMINENCE_SIGNAL_STD_FRACTION = 0.35  # >= 20% of the signal's standard deviation
PEAK_DISTANCE_SIGNAL = 0.01#int(max(1, gt_fs_val * 0.01))
PEAK_DISTANCE_GT = 0.5 # 0.5 seems to work well
PEAK_PROMINENCE_GT = 1.0 # 1.0 seems to work well

# special cases (high frame rate creates more small peaks, so use a higher prominence floor to drop them):
PROMINENCE_SIGNAL_OVERRIDES: dict[str, dict[str, float]] = {
    "new_approach/results/mwir": {
        "std_fraction": 0.8,   # higher frame rate -> noisier; drop small noise peaks
        "floor": PEAK_PROMINENCE_SIGNAL,
    },
    "new_approach/results/vario": {
        "std_fraction": 0.8,   # higher frame rate -> noisier; drop small noise peaks
        "floor": PEAK_PROMINENCE_SIGNAL,
    },
}

# A few mwir ECG ground-truth recordings need the default find_peaks settings
# above relaxed, but they need DIFFERENT relaxations because their ECG lives on
# very different amplitude scales:
#   * ``Ina_20_4_3``    -> normal (~21.8) amplitude scale; the default prominence=1.0
#                          under-detects, so a moderate prominence works best
#                          (distance=0.25, prominence=0.4 -> ~123 bpm).
#   * ``Julian_31_03_2`` -> a tiny (~0.002) amplitude scale where any prominence near
#                          1.0 (or even 0.05) finds NOTHING, so it needs a far smaller,
#                          scale-matched prominence (distance=0.4, prominence=0.0005
#                          -> a stable ~60 bpm).
# Each entry maps a path fragment (matched case-insensitively, slash-agnostically
# against the npz path, only when method="find_peaks") to its OWN ``distance`` (min
# seconds between beats) and ``prominence`` find_peaks settings. Use
# ``prominence=None`` to drop the prominence requirement entirely for a recording.
RELAXED_GT_FIND_PEAKS_SETTINGS: dict[str, dict[str, float | None]] = {
    "mwir/Ina_20_4_3_thermal_data_all_frames": {
        "distance": 0.25,      # allow closer beats (up to ~240 bpm)
        "prominence": 0.4,     # works on Ina's ~21.8 amplitude scale
    },
    "mwir/Julian_31_03_2_Ascii_thermal_data_all_frames": {
        "distance": 0.4,       # wider spacing suits this recording
        "prominence": 0.0005,  # matched to Julian's tiny ~0.002 amplitude scale
    },
}

# Default evaluation parameters
DEFAULT_TOLERANCE_MS = 500.0
DEFAULT_PATTERN = "*.npz"
DEFAULT_PLOT_STEPS = True
DEFAULT_PLOT_BLAND_ALTMAN = True

# Algorithm used to detect R-peaks in the ECG ground-truth signal:
#   "find_peaks" -> scipy.signal.find_peaks (the original behaviour, uses
#                   PEAK_DISTANCE_GT / PEAK_PROMINENCE_GT).
#   "staple"     -> STAPLE consensus of several QRS detectors
#                   (src/metrics/staple_r_peaks.py); more robust on noisy or
#                   pathological ECG. Requires the 'py-ecg-detectors' package.
DEFAULT_GT_PEAK_METHOD = "find_peaks"

# Map the short approach aliases used by scripts/run_batch.py ("new"/"yang"/"nair")
# to the full approach folder names used here, so either form may be passed to
# ``batch_evaluate_folder_structure`` (e.g. approaches=("new",) == ("new_approach",)).
APPROACH_ALIASES: dict[str, str] = {
    "new": "new_approach",
    "new_approach": "new_approach",
    "yang": "yang_et_al",
    "yang_et_al": "yang_et_al",
    "nair": "nair_et_al",
    "nair_et_al": "nair_et_al",
}


def _relaxed_find_peaks_settings_for_path(
    source_path: str | Path | None,
) -> dict[str, float | None] | None:
    """Return the per-recording relaxed find_peaks settings for ``source_path``.

    A few mwir ECG ground-truth recordings (keys of
    ``RELAXED_GT_FIND_PEAKS_SETTINGS``) have an atypical amplitude scale / weak
    R-peaks, so the default ``PEAK_DISTANCE_GT`` / ``PEAK_PROMINENCE_GT`` under- or
    over-detect their beats. Each such recording has its OWN ``distance`` /
    ``prominence``. Returns that settings dict when ``source_path`` matches one of
    the fragments (case-insensitive, slash-agnostic), else ``None``.
    """
    if source_path is None:
        return None
    normalized_path = str(source_path).replace("\\", "/").lower()
    for fragment, settings in RELAXED_GT_FIND_PEAKS_SETTINGS.items():
        if fragment.lower() in normalized_path:
            return settings
    return None


def _prominence_signal_settings_for_path(
    source_path: str | Path | None,
) -> tuple[float, float]:
    """Return ``(floor, std_fraction)`` for the predicted-signal peak prominence.

    Defaults to ``(PEAK_PROMINENCE_SIGNAL, PEAK_PROMINENCE_SIGNAL_STD_FRACTION)``. A
    few approach/camera combinations (keys of ``PROMINENCE_SIGNAL_OVERRIDES``) use a
    different fraction because their predicted signal is noisier -- notably the
    new_approach mwir and vario recordings, whose higher frame rate yields more small
    noise peaks. Returns that override's ``(floor, std_fraction)`` when ``source_path``
    matches one of the fragments (case-insensitive, slash-agnostic), else the defaults.
    """
    floor = PEAK_PROMINENCE_SIGNAL
    std_fraction = PEAK_PROMINENCE_SIGNAL_STD_FRACTION
    if source_path is None:
        return floor, std_fraction
    normalized_path = str(source_path).replace("\\", "/").lower()
    for fragment, settings in PROMINENCE_SIGNAL_OVERRIDES.items():
        if fragment.lower() in normalized_path:
            return (
                float(settings.get("floor", floor)),
                float(settings.get("std_fraction", std_fraction)),
            )
    return floor, std_fraction


def _normalize_approaches(approaches: tuple[str, ...] | str) -> tuple[str, ...]:
    """Map approach names through ``APPROACH_ALIASES`` (short -> full folder names).

    Accepts a single string or an iterable of names so either the short run_batch
    aliases ("new"/"yang"/"nair") or the full folder names ("new_approach"/...) work.
    Unknown names are passed through unchanged (so custom folders still resolve).
    """
    if isinstance(approaches, str):
        approaches = (approaches,)
    normalized: list[str] = []
    for approach in approaches:
        key = str(approach).strip().lower()
        normalized.append(APPROACH_ALIASES.get(key, key))
    return tuple(normalized)


def _normalize_camera_filter(
    cameras: tuple[str, ...] | str | None,
) -> set[str] | None:
    """Return a lowercase set of camera names to keep, or ``None`` for all cameras.

    Accepts a single string, an iterable, or ``None``. Empty / ``None`` entries are
    dropped; if nothing usable remains the result is ``None`` (meaning no filtering).
    """
    if cameras is None:
        return None
    if isinstance(cameras, str):
        cameras = (cameras,)
    filtered = {
        str(camera).strip().lower()
        for camera in cameras
        if camera is not None and str(camera).strip()
    }
    return filtered or None


def _camera_subfolder_of(path: Path, results_dir: Path) -> str:
    """Return the camera subfolder name (e.g. "mwir") of ``path`` under ``results_dir``.

    npz files live at ``<results_dir>/<camera>/.../*.npz``; the camera is the first
    path component relative to ``results_dir``. Returns "" when ``path`` is not nested
    in a subfolder (or not under ``results_dir``), so such files are simply excluded
    by any camera filter.
    """
    try:
        relative = path.relative_to(results_dir)
    except ValueError:
        return ""
    return relative.parts[0].lower() if len(relative.parts) >= 2 else ""


def detect_ground_truth_peaks(
    gt_segment: np.ndarray,
    fs: float,
    method: str = DEFAULT_GT_PEAK_METHOD,
    source_path: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect R-peaks in the ECG ground-truth segment using the chosen method.

    ``method`` selects the algorithm:
      * "find_peaks" -> ``scipy.signal.find_peaks`` via ``detect_peaks_in_signal``
        (uses the module-level ``PEAK_DISTANCE_GT`` / ``PEAK_PROMINENCE_GT``).
      * "staple"     -> the STAPLE consensus of several QRS detectors
        (``src/metrics/staple_r_peaks.py``); more robust on noisy/pathological ECG.

    ``source_path`` is the path of the npz/recording being evaluated. When
    ``method="find_peaks"`` and it matches one of ``RELAXED_GT_FIND_PEAKS_SETTINGS``,
    that recording's OWN relaxed ``distance`` / ``prominence`` are used instead (a
    few mwir recordings whose ECG the defaults under-/over-detect, each on its own
    amplitude scale).

    Both branches return ``(peaks, peak_times)`` with peak sample indices into
    ``gt_segment`` and times in seconds relative to the segment start (identical
    convention), so the result is interchangeable at the call site.
    """
    normalized = str(method).strip().lower()
    if normalized == "staple":
        try:
            from src.metrics.staple_r_peaks import detect_r_peaks_staple
        except ModuleNotFoundError:  # pragma: no cover - allows running file directly
            from staple_r_peaks import detect_r_peaks_staple  # type: ignore
        return detect_r_peaks_staple(gt_segment, fs, min_distance_s=PEAK_DISTANCE_GT)
    if normalized in ("find_peaks", "findpeaks", "scipy", ""):
        min_distance_s = PEAK_DISTANCE_GT
        prominence = PEAK_PROMINENCE_GT
        relaxed_settings = _relaxed_find_peaks_settings_for_path(source_path)
        if relaxed_settings is not None:
            min_distance_s = relaxed_settings["distance"]
            prominence = relaxed_settings["prominence"]
            print(
                f"[metric_pipeline] Relaxed find_peaks GT settings "
                f"(min_distance_s={min_distance_s}, prominence={prominence}) for: {source_path}"
            )
        return detect_peaks_in_signal(
            gt_segment, fs, min_distance_s=min_distance_s, prominence=prominence
        )
    raise ValueError(
        f"gt_peak_method must be 'find_peaks' or 'staple', got {method!r}."
    )


def instantaneous_hr_from_peak_times(peak_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute instantaneous HR (bpm) and timestamps from peak times.

    For consecutive peaks p[i], p[i+1], IBI = p[i+1] - p[i],
    HR = 60 / IBI and timestamp is midpoint (p[i] + p[i+1]) / 2.
    """
    peak_times = np.asarray(peak_times, dtype=float)
    if peak_times.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    ibi = np.diff(peak_times)
    valid = ibi > 0
    if valid.sum() == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    hr_inst = 60.0 / ibi[valid]
    hr_times = (peak_times[:-1][valid] + peak_times[1:][valid]) / 2.0
    return hr_inst.astype(float), hr_times.astype(float)


def _is_finite_scalar(value) -> bool:
    """Safely test whether a (possibly non-numeric) value is finite.

    Returns False for values that cannot be converted to float.
    """
    try:
        v = float(value)
    except Exception:
        return False
    return np.isfinite(v)


def match_by_timestamp_tolerance(
    gt_values: np.ndarray,
    gt_times: np.ndarray,
    pred_values: np.ndarray,
    pred_times: np.ndarray,
    tolerance_ms: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match predicted values to ground-truth values by nearest timestamp within tolerance.

    Returns:
    - matched_gt_values
    - matched_pred_values
    - matched_gt_times
    - matched_pred_times
    """
    gt_values = np.asarray(gt_values, dtype=float)
    gt_times = np.asarray(gt_times, dtype=float)
    pred_values = np.asarray(pred_values, dtype=float)
    pred_times = np.asarray(pred_times, dtype=float)

    if gt_values.size == 0 or pred_values.size == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    tol_s = float(tolerance_ms) / 1000.0
    pred_used = np.zeros(pred_times.shape[0], dtype=bool)

    matched_gt_vals = []
    matched_pred_vals = []
    matched_gt_t = []
    matched_pred_t = []

    # iterate through GT timestamps and find nearest unmatched prediction within tolerance
    for i in range(gt_times.shape[0]):
        dt = np.abs(pred_times - gt_times[i])
        dt[pred_used] = np.inf
        j = int(np.argmin(dt))
        try:
            dtj = float(dt[j])
        except Exception:
            # non-numeric timestamp encountered; skip
            continue
        if np.isfinite(dtj) and dtj <= tol_s:
            matched_gt_vals.append(gt_values[i])
            matched_pred_vals.append(pred_values[j])
            matched_gt_t.append(gt_times[i])
            matched_pred_t.append(pred_times[j])
            pred_used[j] = True

    return (
        np.asarray(matched_gt_vals, dtype=float),
        np.asarray(matched_pred_vals, dtype=float),
        np.asarray(matched_gt_t, dtype=float),
        np.asarray(matched_pred_t, dtype=float),
    )


def match_by_nearest_timestamp(
    gt_values: np.ndarray,
    gt_times: np.ndarray,
    pred_values: np.ndarray,
    pred_times: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Match each ground-truth value to the closest prediction and vice versa.

    This keeps all ground-truth points by matching each one to its nearest prediction,
    and also keeps all predicted points by matching each one to its nearest ground-truth
    value. The returned dictionary contains both directions so neither side is lost.
    """
    gt_values = np.asarray(gt_values, dtype=float)
    gt_times = np.asarray(gt_times, dtype=float)
    pred_values = np.asarray(pred_values, dtype=float)
    pred_times = np.asarray(pred_times, dtype=float)

    if gt_values.size == 0 or pred_values.size == 0:
        empty = np.array([], dtype=float)
        return {
            "gt_to_pred_gt_values": empty,
            "gt_to_pred_pred_values": empty,
            "gt_to_pred_gt_times": empty,
            "gt_to_pred_pred_times": empty,
            "pred_to_gt_pred_values": empty,
            "pred_to_gt_gt_values": empty,
            "pred_to_gt_pred_times": empty,
            "pred_to_gt_gt_times": empty,
        }

    gt_to_pred_gt_vals = []
    gt_to_pred_pred_vals = []
    gt_to_pred_gt_t = []
    gt_to_pred_pred_t = []

    for i in range(gt_times.shape[0]):
        dt = np.abs(pred_times - gt_times[i])
        j = int(np.argmin(dt))
        gt_to_pred_gt_vals.append(gt_values[i])
        gt_to_pred_pred_vals.append(pred_values[j])
        gt_to_pred_gt_t.append(gt_times[i])
        gt_to_pred_pred_t.append(pred_times[j])

    pred_to_gt_pred_vals = []
    pred_to_gt_gt_vals = []
    pred_to_gt_pred_t = []
    pred_to_gt_gt_t = []

    for i in range(pred_times.shape[0]):
        dt = np.abs(gt_times - pred_times[i])
        j = int(np.argmin(dt))
        pred_to_gt_pred_vals.append(pred_values[i])
        pred_to_gt_gt_vals.append(gt_values[j])
        pred_to_gt_pred_t.append(pred_times[i])
        pred_to_gt_gt_t.append(gt_times[j])

    return {
        "gt_to_pred_gt_values": np.asarray(gt_to_pred_gt_vals, dtype=float),
        "gt_to_pred_pred_values": np.asarray(gt_to_pred_pred_vals, dtype=float),
        "gt_to_pred_gt_times": np.asarray(gt_to_pred_gt_t, dtype=float),
        "gt_to_pred_pred_times": np.asarray(gt_to_pred_pred_t, dtype=float),
        "pred_to_gt_pred_values": np.asarray(pred_to_gt_pred_vals, dtype=float),
        "pred_to_gt_gt_values": np.asarray(pred_to_gt_gt_vals, dtype=float),
        "pred_to_gt_pred_times": np.asarray(pred_to_gt_pred_t, dtype=float),
        "pred_to_gt_gt_times": np.asarray(pred_to_gt_gt_t, dtype=float),
    }


def _ihr_time_and_peak_pairs_from_peak_times(
    peak_times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map instantaneous-HR timestamps back to the two peaks forming each interval.

    Returns:
        ihr_times: shape (n_intervals,)
        left_peak_indices: indices into peak_times
        right_peak_indices: indices into peak_times
    """
    peak_times = np.asarray(peak_times, dtype=float)
    if peak_times.size < 2:
        return (
            np.array([], dtype=float),
            np.array([], dtype=int),
            np.array([], dtype=int),
        )

    ibi = np.diff(peak_times)
    valid = ibi > 0
    if valid.sum() == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=int),
            np.array([], dtype=int),
        )

    ihr_times = (peak_times[:-1][valid] + peak_times[1:][valid]) / 2.0
    left = np.nonzero(valid)[0].astype(int)
    right = (left + 1).astype(int)
    return ihr_times.astype(float), left, right


def _nearest_index(sorted_values: np.ndarray, query_value: float) -> int:
    """Return index of nearest value in a sorted 1D array."""
    if sorted_values.size == 0:
        raise ValueError("sorted_values must be non-empty")
    pos = int(np.searchsorted(sorted_values, query_value))
    if pos <= 0:
        return 0
    if pos >= sorted_values.size:
        return int(sorted_values.size - 1)
    left = pos - 1
    right = pos
    if abs(sorted_values[right] - query_value) <= abs(sorted_values[left] - query_value):
        return right
    return left


def plot_matched_peak_pairs(
    *,
    gt_time: np.ndarray,
    gt_signal: np.ndarray,
    gt_peaks: np.ndarray,
    gt_peak_times: np.ndarray,
    pred_time: np.ndarray,
    pred_signal: np.ndarray,
    pred_peaks: np.ndarray,
    pred_peak_times: np.ndarray,
    gt_ihr_times: np.ndarray,
    pred_ihr_times: np.ndarray,
    matched_gt_ihr_times: np.ndarray,
    matched_pred_ihr_times: np.ndarray,
    title: str,
) -> None:
    """Visualize which *detected peaks* contribute to matched peak-to-peak (IHR) pairs.

    The peak-to-peak metrics are computed on instantaneous-HR points (midpoints between
    consecutive peaks). This plot highlights the peaks that form those matched intervals.
    """
    gt_time = np.asarray(gt_time, dtype=float)
    gt_signal = np.asarray(gt_signal, dtype=float)
    pred_time = np.asarray(pred_time, dtype=float)
    pred_signal = np.asarray(pred_signal, dtype=float)
    gt_peaks = np.asarray(gt_peaks, dtype=int)
    pred_peaks = np.asarray(pred_peaks, dtype=int)

    if gt_peaks.size == 0 or pred_peaks.size == 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(gt_time, gt_signal, alpha=0.8, label="Ground truth")
        ax.plot(pred_time, pred_signal, alpha=0.8, label="Predicted (resampled)")
        ax.set_title(title + " (no peaks detected)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return

    gt_interval_times, gt_left, gt_right = _ihr_time_and_peak_pairs_from_peak_times(gt_peak_times)
    pred_interval_times, pred_left, pred_right = _ihr_time_and_peak_pairs_from_peak_times(pred_peak_times)

    gt_matched_peak_mask = np.zeros(gt_peak_times.shape[0], dtype=bool)
    pred_matched_peak_mask = np.zeros(pred_peak_times.shape[0], dtype=bool)

    if gt_interval_times.size > 0 and pred_interval_times.size > 0:
        for t_gt, t_pred in zip(np.asarray(matched_gt_ihr_times, dtype=float), np.asarray(matched_pred_ihr_times, dtype=float)):
            i_gt = _nearest_index(gt_interval_times, float(t_gt))
            i_pred = _nearest_index(pred_interval_times, float(t_pred))

            gt_matched_peak_mask[gt_left[i_gt]] = True
            gt_matched_peak_mask[gt_right[i_gt]] = True
            pred_matched_peak_mask[pred_left[i_pred]] = True
            pred_matched_peak_mask[pred_right[i_pred]] = True

    fig, ax = plt.subplots(figsize=(12, 5))
    gt_line = ax.plot(gt_time, gt_signal, alpha=0.75, label="Ground truth")[0]
    pred_line = ax.plot(pred_time, pred_signal, alpha=0.75, label="Predicted (resampled)")[0]

    # all peaks
    ax.scatter(gt_peak_times, gt_signal[gt_peaks], s=20, color=gt_line.get_color(), alpha=0.55, label="GT peaks")
    ax.scatter(pred_peak_times, pred_signal[pred_peaks], s=20, color=pred_line.get_color(), alpha=0.55, label="Pred peaks")

    # matched peaks (peaks that form matched intervals)
    if gt_matched_peak_mask.any():
        ax.scatter(
            gt_peak_times[gt_matched_peak_mask],
            gt_signal[gt_peaks[gt_matched_peak_mask]],
            s=55,
            facecolors="none",
            edgecolors=gt_line.get_color(),
            linewidths=1.5,
            label="GT peaks used in matched pairs",
        )
    if pred_matched_peak_mask.any():
        ax.scatter(
            pred_peak_times[pred_matched_peak_mask],
            pred_signal[pred_peaks[pred_matched_peak_mask]],
            s=55,
            facecolors="none",
            edgecolors=pred_line.get_color(),
            linewidths=1.5,
            label="Pred peaks used in matched pairs",
        )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def compute_metrics_with_optional_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plot_bland_altman: bool = False,
    title: Optional[str] = None,
) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    """Compute regression + KS + optional Bland-Altman for paired values."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0 or y_pred.size == 0:
        empty_metrics = {
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "ks_pvalue": np.nan,
            "mean_diff": np.nan,
            "loa_low": np.nan,
            "loa_high": np.nan,
            "sd_diff": np.nan,
            "num_pairs": 0,
        }
        return empty_metrics, None

    metrics = compute_regression_metrics(y_true, y_pred)
    metrics["ks_pvalue"] = compute_ks_pvalue(y_true, y_pred)
    ba_stats = bland_altman(y_true, y_pred, plot=plot_bland_altman, title=title)
    metrics.update(ba_stats)
    metrics["num_pairs"] = int(y_true.size)
    return metrics, ba_stats


def _to_json_string(value: object) -> str:
    """Convert arrays/dicts/scalars to a JSON string for CSV storage."""

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    return json.dumps(convert(value), ensure_ascii=False)


def append_evaluation_row_to_csv(
    csv_output_path: str,
    path_signal: Optional[str],
    path_ground_truth: str,
    roi: Optional[str],
    gt_hr_mean: float,
    signal_hr_mean: float,
    gt_ihr_times: np.ndarray,
    gt_ihr: np.ndarray,
    pred_ihr_times: np.ndarray,
    pred_ihr: np.ndarray,
    metrics_mean_hr: Dict[str, float],
    metrics_peak_to_peak_disregard: Dict[str, float],
    metrics_peak_to_peak_closest: Dict[str, float],
) -> None:
    """Append one evaluation result row to a CSV file."""
    os.makedirs(os.path.dirname(csv_output_path) or ".", exist_ok=True)

    fieldnames = [
        "timestamp_utc",
        "path_signal",
        "path_ground_truth",
        "roi",
        "gt_hr_mean",
        "signal_hr_mean",
        "ground_truth_ihr_times",
        "ground_truth_ihr",
        "predicted_ihr_times",
        "predicted_ihr",
        "metrics_mean_hr",
        "metrics_peak_to_peak_disregard",
        "metrics_peak_to_peak_closest",
    ]

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "path_signal": path_signal or "",
        "path_ground_truth": path_ground_truth,
        "roi": roi or "",
        "gt_hr_mean": float(gt_hr_mean) if _is_finite_scalar(gt_hr_mean) else np.nan,
        "signal_hr_mean": float(signal_hr_mean) if _is_finite_scalar(signal_hr_mean) else np.nan,
        "ground_truth_ihr_times": _to_json_string(gt_ihr_times),
        "ground_truth_ihr": _to_json_string(gt_ihr),
        "predicted_ihr_times": _to_json_string(pred_ihr_times),
        "predicted_ihr": _to_json_string(pred_ihr),
        "metrics_mean_hr": _to_json_string(metrics_mean_hr),
        "metrics_peak_to_peak_disregard": _to_json_string(metrics_peak_to_peak_disregard),
        "metrics_peak_to_peak_closest": _to_json_string(metrics_peak_to_peak_closest),
    }

    write_header = not os.path.exists(csv_output_path) or os.path.getsize(csv_output_path) == 0
    with open(csv_output_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# Canonical column order for the aggregate per-npz metrics CSV. The nested metric
# dictionaries from 00_metrics.json are flattened with a prefix; any missing value
# (e.g. Bland-Altman stats when nothing matched) is left blank via restval so every
# row has exactly these columns, regardless of the individual run.
METRICS_SUMMARY_CSV_FIELDNAMES = [
    "timestamp_utc",
    "input_file",
    "plots_folder",
    "metrics_json_path",
    "tolerance_ms",
    "ground_truth_fs",
    "predicted_fs",
    "gt_hr_mean",
    "signal_hr_mean",
    "original_pipeline_bpm_estimate",
    "original_pipeline_bpm_estimate_diff_to_gt_mean",
    "num_gt_peaks",
    "num_pred_peaks",
    "num_matched_disregard",
    "num_matched_closest",
    # metrics_disregard.*
    "metrics_disregard_rmse",
    "metrics_disregard_mae",
    "metrics_disregard_r2",
    "metrics_disregard_ks_pvalue",
    "metrics_disregard_mean_diff",
    "metrics_disregard_loa_low",
    "metrics_disregard_loa_high",
    "metrics_disregard_sd_diff",
    "metrics_disregard_num_pairs",
    "metrics_disregard_num_matched",
    # metrics_closest.*
    "metrics_closest_rmse",
    "metrics_closest_mae",
    "metrics_closest_r2",
    "metrics_closest_ks_pvalue",
    "metrics_closest_mean_diff",
    "metrics_closest_loa_low",
    "metrics_closest_loa_high",
    "metrics_closest_sd_diff",
    "metrics_closest_num_pairs",
    # ba_stats_disregard.*
    "ba_stats_disregard_mean_diff",
    "ba_stats_disregard_loa_low",
    "ba_stats_disregard_loa_high",
    "ba_stats_disregard_sd_diff",
    # ba_stats_closest.*
    "ba_stats_closest_mean_diff",
    "ba_stats_closest_loa_low",
    "ba_stats_closest_loa_high",
    "ba_stats_closest_sd_diff",
]


def _rewrite_csv_with_canonical_header(csv_path: str | Path) -> bool:
    """Ensure an existing summary CSV uses the current ``METRICS_SUMMARY_CSV_FIELDNAMES``.

    The summary-CSV schema grows over time (new columns added, e.g.
    ``original_pipeline_bpm_estimate``). A CSV written with an older schema keeps its
    old header, but new rows are appended in the *new* column order, which misaligns
    every column after the first changed one (so e.g. the ``num_gt_peaks`` column ends
    up showing another field's value even though the per-run JSON is correct). This
    migrates such a file in place to the canonical schema BEFORE any append/remove:

    * header already canonical -> nothing to do (returns ``False``).
    * every data row has the same width as the file's own (old) header -> the file is
      internally consistent, so rows are remapped by column name to the canonical
      schema (new columns become blank, removed ones are dropped).
    * the header is stale but every data row already has the canonical column count
      -> the rows were written in the new order under an old header (the bug above);
      the rows are kept as-is and only the header is rewritten to match.
    * otherwise (mixed/unknown row widths) the file is genuinely inconsistent: it is
      backed up next to itself and replaced by a fresh canonical header-only file so
      the affected recordings can be re-evaluated cleanly.

    Returns ``True`` when the file was migrated/rewritten, ``False`` otherwise.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        try:
            header = next(reader)
        except StopIteration:
            return False
        if header == METRICS_SUMMARY_CSV_FIELDNAMES:
            return False
        data_rows = [row for row in reader if row]

    canonical = METRICS_SUMMARY_CSV_FIELDNAMES
    migrated: list[dict] | None = None
    note = ""
    if data_rows and all(len(row) == len(header) for row in data_rows):
        # Rows are consistent with the file's own (older) header -> map by name.
        migrated = [dict(zip(header, row)) for row in data_rows]
        note = f"remapped {len(migrated)} row(s) by column name"
    elif all(len(row) == len(canonical) for row in data_rows):
        # Stale header but rows already use the current column order (append-after-
        # schema-change symptom) -> keep the rows and just fix the header.
        migrated = [dict(zip(canonical, row)) for row in data_rows]
        note = f"realigned header over {len(migrated)} already-current row(s)"

    if migrated is not None:
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=canonical, extrasaction="ignore", restval=""
            )
            writer.writeheader()
            writer.writerows(migrated)
        print(
            f"[metric_pipeline] Migrated summary CSV to the current column schema "
            f"({note}): {csv_path}"
        )
        return True

    # Inconsistent row widths -> the file can no longer be trusted column-by-column.
    backup_path = csv_path.with_name(
        f"{csv_path.stem}.corrupt-{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak.csv"
    )
    csv_path.rename(backup_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=canonical, extrasaction="ignore", restval=""
        )
        writer.writeheader()
    print(
        f"[metric_pipeline] WARNING: summary CSV header did not match its rows "
        f"(a previous schema change misaligned it). Backed up the old file to "
        f"{backup_path} and started a fresh CSV with the current schema: {csv_path}. "
        f"Re-run the affected recordings (existing_entry_mode='overwrite' or "
        f"overwrite_csv=True) to repopulate it."
    )
    return True


def _flatten_metrics_for_csv(
    output_data: Dict[str, object],
    plots_folder: Optional[str | Path],
    json_path: Optional[str | Path],
) -> Dict[str, object]:
    """Flatten the JSON ``output_data`` (plus plot/JSON locations) into one CSV row.

    Nested metric dictionaries are expanded into ``<prefix>_<key>`` columns so the
    aggregate CSV is directly analysable. Missing nested dicts simply produce no
    keys (the CSV writer fills them with the blank restval).
    """
    data = output_data if isinstance(output_data, dict) else {}
    row: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": data.get("input_file", ""),
        "plots_folder": str(plots_folder) if plots_folder is not None else "",
        "metrics_json_path": str(json_path) if json_path is not None else "",
        "tolerance_ms": data.get("tolerance_ms"),
        "ground_truth_fs": data.get("ground_truth_fs"),
        "predicted_fs": data.get("predicted_fs"),
        "gt_hr_mean": data.get("gt_hr_mean"),
        "signal_hr_mean": data.get("signal_hr_mean"),
        "original_pipeline_bpm_estimate": data.get("original_pipeline_bpm_estimate"),
        "original_pipeline_bpm_estimate_diff_to_gt_mean": data.get(
            "original_pipeline_bpm_estimate_diff_to_gt_mean"
        ),
        "num_gt_peaks": data.get("num_gt_peaks"),
        "num_pred_peaks": data.get("num_pred_peaks"),
        "num_matched_disregard": data.get("num_matched_disregard"),
        "num_matched_closest": data.get("num_matched_closest"),
    }
    for prefix in ("metrics_disregard", "metrics_closest", "ba_stats_disregard", "ba_stats_closest"):
        sub = data.get(prefix)
        if isinstance(sub, dict):
            for key, value in sub.items():
                row[f"{prefix}_{key}"] = value
    return row


def append_metrics_summary_row(
    csv_output_path: str | Path,
    output_data: Dict[str, object],
    plots_folder: Optional[str | Path],
    json_path: Optional[str | Path],
) -> str:
    """Append one flattened metrics row (same data as the per-run JSON) to a CSV.

    The CSV uses a fixed column schema (``METRICS_SUMMARY_CSV_FIELDNAMES``) and a
    pointer to the run's plot folder, so a batch over many npz files builds a single
    aggregate table. Unknown extra keys are ignored and missing ones left blank.
    """
    csv_output_path = str(csv_output_path)
    os.makedirs(os.path.dirname(csv_output_path) or ".", exist_ok=True)
    # Make sure an existing CSV uses the current column schema before appending, so a
    # schema change (added columns) can never misalign the header from the data rows.
    _rewrite_csv_with_canonical_header(csv_output_path)
    row = _flatten_metrics_for_csv(output_data, plots_folder, json_path)
    write_header = not os.path.exists(csv_output_path) or os.path.getsize(csv_output_path) == 0
    with open(csv_output_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=METRICS_SUMMARY_CSV_FIELDNAMES,
            extrasaction="ignore",
            restval="",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return csv_output_path


def _read_existing_csv_input_files(csv_path: str | Path) -> set[str]:
    """Return the set of ``input_file`` values already present in a summary CSV.

    Used to decide which npz files already have a metrics row so a batch can skip
    or overwrite them. Returns an empty set if the CSV does not exist or is empty.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    existing: set[str] = set()
    try:
        with open(csv_path, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                value = row.get("input_file")
                if value:
                    existing.add(value)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[metric_pipeline] Warning: could not read existing CSV {csv_path}: {exc}")
    return existing


def _remove_csv_rows_for_input_files(csv_path: str | Path, input_files: set[str]) -> int:
    """Remove rows whose ``input_file`` is in ``input_files`` from a summary CSV.

    Rewrites the CSV in place keeping every other row. Returns the number of rows
    removed. Used by the "overwrite" existing-entry mode so a recomputed file ends
    up with exactly one (fresh) row instead of a duplicate.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0 or not input_files:
        return 0
    # Heal a stale-schema header first so reading/rewriting stays column-aligned.
    _rewrite_csv_with_canonical_header(csv_path)
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    kept = [row for row in rows if row.get("input_file") not in input_files]
    removed = len(rows) - len(kept)
    if removed == 0:
        return 0
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METRICS_SUMMARY_CSV_FIELDNAMES, extrasaction="ignore", restval="")
        writer.writeheader()
        for row in kept:
            writer.writerow(row)
    return removed


def _select_files_by_existing_entries(
    npz_files: list[Path],
    csv_output_path: str | Path,
    existing_entry_mode: str,
    *,
    log_prefix: str,
) -> list[Path]:
    """Filter npz files based on whether they already have a row in the summary CSV.

    - "skip":      drop files that already have a CSV row (process only new ones).
    - "overwrite": keep all files, but first remove the existing rows of those that
                   already have one so each ends up with a single fresh row.

    Returns the list of files that should be (re)processed.
    """
    mode = str(existing_entry_mode).strip().lower()
    if mode not in ("skip", "overwrite"):
        raise ValueError(
            f"existing_entry_mode must be 'skip' or 'overwrite', got {existing_entry_mode!r}"
        )

    existing_input_files = _read_existing_csv_input_files(csv_output_path)
    if not existing_input_files:
        return list(npz_files)

    already_present = [f for f in npz_files if str(f) in existing_input_files]
    if mode == "skip":
        files_to_process = [f for f in npz_files if str(f) not in existing_input_files]
        if already_present:
            print(
                f"{log_prefix} existing_entry_mode='skip': skipping "
                f"{len(already_present)} file(s) already present in the CSV."
            )
        return files_to_process

    # overwrite: process all files, but drop the stale rows of those already present
    # so each recomputed file ends up with exactly one fresh row.
    to_remove = {str(f) for f in already_present}
    if to_remove:
        removed = _remove_csv_rows_for_input_files(csv_output_path, to_remove)
        print(
            f"{log_prefix} existing_entry_mode='overwrite': removed {removed} existing "
            f"row(s) for {len(to_remove)} file(s) to recompute."
        )
    return list(npz_files)


def resample_signal(sig: np.ndarray, fs_from: float, fs_to: float) -> np.ndarray:
    """Resample 1D signal from fs_from to fs_to using linear interpolation.

    Assumes signal starts at t=0 and samples are uniformly spaced.
    """
    sig = np.asarray(sig, dtype=float)
    if sig.size == 0:
        return sig
    if fs_from == fs_to:
        return sig
    duration = len(sig) / float(fs_from)
    t_from = np.linspace(0.0, duration, num=len(sig), endpoint=False)
    n_to = int(np.floor(duration * fs_to))
    if n_to <= 0:
        return np.array([], dtype=float)
    t_to = np.linspace(0.0, duration, num=n_to, endpoint=False)
    resampled = np.interp(t_to, t_from, sig)
    return resampled


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE and R^2 between two same-length 1D arrays.

    Returns a dict: {"rmse", "mae", "r2"}
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    # Remove NaNs pairwise
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    mse = np.mean((y_true_f - y_pred_f) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true_f - y_pred_f)))

    ss_res = np.sum((y_true_f - y_pred_f) ** 2)
    ss_tot = np.sum((y_true_f - np.mean(y_true_f)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {"rmse": rmse, "mae": mae, "r2": r2}


def compute_ks_pvalue(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute two-sample Kolmogorov-Smirnov test p-value between two arrays.

    Returns p-value. NaNs are removed.
    """
    a = np.asarray(y_true).ravel()
    b = np.asarray(y_pred).ravel()
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float('nan')
    _, pvalue = stats.ks_2samp(a, b)
    return float(pvalue)


def bland_altman(y_true: np.ndarray, y_pred: np.ndarray, plot: bool = True, ax: Optional[plt.Axes] = None, title: Optional[str] = None) -> Dict[str, float]:
    """Compute Bland-Altman statistics and optionally plot.

    Returns dict with: mean_diff, loa_low, loa_high, sd_diff
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"mean_diff": np.nan, "loa_low": np.nan, "loa_high": np.nan, "sd_diff": np.nan}

    t = y_true[mask]
    p = y_pred[mask]
    means = (t + p) / 2.0
    diffs = p - t
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    loa_low = mean_diff - 1.96 * sd_diff
    loa_high = mean_diff + 1.96 * sd_diff

    if plot:
        created_local_axis = ax is None
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(means, diffs, alpha=0.6)
        ax.axhline(mean_diff, color='C1', linestyle='--', label=f'Mean diff = {mean_diff:.3f}')
        ax.axhline(loa_high, color='C2', linestyle='--', label=f'LoA high = {loa_high:.3f}')
        ax.axhline(loa_low, color='C2', linestyle='--', label=f'LoA low = {loa_low:.3f}')
        ax.set_xlabel('Mean of methods')
        ax.set_ylabel('Difference (pred - gt)')
        ax.set_title(title or 'Bland-Altman Plot')
        ax.legend()
        if created_local_axis:
            plt.tight_layout()
            plt.show()

    return {"mean_diff": mean_diff, "loa_low": loa_low, "loa_high": loa_high, "sd_diff": sd_diff}


def evaluate_predictions(
    ground_truth_signal: np.ndarray,
    ground_truth_fs: float,
    predicted_signal: np.ndarray,
    predicted_fs: float,
    align_to_ground_truth: bool = True,
    plot_bland_altman: bool = True,
    return_all: bool = True,
) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    """High-level evaluation routine.

    - Resamples `predicted_signal` to `ground_truth_fs` if sampling rates differ (linear interp).
    - Truncates to minimum length.
    - Computes RMSE, MAE, R2, KS p-value.
    - Computes Bland-Altman stats and optionally plots.

    Returns (metrics_dict, bland_altman_stats)
    """
    gt = np.asarray(ground_truth_signal, dtype=float)
    pred = np.asarray(predicted_signal, dtype=float)

    if align_to_ground_truth and predicted_fs != ground_truth_fs:
        pred = resample_signal(pred, predicted_fs, ground_truth_fs)

    # Trim to common length
    L = min(len(gt), len(pred))
    if L == 0:
        raise ValueError("Empty signals provided")
    gt_cut = gt[:L]
    pred_cut = pred[:L]

    metrics = compute_regression_metrics(gt_cut, pred_cut)
    metrics["ks_pvalue"] = compute_ks_pvalue(gt_cut, pred_cut)

    ba_stats = bland_altman(gt_cut, pred_cut, plot=plot_bland_altman)

    if return_all:
        metrics.update(ba_stats)
        return metrics, ba_stats
    return metrics, None



def evaluate_instantaneous_hr_against_ground_truth(
    results: str | Path,
    flip_ground_truth: bool = False,
    tolerance_ms: float = 150.0,
    plot_steps: bool = True,
    plot_bland_altman: bool = True,
    csv_output_path: str | Path | None = None,
    gt_peak_method: str = DEFAULT_GT_PEAK_METHOD,
) -> Dict[str, object]:
    """Evaluate instantaneous HR using a single `.npy`/`.npz` file as input.

    If ``csv_output_path`` is given, a flattened one-line summary of the metrics
    (the same data written to ``00_metrics.json``) plus the path to this run's plot
    folder is appended to that CSV, so a batch can aggregate every npz file into a
    single table.

    ``gt_peak_method`` selects how R-peaks are found in the ECG ground-truth
    signal: "find_peaks" (scipy) or "staple" (STAPLE consensus of several QRS
    detectors). The predicted-signal peaks are always found with find_peaks.
    """
    if not isinstance(results, (str, Path)):
        raise ValueError("This function only accepts a path to a saved .npy/.npz file as input.")

    signal_path = Path(results)
    if not signal_path.exists():
        raise FileNotFoundError(f"Results file not found: {signal_path}")

    loaded = np.load(signal_path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            print(f"[metric_pipeline] Loaded NPZ archive with keys: {loaded.files}")
        except Exception:
            print("[metric_pipeline] Loaded NPZ archive")
    else:
        print(f"[metric_pipeline] Loaded array object, shape={getattr(loaded, 'shape', None)}, dtype={getattr(loaded, 'dtype', None)}")
    # helper to extract arrays from loaded file
    def _extract_arrays_and_metadata(loaded_obj):
        arrays = {}
        metadata = {}
        # npz archive
        if isinstance(loaded_obj, np.lib.npyio.NpzFile):
            for k in loaded_obj.files:
                try:
                    arr = loaded_obj[k]
                except Exception:
                    continue
                if k.lower() in ("metadata", "meta"):
                    raw = arr
                    if isinstance(raw, np.ndarray) and raw.ndim == 0:
                        raw = raw.item()
                    if isinstance(raw, (bytes, str)):
                        try:
                            metadata = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                        except Exception:
                            metadata = {}
                    elif isinstance(raw, dict):
                        metadata = raw
                else:
                    arrays[k] = np.asarray(arr)
        else:
            # single .npy file: could be object-array containing dict or plain array
            if isinstance(loaded_obj, np.ndarray) and loaded_obj.dtype == object and loaded_obj.shape == ():
                obj = loaded_obj.item()
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k.lower() == "metadata":
                            metadata = v if isinstance(v, dict) else {}
                        else:
                            arrays[k] = np.asarray(v)
                else:
                    arrays["predicted_signal"] = np.asarray(obj)
            else:
                arrays["predicted_signal"] = np.asarray(loaded_obj)
        return arrays, metadata

    arrays, metadata = _extract_arrays_and_metadata(loaded)
    try:
        loaded.close()
    except Exception:
        pass

    # Find predicted and ground-truth arrays among keys; only the canonical names are accepted.
    if "predicted_signal" not in arrays:
        raise ValueError("Saved file must contain a 'predicted_signal' array.")
    if "ground_truth_segment" not in arrays:
        raise ValueError("Saved file must contain a 'ground_truth_segment' array.")

    pred = np.asarray(arrays["predicted_signal"], dtype=float)
    gt = np.asarray(arrays["ground_truth_segment"], dtype=float)
    print(gt.shape, pred.shape)
    print(gt)

    # Read sampling rates from metadata
    predicted_fs_meta = metadata.get("fps") if isinstance(metadata, dict) else None
    gt_fs_meta = metadata.get("fps_ground_truth") if isinstance(metadata, dict) else None

    if predicted_fs_meta is None or gt_fs_meta is None:
        raise ValueError("Saved file metadata must include both 'fps' (predicted) and 'fps_ground_truth' (ground truth).")

    predicted_fs_val = float(predicted_fs_meta)
    gt_fs_val = float(gt_fs_meta)

    print(f"[metric_pipeline] Sampling rates - predicted: {predicted_fs_val} Hz, ground-truth: {gt_fs_val} Hz")

    # Create output folder early so we can save pre-resample plots
    output_root = signal_path.parent / "metric_results"
    output_folder = output_root / signal_path.stem
    os.makedirs(output_folder, exist_ok=True)
    print(f"[metric_pipeline] Created output folder: {output_folder}")

    # Signals are expected to be already cut to the same time span; only resample predicted -> gt
    if predicted_fs_val != gt_fs_val:
        pred_rs = resample_signal(pred, predicted_fs_val, gt_fs_val)
    else:
        pred_rs = pred.copy()

    # If lengths differ after resampling, truncate to the shortest length and build time axis from that length
    if len(gt) != len(pred_rs):
        print(
            f"Warning, after resampling, predicted signal length {len(pred_rs)} does not match ground truth length {len(gt)}. Truncate"
        )
        L = min(len(gt), len(pred_rs))
        gt_segment = gt[:L]
        pred_rs = pred_rs[:L]
    else:
        L = len(gt)
        gt_segment = gt

    print(f"[metric_pipeline] Working length after resample/truncate: {L} samples")

    t = np.arange(L) / float(gt_fs_val)

    # Peak detection
    # Use a prominence that is at least a fraction of the predicted signal's standard
    # deviation (PEAK_PROMINENCE_SIGNAL_STD_FRACTION by default, or a per-approach/camera
    # override), so the threshold scales with the signal's amplitude; the floor acts as
    # an absolute minimum. new_approach mwir/vario use a larger fraction because their
    # higher frame rate makes the signal noisier -- see PROMINENCE_SIGNAL_OVERRIDES.
    prominence_floor, prominence_std_fraction = _prominence_signal_settings_for_path(signal_path)
    pred_signal_std = float(np.nanstd(pred_rs))
    pred_prominence = max(
        prominence_floor, prominence_std_fraction * pred_signal_std
    )
    print(
        f"[metric_pipeline] Predicted-signal peak prominence: {pred_prominence:.6g} "
        f"(>= {prominence_std_fraction:.0%} of signal std {pred_signal_std:.6g}, "
        f"floor {prominence_floor})"
    )
    pred_peaks, _ = detect_peaks_in_signal(pred_rs, gt_fs_val, min_distance_s=PEAK_DISTANCE_SIGNAL, prominence=pred_prominence)
    pred_peak_times = t[pred_peaks]

    gt_peaks, gt_peak_times_rel = detect_ground_truth_peaks(
        gt_segment, gt_fs_val, method=gt_peak_method, source_path=signal_path
    )
    gt_peak_times = gt_peak_times_rel

    print(
        f"[metric_pipeline] Detected peaks (gt method='{gt_peak_method}') - "
        f"predicted: {pred_peaks.size}, ground-truth: {gt_peaks.size}"
    )

    # Instantaneous HR
    pred_ihr, pred_ihr_times = instantaneous_hr_from_peak_times(pred_peak_times)
    gt_ihr, gt_ihr_times = instantaneous_hr_from_peak_times(gt_peak_times)

    # Mean HR-----------------------
    gt_hr_mean = mean_hr_from_peak_times(gt_peak_times)
    signal_hr_mean = mean_hr_from_peak_times(pred_peak_times)

    # Original pipeline HR estimate-----------------------
    # yang_et_al and nair_et_al already produce a single heart-rate estimate (stored
    # as ``heart_rate_estimate_bpm`` in the npz metadata); new_approach leaves it None.
    # Expose that estimate and its signed difference to the ground-truth mean HR
    # (estimate - gt_hr_mean, the same pred-minus-gt convention as Bland-Altman) so the
    # CSV can compare the pipeline's own number against the ground truth directly.
    original_pipeline_bpm_estimate = None
    if isinstance(metadata, dict) and _is_finite_scalar(metadata.get("heart_rate_estimate_bpm")):
        original_pipeline_bpm_estimate = float(metadata["heart_rate_estimate_bpm"])
    original_pipeline_bpm_estimate_diff_to_gt_mean = None
    if original_pipeline_bpm_estimate is not None and _is_finite_scalar(gt_hr_mean):
        original_pipeline_bpm_estimate_diff_to_gt_mean = (
            original_pipeline_bpm_estimate - float(gt_hr_mean)
        )
    print(
        f"[metric_pipeline] Original pipeline HR estimate: "
        f"{original_pipeline_bpm_estimate} bpm; diff to GT mean: "
        f"{original_pipeline_bpm_estimate_diff_to_gt_mean}"
    )

    # Match GT & predicted HRs within a tolerance, others are disregarded-----------------------
    # Compute paired instantaneous-HR metrics using timestamp-based matching
    matched_gt_ihr, matched_pred_ihr, matched_gt_times, matched_pred_times = match_by_timestamp_tolerance(
        gt_values=gt_ihr,
        gt_times=gt_ihr_times,
        pred_values=pred_ihr,
        pred_times=pred_ihr_times,
        tolerance_ms=tolerance_ms,
    )

    print(f"[metric_pipeline] Matched pairs within tolerance ({tolerance_ms} ms): {matched_gt_ihr.size}")

    peak_to_peak_disregard_metrics, ba_stats_disregard = compute_metrics_with_optional_bland_altman(
        matched_gt_ihr,
        matched_pred_ihr,
        plot_bland_altman=plot_bland_altman and not plot_steps,
        title=f"Bland-Altman Instantaneous HR (±{tolerance_ms:.0f} ms match)",
    )
    peak_to_peak_disregard_metrics["num_matched"] = int(matched_gt_ihr.size)

    # Match GT & predicted HRs to closest, don't disregard any-----------------------
    closest_matches = match_by_nearest_timestamp(
        gt_values=gt_ihr,
        gt_times=gt_ihr_times,
        pred_values=pred_ihr,
        pred_times=pred_ihr_times,
    )
    closest_gt_ihr = np.concatenate(
        [
            closest_matches["gt_to_pred_gt_values"],
            closest_matches["pred_to_gt_gt_values"],
        ]
    )
    closest_pred_ihr = np.concatenate(
        [
            closest_matches["gt_to_pred_pred_values"],
            closest_matches["pred_to_gt_pred_values"],
        ]
    )
    closest_gt_times = np.concatenate(
        [
            closest_matches["gt_to_pred_gt_times"],
            closest_matches["pred_to_gt_gt_times"],
        ]
    )
    closest_pred_times = np.concatenate(
        [
            closest_matches["gt_to_pred_pred_times"],
            closest_matches["pred_to_gt_pred_times"],
        ]
    )
    peak_to_peak_closest_metrics, ba_stats_closest = compute_metrics_with_optional_bland_altman(
        closest_gt_ihr,
        closest_pred_ihr,
        plot_bland_altman=False,
        title="Bland-Altman Instantaneous HR (closest timestamp fallback)",
    )

    print(f"[metric_pipeline] Closest-matching pairs combined size: {closest_gt_ihr.size}")

    # Create output folder named after the input NPZ file inside a `metric_results` root
    output_root = signal_path.parent / "metric_results"
    output_folder = output_root / signal_path.stem
    os.makedirs(output_folder, exist_ok=True)

    print(f"[metric_pipeline] Created output folder: {output_folder}")

    # Inform about plots to be generated
    if not plot_steps and not plot_bland_altman:
        print("[metric_pipeline] No plots requested (both plot_steps and plot_bland_altman are False). Skipping plot generation.")

    # Optional plotting: one combined figure with the requested subplots
    if plot_steps:
        fig, axes = plt.subplots(4, 1, figsize=(14, 20))
        fig.suptitle("Instantaneous HR evaluation", fontsize=15)

        # 1) Ground-truth signal with detected peaks
        axes[0].plot(t, gt_segment, label="Ground truth", alpha=0.85, color="C0")
        if gt_peaks.size > 0:
            axes[0].scatter(gt_peak_times, gt_segment[gt_peaks], s=35, color="C0", edgecolors="black", linewidths=0.4, label="GT peaks")
        axes[0].set_title("Ground truth signal with detected peaks")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend(loc="best")
        axes[0].grid(alpha=0.2)

        # 2) Predicted signal with detected peaks
        axes[1].plot(t, pred_rs, label="Predicted (resampled)", alpha=0.75, color="C1")
        if pred_peaks.size > 0:
            axes[1].scatter(pred_peak_times, pred_rs[pred_peaks], s=35, color="C1", edgecolors="black", linewidths=0.4, label="Pred peaks")
        axes[1].set_title("Predicted signal with detected peaks")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Amplitude")
        axes[1].legend(loc="best")
        axes[1].grid(alpha=0.2)

        # 3) IHR plot with disregarded/tolerance matching
        axes[2].plot(gt_ihr_times, gt_ihr, label="GT IHR", color="C0", alpha=0.7, marker="o")
        axes[2].plot(pred_ihr_times, pred_ihr, label="Pred IHR", color="C1", alpha=0.7, marker="o")
        if matched_gt_times.size > 0:
            axes[2].scatter(matched_gt_times, matched_gt_ihr, s=45, color="C0", edgecolors="black", linewidths=0.4, label="Matched GT IHR")
            axes[2].scatter(matched_pred_times, matched_pred_ihr, s=45, color="C1", edgecolors="black", linewidths=0.4, label="Matched Pred IHR")
            for x_gt, y_gt, x_pred, y_pred in zip(matched_gt_times, matched_gt_ihr, matched_pred_times, matched_pred_ihr):
                axes[2].plot([x_gt, x_pred], [y_gt, y_pred], color="gray", alpha=0.45, linewidth=0.8)
        axes[2].set_title(f"IHR with tolerance matching (disregarded unmatched, ±{tolerance_ms:.0f} ms)")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("HR (bpm)")
        axes[2].legend(loc="best")
        axes[2].grid(alpha=0.2)

        # 4) IHR plot with closest matching in both directions
        axes[3].plot(gt_ihr_times, gt_ihr, label="GT IHR", color="C0", alpha=0.45, marker="o")
        axes[3].plot(pred_ihr_times, pred_ihr, label="Pred IHR", color="C1", alpha=0.45, marker="o")
        gt_to_pred_gt_times = closest_matches["gt_to_pred_gt_times"]
        gt_to_pred_pred_times = closest_matches["gt_to_pred_pred_times"]
        gt_to_pred_gt_values = closest_matches["gt_to_pred_gt_values"]
        gt_to_pred_pred_values = closest_matches["gt_to_pred_pred_values"]
        pred_to_gt_pred_times = closest_matches["pred_to_gt_pred_times"]
        pred_to_gt_gt_times = closest_matches["pred_to_gt_gt_times"]
        pred_to_gt_pred_values = closest_matches["pred_to_gt_pred_values"]
        pred_to_gt_gt_values = closest_matches["pred_to_gt_gt_values"]

        if gt_to_pred_gt_times.size > 0:
            axes[3].scatter(gt_to_pred_gt_times, gt_to_pred_gt_values, s=45, color="C0", edgecolors="black", linewidths=0.4, label="GT matched to closest Pred")
            axes[3].scatter(gt_to_pred_pred_times, gt_to_pred_pred_values, s=45, color="C1", edgecolors="black", linewidths=0.4, label="Pred matched from GT")
            for x_gt, y_gt, x_pred, y_pred in zip(gt_to_pred_gt_times, gt_to_pred_gt_values, gt_to_pred_pred_times, gt_to_pred_pred_values):
                axes[3].plot([x_gt, x_pred], [y_gt, y_pred], color="gray", alpha=0.35, linewidth=0.8)

        if pred_to_gt_pred_times.size > 0:
            axes[3].scatter(pred_to_gt_gt_times, pred_to_gt_gt_values, s=30, facecolors="none", edgecolors="C0", linewidths=1.1, label="GT matched from Pred")
            axes[3].scatter(pred_to_gt_pred_times, pred_to_gt_pred_values, s=30, facecolors="none", edgecolors="C1", linewidths=1.1, label="Pred matched to closest GT")
            for x_pred, y_pred, x_gt, y_gt in zip(pred_to_gt_pred_times, pred_to_gt_pred_values, pred_to_gt_gt_times, pred_to_gt_gt_values):
                axes[3].plot([x_pred, x_gt], [y_pred, y_gt], color="tab:purple", alpha=0.25, linewidth=0.7)

        axes[3].set_title("IHR with closest matching in both directions")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylabel("HR (bpm)")
        axes[3].legend(loc="best", ncol=2)
        axes[3].grid(alpha=0.2)

        plt.tight_layout()
        combined_plot_path = output_folder / "01_combined_evaluation.png"
        print(f"[metric_pipeline] Saving combined plot to: {combined_plot_path}")
        plt.savefig(combined_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    # Generate and save Bland-Altman plots for both matching strategies
    if plot_bland_altman or plot_steps:
        # Disregard/tolerance matching Bland-Altman plot
        if matched_gt_ihr.size > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            bland_altman(
                matched_gt_ihr,
                matched_pred_ihr,
                plot=True,
                ax=ax,
                title=f"Bland-Altman: Tolerance matching (±{tolerance_ms:.0f} ms)",
            )
            ba_disregard_path = output_folder / "02_bland_altman_disregard.png"
            print(f"[metric_pipeline] Saving Bland-Altman (disregard) to: {ba_disregard_path}")
            fig.tight_layout()
            plt.savefig(ba_disregard_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        # Closest matching Bland-Altman plot
        if closest_gt_ihr.size > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            bland_altman(
                closest_gt_ihr,
                closest_pred_ihr,
                plot=True,
                ax=ax,
                title="Bland-Altman: Closest matching (both directions)",
            )
            ba_closest_path = output_folder / "03_bland_altman_closest.png"
            print(f"[metric_pipeline] Saving Bland-Altman (closest) to: {ba_closest_path}")
            fig.tight_layout()
            plt.savefig(ba_closest_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Save output data as JSON for easy inspection
    def _serialize_for_json(obj):
        """Convert numpy types to JSON-serializable Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _serialize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialize_for_json(v) for v in obj]
        return obj

    output_data = {
        "input_file": str(signal_path),
        "gt_peak_method": str(gt_peak_method),
        "tolerance_ms": tolerance_ms,
        "ground_truth_fs": float(gt_fs_val),
        "predicted_fs": float(predicted_fs_val),
        "metrics_disregard": _serialize_for_json(peak_to_peak_disregard_metrics),
        "ba_stats_disregard": _serialize_for_json(ba_stats_disregard),
        "metrics_closest": _serialize_for_json(peak_to_peak_closest_metrics),
        "ba_stats_closest": _serialize_for_json(ba_stats_closest),
        "gt_hr_mean": float(gt_hr_mean) if _is_finite_scalar(gt_hr_mean) else None,
        "signal_hr_mean": float(signal_hr_mean) if _is_finite_scalar(signal_hr_mean) else None,
        "original_pipeline_bpm_estimate": original_pipeline_bpm_estimate,
        "original_pipeline_bpm_estimate_diff_to_gt_mean": original_pipeline_bpm_estimate_diff_to_gt_mean,
        "num_gt_peaks": int(gt_peaks.size),
        "num_pred_peaks": int(pred_peaks.size),
        "num_matched_disregard": int(matched_gt_ihr.size),
        "num_matched_closest": int(closest_gt_ihr.size),
    }

    output_json_path = output_folder / "00_metrics.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"[metric_pipeline] Saved metrics JSON to: {output_json_path}")

    # Additionally append a flattened summary row (the same data as the JSON, plus a
    # pointer to this run's plot folder) to an aggregate CSV. Running a batch over
    # many npz files therefore yields one CSV with the metrics for all of them.
    if csv_output_path is not None:
        append_metrics_summary_row(
            csv_output_path=csv_output_path,
            output_data=output_data,
            plots_folder=output_folder,
            json_path=output_json_path,
        )
        print(f"[metric_pipeline] Appended metrics summary row to CSV: {csv_output_path}")

    out = {
        "peak_to_peak_disregard_metrics": peak_to_peak_disregard_metrics,
        "peak_to_peak_disregard_ba_stats": ba_stats_disregard,
        "peak_to_peak_closest_metrics": peak_to_peak_closest_metrics,
        "peak_to_peak_closest_ba_stats": ba_stats_closest,
        "gt_hr_mean": gt_hr_mean,
        "signal_hr_mean": signal_hr_mean,
        "ground_truth_ihr": gt_ihr,
        "ground_truth_ihr_times": gt_ihr_times,
        "predicted_ihr": pred_ihr,
        "predicted_ihr_times": pred_ihr_times,
        "matched_ground_truth_ihr_disregard": matched_gt_ihr,
        "matched_predicted_ihr_disregard": matched_pred_ihr,
        "matched_ground_truth_times_disregard": matched_gt_times,
        "matched_predicted_times_disregard": matched_pred_times,
        "closest_ground_truth_ihr": closest_gt_ihr,
        "closest_predicted_ihr": closest_pred_ihr,
        "closest_ground_truth_times": closest_gt_times,
        "closest_predicted_times": closest_pred_times,
        "ground_truth_fs": gt_fs_val,
        "predicted_resampled_fs": gt_fs_val,
        "tolerance_ms": tolerance_ms,
    }
    out["output_folder"] = str(output_folder)
    out["metrics_json_path"] = str(output_json_path)

    return out


def batch_evaluate_folder(
    folder: str | Path,
    pattern: str = "*.npz",
    flip_ground_truth: bool = False,
    tolerance_ms: float = 150.0,
    plot_steps: bool = True,
    plot_bland_altman: bool = True,
    csv_output_path: str | Path | None = None,
    overwrite_csv: bool = False,
    existing_entry_mode: str = "overwrite",
    gt_peak_method: str = DEFAULT_GT_PEAK_METHOD,
) -> Dict[str, object]:
    """Run evaluate_instantaneous_hr_against_ground_truth for all archive files in `folder`.

    ``existing_entry_mode`` controls files that already have a row in the aggregate
    CSV: "skip" only processes files not yet in the CSV, while "overwrite" recomputes
    them and replaces their existing row. Set ``overwrite_csv=True`` to wipe the whole
    CSV before starting (then every file is processed fresh).

    For each found file, a subfolder named after the file stem is created (if not existing)
    and figures are saved inside that folder (the called function already saves into
    `signal_path.parent / signal_path.stem`). A single aggregate CSV (one row per npz,
    same fields as each run's JSON plus the plot-folder path) is written to
    ``csv_output_path`` (default: ``folder / 'all_npz_metrics.csv'``). Returns a dict
    mapping file paths to the returned result dict or an error description.
    """
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder not found: {folder}")

    if csv_output_path is None:
        csv_output_path = folder / "all_npz_metrics.csv"
    csv_output_path = Path(csv_output_path)
    if overwrite_csv and csv_output_path.exists():
        print(f"[metric_pipeline] Overwriting existing summary CSV: {csv_output_path}")
        csv_output_path.unlink()

    # Migrate any existing CSV to the current column schema up-front, so reading the
    # already-present files and rewriting rows stay column-aligned (heals CSVs written
    # before new summary columns were added).
    _rewrite_csv_with_canonical_header(csv_output_path)

    results: dict[str, object] = {}
    files = sorted(folder.glob(pattern))
    # Skip or overwrite files that already have a row in the aggregate CSV.
    files = _select_files_by_existing_entries(
        files,
        csv_output_path,
        existing_entry_mode,
        log_prefix="[metric_pipeline] Batch:",
    )
    if not files:
        print(
            f"[metric_pipeline] Batch: nothing to process; all files already present "
            f"in CSV (existing_entry_mode='{existing_entry_mode}')."
        )
        return results
    for f in files:
        print(f"[metric_pipeline] Batch: processing file: {f}")
        try:
            out = evaluate_instantaneous_hr_against_ground_truth(
                results=f,
                flip_ground_truth=flip_ground_truth,
                tolerance_ms=tolerance_ms,
                plot_steps=plot_steps,
                plot_bland_altman=plot_bland_altman,
                csv_output_path=csv_output_path,
                gt_peak_method=gt_peak_method,
            )
            results[str(f)] = out
            of = out.get("output_folder", "N/A")
            matched = len(out.get("matched_ground_truth_ihr_disregard", []))
            print(f"[metric_pipeline] Batch: finished {f}; output_folder={of}; matched_within_tolerance={matched}")
        except Exception as exc:
            results[str(f)] = {"error": str(exc)}
            print(f"[metric_pipeline] Batch: error processing {f}: {exc}")

    print(f"[metric_pipeline] Batch complete. Aggregate metrics CSV: {csv_output_path}")
    return results


def batch_evaluate_folder_structure(
    root: str | Path,
    approaches: tuple[str, ...] = ("nair_et_al", "yang_et_al", "new_approach"),
    cameras: tuple[str, ...] | str | None = None,
    results_subdir: str = "results",
    pattern: str = "*.npz",
    flip_ground_truth: bool = False,
    tolerance_ms: float = 150.0,
    plot_steps: bool = True,
    plot_bland_altman: bool = True,
    csv_output_path: str | Path | None = None,
    overwrite_csv: bool = False,
    existing_entry_mode: str = "overwrite",
    gt_peak_method: str = DEFAULT_GT_PEAK_METHOD,
) -> Dict[str, object]:
    """Evaluate every npz under ``<root>/<approach>/<results_subdir>/**`` for each approach.

    ``approaches`` selects which approach subfolders to walk and accepts either the
    short run_batch aliases ("new"/"yang"/"nair") or the full folder names
    ("new_approach"/"yang_et_al"/"nair_et_al"). ``cameras`` optionally restricts the
    run to specific camera subfolders (the first path component under ``results``,
    e.g. ("mwir", "vario")); ``None`` evaluates every camera. Both mirror
    run_batch.py's ``pipelines`` / ``cameras`` configuration.

    ``existing_entry_mode`` controls npz files that already have a row in the
    aggregate CSV: "skip" only computes metrics for files not yet in the CSV, while
    "overwrite" recomputes them and replaces their existing row. Set
    ``overwrite_csv=True`` to wipe the whole CSV first (then every file is fresh).

    The expected layout is::

        root/
          nair_et_al/results/<...>/*.npz
          yang_et_al/results/<...>/*.npz
          new_approach/results/<...>/*.npz

    Every matching npz is evaluated (saving its own JSON + plots, as before) and a
    single aggregate CSV (one row per npz, same fields as each JSON plus the path to
    that run's plot folder) is written to ``csv_output_path`` (default:
    ``root / 'all_npz_metrics.csv'``). Returns a dict mapping file paths to the
    returned result dict or an error description.
    """
    root = Path(root)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Root folder not found: {root}")

    # Accept short ("new"/"yang"/"nair") or full approach names, like run_batch.py.
    approaches = _normalize_approaches(approaches)
    # Optional camera filter: the immediate subfolder of each approach's results dir
    # (e.g. "mwir"/"vario"/"lepton"), matched case-insensitively. None = all cameras.
    camera_filter = _normalize_camera_filter(cameras)

    if csv_output_path is None:
        csv_output_path = root / "all_npz_metrics.csv"
    csv_output_path = Path(csv_output_path)
    if overwrite_csv and csv_output_path.exists():
        print(f"[metric_pipeline] Overwriting existing summary CSV: {csv_output_path}")
        csv_output_path.unlink()

    # Migrate any existing CSV to the current column schema up-front (heals CSVs
    # written before new summary columns were added; see _rewrite_csv_with_canonical_header).
    _rewrite_csv_with_canonical_header(csv_output_path)

    print(f"[metric_pipeline] Structure batch: approach(es): {list(approaches)}")
    if camera_filter is not None:
        print(f"[metric_pipeline] Structure batch: restricting to camera(s): {sorted(camera_filter)}")

    # Collect all npz files across the approach/results trees first (so the summary
    # print is meaningful and missing folders are reported clearly).
    npz_files: list[Path] = []
    for approach in approaches:
        results_dir = root / approach / results_subdir
        if not results_dir.exists() or not results_dir.is_dir():
            print(f"[metric_pipeline] Structure batch: skipping missing results dir: {results_dir}")
            continue
        found = sorted(results_dir.rglob(pattern))
        if camera_filter is not None:
            found = [f for f in found if _camera_subfolder_of(f, results_dir) in camera_filter]
        print(f"[metric_pipeline] Structure batch: {approach}: found {len(found)} '{pattern}' file(s) under {results_dir}")
        npz_files.extend(found)

    results: dict[str, object] = {}
    if not npz_files:
        print(f"[metric_pipeline] Structure batch: no '{pattern}' files found under any approach results dir in {root}")
        return results

    # Skip or overwrite npz files that already have a row in the aggregate CSV.
    files_to_process = _select_files_by_existing_entries(
        npz_files,
        csv_output_path,
        existing_entry_mode,
        log_prefix="[metric_pipeline] Structure batch:",
    )
    if not files_to_process:
        print(
            f"[metric_pipeline] Structure batch: all {len(npz_files)} file(s) already "
            f"present in CSV; nothing to do (existing_entry_mode='{existing_entry_mode}')."
        )
        return results

    print(f"[metric_pipeline] Structure batch: evaluating {len(files_to_process)} of {len(npz_files)} file(s); aggregate CSV: {csv_output_path}")
    for f in files_to_process:
        print(f"[metric_pipeline] Structure batch: processing file: {f}")
        try:
            out = evaluate_instantaneous_hr_against_ground_truth(
                results=f,
                flip_ground_truth=flip_ground_truth,
                tolerance_ms=tolerance_ms,
                plot_steps=plot_steps,
                plot_bland_altman=plot_bland_altman,
                csv_output_path=csv_output_path,
                gt_peak_method=gt_peak_method,
            )
            results[str(f)] = out
            of = out.get("output_folder", "N/A")
            matched = len(out.get("matched_ground_truth_ihr_disregard", []))
            print(f"[metric_pipeline] Structure batch: finished {f}; output_folder={of}; matched_within_tolerance={matched}")
        except Exception as exc:
            results[str(f)] = {"error": str(exc)}
            print(f"[metric_pipeline] Structure batch: error processing {f}: {exc}")

    print(f"[metric_pipeline] Structure batch complete. Aggregate metrics CSV: {csv_output_path}")
    return results


if __name__ == "__main__":
    # Choose how to run:
    #   "single"    -> evaluate one folder (all matching files in it) or one file, as before.
    #   "structure" -> walk <root>/{nair_et_al,yang_et_al,new_approach}/results/**/*.npz
    #                  and aggregate every npz's metrics into one CSV at the root.
    RUN_MODE = "structure"  # "single" or "structure"

    # For RUN_MODE == "structure": how to treat npz files that already have a row in
    # the aggregate CSV.
    #   "skip"      -> only compute metrics for npz files that are NOT yet in the CSV.
    #   "overwrite" -> recompute every npz file and replace its existing CSV row.
    EXISTING_ENTRY_MODE = "overwrite"  # "skip" or "overwrite"

    # Algorithm used to detect R-peaks in the ECG ground-truth signal:
    #   "find_peaks" -> scipy.signal.find_peaks (original behaviour).
    #   "staple"     -> STAPLE consensus of several QRS detectors (more robust on
    #                   noisy/pathological ECG); needs the 'py-ecg-detectors' package.
    GT_PEAK_METHOD = "find_peaks"  # "find_peaks" or "staple"

    # For RUN_MODE == "structure": restrict which approach(es) and camera(s) to
    # evaluate (similar to run_batch.py's `pipelines` / `cameras`). APPROACHES_TO_RUN
    # accepts short ("new"/"yang"/"nair") or full ("new_approach"/...) names; set
    # CAMERAS_TO_RUN to e.g. ("mwir", "vario") to evaluate only those cameras, or None
    # for all cameras.
    APPROACHES_TO_RUN = ("nair_et_al", "yang_et_al", "new_approach") #("nair_et_al", "yang_et_al", "new_approach")
    CAMERAS_TO_RUN = None#("mwir", "vario", "l")  # e.g. ("mwir", "vario") or None for all

    # Shared evaluation parameters
    HARD_CODED_PATTERN = DEFAULT_PATTERN
    HARD_CODED_FLIP = False
    HARD_CODED_TOLERANCE_MS = DEFAULT_TOLERANCE_MS
    HARD_CODED_PLOT_STEPS = DEFAULT_PLOT_STEPS
    HARD_CODED_PLOT_BA = DEFAULT_PLOT_BLAND_ALTMAN

    if RUN_MODE == "structure":
        # Root that contains the approach subfolders (each with a `results` dir).
        HARD_CODED_ROOT = Path("saved_results")
        if HARD_CODED_ROOT.exists() and HARD_CODED_ROOT.is_dir():
            batch_evaluate_folder_structure(
                root=HARD_CODED_ROOT,
                approaches=APPROACHES_TO_RUN,
                cameras=CAMERAS_TO_RUN,
                pattern=HARD_CODED_PATTERN,
                flip_ground_truth=HARD_CODED_FLIP,
                tolerance_ms=HARD_CODED_TOLERANCE_MS,
                plot_steps=HARD_CODED_PLOT_STEPS,
                plot_bland_altman=HARD_CODED_PLOT_BA,
                existing_entry_mode=EXISTING_ENTRY_MODE,
                gt_peak_method=GT_PEAK_METHOD,
            )
        else:
            print(
                f"Hard-coded root not found: {HARD_CODED_ROOT}. Update HARD_CODED_ROOT "
                "to point at the folder containing nair_et_al/yang_et_al/new_approach."
            )
    else:
        # Hard-coded parameters for single folder/file evaluation.
        HARD_CODED_PATH = Path("saved_results/new_approach/results/lepton")

        if HARD_CODED_PATH.exists():
            if HARD_CODED_PATH.is_dir():
                batch_evaluate_folder(
                    folder=HARD_CODED_PATH,
                    pattern=HARD_CODED_PATTERN,
                    flip_ground_truth=HARD_CODED_FLIP,
                    tolerance_ms=HARD_CODED_TOLERANCE_MS,
                    plot_steps=HARD_CODED_PLOT_STEPS,
                    plot_bland_altman=HARD_CODED_PLOT_BA,
                    gt_peak_method=GT_PEAK_METHOD,
                )
            elif HARD_CODED_PATH.is_file():
                evaluate_instantaneous_hr_against_ground_truth(
                    results=str(HARD_CODED_PATH),
                    flip_ground_truth=HARD_CODED_FLIP,
                    tolerance_ms=HARD_CODED_TOLERANCE_MS,
                    plot_steps=HARD_CODED_PLOT_STEPS,
                    plot_bland_altman=HARD_CODED_PLOT_BA,
                    csv_output_path=HARD_CODED_PATH.parent / "all_npz_metrics.csv",
                    gt_peak_method=GT_PEAK_METHOD,
                )
        else:
            print(f"Hard-coded path not found: {HARD_CODED_PATH}. Update the path in the script to point to your .npz folder or file.")
