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
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from src.approaches.yang_et_al.utils_ground_truth import (
    load_ground_truth_signal,
    get_ground_truth_segment,
    detect_peaks_in_signal,
    mean_hr_from_peak_times,
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
        if np.isfinite(dt[j]) and dt[j] <= tol_s:
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match each ground-truth value to the closest prediction by timestamp.

    Unlike tolerance matching, this keeps all ground-truth points (if predictions exist)
    and allows reusing predicted points.
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

    matched_gt_vals = []
    matched_pred_vals = []
    matched_gt_t = []
    matched_pred_t = []

    for i in range(gt_times.shape[0]):
        dt = np.abs(pred_times - gt_times[i])
        j = int(np.argmin(dt))
        matched_gt_vals.append(gt_values[i])
        matched_pred_vals.append(pred_values[j])
        matched_gt_t.append(gt_times[i])
        matched_pred_t.append(pred_times[j])

    return (
        np.asarray(matched_gt_vals, dtype=float),
        np.asarray(matched_pred_vals, dtype=float),
        np.asarray(matched_gt_t, dtype=float),
        np.asarray(matched_pred_t, dtype=float),
    )


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
        "gt_hr_mean": float(gt_hr_mean) if np.isfinite(gt_hr_mean) else np.nan,
        "signal_hr_mean": float(signal_hr_mean) if np.isfinite(signal_hr_mean) else np.nan,
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


def evaluate_against_ground_truth(
    predicted_signal: np.ndarray,
    predicted_fs: float,
    path_ground_truth: str,
    timespan: Optional[Tuple[float, float]] = None,
    time_first_frame_signal: float = 0.0,
    time_first_frame_ecg: float = 0.0,
    flip_ground_truth: bool = False,
    plot_steps: bool = True,
) -> Dict[str, object]:
    """Run the full ground-truth evaluation pipeline as in yang_et_al.

    Parameters
    - predicted_signal, predicted_fs: prediction from pipeline
    - path_ground_truth: path to .npz ground truth file (loaded with load_ground_truth_signal)
    - timespan: (start_time, end_time) in seconds to extract from ground truth; if None uses whole signal
    - time_first_frame_signal: offset of first frame of predicted signal (seconds)
    - time_first_frame_ecg: offset of first frame of ground truth (seconds)
    - flip_ground_truth: pass to load_ground_truth_signal
    - plot_steps: whether to show Bland-Altman plot

    Returns a dict with keys: metrics (RMSE, MAE, R2, ks_pvalue, mean_diff, loa_low, loa_high),
    gt_hr_mean, signal_hr_mean, signal_peak_times, ground_truth_peak_times, and raw arrays.
    """
    # Load ground truth
    gt_signal, gt_fs = load_ground_truth_signal(path_ground_truth, flip=flip_ground_truth)

    if timespan is not None:
        start_time, end_time = timespan
        signal_offset = float(time_first_frame_signal or 0.0)
        ecg_offset = float(time_first_frame_ecg or 0.0)
        gt_segment, gt_time = get_ground_truth_segment(
            gt_signal, gt_fs, start_time, end_time, signal_offset=signal_offset, ecg_offset=ecg_offset
        )
    else:
        gt_segment = gt_signal
        gt_time = np.arange(len(gt_segment)) / gt_fs
        start_time = 0.0

    # Build predicted signal time axis aligned to start_time
    signal_time = np.arange(len(predicted_signal)) / float(predicted_fs) + start_time

    # Detect peaks in predicted and ground truth
    try:
        signal_peaks, _ = detect_peaks_in_signal(predicted_signal, predicted_fs, min_distance_s=0.3)
        signal_peak_times = signal_time[signal_peaks]
    except (ValueError, RuntimeError):
        signal_peaks = np.array([], dtype=int)
        signal_peak_times = np.array([])

    try:
        gt_peaks, gt_peak_times = detect_peaks_in_signal(gt_segment, gt_fs, min_distance_s=0.5, prominence=1.0)
        gt_peak_times = gt_peak_times + start_time
    except (ValueError, RuntimeError):
        gt_peaks = np.array([], dtype=int)
        gt_peak_times = np.array([])

    gt_hr_mean = mean_hr_from_peak_times(gt_peak_times)
    signal_hr_mean = mean_hr_from_peak_times(signal_peak_times)

    if plot_steps:
        plt.figure(figsize=(10, 5))
        signal_line, = plt.plot(signal_time, predicted_signal, label="Selected PCA component")
        ground_truth_line, = plt.plot(gt_time, gt_segment, label="Ground truth", alpha=0.7)

        if signal_peaks.size > 0:
            plt.scatter(
                signal_peak_times,
                predicted_signal[signal_peaks],
                label="_nolegend_",
                s=25,
                color=signal_line.get_color(),
            )
        if gt_peaks.size > 0:
            plt.scatter(
                gt_peak_times,
                gt_segment[gt_peaks],
                label="_nolegend_",
                s=25,
                color=ground_truth_line.get_color(),
            )
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Selected PCA Component vs Ground Truth")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Compute standard metrics and Bland-Altman using evaluate_predictions
    metrics, ba_stats = evaluate_predictions(
        ground_truth_signal=gt_segment,
        ground_truth_fs=gt_fs,
        predicted_signal=predicted_signal,
        predicted_fs=predicted_fs,
        align_to_ground_truth=True,
        plot_bland_altman=plot_steps,
        return_all=True,
    )

    out = {
        "metrics": metrics,
        "ba_stats": ba_stats,
        "gt_hr_mean": gt_hr_mean,
        "signal_hr_mean": signal_hr_mean,
        "signal_peak_times": signal_peak_times,
        "ground_truth_peak_times": gt_peak_times,
        "ground_truth_fs": gt_fs,
        "predicted_fs": predicted_fs,
    }

    return out


def evaluate_instantaneous_hr_against_ground_truth(
    predicted_signal: np.ndarray,
    predicted_fs: float,
    path_ground_truth: str,
    path_signal: Optional[str] = None,
    roi: Optional[str] = None,
    csv_output_path: Optional[str] = "src/ground_truth_methods/evaluation_results.csv",
    timespan: Optional[Tuple[float, float]] = None,
    time_first_frame_signal: float = 0.0,
    time_first_frame_ecg: float = 0.0,
    flip_ground_truth: bool = False,
    tolerance_ms: float = 150.0,
    plot_steps: bool = True,
    plot_bland_altman: bool = True,
) -> Dict[str, object]:
    """Evaluate instantaneous HR by timestamp-based peak matching.

    Pipeline:
    1) load + optional timespan extraction of ground truth
    2) interpolate/resample predicted signal to ground-truth fs
    3) detect peaks in both signals
    4) compute instantaneous HR + timestamps from peak times
     5) compute mean HR for full predicted and full ground-truth signals
     6) compute peak-to-peak metrics in two modes:
         a) disregard unmatched peaks (within ±tolerance_ms)
         b) fallback to closest timestamp when no match in tolerance exists
    """
    # 1) ground truth loading and optional segmentation
    gt_signal, gt_fs = load_ground_truth_signal(path_ground_truth, flip=flip_ground_truth)

    t_gt_full = np.arange(len(gt_signal)) / float(gt_fs)

    if timespan is not None:
        start_time, end_time = timespan
        signal_offset = float(time_first_frame_signal or 0.0)
        ecg_offset = float(time_first_frame_ecg or 0.0)
        gt_segment, gt_time = get_ground_truth_segment(
            gt_signal,
            gt_fs,
            start_time,
            end_time,
            signal_offset=signal_offset,
            ecg_offset=ecg_offset,
        )
    else:
        start_time = 0.0
        gt_segment = np.asarray(gt_signal, dtype=float)
        gt_time = np.arange(len(gt_segment)) / float(gt_fs)

    # 2) resample prediction to GT fs, then keep same duration as GT segment
    pred = np.asarray(predicted_signal, dtype=float)
    pred_time_original = np.arange(len(pred)) / float(predicted_fs) + start_time
    pred_rs = resample_signal(pred, predicted_fs, gt_fs)
    L = min(len(gt_segment), len(pred_rs))
    if L == 0:
        raise ValueError("Empty signals after segmentation/resampling.")
    gt_segment = gt_segment[:L]
    gt_time = gt_time[:L]
    pred_rs = pred_rs[:L]
    pred_time = np.arange(L) / float(gt_fs) + start_time

    # 3) peak detection
    pred_peaks, _ = detect_peaks_in_signal(pred_rs, gt_fs, min_distance_s=0.3)
    pred_peak_times = pred_time[pred_peaks]

    gt_peaks, gt_peak_times_rel = detect_peaks_in_signal(
        gt_segment,
        gt_fs,
        min_distance_s=0.5,
        prominence=1.0,
    )
    gt_peak_times = gt_peak_times_rel + start_time

    # 4) instantaneous HR + timestamps
    pred_ihr, pred_ihr_times = instantaneous_hr_from_peak_times(pred_peak_times)
    gt_ihr, gt_ihr_times = instantaneous_hr_from_peak_times(gt_peak_times)

    gt_hr_mean = mean_hr_from_peak_times(gt_peak_times)
    signal_hr_mean = mean_hr_from_peak_times(pred_peak_times)

    mean_hr_metrics, mean_hr_ba_stats = compute_metrics_with_optional_bland_altman(
        np.array([gt_hr_mean], dtype=float),
        np.array([signal_hr_mean], dtype=float),
        plot_bland_altman=False,
        title="Bland-Altman Mean HR (whole signal)",
    )

    # 5) timestamp-based matching within tolerance
    matched_gt_ihr, matched_pred_ihr, matched_gt_times, matched_pred_times = match_by_timestamp_tolerance(
        gt_values=gt_ihr,
        gt_times=gt_ihr_times,
        pred_values=pred_ihr,
        pred_times=pred_ihr_times,
        tolerance_ms=tolerance_ms,
    )

    peak_to_peak_disregard_metrics, ba_stats_disregard = compute_metrics_with_optional_bland_altman(
        matched_gt_ihr,
        matched_pred_ihr,
        plot_bland_altman=plot_bland_altman and not plot_steps,
        title=f"Bland-Altman Instantaneous HR (±{tolerance_ms:.0f} ms match)",
    )
    peak_to_peak_disregard_metrics["num_matched"] = int(matched_gt_ihr.size)

    closest_gt_ihr, closest_pred_ihr, closest_gt_times, closest_pred_times = match_by_nearest_timestamp(
        gt_values=gt_ihr,
        gt_times=gt_ihr_times,
        pred_values=pred_ihr,
        pred_times=pred_ihr_times,
    )
    peak_to_peak_closest_metrics, ba_stats_closest = compute_metrics_with_optional_bland_altman(
        closest_gt_ihr,
        closest_pred_ihr,
        plot_bland_altman=False,
        title="Bland-Altman Instantaneous HR (closest timestamp fallback)",
    )

    if plot_steps:
        fig, axes = plt.subplots(4, 2, figsize=(18, 16))
        fig.suptitle("Instantaneous HR evaluation steps", fontsize=14)
        axes = axes.ravel()

        axes[0].plot(t_gt_full, gt_signal, color="C0", label="Ground truth (full)")
        axes[0].set_title("Step 1: Full ground truth signal")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend()

        axes[1].plot(gt_time, gt_segment, color="C1", label="Ground truth (segment)")
        axes[1].set_title("Step 2: Ground truth segment")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Amplitude")
        axes[1].legend()

        axes[2].plot(pred_time_original, pred, color="C3", alpha=0.75, label="Predicted (before)")
        axes[2].plot(pred_time, pred_rs, color="C2", alpha=0.85, label="Predicted (after interpolation)")
        axes[2].set_title("Step 3: Interpolation comparison")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Amplitude")
        axes[2].legend()

        axes[3].plot(gt_time, gt_segment, alpha=0.8, label="Ground truth")
        axes[3].plot(pred_time, pred_rs, alpha=0.8, label="Predicted (resampled)")
        axes[3].set_title("Step 4: GT vs resampled prediction")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylabel("Amplitude")
        axes[3].legend()

        gt_line = axes[4].plot(gt_time, gt_segment, label="Ground truth", alpha=0.8)[0]
        pred_line = axes[4].plot(pred_time, pred_rs, label="Predicted (resampled)", alpha=0.8)[0]
        if gt_peaks.size > 0:
            axes[4].scatter(gt_peak_times, gt_segment[gt_peaks], s=20, color=gt_line.get_color(), label="GT peaks")
        if pred_peaks.size > 0:
            axes[4].scatter(pred_peak_times, pred_rs[pred_peaks], s=20, color=pred_line.get_color(), label="Pred peaks")
        axes[4].set_title("Step 5: Peak detection")
        axes[4].set_xlabel("Time (s)")
        axes[4].set_ylabel("Amplitude")
        axes[4].legend()

        if gt_ihr.size > 0:
            axes[5].plot(gt_ihr_times, gt_ihr, marker="o", linestyle="-", label="GT IHR")
        if pred_ihr.size > 0:
            axes[5].plot(pred_ihr_times, pred_ihr, marker="o", linestyle="-", label="Pred IHR")
        axes[5].set_title("Step 6: Instantaneous HR")
        axes[5].set_xlabel("Time (s)")
        axes[5].set_ylabel("HR (bpm)")
        axes[5].legend()

        if gt_ihr.size > 0:
            axes[6].plot(gt_ihr_times, gt_ihr, alpha=0.35, label="GT IHR (all)", color="C0")
        if pred_ihr.size > 0:
            axes[6].plot(pred_ihr_times, pred_ihr, alpha=0.35, label="Pred IHR (all)", color="C1")
        if matched_gt_ihr.size > 0:
            axes[6].scatter(matched_gt_times, matched_gt_ihr, color="C0", s=28, label="Matched GT")
            axes[6].scatter(matched_pred_times, matched_pred_ihr, color="C1", s=28, label="Matched Pred")
            for i in range(matched_gt_ihr.size):
                axes[6].plot(
                    [matched_gt_times[i], matched_pred_times[i]],
                    [matched_gt_ihr[i], matched_pred_ihr[i]],
                    color="gray",
                    alpha=0.35,
                    linewidth=0.7,
                )
        axes[6].set_title(f"Step 7: Timestamp matching (±{tolerance_ms:.0f} ms)")
        axes[6].set_xlabel("Time (s)")
        axes[6].set_ylabel("HR (bpm)")
        axes[6].legend()

        if matched_gt_ihr.size > 0:
            ba_stats_subplot = bland_altman(
                matched_gt_ihr,
                matched_pred_ihr,
                plot=True,
                ax=axes[7],
                title="Step 8: Bland-Altman (matched IHR)",
            )
            if ba_stats_disregard is None:
                ba_stats_disregard = ba_stats_subplot
        else:
            axes[7].text(0.5, 0.5, "No matched pairs for\nBland-Altman", ha="center", va="center")
            axes[7].set_title("Step 8: Bland-Altman")
            axes[7].set_axis_off()

        plt.tight_layout()
        plt.show()

    out = {
        "metrics": peak_to_peak_disregard_metrics,
        "mean_hr_metrics": mean_hr_metrics,
        "mean_hr_ba_stats": mean_hr_ba_stats,
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
        "matched_ground_truth_ihr": matched_gt_ihr,
        "matched_predicted_ihr": matched_pred_ihr,
        "matched_ground_truth_times": matched_gt_times,
        "matched_predicted_times": matched_pred_times,
        "closest_ground_truth_ihr": closest_gt_ihr,
        "closest_predicted_ihr": closest_pred_ihr,
        "closest_ground_truth_times": closest_gt_times,
        "closest_predicted_times": closest_pred_times,
        "ground_truth_peak_times": gt_peak_times,
        "ground_truth_peak_indices": gt_peaks,
        "predicted_peak_times": pred_peak_times,
        "ground_truth_fs": gt_fs,
        "predicted_resampled_fs": gt_fs,
        "tolerance_ms": tolerance_ms,
    }

    if csv_output_path:
        append_evaluation_row_to_csv(
            csv_output_path=csv_output_path,
            path_signal=path_signal,
            path_ground_truth=path_ground_truth,
            roi=roi,
            gt_hr_mean=gt_hr_mean,
            signal_hr_mean=signal_hr_mean,
            gt_ihr_times=gt_ihr_times,
            gt_ihr=gt_ihr,
            pred_ihr_times=pred_ihr_times,
            pred_ihr=pred_ihr,
            metrics_mean_hr=mean_hr_metrics,
            metrics_peak_to_peak_disregard=peak_to_peak_disregard_metrics,
            metrics_peak_to_peak_closest=peak_to_peak_closest_metrics,
        )

    return out


if __name__ == "__main__":
    print("This module provides evaluation utilities for ground truth comparison.")
