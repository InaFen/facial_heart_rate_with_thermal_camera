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



def evaluate_instantaneous_hr_against_ground_truth(
    results: str | Path,
    flip_ground_truth: bool = False,
    tolerance_ms: float = 150.0,
    plot_steps: bool = True,
    plot_bland_altman: bool = True,
) -> Dict[str, object]:
    """Evaluate instantaneous HR using a single `.npy`/`.npz` file as input.
    """
    if not isinstance(results, (str, Path)):
        raise ValueError("This function only accepts a path to a saved .npy/.npz file as input.")

    signal_path = Path(results)
    if not signal_path.exists():
        raise FileNotFoundError(f"Results file not found: {signal_path}")

    loaded = np.load(signal_path, allow_pickle=True)

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

    # Read sampling rates from metadata
    predicted_fs_meta = metadata.get("fps") if isinstance(metadata, dict) else None
    gt_fs_meta = metadata.get("fps_ground_truth") if isinstance(metadata, dict) else None

    if predicted_fs_meta is None or gt_fs_meta is None:
        raise ValueError("Saved file metadata must include both 'fps' (predicted) and 'fps_ground_truth' (ground truth).")

    predicted_fs_val = float(predicted_fs_meta)
    gt_fs_val = float(gt_fs_meta)

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

    t = np.arange(L) / float(gt_fs_val)

    # Peak detection
    PEAK_DISTANCE_SIGNAL = 0.3
    PEAK_PROMINENCE_SIGNAL = 0.5
    PEAK_DISTANCE_GT = 0.5
    PEAK_PROMINENCE_GT = 1.0

    pred_peaks, _ = detect_peaks_in_signal(pred_rs, gt_fs_val, min_distance_s=PEAK_DISTANCE_SIGNAL, prominence=PEAK_PROMINENCE_SIGNAL)
    pred_peak_times = t[pred_peaks]

    gt_peaks, gt_peak_times_rel = detect_peaks_in_signal(gt_segment, gt_fs_val, min_distance_s=PEAK_DISTANCE_GT, prominence=PEAK_PROMINENCE_GT)
    gt_peak_times = gt_peak_times_rel

    # Instantaneous HR
    pred_ihr, pred_ihr_times = instantaneous_hr_from_peak_times(pred_peak_times)
    gt_ihr, gt_ihr_times = instantaneous_hr_from_peak_times(gt_peak_times)

    # Mean HR-----------------------
    gt_hr_mean = mean_hr_from_peak_times(gt_peak_times)
    signal_hr_mean = mean_hr_from_peak_times(pred_peak_times)

    # Match GT & predicted HRs within a tolerance, others are disregarded-----------------------
    # Compute paired instantaneous-HR metrics using timestamp-based matching
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

    # Match GT & predicted HRs to closest, don't disregard any-----------------------
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

    # Optional plotting (kept simple)
    if plot_steps:
        plt.figure(figsize=(10, 5))
        plt.plot(t, pred_rs, label="Predicted (resampled)")
        plt.plot(t, gt_segment, label="Ground truth", alpha=0.8)
        if pred_peaks.size > 0:
            plt.scatter(pred_peak_times, pred_rs[pred_peaks], s=25, color="C1")
        if gt_peaks.size > 0:
            plt.scatter(gt_peak_times, gt_segment[gt_peaks], s=25, color="C2")
        plt.legend()
        plt.title("Predicted vs Ground Truth (resampled)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

        if matched_gt_ihr.size > 0:
            ba = bland_altman(matched_gt_ihr, matched_pred_ihr, plot=True, title="Bland-Altman (matched IHR)")
            if ba_stats_disregard is None:
                ba_stats_disregard = ba

    out = {
        "metrics": peak_to_peak_disregard_metrics,
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
        "ground_truth_fs": gt_fs_val,
        "predicted_resampled_fs": gt_fs_val,
        "tolerance_ms": tolerance_ms,
    }

    return out


if __name__ == "__main__":
    evaluate_instantaneous_hr_against_ground_truth(
        results="saved_results/yang_et_al/05032026_before_after_sport_interpolated_temp_0_00-10_00s.npz",
        flip_ground_truth=False,
        tolerance_ms=150.0,
        plot_steps=True,
        plot_bland_altman=True,
    )
