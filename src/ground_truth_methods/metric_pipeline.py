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
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from src.approaches.yang_et_al.utils_ground_truth import (
    load_ground_truth_signal,
    get_ground_truth_segment,
    detect_peaks_in_signal,
    mean_hr_from_peak_times,
)


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
    statistic, pvalue = stats.ks_2samp(a, b)
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
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(means, diffs, alpha=0.6)
        ax.axhline(mean_diff, color='C1', linestyle='--', label=f'Mean diff = {mean_diff:.3f}')
        ax.axhline(loa_high, color='C2', linestyle='--', label=f'LoA high = {loa_high:.3f}')
        ax.axhline(loa_low, color='C2', linestyle='--', label=f'LoA low = {loa_low:.3f}')
        ax.set_xlabel('Mean of methods')
        ax.set_ylabel('Difference (pred - gt)')
        ax.set_title(title or 'Bland-Altman Plot')
        ax.legend()
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
    min_distance_signal = int(max(1, np.floor(0.3 * predicted_fs)))
    try:
        signal_peaks, _ = detect_peaks_in_signal(predicted_signal, predicted_fs, min_distance_s=0.3)
        signal_peak_times = signal_time[signal_peaks]
    except Exception:
        signal_peaks = np.array([], dtype=int)
        signal_peak_times = np.array([])

    try:
        gt_peaks, gt_peak_times = detect_peaks_in_signal(gt_segment, gt_fs, min_distance_s=0.5, prominence=1.0)
        gt_peak_times = gt_peak_times + start_time
    except Exception:
        gt_peaks = np.array([], dtype=int)
        gt_peak_times = np.array([])

    gt_hr_mean = mean_hr_from_peak_times(gt_peak_times)
    signal_hr_mean = mean_hr_from_peak_times(signal_peak_times)

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


if __name__ == "__main__":
    print("This module provides evaluation utilities for ground truth comparison.")
# TODO test metric pipelien