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
import numpy.typing as npt

import matplotlib

# Use the non-interactive "Agg" backend: this module only saves figures to disk
# (never displays them), so it must not try to open an X window. Running under
# tmux/SSH the interactive backend connects to the forwarded X display
# (e.g. localhost:10); when that SSH session detaches the X connection breaks with
# "X connection to localhost:10 broken (explicit kill or shutdown)" and kills the
# run. Forcing Agg removes the X dependency entirely. Set before importing pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
from helpers_metric_pipeline import save_single_plot

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
        "std_fraction": 0.8,   #0.8 # higher frame rate -> noisier; drop small noise peaks
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

# A few recordings detect roughly twice as many predicted peaks as there are real
# heartbeats (so their instantaneous HR comes out ~2x too high). For these, an extra
# "halved" variant is additionally evaluated where every second detected predicted peak
# is dropped (keep the 1st, 3rd, 5th, ...) before all metrics are recomputed. Each rule
# is an ``(path_fragment, roi)`` pair; BOTH must appear in the npz path (matched
# case-insensitively, slash-agnostically) for the rule to fire. Currently this targets
# the new_approach mwir manual_rectangle recordings only.
DROP_EVERY_SECOND_PEAK_RULES: tuple[dict[str, str], ...] = (
    {"path_fragment": "new_approach/results/mwir", "roi": "manual_rectangle"},
    {"path_fragment": "new_approach/results/vario", "roi": "manual_rectangle"},
    {"path_fragment": "new_approach/results/lepton", "roi": "manual_rectangle"},
    {"path_fragment": "yang_et_al/results/mwir", "roi": "manual_rectangle"},
    {"path_fragment": "yang_et_al/results/vario", "roi": "manual_rectangle"},
    {"path_fragment": "yang_et_al/results/lepton", "roi": "manual_rectangle"},
    {"path_fragment": "nair_et_al/results/mwir", "roi": "manual_rectangle"},
    {"path_fragment": "nair_et_al/results/vario", "roi": "manual_rectangle"},
    {"path_fragment": "nair_et_al/results/lepton", "roi": "manual_rectangle"},
)

# Two heart-rate estimates "agree" at a moment when their instantaneous HR differs by
# no more than this many bpm. The total time (in seconds) of such agreement across the
# recording is reported as a metric (``hr_agreement_seconds``) and used to shade the
# HR-agreement plot green (within the threshold) vs grey (outside / HR undefined).
HR_AGREEMENT_MAX_BPM_DIFF = 5.0

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

import re
from pathlib import Path


def _extract_metadata_from_metrics_path(metrics_json_path):
    parts = Path(metrics_json_path).parts

    metadata = {
        "approach": "",
        "camera": "",
        "participant": "",
        "run": "",
        "roi": "",
    }

    try:
        results_idx = parts.index("results")

        metadata["approach"] = parts[results_idx - 1]
        metadata["camera"] = parts[results_idx + 1]

        folder_name = Path(metrics_json_path).parent.name

        # Extract participant and run number.
        # Supports names such as:
        # Julian4_interpolated_...
        # Julian_31_03_4_Ascii_thermal_data_...
        # Julian1_thermal_data_...
        participant_match = re.match(
            r"^(?P<participant>[A-Za-z]+)",
            folder_name,
        )

        if participant_match:
            metadata["participant"] = participant_match.group("participant")

        # Extract run number from the folder name.
        # Prefer the number immediately after the participant,
        # or the final run number in the date-based MWIR naming.
        run_match = re.match(
            r"^[A-Za-z]+(?P<run>\d+)(?:_|_interpolated)",
            folder_name,
        )

        if run_match:
            run_number = int(run_match.group("run"))
        else:
            mwir_match = re.match(
                r"^[A-Za-z]+_\d+_\d+_(?P<run>\d+)_",
                folder_name,
            )
            run_number = int(mwir_match.group("run")) if mwir_match else None

        if run_number in (1, 2):
            metadata["run"] = "before sport"
        elif run_number in (3, 4):
            metadata["run"] = "after sport"

        # Extract ROI after the relevant marker.
        roi_match = re.search(
            r"(?:interpolated_|frames_)(.+?)_\d{2}_\d{2}-\d{2}_\d{2}s$",
            folder_name,
        )

        if roi_match:
            metadata["roi"] = roi_match.group(1)

    except (ValueError, IndexError):
        pass

    return metadata


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


def _should_drop_every_second_signal_peak(source_path: str | Path | None) -> bool:
    """Whether to additionally drop every second predicted peak for ``source_path``.

    A few recordings detect roughly twice as many predicted peaks as there are real
    heartbeats, which doubles the instantaneous HR. ``DROP_EVERY_SECOND_PEAK_RULES``
    lists those as ``(path_fragment, roi)`` pairs; both fragments must appear in the npz
    path (case-insensitive, slash-agnostic) for the rule to fire -- notably the
    new_approach mwir manual_rectangle recordings. Returns ``True`` on a match.
    """
    if source_path is None:
        return False
    normalized_path = str(source_path).replace("\\", "/").lower()
    for rule in DROP_EVERY_SECOND_PEAK_RULES:
        fragment = str(rule.get("path_fragment", "")).lower()
        roi_token = str(rule.get("roi", "")).lower()
        if fragment and roi_token and fragment in normalized_path and roi_token in normalized_path:
            return True
    return False


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


def _normalize_individual_filter(
    individuals: tuple[str, ...] | str | None,
) -> set[str] | None:
    """Return a lowercase set of individual name fragments to keep, or ``None`` for all.

    Accepts a single string, an iterable, or ``None``. Each entry is matched as a
    case-insensitive substring against the npz filename (so "Ina" selects every
    ``Ina_*`` / ``ina*`` recording across cameras, while a precise stem like
    "Ina_20_4_3" targets a single recording). Empty / ``None`` entries are dropped;
    if nothing usable remains the result is ``None`` (meaning no filtering).
    """
    if individuals is None:
        return None
    if isinstance(individuals, str):
        individuals = (individuals,)
    filtered = {
        str(individual).strip().lower()
        for individual in individuals
        if individual is not None and str(individual).strip()
    }
    return filtered or None


def _matches_individual_filter(path: Path, individual_filter: set[str] | None) -> bool:
    """Return ``True`` if ``path``'s filename matches the individual filter.

    ``individual_filter`` is a set of lowercase substrings; a file matches when its
    (lowercased) filename contains ANY of them. ``None`` matches every file.
    """
    if individual_filter is None:
        return True
    name = path.name.lower()
    return any(fragment in name for fragment in individual_filter)


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

def compute_distance_based_f1_score(
    pred_peaks: npt.NDArray[np.int64],
    pred_fs: float,
    gt_peaks: npt.NDArray[np.int64],
    gt_fs: float,
    pred_peak_times,
    gt_peak_times,
    limit_pat_window: bool = True,
    lower_pat_limit: float = 50.0,
    higher_pat_limit: float = 350.0
) -> dict:
    """Calculate F1 score using interval-based peak matching.

    For each ground-truth ECG peak, find all predicted peaks that occur
    within the current ECG interval.

    If limit_pat_window is True, only predicted peaks within the
    specified PAT window are eligible for TP matching.

    The first eligible predicted peak is a true positive (TP).
    All additional predicted peaks remain false positives (FP).
    If no eligible predicted peak exists, the ECG peak is a
    false negative (FN).

    PAT is calculated as the time difference between the first predicted
    peak and its corresponding ground-truth ECG peak.

    Returns:
        tp, fp, fn, precision, recall, f1,
        mean_pat, std_pat (in milliseconds),
        pat_values (in milliseconds),
        pred_matched, gt_matched.
    """

    # Validate PAT window limits.
    if limit_pat_window:
        if lower_pat_limit < 0:
            raise ValueError("lower_pat_limit must be >= 0.")

        if higher_pat_limit < lower_pat_limit:
            raise ValueError(
                "higher_pat_limit must be >= lower_pat_limit."
            )

    # Initialize matching masks.
    # True means the peak is matched (TP).
    pred_matched = np.zeros(len(pred_peaks), dtype=bool)
    gt_matched = np.zeros(len(gt_peaks), dtype=bool)

    # Store pulse arrival times (PAT) in milliseconds.
    pat_values = []
    pat_gt_timestamps = []

    # Iterate over every ground-truth ECG peak.
    for i, current_gt_time in enumerate(gt_peak_times):

        # Define the end of the current ECG interval.
        # The interval ends immediately before the next ECG peak.
        if i < len(gt_peak_times) - 1:
            next_gt_time = gt_peak_times[i + 1]
        else:
            # The final interval ends at the end of the predicted signal.
            next_gt_time = len(pred_peak_times) * 0 + (
                len(pred_peaks) / pred_fs
            )

        # Define the valid PAT interval.
        if limit_pat_window:
            lower_time = current_gt_time + lower_pat_limit / 1000.0 # as time windows are in ms, not in seconds
            higher_time = current_gt_time + higher_pat_limit / 1000.0
        else:
            lower_time = current_gt_time
            higher_time = next_gt_time

        # Restrict the PAT interval to the current ECG interval.
        higher_time = min(higher_time, next_gt_time)

        # Find predicted peaks within the valid interval.
        interval_mask = (
            (pred_peak_times >= lower_time) &
            (pred_peak_times < higher_time)
        )

        # Get the indices of all predicted peaks in this interval.
        interval_indices = np.flatnonzero(interval_mask)

        # Check whether at least one predicted peak exists.
        if len(interval_indices) > 0:

            # The first predicted peak is the true positive.
            first_pred_idx = interval_indices[0]

            pred_matched[first_pred_idx] = True
            gt_matched[i] = True

            # Calculate PAT in milliseconds.
            pat_ms = (
                pred_peak_times[first_pred_idx] - current_gt_time
            ) * 1000.0

            pat_values.append(pat_ms)
            pat_gt_timestamps.append(current_gt_time)

            # All remaining predicted peaks in this interval
            # remain unmatched and are therefore false positives.

        # If no predicted peak exists, the GT peak remains unmatched
        # and is therefore a false negative.

    # Calculate TP, FP and FN.
    tp = int(np.count_nonzero(pred_matched))
    fp = int(len(pred_peaks) - tp)
    fn = int(len(gt_peaks) - tp)

    # Calculate precision.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Calculate recall.
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Calculate F1 score.
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    # Calculate mean and standard deviation of PAT.
    mean_pat = float(np.mean(pat_values)) if pat_values else 0.0
    std_pat = float(np.std(pat_values)) if pat_values else 0.0

    return {
        "tp_distance_based": tp,
        "fp_distance_based": fp,
        "fn_distance_based": fn,
        "precision_distance_based": precision,
        "recall_distance_based": recall,
        "f1_distance_based": f1,
        "mean_pat_distance_based": mean_pat,
        "std_pat_distance_based": std_pat,
        "pat_values_distance_based": np.asarray(pat_values),
        "pred_matched_distance_based": pred_matched,
        "gt_matched_distance_based": gt_matched,
        "pat_values_timestamps": pat_gt_timestamps
    }


def plot_distance_based_peak_matching(
    pred_signal: npt.NDArray[np.float64],
    pred_fs: float,
    pred_peaks: npt.NDArray[np.int64],
    pred_matched: npt.NDArray[np.bool_],
    gt_signal: npt.NDArray[np.float64],
    gt_fs: float,
    gt_peaks: npt.NDArray[np.int64],
    gt_matched: npt.NDArray[np.bool_],
    title: str,
    f1_metrics: dict,
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    # Convert sample indices to seconds
    t_pred = np.arange(len(pred_signal)) / pred_fs
    t_gt = np.arange(len(gt_signal)) / gt_fs

    pred_peak_times = pred_peaks / pred_fs
    gt_peak_times = gt_peaks / gt_fs

    fig, axes = plt.subplots(
        2, 1,
        figsize=(15, 8),
        sharex=True,
    )

    # Plot ECG and PPG signals
    axes[0].plot(t_gt, gt_signal, color="darkgrey", label="ECG")
    axes[1].plot(t_pred, pred_signal, color="darkgrey", label="PPG")

    # Plot predicted peaks: TP in green, FP in red
    axes[1].scatter(
        pred_peak_times[pred_matched],
        pred_signal[pred_peaks[pred_matched]],
        color="green",
        label="TP",
        zorder=5,
    )
    axes[1].scatter(
        pred_peak_times[~pred_matched],
        pred_signal[pred_peaks[~pred_matched]],
        color="darkred",
        label="FP",
        zorder=5,
    )

    # Plot ground-truth peaks: matched in green, FN in red
    axes[0].scatter(
        gt_peak_times[gt_matched],
        gt_signal[gt_peaks[gt_matched]],
        color="green",
        label="TP",
        zorder=5,
    )
    axes[0].scatter(
        gt_peak_times[~gt_matched],
        gt_signal[gt_peaks[~gt_matched]],
        color="red",
        label="FN",
        zorder=5,
    )

    # Shade only the time between each GT peak and its TP
    interval_colors = ["lightblue", "moccasin", "thistle"]
    color_idx = 0

    for gt_idx, gt_time in enumerate(gt_peak_times):
        # Skip ground-truth peaks without a true positive
        if not gt_matched[gt_idx]:
            continue

        next_gt_time = (
            gt_peak_times[gt_idx + 1]
            if gt_idx < len(gt_peak_times) - 1
            else min(len(pred_signal) / pred_fs, len(gt_signal) / gt_fs)
        )

        # Find predicted peaks in the current GT interval
        interval_mask = (
            (pred_peak_times >= gt_time)
            & (pred_peak_times < next_gt_time)
        )
        interval_indices = np.flatnonzero(interval_mask)

        if len(interval_indices) == 0:
            continue

        # The earliest predicted peak is the TP
        tp_idx = interval_indices[0]
        tp_time = pred_peak_times[tp_idx]

        color = interval_colors[color_idx % len(interval_colors)]
        color_idx += 1

        # Shade only from the GT peak to the TP
        axes[0].axvspan(
            gt_time, tp_time,
            color=color,
            alpha=0.25,
            zorder=0,
        )
        axes[1].axvspan(
            gt_time, tp_time,
            color=color,
            alpha=0.25,
            zorder=0,
        )

    # Labels and formatting
    axes[0].set_ylabel("ECG amplitude")
    axes[1].set_ylabel("PPG amplitude")
    axes[1].set_xlabel("Time (s)")

    axes[0].set_title(
        f"{title}\n"
        f"Precision: {f1_metrics['precision_distance_based']:.3f} | "
        f"Recall: {f1_metrics['recall_distance_based']:.3f} | "
        f"F1: {f1_metrics['f1_distance_based']:.3f} | "
        f"Mean PAT: {f1_metrics['mean_pat_distance_based']:.2f} ms | "
        f"Std PAT: {f1_metrics['std_pat_distance_based']:.2f} ms"
    )

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

def plot_pat_over_time(
    pat_values,
    title,
    plot_path,
    pat_times,
    gt_ihr,
    gt_ihr_times
):
    """
    Plot PAT and ground-truth instantaneous heart rate (IHR) over time.

    Parameters
    ----------
    pat_values : array-like
        PAT values in milliseconds.
    title : str
        Plot title.
    plot_path : str
        Path where the figure will be saved.
    pat_times : array-like
        Timestamps corresponding to PAT values (seconds).
    gt_ihr : array-like
        Ground-truth instantaneous heart rate values (BPM).
    gt_ihr_times : array-like
        Timestamps corresponding to GT IHR values (seconds).
    """

    # Convert inputs to numpy arrays
    pat_values = np.asarray(pat_values)
    pat_times = np.asarray(pat_times)
    gt_ihr = np.asarray(gt_ihr)
    gt_ihr_times = np.asarray(gt_ihr_times)

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot PAT on the left y-axis
    ax1.plot(
        pat_times,
        pat_values,
            color="tab:red",
            marker = 'o',
        linestyle="-",
        label="PAT",
        
    )

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("PAT (ms)")
    ax1.grid(True, alpha=0.3)

    # Create secondary y-axis for instantaneous HR
    ax2 = ax1.twinx()

    ax2.plot(
        gt_ihr_times,
        gt_ihr,
    color="tab:blue",
                marker = 'o',
        linestyle="-",
        label="Ground Truth IHR",
    )

    ax2.set_ylabel("Ground Truth IHR (BPM)")

    # Add title
    ax1.set_title("PAT length over time, compared to ECG heart rate for: " + str(title))

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best"
    )

    # Improve layout
    fig.tight_layout()

    # Save figure
    if plot_path is not None:
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, ax1, ax2
    
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


def compute_hr_agreement_over_time(
    gt_ihr_times: np.ndarray,
    gt_ihr: np.ndarray,
    pred_ihr_times: np.ndarray,
    pred_ihr: np.ndarray,
    time_grid: np.ndarray,
    max_bpm_diff: float = HR_AGREEMENT_MAX_BPM_DIFF,
) -> Dict[str, object]:
    """Find where predicted & ground-truth instantaneous HR agree within ``max_bpm_diff``.

    Both instantaneous-HR series are linearly interpolated onto ``time_grid`` (the dense
    signal time axis). Only grid samples that lie inside BOTH series' time spans are
    "comparable" -- outside those spans the instantaneous HR is undefined. A grid sample
    is "in agreement" when it is comparable AND the absolute HR difference there is
    ``<= max_bpm_diff`` bpm. The total agreement duration is simply the number of
    in-agreement grid samples times the grid spacing, i.e. how many seconds of the
    signal track the ground-truth HR to within ``max_bpm_diff`` bpm.

    Returns a dict with:
      - ``abs_diff``: |pred - gt| on the grid (NaN where not comparable),
      - ``comparable_mask`` / ``agreement_mask``: bool arrays over ``time_grid``,
      - ``agreement_seconds`` / ``comparable_seconds``: durations in seconds,
      - ``agreement_fraction``: agreement_seconds / comparable_seconds (NaN if none),
      - ``max_bpm_diff``: the threshold used.
    """
    time_grid = np.asarray(time_grid, dtype=float)
    gt_ihr_times = np.asarray(gt_ihr_times, dtype=float)
    gt_ihr = np.asarray(gt_ihr, dtype=float)
    pred_ihr_times = np.asarray(pred_ihr_times, dtype=float)
    pred_ihr = np.asarray(pred_ihr, dtype=float)

    n = int(time_grid.size)
    abs_diff = np.full(n, np.nan, dtype=float)
    comparable_mask = np.zeros(n, dtype=bool)
    agreement_mask = np.zeros(n, dtype=bool)

    dt = float(np.median(np.diff(time_grid))) if time_grid.size >= 2 else 0.0

    if gt_ihr_times.size >= 2 and pred_ihr_times.size >= 2 and dt > 0:
        # Overlap of the two series' time spans = where both HRs are defined.
        lo = max(float(gt_ihr_times.min()), float(pred_ihr_times.min()))
        hi = min(float(gt_ihr_times.max()), float(pred_ihr_times.max()))
        if hi > lo:
            comparable_mask = (time_grid >= lo) & (time_grid <= hi)
            if np.any(comparable_mask):
                gt_on_grid = np.interp(time_grid, gt_ihr_times, gt_ihr)
                pred_on_grid = np.interp(time_grid, pred_ihr_times, pred_ihr)
                diff = np.abs(pred_on_grid - gt_on_grid)
                abs_diff = np.where(comparable_mask, diff, np.nan)
                agreement_mask = comparable_mask & (abs_diff <= float(max_bpm_diff))

    agreement_seconds = float(np.count_nonzero(agreement_mask) * dt)
    comparable_seconds = float(np.count_nonzero(comparable_mask) * dt)
    agreement_fraction = (
        agreement_seconds / comparable_seconds if comparable_seconds > 0 else float("nan")
    )
    return {
        "abs_diff": abs_diff,
        "comparable_mask": comparable_mask,
        "agreement_mask": agreement_mask,
        "agreement_seconds": agreement_seconds,
        "comparable_seconds": comparable_seconds,
        "agreement_fraction": agreement_fraction,
        "max_bpm_diff": float(max_bpm_diff),
    }


def _mask_to_spans(
    time_grid: np.ndarray, mask: np.ndarray
) -> list[tuple[float, float]]:
    """Convert a boolean ``mask`` over ``time_grid`` into ``(start, end)`` time spans.

    Each maximal run of ``True`` samples becomes one ``(start_time, end_time)`` pair; the
    end is extended by one grid step so adjacent shaded spans tile the axis without gaps.
    """
    time_grid = np.asarray(time_grid, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if time_grid.size == 0 or mask.size == 0 or not mask.any():
        return []
    dt = float(np.median(np.diff(time_grid))) if time_grid.size >= 2 else 0.0
    spans: list[tuple[float, float]] = []
    in_run = False
    start_idx = 0
    for i, flag in enumerate(mask):
        if flag and not in_run:
            in_run = True
            start_idx = i
        elif not flag and in_run:
            in_run = False
            spans.append((float(time_grid[start_idx]), float(time_grid[i - 1] + dt)))
    if in_run:
        spans.append((float(time_grid[start_idx]), float(time_grid[-1] + dt)))
    return spans


def _shade_agreement_background(
    ax: plt.Axes,
    time_grid: np.ndarray,
    agreement_mask: np.ndarray,
    *,
    agreement_color: str = "#b6e8b6",
    background_color: str = "#e0e0e0",
) -> None:
    """Shade ``ax`` grey everywhere and light green where ``agreement_mask`` is True.

    ``time_grid`` and ``agreement_mask`` must be aligned. The green spans are drawn
    behind the data (``zorder=0``). Used by the HR-agreement plot so the signal, ECG
    and HR subplots share one green (<= threshold) / grey (else) backdrop.
    """
    ax.set_facecolor(background_color)
    for start, end in _mask_to_spans(time_grid, agreement_mask):
        ax.axvspan(start, end, facecolor=agreement_color, edgecolor="none", zorder=0)


def _save_hr_agreement_plot(
    output_path: str | Path,
    *,
    time_grid: np.ndarray,
    hr_agreement: Dict[str, object],
    pred_signal: np.ndarray,
    pred_peaks: np.ndarray,
    pred_peak_times: np.ndarray,
    pred_ihr_times: np.ndarray,
    pred_ihr: np.ndarray,
    gt_segment: np.ndarray,
    gt_peaks: np.ndarray,
    gt_peak_times: np.ndarray,
    gt_ihr_times: np.ndarray,
    gt_ihr: np.ndarray,
    max_bpm_diff: float = HR_AGREEMENT_MAX_BPM_DIFF,
    pred_color: str = "C1",
    pred_signal_label: str = "Predicted signal",
    pred_peaks_label: str = "Predicted peaks",
    pred_ihr_label: str = "Predicted HR",
    panel1_title: str = "Predicted signal",
    suptitle_note: str = "",
) -> str:
    """Save the 3-panel HR-agreement figure (predicted signal, GT ECG, compared IHR).

    The three panels share one time axis and a light-green (|Pred - GT| instantaneous
    HR <= ``max_bpm_diff`` bpm) / grey (else / HR undefined) background driven by
    ``hr_agreement['agreement_mask']`` (see compute_hr_agreement_over_time /
    _shade_agreement_background). ``suptitle_note`` is appended to the figure title and
    the predicted curves use ``pred_color`` / the given labels, so the same routine
    renders both the primary and the every-2nd-peak-dropped ("halved") variants.
    """
    time_grid = np.asarray(time_grid, dtype=float)
    pred_signal = np.asarray(pred_signal, dtype=float)
    gt_segment = np.asarray(gt_segment, dtype=float)
    pred_peaks = np.asarray(pred_peaks, dtype=int)
    gt_peaks = np.asarray(gt_peaks, dtype=int)
    agreement_mask = np.asarray(hr_agreement["agreement_mask"], dtype=bool)
    thr = float(max_bpm_diff)
    agreement_seconds = float(hr_agreement["agreement_seconds"])
    comparable_seconds = float(hr_agreement["comparable_seconds"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
    #fig.suptitle(
    #    f"HR agreement within +/-{thr:.0f} bpm{suptitle_note} "
    #    f"(green = agreement; total {agreement_seconds:.1f} s "
    #    f"of {comparable_seconds:.1f} comparable s)",
    #    fontsize=15,
    #)

    # Background-highlight legend patches – defined once, reused in all three panels.
    bg_handles = [
        Patch(facecolor="#b6e8b6", edgecolor="none", label=f"<= {thr:.0f} bpm difference between Pred & GT"),
        Patch(facecolor="#e0e0e0", edgecolor="none", label=f"> {thr:.0f} bpm difference between Pred & GT"),
    ]
    _spacer = Patch(facecolor="none", edgecolor="none", label="")

    def _add_bg_legend(ax: "plt.Axes") -> None:
        """Append bg_handles to ax's legend, always on the second line (ncol=2).

        Line handles fill row 1; if there is an odd number the row is padded with an
        invisible spacer so the two coloring patches always start together on row 2.
        """
        lh, ll = ax.get_legend_handles_labels()
        if len(lh) % 2 == 1:
            lh = lh + [_spacer]
            ll = ll + [""]
        ax.legend(
            lh + bg_handles,
            ll + [h.get_label() for h in bg_handles],
            loc="upper right",
            ncol=2,
        )

    # 1) Predicted signal with detected peaks
    _shade_agreement_background(axes[0], time_grid, agreement_mask)
    axes[0].grid(True, which='major', color='0.85', linewidth=0.8)
    axes[0].plot(time_grid, pred_signal, color=pred_color, alpha=0.85, label=pred_signal_label)
    if pred_peaks.size > 0:
        axes[0].scatter(pred_peak_times, pred_signal[pred_peaks], s=22, color=pred_color, edgecolors="black", linewidths=0.4)#, label=pred_peaks_label)
    axes[0].set_ylabel("Amplitude (arbitrary units)")
    _add_bg_legend(axes[0])

    # 2) Ground-truth ECG with detected peaks
    _shade_agreement_background(axes[1], time_grid, agreement_mask)
    axes[1].grid(True, which='major', color='0.85', linewidth=0.8)
    axes[1].plot(time_grid, gt_segment, color="C0", alpha=0.85, label="Ground truth (ECG)")
    if gt_peaks.size > 0:
        axes[1].scatter(gt_peak_times, gt_segment[gt_peaks], s=22, color="C0", edgecolors="black", linewidths=0.4)#, label="GT peaks")
    axes[1].set_ylabel("Amplitude (arbitrary units)")
    _add_bg_legend(axes[1])

    # 3) Compared instantaneous heart rates
    _shade_agreement_background(axes[2], time_grid, agreement_mask)
    axes[2].grid(True, which='major', color='0.85', linewidth=0.8)
    axes[2].plot(gt_ihr_times, gt_ihr, color="C0", alpha=0.8, marker="o", label="Ground truth instanteous HR")
    axes[2].plot(pred_ihr_times, pred_ihr, color=pred_color, alpha=0.8, marker="o", label=pred_ihr_label)
    #axes[2].set_title(f"Instantaneous HR (green where |Pred - GT| <= {thr:.0f} bpm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("HR (bpm)")
    _add_bg_legend(axes[2])
    axes[2].grid(alpha=0.2)

    plt.tight_layout()
    output_path = Path(output_path)
    print(f"[metric_pipeline] Saving HR-agreement plot to: {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Individual panel saves – each is a standalone figure that includes
    # the same background shading and bg_handles legend entries.
    # ------------------------------------------------------------------
    def _draw_signal(ax_i: "plt.Axes") -> None:
        ax_i.plot(time_grid, pred_signal, color=pred_color, alpha=0.85, label=pred_signal_label)
        if pred_peaks.size > 0:
            ax_i.scatter(pred_peak_times, pred_signal[pred_peaks], s=22, color=pred_color,
                         edgecolors="black", linewidths=0.4)

    def _draw_gt(ax_i: "plt.Axes") -> None:
        ax_i.plot(time_grid, gt_segment, color="C0", alpha=0.85, label="Ground truth (ECG)")
        if gt_peaks.size > 0:
            ax_i.scatter(gt_peak_times, gt_segment[gt_peaks], s=22, color="C0",
                         edgecolors="black", linewidths=0.4)

    def _draw_ihr(ax_i: "plt.Axes") -> None:
        ax_i.plot(gt_ihr_times, gt_ihr, color="C0", alpha=0.8, marker="o", label="Ground truth HR")
        ax_i.plot(pred_ihr_times, pred_ihr, color=pred_color, alpha=0.8, marker="o", label=pred_ihr_label)
        ax_i.set_xlabel("Time (s)")
        ax_i.grid(alpha=0.2)

    for suffix, draw_fn, ylabel in [
        ("signal", _draw_signal, "Amplitude (arbitrary units)"),
        ("gt",     _draw_gt,     "Amplitude (arbitrary units)"),
        ("ihr",    _draw_ihr,    "HR (bpm)"),
    ]:
        fig_i, ax_i = plt.subplots(1, 1, figsize=(14, 5))
        _shade_agreement_background(ax_i, time_grid, agreement_mask)
        ax_i.grid(True, which="major", color="0.85", linewidth=0.8)
        draw_fn(ax_i)
        ax_i.set_ylabel(ylabel)
        _add_bg_legend(ax_i)
        ind_path = output_path.parent / f"{output_path.stem}_{suffix}{output_path.suffix}"
        plt.tight_layout()
        print(f"[metric_pipeline] Saving individual panel to: {ind_path}")
        plt.savefig(ind_path, dpi=150, bbox_inches="tight")
        plt.close(fig_i)

    return str(output_path)


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


def _match_and_score_ihr(
    gt_ihr: np.ndarray,
    gt_ihr_times: np.ndarray,
    pred_ihr: np.ndarray,
    pred_ihr_times: np.ndarray,
    tolerance_ms: float,
) -> Dict[str, object]:
    """Match predicted vs ground-truth instantaneous HR and score both strategies.

    Runs the same two matchings used by the main evaluation and scores each with
    regression + KS + Bland-Altman metrics:
      * tolerance / "disregard": nearest match within ``tolerance_ms`` (unmatched points
        dropped);
      * "closest": every GT point matched to its nearest prediction and vice versa
        (nothing dropped).
    Returns a dict with the matched arrays, the combined closest arrays, the raw
    closest-match mapping and the metric/Bland-Altman dicts, so callers can both report
    and plot the result. Used for the drop-every-second-peak ("halved") variant.
    """
    matched_gt_ihr, matched_pred_ihr, matched_gt_times, matched_pred_times = (
        match_by_timestamp_tolerance(
            gt_values=gt_ihr,
            gt_times=gt_ihr_times,
            pred_values=pred_ihr,
            pred_times=pred_ihr_times,
            tolerance_ms=tolerance_ms,
        )
    )
    disregard_metrics, ba_disregard = compute_metrics_with_optional_bland_altman(
        matched_gt_ihr, matched_pred_ihr, plot_bland_altman=False
    )
    disregard_metrics["num_matched"] = int(matched_gt_ihr.size)

    closest_matches = match_by_nearest_timestamp(
        gt_values=gt_ihr,
        gt_times=gt_ihr_times,
        pred_values=pred_ihr,
        pred_times=pred_ihr_times,
    )
    closest_gt_ihr = np.concatenate(
        [closest_matches["gt_to_pred_gt_values"], closest_matches["pred_to_gt_gt_values"]]
    )
    closest_pred_ihr = np.concatenate(
        [closest_matches["gt_to_pred_pred_values"], closest_matches["pred_to_gt_pred_values"]]
    )
    closest_gt_times = np.concatenate(
        [closest_matches["gt_to_pred_gt_times"], closest_matches["pred_to_gt_gt_times"]]
    )
    closest_pred_times = np.concatenate(
        [closest_matches["gt_to_pred_pred_times"], closest_matches["pred_to_gt_pred_times"]]
    )
    closest_metrics, ba_closest = compute_metrics_with_optional_bland_altman(
        closest_gt_ihr, closest_pred_ihr, plot_bland_altman=False
    )
    return {
        "matched_gt_ihr": matched_gt_ihr,
        "matched_pred_ihr": matched_pred_ihr,
        "matched_gt_times": matched_gt_times,
        "matched_pred_times": matched_pred_times,
        "disregard_metrics": disregard_metrics,
        "ba_disregard": ba_disregard,
        "closest_matches": closest_matches,
        "closest_gt_ihr": closest_gt_ihr,
        "closest_pred_ihr": closest_pred_ihr,
        "closest_gt_times": closest_gt_times,
        "closest_pred_times": closest_pred_times,
        "closest_metrics": closest_metrics,
        "ba_closest": ba_closest,
    }


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
    "approach",
    "camera",
    "participant",
    "run",
    "roi",
        #f1 
    "tp_distance_based",
    "fp_distance_based",
    "fn_distance_based",
    "precision_distance_based",
    "recall_distance_based",
    "f1_distance_based",
    "mean_pat_distance_based",
    "std_pat_distance_based",
    #paths
    "timestamp_utc",
    "input_file",
    "plots_folder",
    "metrics_json_path",
    "tolerance_ms",
    "ground_truth_fs",
    "predicted_fs",
    "gt_hr_mean",
    "signal_hr_mean",
    "signal_hr_mean_diff_to_gt_mean",
    "signal_hr_mean_abs_diff_to_gt_mean",
    "original_pipeline_bpm_estimate",
    "original_pipeline_bpm_estimate_diff_to_gt_mean",
    "original_pipeline_bpm_estimate_abs_diff_to_gt_mean",
    "num_gt_peaks",
    "num_pred_peaks",
    # --- per-moment HR agreement: total seconds where |pred - gt| instantaneous HR is
    #     within hr_agreement_max_bpm_diff bpm (and the comparable overlap / fraction). ---
    "hr_agreement_max_bpm_diff",
    "hr_agreement_seconds",
    "hr_comparable_seconds",
    "hr_agreement_fraction",
    # --- closest matching: all 'closest' columns are grouped before any 'disregard' ---
    "num_matched_closest",
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
    # ba_stats_closest.*
    "ba_stats_closest_mean_diff",
    "ba_stats_closest_loa_low",
    "ba_stats_closest_loa_high",
    "ba_stats_closest_sd_diff",
    # --- disregard / tolerance matching: all 'disregard' columns come after ---
    "num_matched_disregard",
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
    # ba_stats_disregard.*
    "ba_stats_disregard_mean_diff",
    "ba_stats_disregard_loa_low",
    "ba_stats_disregard_loa_high",
    "ba_stats_disregard_sd_diff",
    # --- halved variant: every second predicted peak dropped (only populated for the
    #     new_approach mwir manual_rectangle recordings; blank elsewhere). Same layout
    #     as above, all 'closest' columns before any 'disregard' column. ---
    "signal_hr_mean_halved",
    "signal_hr_mean_halved_diff_to_gt_mean",
    "signal_hr_mean_halved_abs_diff_to_gt_mean",
    "num_pred_peaks_halved",
    "hr_agreement_seconds_halved",
    "hr_comparable_seconds_halved",
    "hr_agreement_fraction_halved",
    "num_matched_closest_halved",
    "metrics_closest_halved_rmse",
    "metrics_closest_halved_mae",
    "metrics_closest_halved_r2",
    "metrics_closest_halved_ks_pvalue",
    "metrics_closest_halved_mean_diff",
    "metrics_closest_halved_loa_low",
    "metrics_closest_halved_loa_high",
    "metrics_closest_halved_sd_diff",
    "metrics_closest_halved_num_pairs",
    "ba_stats_closest_halved_mean_diff",
    "ba_stats_closest_halved_loa_low",
    "ba_stats_closest_halved_loa_high",
    "ba_stats_closest_halved_sd_diff",
    "num_matched_disregard_halved",
    "metrics_disregard_halved_rmse",
    "metrics_disregard_halved_mae",
    "metrics_disregard_halved_r2",
    "metrics_disregard_halved_ks_pvalue",
    "metrics_disregard_halved_mean_diff",
    "metrics_disregard_halved_loa_low",
    "metrics_disregard_halved_loa_high",
    "metrics_disregard_halved_sd_diff",
    "metrics_disregard_halved_num_pairs",
    "metrics_disregard_halved_num_matched",
    "ba_stats_disregard_halved_mean_diff",
    "ba_stats_disregard_halved_loa_low",
    "ba_stats_disregard_halved_loa_high",
    "ba_stats_disregard_halved_sd_diff",
]


def _backfill_derived_summary_columns(rows: list[dict]) -> int:
    """Fill blank derived columns of already-migrated summary rows in place.

    A few summary columns are pure functions of other columns that older CSVs already
    contain, so when an old file is migrated to the current schema they can be computed
    rather than left blank:

    * ``signal_hr_mean_diff_to_gt_mean``      = ``signal_hr_mean`` - ``gt_hr_mean``
    * ``signal_hr_mean_abs_diff_to_gt_mean``  = ``abs(signal_hr_mean_diff_to_gt_mean)``
    * ``original_pipeline_bpm_estimate_abs_diff_to_gt_mean`` =
      ``abs(original_pipeline_bpm_estimate_diff_to_gt_mean)``

    Only cells that are currently blank/missing are filled, so genuine values written by
    the pipeline are never overwritten. Returns the number of rows that were touched.
    """

    def _num(value: object) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if np.isfinite(parsed) else None

    touched = 0
    for row in rows:
        changed = False
        gt = _num(row.get("gt_hr_mean"))
        sig = _num(row.get("signal_hr_mean"))
        if gt is not None and sig is not None:
            diff = sig - gt
            if not row.get("signal_hr_mean_diff_to_gt_mean"):
                row["signal_hr_mean_diff_to_gt_mean"] = diff
                changed = True
            if not row.get("signal_hr_mean_abs_diff_to_gt_mean"):
                row["signal_hr_mean_abs_diff_to_gt_mean"] = abs(diff)
                changed = True
        orig_diff = _num(row.get("original_pipeline_bpm_estimate_diff_to_gt_mean"))
        if orig_diff is not None and not row.get(
            "original_pipeline_bpm_estimate_abs_diff_to_gt_mean"
        ):
            row["original_pipeline_bpm_estimate_abs_diff_to_gt_mean"] = abs(orig_diff)
            changed = True
        if changed:
            touched += 1
    return touched


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
        backfilled = _backfill_derived_summary_columns(migrated)
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=canonical, extrasaction="ignore", restval=""
            )
            writer.writeheader()
            writer.writerows(migrated)
        if backfilled:
            note = f"{note}; backfilled derived columns on {backfilled} row(s)"
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

    data = output_data if isinstance(output_data, dict) else {}

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": data.get("input_file", ""),
        "plots_folder": str(plots_folder) if plots_folder is not None else "",
        "metrics_json_path": str(json_path) if json_path is not None else "",
    }

    metadata = _extract_metadata_from_metrics_path(json_path)

    row.update(metadata)

    # Automatically flatten nested metric dictionaries
    for key, value in data.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                row[f"{key}_{subkey}"] = subvalue
        else:
            row[key] = value

    return {
        field: row.get(field, "")
        for field in METRICS_SUMMARY_CSV_FIELDNAMES
    }


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



def evaluate_prediction_against_ground_truth(
    results: str | Path,
    flip_ground_truth: bool = False,
    tolerance_ms: float = 0.0,
    plot_steps: bool = True,
    plot_bland_altman: bool = True,
    csv_output_path: str | Path | None = None,
    gt_peak_method: str = DEFAULT_GT_PEAK_METHOD,
    output_root = None
) -> Dict[str, object]:
    """Evaluate prediction using a single `.npy`/`.npz` file as input.

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
    if output_root is None:
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

    # F1 score of distances
    distance_based_f1_metrics = compute_distance_based_f1_score(pred_peaks=pred_peaks, pred_fs=predicted_fs_val,
                                    gt_peaks=gt_peaks, gt_fs=gt_fs_val, pred_peak_times= pred_peak_times, gt_peak_times= gt_peak_times)

    pat_over_time_plot_path = output_folder / "b_pat_over_time.png"
    peak_matching_plot_path = output_folder / "a_distance_based_peak_matching.png"
    

    plot_distance_based_peak_matching(
        pred_signal=pred_rs,
        pred_fs=predicted_fs_val,
        pred_peaks=pred_peaks,
        pred_matched=distance_based_f1_metrics["pred_matched_distance_based"],
        gt_signal=gt_segment,
        gt_fs=predicted_fs_val,
        gt_peaks=gt_peaks,
        gt_matched=distance_based_f1_metrics["gt_matched_distance_based"],
        title="Distance-based peak matching",
        f1_metrics=distance_based_f1_metrics,
        save_path=peak_matching_plot_path
    )

    # Instantaneous HR
    pred_ihr, pred_ihr_times = instantaneous_hr_from_peak_times(pred_peak_times)
    gt_ihr, gt_ihr_times = instantaneous_hr_from_peak_times(gt_peak_times)

    plot_pat_over_time(pat_values= distance_based_f1_metrics["pat_values_distance_based"], 
                           title = results, 
                           plot_path = pat_over_time_plot_path, 
                           pat_times = distance_based_f1_metrics["pat_values_timestamps"],
                           gt_ihr = gt_ihr,
                           gt_ihr_times = gt_ihr_times)

    # Mean HR-----------------------
    gt_hr_mean = mean_hr_from_peak_times(gt_peak_times)
    signal_hr_mean = mean_hr_from_peak_times(pred_peak_times)

    # Difference of the peak-based signal mean HR to the ground-truth mean HR
    # (signal_hr_mean - gt_hr_mean, the same pred-minus-gt convention as Bland-Altman;
    # positive = the signal over-estimates) plus its absolute value. Both are None when
    # either mean is missing / non-finite.
    signal_hr_mean_diff_to_gt_mean = None
    signal_hr_mean_abs_diff_to_gt_mean = None
    if _is_finite_scalar(signal_hr_mean) and _is_finite_scalar(gt_hr_mean):
        signal_hr_mean_diff_to_gt_mean = float(signal_hr_mean) - float(gt_hr_mean)
        signal_hr_mean_abs_diff_to_gt_mean = abs(signal_hr_mean_diff_to_gt_mean)

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
    original_pipeline_bpm_estimate_abs_diff_to_gt_mean = None
    if original_pipeline_bpm_estimate is not None and _is_finite_scalar(gt_hr_mean):
        original_pipeline_bpm_estimate_diff_to_gt_mean = (
            original_pipeline_bpm_estimate - float(gt_hr_mean)
        )
        original_pipeline_bpm_estimate_abs_diff_to_gt_mean = abs(
            original_pipeline_bpm_estimate_diff_to_gt_mean
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

    # Per-moment HR agreement: total time (s) where the predicted and ground-truth
    # instantaneous HRs differ by <= HR_AGREEMENT_MAX_BPM_DIFF bpm. Both IHR series are
    # interpolated onto the signal time axis ``t``; only the overlap where both are
    # defined is comparable. Drives the 06_hr_agreement plot's green/grey background.
    hr_agreement = compute_hr_agreement_over_time(
        gt_ihr_times=gt_ihr_times,
        gt_ihr=gt_ihr,
        pred_ihr_times=pred_ihr_times,
        pred_ihr=pred_ihr,
        time_grid=t,
        max_bpm_diff=HR_AGREEMENT_MAX_BPM_DIFF,
    )
    print(
        f"[metric_pipeline] HR agreement within +/-{HR_AGREEMENT_MAX_BPM_DIFF:.0f} bpm: "
        f"{hr_agreement['agreement_seconds']:.2f} s of "
        f"{hr_agreement['comparable_seconds']:.2f} comparable s "
        f"(fraction={hr_agreement['agreement_fraction']})"
    )

    # Optional "halved" variant: a few recordings detect ~2x as many predicted peaks as
    # there are real heartbeats (see DROP_EVERY_SECOND_PEAK_RULES -- notably the
    # new_approach mwir manual_rectangle recordings), which roughly doubles the
    # instantaneous HR. For those, additionally drop every second detected predicted
    # peak (keep the 1st, 3rd, 5th, ...) and recompute every metric, so the doubled-peak
    # result can be compared against the halved one. All halved values stay None/empty
    # for every other recording.
    drop_every_second_peak = _should_drop_every_second_signal_peak(signal_path)
    pred_peaks_halved = np.array([], dtype=int)
    pred_peak_times_halved = np.array([], dtype=float)
    pred_ihr_halved = np.array([], dtype=float)
    pred_ihr_times_halved = np.array([], dtype=float)
    signal_hr_mean_halved = None
    signal_hr_mean_halved_diff_to_gt_mean = None
    signal_hr_mean_halved_abs_diff_to_gt_mean = None
    halved: Dict[str, object] | None = None
    hr_agreement_halved: Dict[str, object] | None = None
    if drop_every_second_peak:
        pred_peaks_halved = np.asarray(pred_peaks, dtype=int)[::2]
        pred_peak_times_halved = (
            t[pred_peaks_halved] if pred_peaks_halved.size else np.array([], dtype=float)
        )
        pred_ihr_halved, pred_ihr_times_halved = instantaneous_hr_from_peak_times(
            pred_peak_times_halved
        )
        signal_hr_mean_halved = mean_hr_from_peak_times(pred_peak_times_halved)
        if _is_finite_scalar(signal_hr_mean_halved) and _is_finite_scalar(gt_hr_mean):
            signal_hr_mean_halved_diff_to_gt_mean = (
                float(signal_hr_mean_halved) - float(gt_hr_mean)
            )
            signal_hr_mean_halved_abs_diff_to_gt_mean = abs(
                signal_hr_mean_halved_diff_to_gt_mean
            )
        halved = _match_and_score_ihr(
            gt_ihr=gt_ihr,
            gt_ihr_times=gt_ihr_times,
            pred_ihr=pred_ihr_halved,
            pred_ihr_times=pred_ihr_times_halved,
            tolerance_ms=tolerance_ms,
        )
        hr_agreement_halved = compute_hr_agreement_over_time(
            gt_ihr_times=gt_ihr_times,
            gt_ihr=gt_ihr,
            pred_ihr_times=pred_ihr_times_halved,
            pred_ihr=pred_ihr_halved,
            time_grid=t,
            max_bpm_diff=HR_AGREEMENT_MAX_BPM_DIFF,
        )
        print(
            f"[metric_pipeline] Drop-every-second-peak variant active: predicted peaks "
            f"{pred_peaks.size} -> {pred_peaks_halved.size}; signal_hr_mean "
            f"{signal_hr_mean} -> {signal_hr_mean_halved}"
        )
        print(
            f"[metric_pipeline] HR agreement (halved) within +/-{HR_AGREEMENT_MAX_BPM_DIFF:.0f} bpm: "
            f"{hr_agreement_halved['agreement_seconds']:.2f} s of "
            f"{hr_agreement_halved['comparable_seconds']:.2f} comparable s "
            f"(fraction={hr_agreement_halved['agreement_fraction']})"
        )

    # Create output folder named after the input NPZ file inside a `metric_results` root
    output_root = signal_path.parent / "metric_results"
    output_folder = output_root / signal_path.stem
    os.makedirs(output_folder, exist_ok=True)

    print(f"[metric_pipeline] Created output folder: {output_folder}")

    # Inform about plots to be generated
    if not plot_steps and not plot_bland_altman:
        print("[metric_pipeline] No plots requested (both plot_steps and plot_bland_altman are False). Skipping plot generation.")

    # Optional plotting: one combined figure with the requested subplots. The halved-
    # peak variant (when active) adds two extra panels at the bottom.
    if plot_steps:
        n_panels = 6 if (drop_every_second_peak and halved is not None) else 4
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 5 * n_panels))
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
        axes[3].plot(gt_ihr_times, gt_ihr, label="Ground truth HR", color="C0", alpha=0.45, marker="o")
        axes[3].plot(pred_ihr_times, pred_ihr, label="Predicted HR", color="C1", alpha=0.45, marker="o")
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

        #axes[3].set_title("IHR with closest matching in both directions")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylabel("HR (bpm)")
        axes[3].legend(loc="best", ncol=2)
        axes[3].grid(alpha=0.2)

        # 5) + 6) Halved-peak variant (only for the drop-every-second-peak recordings):
        # show which predicted peaks survive and the resulting closest-matched IHR.
        if drop_every_second_peak and halved is not None:
            axes[4].plot(t, pred_rs, label="Predicted (resampled)", alpha=0.5, color="C1")
            if pred_peaks.size > 0:
                axes[4].scatter(
                    pred_peak_times,
                    pred_rs[pred_peaks],
                    s=25,
                    color="C1",
                    alpha=0.35,
                    label="All detected peaks",
                )
            if pred_peaks_halved.size > 0:
                axes[4].scatter(
                    pred_peak_times_halved,
                    pred_rs[pred_peaks_halved],
                    s=55,
                    color="C3",
                    edgecolors="black",
                    linewidths=0.4,
                    label="Kept peaks (every 2nd dropped)",
                )
            axes[4].set_title("Predicted signal: every second detected peak dropped")
            axes[4].set_xlabel("Time (s)")
            axes[4].set_ylabel("Amplitude")
            axes[4].legend(loc="best")
            axes[4].grid(alpha=0.2)

            axes[5].plot(gt_ihr_times, gt_ihr, label="Ground truth HR", color="C0", alpha=0.5, marker="o")
            axes[5].plot(
                pred_ihr_times_halved,
                pred_ihr_halved,
                label="Predicted HR (every 2nd peak dropped)",
                color="C3",
                alpha=0.75,
                marker="o",
            )
            halved_closest = halved["closest_matches"]
            if halved_closest["gt_to_pred_gt_times"].size > 0:
                axes[5].scatter(
                    halved_closest["gt_to_pred_gt_times"],
                    halved_closest["gt_to_pred_gt_values"],
                    s=40,
                    color="C0",
                    edgecolors="black",
                    linewidths=0.4,
                    label="GT matched to closest Pred",
                )
                axes[5].scatter(
                    halved_closest["gt_to_pred_pred_times"],
                    halved_closest["gt_to_pred_pred_values"],
                    s=40,
                    color="C3",
                    edgecolors="black",
                    linewidths=0.4,
                    label="Pred matched from GT",
                )
                for x_gt, y_gt, x_pred, y_pred in zip(
                    halved_closest["gt_to_pred_gt_times"],
                    halved_closest["gt_to_pred_gt_values"],
                    halved_closest["gt_to_pred_pred_times"],
                    halved_closest["gt_to_pred_pred_values"],
                ):
                    axes[5].plot([x_gt, x_pred], [y_gt, y_pred], color="gray", alpha=0.3, linewidth=0.0)
            #axes[5].set_title("IHR (closest matching) with every second predicted peak dropped")
            axes[5].set_xlabel("Time (s)")
            axes[5].set_ylabel("HR (bpm)")
            axes[5].legend(loc="best", ncol=2)
            #axes[5].grid(alpha=0.2)

        plt.tight_layout()
        combined_plot_path = output_folder / "01_combined_evaluation.png"
        print(f"[metric_pipeline] Saving combined plot to: {combined_plot_path}")
        plt.savefig(combined_plot_path, dpi=150, bbox_inches="tight")

        save_single_plot(axes[0], output_folder / "01_1_gt_signal_peaks.png")
        save_single_plot(axes[1], output_folder / "01_2_pred_signal_peaks.png")
        save_single_plot(axes[2], output_folder / "01_3_ihr_tolerance_matching.png")
        save_single_plot(axes[3], output_folder / "01_4_ihr_closest_matching.png")

        if drop_every_second_peak and halved is not None:
            save_single_plot(axes[4], output_folder / "01_5_halved_peaks_signal.png")
            save_single_plot(axes[5], output_folder / "01_6_halved_peaks_ihr.png")

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

    # Halved-peak variant Bland-Altman plots: only generated for the
    # drop-every-second-peak recordings (new_approach mwir manual_rectangle).
    if (plot_bland_altman or plot_steps) and drop_every_second_peak and halved is not None:
        if halved["matched_gt_ihr"].size > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            bland_altman(
                halved["matched_gt_ihr"],
                halved["matched_pred_ihr"],
                plot=True,
                ax=ax,
                title=f"Bland-Altman: Tolerance matching, every 2nd peak dropped (±{tolerance_ms:.0f} ms)",
            )
            ba_disregard_halved_path = output_folder / "04_bland_altman_disregard_halved.png"
            print(
                f"[metric_pipeline] Saving Bland-Altman (disregard, halved) to: "
                f"{ba_disregard_halved_path}"
            )
            fig.tight_layout()
            plt.savefig(ba_disregard_halved_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        if halved["closest_gt_ihr"].size > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            bland_altman(
                halved["closest_gt_ihr"],
                halved["closest_pred_ihr"],
                plot=True,
                ax=ax,
                title="Bland-Altman: Closest matching, every 2nd peak dropped",
            )
            ba_closest_halved_path = output_folder / "05_bland_altman_closest_halved.png"
            print(
                f"[metric_pipeline] Saving Bland-Altman (closest, halved) to: "
                f"{ba_closest_halved_path}"
            )
            fig.tight_layout()
            plt.savefig(ba_closest_halved_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # HR-agreement plot: the predicted signal, the ground-truth ECG and the compared
    # instantaneous HRs share one time axis, with a light-green background wherever the
    # two HR estimates agree within +/-HR_AGREEMENT_MAX_BPM_DIFF bpm and a grey
    # background everywhere else (outside the threshold or where HR is undefined). See
    # compute_hr_agreement_over_time / _shade_agreement_background. The same plot is
    # additionally produced for the every-2nd-peak-dropped ("halved") variant whenever
    # that variant was computed for this recording.
    if plot_steps or plot_bland_altman:
        _save_hr_agreement_plot(
            output_folder / "06_hr_agreement_highlighted.png",
            time_grid=t,
            hr_agreement=hr_agreement,
            pred_signal=pred_rs,
            pred_peaks=pred_peaks,
            pred_peak_times=pred_peak_times,
            pred_ihr_times=pred_ihr_times,
            pred_ihr=pred_ihr,
            gt_segment=gt_segment,
            gt_peaks=gt_peaks,
            gt_peak_times=gt_peak_times,
            gt_ihr_times=gt_ihr_times,
            gt_ihr=gt_ihr,
            max_bpm_diff=HR_AGREEMENT_MAX_BPM_DIFF,
        )

        if drop_every_second_peak and hr_agreement_halved is not None:
            _save_hr_agreement_plot(
                output_folder / "07_hr_agreement_highlighted_halved.png",
                time_grid=t,
                hr_agreement=hr_agreement_halved,
                pred_signal=pred_rs,
                pred_peaks=pred_peaks_halved,
                pred_peak_times=pred_peak_times_halved,
                pred_ihr_times=pred_ihr_times_halved,
                pred_ihr=pred_ihr_halved,
                gt_segment=gt_segment,
                gt_peaks=gt_peaks,
                gt_peak_times=gt_peak_times,
                gt_ihr_times=gt_ihr_times,
                gt_ihr=gt_ihr,
                max_bpm_diff=HR_AGREEMENT_MAX_BPM_DIFF,
                pred_color="C3",
                pred_peaks_label="Kept peaks (every 2nd dropped)",
                pred_ihr_label="Pred IHR (every 2nd peak dropped)",
                panel1_title="Predicted signal: every second detected peak dropped",
                suptitle_note=", every 2nd predicted peak dropped",
            )

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
        "signal_hr_mean_diff_to_gt_mean": signal_hr_mean_diff_to_gt_mean,
        "signal_hr_mean_abs_diff_to_gt_mean": signal_hr_mean_abs_diff_to_gt_mean,
        "original_pipeline_bpm_estimate": original_pipeline_bpm_estimate,
        "original_pipeline_bpm_estimate_diff_to_gt_mean": original_pipeline_bpm_estimate_diff_to_gt_mean,
        "original_pipeline_bpm_estimate_abs_diff_to_gt_mean": original_pipeline_bpm_estimate_abs_diff_to_gt_mean,
        "num_gt_peaks": int(gt_peaks.size),
        "num_pred_peaks": int(pred_peaks.size),
        "num_matched_disregard": int(matched_gt_ihr.size),
        "num_matched_closest": int(closest_gt_ihr.size),
        "hr_agreement_max_bpm_diff": float(HR_AGREEMENT_MAX_BPM_DIFF),
        "hr_agreement_seconds": float(hr_agreement["agreement_seconds"]),
        "hr_comparable_seconds": float(hr_agreement["comparable_seconds"]),
        "hr_agreement_fraction": (
            float(hr_agreement["agreement_fraction"])
            if _is_finite_scalar(hr_agreement["agreement_fraction"]) else None
        ),
        "signal_hr_mean_halved": (
            float(signal_hr_mean_halved) if _is_finite_scalar(signal_hr_mean_halved) else None
        ),
        "signal_hr_mean_halved_diff_to_gt_mean": signal_hr_mean_halved_diff_to_gt_mean,
        "signal_hr_mean_halved_abs_diff_to_gt_mean": signal_hr_mean_halved_abs_diff_to_gt_mean,
        "num_pred_peaks_halved": int(pred_peaks_halved.size) if drop_every_second_peak else None,
        "hr_agreement_seconds_halved": (
            float(hr_agreement_halved["agreement_seconds"]) if hr_agreement_halved is not None else None
        ),
        "hr_comparable_seconds_halved": (
            float(hr_agreement_halved["comparable_seconds"]) if hr_agreement_halved is not None else None
        ),
        "hr_agreement_fraction_halved": (
            float(hr_agreement_halved["agreement_fraction"])
            if hr_agreement_halved is not None and _is_finite_scalar(hr_agreement_halved["agreement_fraction"])
            else None
        ),
        "num_matched_disregard_halved": (
            int(halved["matched_gt_ihr"].size) if halved is not None else None
        ),
        "num_matched_closest_halved": (
            int(halved["closest_gt_ihr"].size) if halved is not None else None
        ),
        "metrics_disregard_halved": (
            _serialize_for_json(halved["disregard_metrics"]) if halved is not None else None
        ),
        "ba_stats_disregard_halved": (
            _serialize_for_json(halved["ba_disregard"]) if halved is not None else None
        ),
        "metrics_closest_halved": (
            _serialize_for_json(halved["closest_metrics"]) if halved is not None else None
        ),
        "ba_stats_closest_halved": (
            _serialize_for_json(halved["ba_closest"]) if halved is not None else None
        ),
        "tp_distance_based": _serialize_for_json(distance_based_f1_metrics["tp_distance_based"]),
        "fp_distance_based": _serialize_for_json(distance_based_f1_metrics["fp_distance_based"]),
        "fn_distance_based": _serialize_for_json(distance_based_f1_metrics["fn_distance_based"]),
        "precision_distance_based": _serialize_for_json(distance_based_f1_metrics["precision_distance_based"]),
        "recall_distance_based": _serialize_for_json(distance_based_f1_metrics["recall_distance_based"]),
        "f1_distance_based": _serialize_for_json(distance_based_f1_metrics["f1_distance_based"]),
        "mean_pat_distance_based": _serialize_for_json(distance_based_f1_metrics["mean_pat_distance_based"]),
        "std_pat_distance_based": _serialize_for_json(distance_based_f1_metrics["std_pat_distance_based"]),
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
    out["hr_agreement_seconds"] = hr_agreement["agreement_seconds"]
    out["hr_comparable_seconds"] = hr_agreement["comparable_seconds"]
    out["hr_agreement_fraction"] = hr_agreement["agreement_fraction"]
    out["hr_agreement_max_bpm_diff"] = float(HR_AGREEMENT_MAX_BPM_DIFF)
    out["hr_agreement_seconds_halved"] = (
        hr_agreement_halved["agreement_seconds"] if hr_agreement_halved is not None else None
    )
    out["hr_comparable_seconds_halved"] = (
        hr_agreement_halved["comparable_seconds"] if hr_agreement_halved is not None else None
    )
    out["hr_agreement_fraction_halved"] = (
        hr_agreement_halved["agreement_fraction"] if hr_agreement_halved is not None else None
    )

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
    individuals: tuple[str, ...] | str | None = None,
) -> Dict[str, object]:
    """Run evaluate_prediction_against_ground_truth for all archive files in `folder`.

    ``existing_entry_mode`` controls files that already have a row in the aggregate
    CSV: "skip" only processes files not yet in the CSV, while "overwrite" recomputes
    them and replaces their existing row. Set ``overwrite_csv=True`` to wipe the whole
    CSV before starting (then every file is processed fresh).

    ``individuals`` optionally restricts the run to one (or more) specific individuals,
    matched as a case-insensitive substring of the npz filename (e.g. "Ina" for every
    Ina recording, or "Ina_20_4_3" for a single one); ``None`` evaluates everyone.

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
    # Optional individual filter: keep only files whose name matches one of the
    # requested individuals (case-insensitive substring of the npz filename).
    individual_filter = _normalize_individual_filter(individuals)
    if individual_filter is not None:
        before = len(files)
        files = [f for f in files if _matches_individual_filter(f, individual_filter)]
        print(
            f"[metric_pipeline] Batch: restricting to individual(s) {sorted(individual_filter)}: "
            f"{len(files)} of {before} file(s) match."
        )
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
            out = evaluate_prediction_against_ground_truth(
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
    individuals: tuple[str, ...] | str | None = None,
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

    ``individuals`` optionally restricts the run to one (or more) specific individuals,
    matched as a case-insensitive substring of the npz filename (e.g. "Ina" for every
    Ina recording across cameras, or "Ina_20_4_3" for a single one); ``None``
    evaluates everyone.

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
    # Optional individual filter: a case-insensitive substring of the npz filename
    # (e.g. "Ina" -> all Ina recordings, "Ina_20_4_3" -> one). None = all individuals.
    individual_filter = _normalize_individual_filter(individuals)

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
    if individual_filter is not None:
        print(f"[metric_pipeline] Structure batch: restricting to individual(s): {sorted(individual_filter)}")

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
        if individual_filter is not None:
            found = [f for f in found if _matches_individual_filter(f, individual_filter)]
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
            out = evaluate_prediction_against_ground_truth(
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
    APPROACHES_TO_RUN = ("new_approach") #("nair_et_al", "yang_et_al", "new_approach")
    CAMERAS_TO_RUN = ("lepton")#None#("mwir", "vario", "l")  # e.g. ("mwir", "vario") or None for all

    # Optionally restrict the run to ONE specific individual (or a few). Matched as a
    # case-insensitive substring of the npz filename, so "Ina" runs every Ina recording
    # across cameras while "Ina_20_4_3" targets a single recording. Accepts a string, a
    # tuple of strings, or None for everyone. Applies to both RUN_MODEs.
    INDIVIDUALS_TO_RUN = "Julian"  # e.g. "Ina" or ("Ina", "Julian") or "Ina_20_4_3"

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
                individuals=INDIVIDUALS_TO_RUN,
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
                    individuals=INDIVIDUALS_TO_RUN,
                )
            elif HARD_CODED_PATH.is_file():
                evaluate_prediction_against_ground_truth(
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
