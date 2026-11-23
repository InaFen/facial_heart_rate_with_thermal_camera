
from typing import Tuple, Dict, Optional
import numpy as np
import numpy.typing as npt


import matplotlib.pyplot as plt

def compute_distance_based_f1_score(
    pred_peaks: npt.NDArray[np.int64],
    pred_fs: float,
    gt_peaks: npt.NDArray[np.int64],
    gt_fs: float,
    pred_peak_times,
    gt_peak_times,
    limit_pat_window: bool = False,
    lower_pat_limit: Optional[float] = None,
    higher_pat_limit: Optional[float] = None
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
    pred_peak_times,
    gt_peak_times,
    time,
    save_path: Optional[str] = None,
    limit_pat_window: bool = False,
    lower_pat_limit: Optional[float] = None,
    higher_pat_limit: Optional[float] = None,
):

    fig, axes = plt.subplots(
        2, 1,
        figsize=(15, 8),
        sharex=True,
    )

    # Plot ECG and PPG signals
    axes[0].plot(time, gt_signal, color="darkgrey", label="ECG")
    axes[1].plot(time, pred_signal, color="darkgrey", label="PPG")

    # Shade time windows
    if limit_pat_window:
        for gt_idx, gt_time in enumerate(gt_peak_times):

            next_gt_time = (
                gt_peak_times[gt_idx + 1]
                if gt_idx < len(gt_peak_times) - 1
                else len(pred_signal) / pred_fs
            )

            window_start = gt_time + lower_pat_limit / 1000.0
            window_end = gt_time + higher_pat_limit / 1000.0

            # Apply exactly the same clipping as in F1 computation
            window_end = min(window_end, next_gt_time)

            for ax in axes:
                ax.axvspan(
                    window_start,
                    window_end,
                    color="lightblue",
                    alpha=0.25,
                    zorder=0,
                )

    else:
        # Original GT-to-TP interval shading
        interval_colors = ["lightblue", "moccasin", "thistle"]
        color_idx = 0

        for gt_idx, gt_time in enumerate(gt_peak_times):
            # Skip ground-truth peaks without a true positive
            if not gt_matched[gt_idx]:
                continue

            next_gt_time = (
                gt_peak_times[gt_idx + 1]
                if gt_idx < len(gt_peak_times) - 1
                else min(
                    len(pred_signal) / pred_fs,
                    len(gt_signal) / gt_fs,
                )
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
            for ax in axes:
                ax.axvspan(
                    gt_time,
                    tp_time,
                    color=color,
                    alpha=0.25,
                    zorder=0,
                )

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

