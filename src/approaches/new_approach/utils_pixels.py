import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_subroi_means(frame_values_list: list[np.ndarray], coords_per_frame: list[np.ndarray], n_frames: int, n_cells: int, SUBROI_GRID_ROWS: int, SUBROI_GRID_COLS: int) -> np.ndarray:
        subroi_means = np.full((n_frames, n_cells), np.nan, dtype=float)
        for frame_idx in range(n_frames):
            coords = coords_per_frame[frame_idx]
            values = np.asarray(frame_values_list[frame_idx], dtype=float)
            if coords.shape[0] == 0 or values.size == 0:
                continue
            xs = coords[:, 0]
            ys = coords[:, 1]
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            x_span = max(x_max - x_min + 1, 1)
            y_span = max(y_max - y_min + 1, 1)
            col_idx = np.clip(((xs - x_min) * SUBROI_GRID_COLS) // x_span, 0, SUBROI_GRID_COLS - 1)
            row_idx = np.clip(((ys - y_min) * SUBROI_GRID_ROWS) // y_span, 0, SUBROI_GRID_ROWS - 1)
            cell_idx = row_idx * SUBROI_GRID_COLS + col_idx

            finite = np.isfinite(values)
            sums = np.zeros(n_cells, dtype=float)
            counts = np.zeros(n_cells, dtype=float)
            np.add.at(sums, cell_idx, np.where(finite, np.nan_to_num(values, nan=0.0), 0.0))
            np.add.at(counts, cell_idx, finite.astype(float))
            with np.errstate(divide="ignore", invalid="ignore"):
                subroi_means[frame_idx, :] = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
        return subroi_means


def write_subrois_video(writer, n_video: int, source_frames: list[np.ndarray], coords_per_frame: list[np.ndarray], video_path: str, SUBROI_GRID_ROWS: int, SUBROI_GRID_COLS: int, n_cells: int, cell_colors_bgr: list[tuple[int, int, int]], frame_height: int, frame_width: int) -> None:
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
            # Per-frame sub-ROI grid (same per-frame bounding box as the
            # sub-ROI mean plots), so the overlay matches the analyzed sub-ROIs.
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


def plot_subroi_means(n_cells: int, subroi_means: np.ndarray, time_axis: np.ndarray, label: str, roi_dir: str, prefix: str, SUBROI_GRID_ROWS: int, SUBROI_GRID_COLS: int) -> None:
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(
            SUBROI_GRID_ROWS,
            SUBROI_GRID_COLS,
            figsize=(4.0 * SUBROI_GRID_COLS, 2.6 * SUBROI_GRID_ROWS),
            sharex=True,
            squeeze=False,
        )
    for cell in range(n_cells):
            row = cell // SUBROI_GRID_COLS
            col = cell % SUBROI_GRID_COLS
            ax = axes[row][col]
            ax.plot(time_axis, subroi_means[:, cell], color=cmap(cell % 10), linewidth=1.2)
            ax.set_title(f"sub-ROI r{row}c{col}", fontsize=9)
            ax.grid(alpha=0.25)
            if row == SUBROI_GRID_ROWS - 1:
                ax.set_xlabel("Time (s)")
            if col == 0:
                ax.set_ylabel("Temperature")
    fig.suptitle(f"{label} sub-ROI mean signals ({SUBROI_GRID_ROWS}x{SUBROI_GRID_COLS} grid)")
    fig.tight_layout()
    fig.savefig(roi_dir / f"{prefix}_subrois.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


import numpy as np
from scipy.signal import butter, filtfilt


def interpolate_small_gaps(signal, max_gap_frames):
    signal = signal.astype(float).copy()

    n = len(signal)
    valid = np.isfinite(signal)

    if valid.sum() < 2:
        return signal

    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue

        start = i
        while i < n and not valid[i]:
            i += 1
        end = i

        gap_len = end - start

        # only interpolate internal gaps
        if (
            gap_len <= max_gap_frames
            and start > 0
            and end < n
            and valid[start - 1]
            and valid[end]
        ):
            signal[start:end] = np.interp(
                np.arange(start, end),
                [start - 1, end],
                [signal[start - 1], signal[end]],
            )

    return signal


def highpass_subroi_means(
    subroi_means,
    fs,
    cutoff=0.7,
    order=4,
    max_gap_frames=5,
    max_missing_fraction=0.2,
):
    nyquist = fs / 2
    b, a = butter(order, cutoff / nyquist, btype="high")

    filtered = np.full_like(subroi_means, np.nan, dtype=float)

    for cell in range(subroi_means.shape[1]):

        signal = subroi_means[:, cell]

        # interpolate only small gaps
        signal_interp = interpolate_small_gaps(
            signal,
            max_gap_frames=max_gap_frames,
        )

        remaining_missing = np.mean(~np.isfinite(signal_interp))

        # reject signal if too much missing data remains
        if remaining_missing > max_missing_fraction:
            continue

        valid = np.isfinite(signal_interp)

        if valid.sum() < 3 * max(len(a), len(b)):
            continue

        filtered_signal = filtfilt(
            b,
            a,
            signal_interp[valid],
        )

        filtered[valid, cell] = filtered_signal

    return filtered

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


def plot_subroi_psds(
    subroi_means,
    fs,
    fmin=0.0,
    fmax=None,
    nperseg=None,
    roi_dir=None,
    prefix="",
):
    """
    Plot PSDs of all subROI signals.

    Parameters
    ----------
    subroi_means : ndarray, shape (n_frames, n_cells)
        One column per subROI.
    fs : float
        Sampling frequency (video frame rate).
    fmin, fmax : float
        Frequency limits for display.
    nperseg : int or None
        Welch segment length. If None, chosen automatically.
    """

    n_frames, n_cells = subroi_means.shape

    if nperseg is None:
        nperseg = min(256, n_frames)

    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(10, 6))

    for cell in range(n_cells):

        signal = subroi_means[:, cell]

        valid = np.isfinite(signal)

        # skip signals with insufficient data
        if valid.sum() < nperseg:
            continue

        sig = signal[valid]

        freqs, psd = welch(
            sig,
            fs=fs,
            nperseg=min(nperseg, len(sig)),
            detrend="constant",
        )

        color = cmap(cell % 20)

        ax.plot(
            freqs,
            psd,
            color=color,
            linewidth=1.0,
            alpha=0.8,
            label=f"ROI {cell}",
        )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("SubROI Power Spectral Densities")

    if fmax is not None:
        ax.set_xlim(fmin, fmax)
    else:
        ax.set_xlim(fmin, fs / 2)

    ax.grid(True, alpha=0.3)

    # Comment out if too many subROIs
    if n_cells <= 20:
        ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    #save plot

    fig.savefig(roi_dir / f"{prefix}_psds.png", dpi=200, bbox_inches="tight")
