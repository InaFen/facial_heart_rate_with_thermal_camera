import os
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
import numpy as np
from scipy.signal import butter, filtfilt

from src.approaches.yang_et_al.utils_post_processing import apply_butterworth_bandpass_filter


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


def draw_subrois_on_frame(
    frame: np.ndarray,
    coords: np.ndarray,
    SUBROI_GRID_ROWS: int,
    SUBROI_GRID_COLS: int,
    n_cells: int,
    cell_colors_bgr: list[tuple[int, int, int]],
    frame_height: int,
    frame_width: int,
    highlight_cell: int | None = None,
    highlight_color_bgr: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Render a single grayscale source frame with the sub-ROI grid overlay.

    Returns a BGR uint8 image: the contrast-normalised frame with each sub-ROI
    filled in a translucent distinct color and outlined. Used both for the
    overlay video (per frame) and for the saved first-frame PNG.

    If ``highlight_cell`` is given, that sub-ROI (row-major index
    ``row * SUBROI_GRID_COLS + col``) is emphasised with a thicker outline in
    ``highlight_color_bgr`` plus its index label, so it is clearly visible which
    sub-ROI was selected (e.g. the one chosen for heart-rate estimation).
    """
    vmin, vmax = np.percentile(frame, [1, 99])
    frame_clipped = np.clip(frame, vmin, vmax)
    frame_norm = cv2.normalize(frame_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)

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

        # Emphasize the selected sub-ROI (e.g. the one chosen for HR) so it is
        # clearly visible which sub-ROI was used: a thicker, distinct outline
        # plus its index label near the cell centroid.
        if highlight_cell is not None and 0 <= int(highlight_cell) < n_cells:
            sel = (cell_idx == int(highlight_cell)) & in_bounds
            if np.any(sel):
                cell_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                cell_mask[ys[sel], xs[sel]] = 255
                contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame_bgr, contours, -1, highlight_color_bgr, 2)
                label_x = int(round(float(np.mean(xs[sel]))))
                label_y = int(round(float(np.mean(ys[sel]))))
                cv2.putText(
                    frame_bgr,
                    str(int(highlight_cell)),
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    highlight_color_bgr,
                    1,
                    cv2.LINE_AA,
                )

    return frame_bgr


def write_subrois_video(writer, n_video: int, source_frames: list[np.ndarray], coords_per_frame: list[np.ndarray], video_path: str, SUBROI_GRID_ROWS: int, SUBROI_GRID_COLS: int, n_cells: int, cell_colors_bgr: list[tuple[int, int, int]], frame_height: int, frame_width: int, first_frame_png_path: str | None = None) -> None:
    for frame_idx in range(n_video):
        frame_bgr = draw_subrois_on_frame(
            source_frames[frame_idx],
            coords_per_frame[frame_idx],
            SUBROI_GRID_ROWS,
            SUBROI_GRID_COLS,
            n_cells,
            cell_colors_bgr,
            frame_height,
            frame_width,
        )

        # Save the first overlaid frame as a still PNG (alongside the video).
        if frame_idx == 0 and first_frame_png_path is not None:
            cv2.imwrite(str(first_frame_png_path), frame_bgr)
            print(f"Saved sub-ROI overlay first-frame PNG to: {first_frame_png_path}")

        writer.write(frame_bgr)

    writer.release()
    print(f"Saved sub-ROI overlay video to: {video_path}")


def save_roi_overlay_green(
    source_frame: np.ndarray,
    roi_coords: np.ndarray,
    output_path: str | Path,
    *,
    fill_color_bgr: tuple[int, int, int] = (0, 255, 0),
    outline_color_bgr: tuple[int, int, int] = (0, 255, 0),
    fill_alpha: float = 0.35,
    outline_thickness: int = 2,
) -> str:
    """Save a single frame with ONE whole ROI overlaid: green fill + green outline.

    Unlike ``draw_subrois_on_frame`` (which colors each sub-ROI cell separately),
    this draws exactly one translucent green fill plus a solid green outline for
    the entire ``roi_coords`` mask -- used by the visualization-run mode to mark
    the ROI on the first frame of ``video_facemesh.mp4`` / ``video_tracked_face.mp4``.
    Returns the written path.
    """
    frame_height, frame_width = source_frame.shape[:2]
    if source_frame.ndim == 3 and source_frame.shape[2] == 3:
        # Already a color (e.g. TURBO colormapped) BGR frame -- use it as-is,
        # preserving the same color filter that the source video has.
        frame_bgr = source_frame.copy()
        if frame_bgr.dtype != np.uint8:
            vmin, vmax = np.percentile(frame_bgr, [1, 99])
            frame_clipped = np.clip(frame_bgr, vmin, vmax)
            frame_bgr = cv2.normalize(frame_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        vmin, vmax = np.percentile(source_frame, [1, 99])
        frame_clipped = np.clip(source_frame, vmin, vmax)
        frame_norm = cv2.normalize(frame_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)

    coords = np.asarray(roi_coords, dtype=int)
    if coords.size > 0:
        xs = coords[:, 0]
        ys = coords[:, 1]
        in_bounds = (ys >= 0) & (ys < frame_height) & (xs >= 0) & (xs < frame_width)
        xs, ys = xs[in_bounds], ys[in_bounds]

        if xs.size > 0:
            mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            mask[ys, xs] = 255

            overlay = frame_bgr.copy()
            overlay[mask > 0] = fill_color_bgr
            frame_bgr = cv2.addWeighted(overlay, fill_alpha, frame_bgr, 1.0 - fill_alpha, 0)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame_bgr, contours, -1, outline_color_bgr, outline_thickness)

    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, frame_bgr)
    print(f"[visualization] Saved green ROI overlay to: {output_path}")
    return output_path


def save_first_frame_of_video(video_path: str | Path) -> np.ndarray | None:
    """Read and return the first frame of a video (BGR), or None if unreadable.

    Small helper for the visualization-run mode, used to grab the first frame of
    ``video_facemesh.mp4`` / ``video_tracked_face.mp4`` for the green ROI overlay.
    """
    video_path = str(video_path)
    if not os.path.isfile(video_path):
        print(f"[visualization] Video not found, cannot extract first frame: {video_path}")
        return None
    cap = cv2.VideoCapture(video_path)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        print(f"[visualization] Could not read first frame from: {video_path}")
        return None
    return frame


def save_gaussian_before_after_first_frame(
    raw_image: np.ndarray,
    smoothed_image: np.ndarray,
    output_path: str | Path,
    *,
    label: str = "",
) -> str:
    """Save a SINGLE image of the first frame's ROI crop where one half shows
    the raw (pre-Gaussian) pixels and the other half shows the Gaussian-smoothed
    pixels, split down the vertical midline of the ROI's own bounding box.

    Showing both halves within the SAME image (rather than two separate
    side-by-side panels) makes it immediately obvious that both halves cover
    the exact same ROI/extent -- only whichever half was filtered looks smoother.

    ``raw_image`` / ``smoothed_image`` are the reconstructed ROI bounding-box
    images (same convention as the Gaussian-smoothing step in
    ``main_new_approach.py``: a small 2D array with 0 outside the ROI mask, and
    both arrays share the identical shape).
    Used only by the visualization-run mode. Returns the written path.
    """
    height, width = raw_image.shape[:2]
    split_col = width // 2

    combined = np.empty_like(raw_image)
    combined[:, :split_col] = raw_image[:, :split_col]
    combined[:, split_col:] = smoothed_image[:, split_col:]

    vmin = float(min(np.nanmin(raw_image), np.nanmin(smoothed_image)))
    vmax = float(max(np.nanmax(raw_image), np.nanmax(smoothed_image)))
    extent = (0, width, height, 0)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(
        combined, cmap="gray", vmin=vmin, vmax=vmax, extent=extent, aspect="equal", interpolation="nearest"
    )
    ax.axvline(split_col, color="red", linewidth=3.0, linestyle="--")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(f"{label} - Left: before Gaussian filter | Right: after Gaussian filter".strip(" -"))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Temperature (deg C)")
    fig.tight_layout()
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[visualization] Saved Gaussian before/after split-ROI comparison to: {output_path}")
    return output_path


def plot_subroi_signals_before_after_bandpass(
    n_cells: int,
    raw_subroi_means: np.ndarray,
    filtered_subroi_means: np.ndarray,
    time_axis: np.ndarray,
    label: str,
    roi_dir: str | Path,
    prefix: str,
    SUBROI_GRID_ROWS: int,
    SUBROI_GRID_COLS: int,
) -> None:
    """Save ONE standalone PNG per sub-ROI, each with TWO stacked subplots:
    the signal before bandpass filtering (top) and after (bottom), with proper
    axis labels on both, into a dedicated 'before_after_bandpass' subfolder.

    Used only by the visualization-run mode.
    """
    cmap = plt.get_cmap("tab10")

    output_dir = Path(roi_dir) / "before_after_bandpass"
    output_dir.mkdir(parents=True, exist_ok=True)

    for cell in range(n_cells):
        row = cell // SUBROI_GRID_COLS
        col = cell % SUBROI_GRID_COLS

        fig, (ax_before, ax_after) = plt.subplots(2, 1, figsize=(8, 6))

        ax_before.plot(time_axis, raw_subroi_means[:, cell], color=cmap(cell % 10), linewidth=1.0)
        ax_before.set_title(f"sub-ROI r{row}c{col} - before bandpass")
        ax_before.set_xlabel("Time (s)")
        ax_before.set_ylabel("Temperature (deg C)")
        ax_before.grid(alpha=0.25)

        ax_after.plot(time_axis, filtered_subroi_means[:, cell], color=cmap(cell % 10), linewidth=1.0)
        ax_after.set_title("After bandpass")
        ax_after.set_xlabel("Time (s)")
        ax_after.set_ylabel("Amplitude (arbitrary units)")
        ax_after.grid(alpha=0.25)

        fig.suptitle(f"{label} - sub-ROI r{row}c{col}")
        fig.tight_layout()
        output_path = output_dir / f"{prefix}_r{row}c{col}_before_after_bandpass.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[visualization] Saved before/after bandpass sub-ROI plot to: {output_path}")


def compute_subroi_location_info(
    best_idx: int,
    grid_rows: int,
    grid_cols: int,
    roi_name: str | None = None,
) -> dict:
    """Describe a selected sub-ROI's location in a person-comparable way.

    The sub-ROI grid is built per frame from the ROI's own bounding box with
    row 0 = top (min y) and col 0 = left (min x), so the same (row, col) refers
    to the same relative location for everyone. Returns:

    - ``best_subroi_index``: row-major index (``row * grid_cols + col``);
      directly comparable across persons for a fixed ROI/grid size.
    - ``row`` / ``col`` and the grid dimensions.
    - ``row_normalized`` / ``col_normalized``: the sub-ROI's fractional center
      within the ROI bounding box in [0, 1]; comparable across persons AND across
      different grid sizes (e.g. forehead 8x8 vs glabella 3x3).
    - ``subroi_name`` (when ``roi_name`` is given), matching the naming used by
      ``split_roi_pixel_data_into_subrois`` (``{roi_name}_r{row}c{col}``).
    """
    best_idx = int(best_idx)
    grid_rows = int(grid_rows)
    grid_cols = int(grid_cols)
    row = best_idx // grid_cols
    col = best_idx % grid_cols
    info = {
        "best_subroi_index": best_idx,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "n_subrois": grid_rows * grid_cols,
        "row": int(row),
        "col": int(col),
        "row_normalized": float((row + 0.5) / grid_rows) if grid_rows > 0 else None,
        "col_normalized": float((col + 0.5) / grid_cols) if grid_cols > 0 else None,
    }
    if roi_name is not None:
        info["roi_name"] = str(roi_name)
        info["subroi_name"] = f"{roi_name}_r{int(row)}c{int(col)}"
    return info


def save_subroi_overlay_with_highlight(
    source_frame: np.ndarray,
    coords: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    n_cells: int,
    cell_colors_bgr: list[tuple[int, int, int]],
    output_path: str | Path,
    highlight_cell: int,
    highlight_color_bgr: tuple[int, int, int] = (255, 255, 255),
) -> str:
    """Save a single first-frame sub-ROI overlay PNG with one sub-ROI highlighted.

    Mirrors the per-frame overlay drawn for the video, but marks ``highlight_cell``
    (the selected sub-ROI) with a thick outline + label so it is obvious which
    sub-ROI was used. Returns the written path.
    """
    frame_height, frame_width = source_frame.shape[:2]
    image = draw_subrois_on_frame(
        source_frame,
        np.asarray(coords),
        grid_rows,
        grid_cols,
        n_cells,
        cell_colors_bgr,
        frame_height,
        frame_width,
        highlight_cell=highlight_cell,
        highlight_color_bgr=highlight_color_bgr,
    )
    output_path = str(output_path)
    cv2.imwrite(output_path, image)
    print(f"Saved selected sub-ROI overlay (highlight={highlight_cell}) to: {output_path}")
    return output_path


def split_roi_pixel_data_into_subrois(
    roi_pixel_data: list[dict],
    roi_name: str,
    grid_rows: int,
    grid_cols: int,
) -> tuple[list[dict], list[str], list[np.ndarray]]:
    """Split a single ROI in ``roi_pixel_data`` into a grid of sub-ROIs.

    For every frame the ROI's pixels are assigned to ``grid_rows`` x ``grid_cols``
    cells using that frame's own bounding box (the identical scheme used by
    ``compute_subroi_means`` and the overlay video), so each sub-ROI carries its
    own ``coords`` / ``adc_values`` / ``mean_value``. This lets approaches that
    treat each ROI as a building block (Yang's spatio-temporal graph) or a
    subregion (Nair's SVD) operate on the sub-ROIs of a single ROI.

    Every frame dict contains all sub-ROI names (empty cells get empty coords and
    a NaN mean), so the block/subregion count stays constant across frames. The
    sub-ROIs are returned in row-major (cell) order so their index/name lines up
    with the colors drawn by ``write_subrois_video``.

    Returns:
        new_roi_pixel_data: list (per frame) of {subroi_name: {"coords", "adc_values", "mean_value"}}
        subroi_names: ordered list of sub-ROI names (length grid_rows*grid_cols)
        coords_per_frame: list of the original ROI coords (np.ndarray (N, 2)) per frame, for the overlay video
    """
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid_rows and grid_cols must be positive integers")

    n_cells = grid_rows * grid_cols
    # Row-major order: index == row * grid_cols + col == the cell index used by the overlay.
    subroi_names = [
        f"{roi_name}_r{row}c{col}"
        for row in range(grid_rows)
        for col in range(grid_cols)
    ]

    new_roi_pixel_data: list[dict] = []
    coords_per_frame: list[np.ndarray] = []

    for frame_data in roi_pixel_data:
        # Default: every sub-ROI present but empty (keeps the count constant per frame).
        frame_subrois = {
            name: {"coords": [], "adc_values": [], "mean_value": np.nan}
            for name in subroi_names
        }

        roi_info = frame_data.get(roi_name) if isinstance(frame_data, dict) else None
        if roi_info is None:
            new_roi_pixel_data.append(frame_subrois)
            coords_per_frame.append(np.zeros((0, 2), dtype=int))
            continue

        coords = np.asarray(roi_info.get("coords", []), dtype=int)
        if coords.size == 0:
            coords = coords.reshape(0, 2)
        elif coords.ndim != 2 or coords.shape[1] != 2:
            coords = coords.reshape(-1, 2)
        values = np.asarray(roi_info.get("adc_values", []), dtype=float)
        coords_per_frame.append(coords)

        if coords.shape[0] == 0 or values.size == 0:
            new_roi_pixel_data.append(frame_subrois)
            continue

        xs = coords[:, 0]
        ys = coords[:, 1]
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        x_span = max(x_max - x_min + 1, 1)
        y_span = max(y_max - y_min + 1, 1)
        col_idx = np.clip(((xs - x_min) * grid_cols) // x_span, 0, grid_cols - 1)
        row_idx = np.clip(((ys - y_min) * grid_rows) // y_span, 0, grid_rows - 1)
        cell_idx = row_idx * grid_cols + col_idx

        for cell in range(n_cells):
            sel = cell_idx == cell
            if not np.any(sel):
                continue
            cell_coords = coords[sel]
            cell_values = values[sel]
            finite = np.isfinite(cell_values)
            mean_value = float(np.mean(cell_values[finite])) if np.any(finite) else float("nan")
            frame_subrois[subroi_names[cell]] = {
                "coords": [(int(x), int(y)) for x, y in cell_coords],
                "adc_values": cell_values.tolist(),
                "mean_value": mean_value,
            }

        new_roi_pixel_data.append(frame_subrois)

    return new_roi_pixel_data, subroi_names, coords_per_frame


def load_overlay_source_frames(configs, src_start_frame: int, src_end_frame: int | None) -> list[np.ndarray]:
    """Load the thermal source frames used to draw the sub-ROI overlay video.

    Mirrors the loading logic in ``main_new_approach``: supports a single ``.npy``
    stack, a directory of ``.asc`` files, or a directory of ``.gray`` frames,
    restricted to ``[src_start_frame, src_end_frame)``. Imports the gray/asc
    helpers lazily to avoid import cycles.
    """
    from src.approaches.new_approach.utils.utils_gray_files import sort_grayfiles_with_interpolations
    from src.approaches.new_approach.utils.utils_landmark_detection import (
        _asc_to_kelvin_x100,
        _parse_asc_file,
        load_gray_frame,
    )

    source_frames: list[np.ndarray] = []
    gray_path = configs.path_gray_frames
    if os.path.isfile(gray_path) and gray_path.endswith(".npy"):
        stack = np.load(gray_path, allow_pickle=False)
        end = src_end_frame if src_end_frame is not None else stack.shape[0]
        source_frames = [stack[i] for i in range(src_start_frame, min(end, stack.shape[0]))]
    elif os.path.isdir(gray_path):
        files = [f for f in os.listdir(gray_path) if os.path.isfile(os.path.join(gray_path, f))]
        ext = os.path.splitext(files[0])[1].lower() if files else ""
        if ext == ".asc":
            asc_paths = sorted(os.path.join(gray_path, f) for f in files if f.lower().endswith(".asc"))
            end = src_end_frame if src_end_frame is not None else len(asc_paths)
            for p in asc_paths[src_start_frame:end]:
                fr = _parse_asc_file(p)
                if getattr(configs, "asc_unit", None) == "kelvin":
                    fr = _asc_to_kelvin_x100(fr)
                source_frames.append(fr)
        else:
            _, gray_files = sort_grayfiles_with_interpolations(gray_path)
            end = src_end_frame if src_end_frame is not None else len(gray_files)
            for fname in gray_files[src_start_frame:end]:
                source_frames.append(load_gray_frame(os.path.join(gray_path, fname)))
    return source_frames


def save_subroi_overlay_outputs(
    source_frames: list[np.ndarray],
    coords_per_frame: list[np.ndarray],
    grid_rows: int,
    grid_cols: int,
    fps: float,
    output_dir: str | Path,
    n_frames: int | None = None,
    video_name: str = "subrois_overlay.mp4",
    image_name: str = "subrois_overlay_first_frame.png",
) -> None:
    """Write the sub-ROI overlay video and a first-frame PNG into ``output_dir``.

    Reuses ``write_subrois_video`` (which also saves the first overlaid frame as a
    PNG). Colors per sub-ROI follow the same tab10 colormap used in the plots.
    """
    if not source_frames:
        print("[sub-ROI overlay] No source frames available; overlay video/image skipped.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_cells = grid_rows * grid_cols
    subroi_cmap = plt.get_cmap("tab10")
    cell_colors_bgr = [
        tuple(int(round(c * 255)) for c in subroi_cmap(cell % 10)[:3][::-1]) for cell in range(n_cells)
    ]

    total = len(source_frames) if n_frames is None else min(len(source_frames), n_frames)
    n_video = min(total, len(coords_per_frame))
    frame_height, frame_width = source_frames[0].shape[:2]
    video_path = str(output_dir / video_name)
    first_frame_png_path = str(output_dir / image_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    write_subrois_video(
        writer=writer,
        n_video=n_video,
        source_frames=source_frames,
        coords_per_frame=coords_per_frame,
        video_path=video_path,
        SUBROI_GRID_ROWS=grid_rows,
        SUBROI_GRID_COLS=grid_cols,
        n_cells=n_cells,
        cell_colors_bgr=cell_colors_bgr,
        frame_height=frame_height,
        frame_width=frame_width,
        first_frame_png_path=first_frame_png_path,
    )


def maybe_split_single_roi_and_save_overlay(
    roi_pixel_data: list[dict],
    configs,
    output_dir: str | Path,
    subroi_grid_by_roi: dict,
) -> tuple[list[dict], bool]:
    """Split a single ROI into sub-ROIs and save an overlay video + first-frame image.

    If ``roi_pixel_data`` contains exactly one ROI and that ROI name is a key in
    ``subroi_grid_by_roi``, the ROI is split into a ``rows x rows`` grid of
    sub-ROIs (``rows`` taken from the mapping) so downstream approaches treat each
    sub-ROI as its own block/subregion. The sub-ROI overlay video and a PNG of the
    first overlaid frame are written to ``output_dir`` (mirroring main_new_approach).

    Returns ``(roi_pixel_data, was_split)`` -- the (possibly replaced) data and a
    flag indicating whether the split happened. When not applicable the input is
    returned unchanged with ``was_split=False``.
    """
    if not roi_pixel_data:
        return roi_pixel_data, False
    roi_names = list(roi_pixel_data[0].keys())
    if len(roi_names) != 1:
        return roi_pixel_data, False

    roi_name = roi_names[0]
    grid_lookup = {str(k).strip().lower(): int(v) for k, v in dict(subroi_grid_by_roi).items()}
    key = str(roi_name).strip().lower()
    if key not in grid_lookup:
        return roi_pixel_data, False

    grid_rows = grid_lookup[key]
    grid_cols = grid_rows
    print(
        f"[sub-ROI split] Single ROI '{roi_name}' is in the sub-ROI grid map; "
        f"splitting into {grid_rows}x{grid_cols} ({grid_rows * grid_cols}) sub-ROIs."
    )
    new_data, _subroi_names, coords_per_frame = split_roi_pixel_data_into_subrois(
        roi_pixel_data, roi_name, grid_rows, grid_cols
    )

    # Frame range used by the analysis (same as get_roi_data_pipeline).
    fps = float(configs.fps)
    timespan = getattr(configs, "timespan_signal_to_look_at", None)
    if timespan is not None:
        start_t, end_t = timespan
        src_start_frame = int(float(start_t) * fps)
        src_end_frame = int(float(end_t) * fps)
    else:
        src_start_frame = 0
        src_end_frame = None

    try:
        source_frames = load_overlay_source_frames(configs, src_start_frame, src_end_frame)
        save_subroi_overlay_outputs(
            source_frames=source_frames,
            coords_per_frame=coords_per_frame,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            fps=fps,
            output_dir=output_dir,
            n_frames=len(new_data),
        )
    except Exception as exc:
        print(f"[sub-ROI overlay] Failed to create overlay video/image: {exc}")

    return new_data, True


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
    plt.close(fig)


def bandpass_subroi_means(
    subroi_means,
    fs,
    lowcut,
    highcut,
    order,
    max_gap_frames=5,
    max_missing_fraction=0.2,
):
    filtered = np.full_like(subroi_means, np.nan, dtype=float)

    for cell in range(subroi_means.shape[1]):
        signal = subroi_means[:, cell]

        # Interpolate only small gaps
        signal_interp = interpolate_small_gaps(
            signal,
            max_gap_frames=max_gap_frames,
        )

        remaining_missing = np.mean(~np.isfinite(signal_interp))

        # Reject signal if too much missing data remains
        if remaining_missing > max_missing_fraction:
            continue

        valid = np.isfinite(signal_interp)

        # Need enough valid samples for filtering
        if valid.sum() < 3 * (order + 1):
            continue

        filtered_signal = apply_butterworth_bandpass_filter(
            signal_interp[valid][:, np.newaxis],
            lowcut=lowcut,
            highcut=highcut,
            fs=fs,
            order=order,
        ).squeeze()

        filtered[valid, cell] = filtered_signal

    return filtered


def select_best_subroi_by_hr_band_power(
    subroi_signals,
    fs,
    hr_band=(0.7, 3.0),
):
    """
    Select the sub-ROI whose signal has the highest fraction of power
    within the heart-rate frequency band.

    Parameters
    ----------
    subroi_signals : array, shape (T, N)
        One signal per sub-ROI.
    fs : float
        Sampling frequency in Hz.
    hr_band : tuple(float, float)
        Heart-rate band in Hz (default: 0.7–3.0 Hz = 42–180 bpm).

    Returns
    -------
    best_idx : int
        Index of the selected sub-ROI.
    band_power_ratios : ndarray, shape (N,)
        HR-band power / total power for each sub-ROI.
    """

    band_power_ratios = compute_hr_band_power_ratios(subroi_signals, fs, hr_band)

    if not np.any(np.isfinite(band_power_ratios)):
        raise ValueError(
            "No usable sub-ROI signal for HR-band selection: every sub-ROI is "
            "empty or has no power. This usually means the sub-ROI grid is too "
            "fine (too many empty cells) or too much of the signal is missing. "
            "Try a coarser SUBROI_GRID, or relax max_gap_frames / "
            "max_missing_fraction in bandpass_subroi_means."
        )

    best_idx = int(np.nanargmax(band_power_ratios))
    print(f"Selected sub-ROI {best_idx} with HR-band power ratio {band_power_ratios[best_idx]:.3f}")

    return best_idx, band_power_ratios


def _normalize_01(values):
    """Min-max normalise a 1D array to [0, 1], ignoring NaNs.

    NaNs stay NaN; a constant (zero-range) set of finite values maps to 0.5 so no
    sub-ROI is unfairly favoured when a criterion happens to be identical everywhere.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    out = np.full_like(values, np.nan)
    if finite.sum() == 0:
        return out
    vmin = float(np.min(values[finite]))
    vmax = float(np.max(values[finite]))
    if vmax - vmin <= 0:
        out[finite] = 0.5
    else:
        out[finite] = (values[finite] - vmin) / (vmax - vmin)
    return out


def compute_hr_band_power_ratios(subroi_signals, fs, hr_band=(0.7, 3.0)):
    """HR-band power / total power for each sub-ROI (NaN where unusable).

    Factored out of :func:`select_best_subroi_by_hr_band_power` so the same ratio can
    be reused by the combined selector without recomputing it.
    """
    n_subrois = subroi_signals.shape[1]
    band_power_ratios = np.full(n_subrois, np.nan)
    for i in range(n_subrois):
        signal = subroi_signals[:, i].astype(float)
        finite = np.isfinite(signal)
        # Skip sub-ROIs that are entirely empty or still have residual NaNs (long
        # gaps the small-gap interpolation in the bandpass step left behind).
        if finite.sum() < 2 or not finite.all():
            continue
        freqs, psd = welch(signal, fs=fs)
        total_power = np.trapz(psd, freqs)
        hr_mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
        hr_power = np.trapz(psd[hr_mask], freqs[hr_mask])
        if total_power > 0:
            band_power_ratios[i] = hr_power / total_power
    return band_power_ratios


def compute_peak_clarity_scores(subroi_signals, fs, hr_band=(0.7, 3.0)):
    """Compute a peak-clarity score for each sub-ROI signal, based on the shape of the autocorrelation and the presence of extra peaks in the signal. NaN where
unusable.
    """
    n_subrois = subroi_signals.shape[1]
    clarity = np.full(n_subrois, np.nan)
    periodicity = np.full(n_subrois, np.nan)
    ripple = np.full(n_subrois, np.nan)

    for i in range(n_subrois):
        signal = subroi_signals[:, i].astype(float)
        finite = np.isfinite(signal)
        # Need enough finite, gap-free samples to judge peak shape reliably.
        if finite.sum() < 8 or not finite.all():
            continue
        sig = signal - np.mean(signal)
        std = float(np.std(sig))
        if std <= 0:
            continue
        sig = sig / std
        n = sig.size

        # (1) Autocorrelation periodicity at the HR period.
        autocorr = np.correlate(sig, sig, mode="full")[n - 1:]
        autocorr = autocorr / autocorr[0]
        lo_lag = max(1, int(np.floor(fs / hr_band[1])))
        hi_lag = min(n - 1, int(np.ceil(fs / hr_band[0])))
        if hi_lag > lo_lag:
            periodicity[i] = float(np.clip(np.max(autocorr[lo_lag:hi_lag + 1]), 0.0, 1.0))

        # (2) Ripple / extra-peak penalty: count ALL local maxima (any tiny ripple
        #     counts) and compare to the number of beats expected from the dominant
        #     HR-band frequency.
        local_maxima, _ = find_peaks(sig)
        freqs, psd = welch(sig, fs=fs)
        band = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
        if local_maxima.size > 0 and np.any(band):
            f_dom = float(freqs[band][int(np.argmax(psd[band]))])
            expected_beats = f_dom * (n / fs)
            if expected_beats > 0:
                ripple[i] = float(np.clip(expected_beats / local_maxima.size, 0.0, 1.0))

        parts = [v for v in (periodicity[i], ripple[i]) if np.isfinite(v)]
        if parts:
            clarity[i] = float(np.mean(parts))

    return clarity, {"periodicity": periodicity, "ripple": ripple}


def select_best_subroi(
    subroi_signals,
    fs,
    hr_band=(0.7, 3.0),
    weight_band_power=0.5,
    weight_peak_clarity=0.5,
    min_peak_clarity=None,
):
    """Select the best sub-ROI from the HR-band power and/or a peak-clarity score.

    The two (normalised) criteria are weighted and combined. A criterion with a
    zero weight is ignored entirely -- including its NaNs -- so e.g.
    ``weight_peak_clarity=0`` gives a purely SQI-based selection in which a strong
    sub-ROI is never dropped just because its peak-clarity score is undefined.

    Parameters
    ----------
    subroi_signals : array, shape (T, N)
        One signal per sub-ROI (already bandpassed).
    fs : float
        Sampling frequency in Hz.
    hr_band : tuple(float, float)
        Heart-rate band in Hz.
    weight_band_power, weight_peak_clarity : float
        Relative weights of the two (normalised) criteria (their sum must be > 0).
        Set ``weight_peak_clarity=0`` for an HR-band-only (SQI) selection, or
        ``weight_band_power=0`` for a peak-clarity-only selection.
    min_peak_clarity : float or None
        Optional hard threshold on the raw peak-clarity score in [0, 1]; applied
        only when peak clarity carries weight.

    Returns
    -------
    best_idx : int
    scores : dict[str, ndarray]
        Per-sub-ROI 'band_power_ratio', 'peak_clarity', 'periodicity', 'ripple',
        'band_power_norm', 'peak_clarity_norm' and 'combined' (NaN where unusable).
    """
    band_power_ratio = compute_hr_band_power_ratios(subroi_signals, fs, hr_band)
    peak_clarity, clarity_parts = compute_peak_clarity_scores(subroi_signals, fs, hr_band)

    weight_band_power = float(weight_band_power)
    weight_peak_clarity = float(weight_peak_clarity)
    weight_sum = weight_band_power + weight_peak_clarity
    if weight_sum <= 0:
        raise ValueError("weight_band_power + weight_peak_clarity must be > 0.")
    use_band_power = weight_band_power > 0
    use_peak_clarity = weight_peak_clarity > 0

    # A sub-ROI is usable only if every criterion that actually carries weight is finite
    # for it. A zero-weight criterion is ignored entirely (including its NaNs), so e.g.
    # an SQI-only selection (weight_peak_clarity == 0) never drops a strong sub-ROI just
    # because its peak-clarity score happens to be undefined.
    usable = np.ones(subroi_signals.shape[1], dtype=bool)
    if use_band_power:
        usable &= np.isfinite(band_power_ratio)
    if use_peak_clarity:
        usable &= np.isfinite(peak_clarity)
    if not np.any(usable):
        raise ValueError(
            "No usable sub-ROI signal for selection: every sub-ROI is empty, "
            "has residual NaNs, or has no measurable peaks. Try a coarser SUBROI_GRID, "
            "or relax max_gap_frames / max_missing_fraction in bandpass_subroi_means."
        )

    # Optional hard gate: only keep sub-ROIs with sufficiently clear peaks (applies only
    # when peak clarity is actually considered).
    candidate = usable.copy()
    if use_peak_clarity and min_peak_clarity is not None:
        gated = candidate & (peak_clarity >= float(min_peak_clarity))
        if np.any(gated):
            candidate = gated
        else:
            print(
                f"[select_best_subroi] No sub-ROI reaches min_peak_clarity="
                f"{min_peak_clarity}; ignoring the gate and ranking all usable sub-ROIs."
            )

    # Normalise each criterion across the candidate sub-ROIs, then combine. Only the
    # weighted criteria contribute; an unweighted criterion is left out so its NaNs can
    # neither influence nor invalidate the combined score.
    bp_norm = _normalize_01(np.where(candidate, band_power_ratio, np.nan))
    pc_norm = _normalize_01(np.where(candidate, peak_clarity, np.nan))

    combined = np.full(subroi_signals.shape[1], np.nan)
    if use_band_power and use_peak_clarity:
        valid = np.isfinite(bp_norm) & np.isfinite(pc_norm)
        combined[valid] = (
            weight_band_power * bp_norm[valid] + weight_peak_clarity * pc_norm[valid]
        ) / weight_sum
    elif use_band_power:
        combined = np.where(np.isfinite(bp_norm), bp_norm, np.nan)
    else:  # peak clarity only
        combined = np.where(np.isfinite(pc_norm), pc_norm, np.nan)

    best_idx = int(np.nanargmax(combined))
    print(
        f"Selected sub-ROI {best_idx}: combined={combined[best_idx]:.3f} "
        f"(HR-band power ratio={band_power_ratio[best_idx]:.3f}, "
        f"peak clarity={peak_clarity[best_idx]:.3f})"
    )

    scores = {
        "band_power_ratio": band_power_ratio,
        "peak_clarity": peak_clarity,
        "periodicity": clarity_parts["periodicity"],
        "ripple": clarity_parts["ripple"],
        "band_power_norm": bp_norm,
        "peak_clarity_norm": pc_norm,
        "combined": combined,
    }
    return best_idx, scores