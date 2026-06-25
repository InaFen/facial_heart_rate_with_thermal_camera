"""Shared helpers for the water experiment.

The new-approach pipeline determines, from the RAW signal of the selected sub-ROI,
every frame whose temperature is >= a threshold ("high-temperature" frames) and
saves their timestamps to a single, well-known JSON file. The Yang et al. and Nair
et al. pipelines reuse those timestamps to shade the same time spans on their final
component plots, so all three approaches highlight identical periods.

Reuse contract
--------------
* Timestamps are stored as ``high_temperature_bands_s`` -> a list of ``[start_s, end_s]``
  spans (seconds from the segment start), already merged over consecutive frames. They
  are independent of the frame rate, so any pipeline can shade them directly via
  :func:`shade_high_temp_bands` without re-deriving anything.
* The frame indices / fps are stored as well so the bands can be recomputed if needed
  (:func:`load_high_temp_bands` falls back to them).
* The file location defaults to ``saved_results/water_experiment/high_temp_timestamps.json``
  and is overridable via the ``WATER_EXPERIMENT_TIMESTAMPS_PATH`` env var (so run_batch
  can point all three pipelines at the same per-recording file).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

# Single, well-known location of the reusable water-experiment timestamps. The
# new-approach pipeline writes it; Yang/Nair read it.
DEFAULT_WATER_TIMESTAMPS_PATH = "saved_results/water_experiment/high_temp_timestamps.json"
WATER_TIMESTAMPS_PATH_ENV_VAR = "WATER_EXPERIMENT_TIMESTAMPS_PATH"

# Default highlight style (kept here so the blue shading is identical across all three
# pipelines).
WATER_HIGHLIGHT_COLOR = "blue"
WATER_HIGHLIGHT_ALPHA = 0.15

PathLike = Union[str, "os.PathLike[str]"]
Band = Sequence[float]


def get_water_timestamps_path(default: str = DEFAULT_WATER_TIMESTAMPS_PATH) -> Path:
    """Return the path of the reusable water-experiment timestamps file.

    Uses the ``WATER_EXPERIMENT_TIMESTAMPS_PATH`` env var when set, otherwise the
    project default.
    """
    raw = os.environ.get(WATER_TIMESTAMPS_PATH_ENV_VAR)
    return Path(raw) if raw else Path(default)


def compute_high_temp_bands(frame_indices, fps: float) -> list[list[float]]:
    """Merge consecutive high-temperature frame indices into ``[start_s, end_s]`` bands.

    Consecutive frames are merged into a single band so a translucent highlight does
    not stack (darken) at run boundaries. Each band is widened by half a frame on each
    side so isolated frames stay visible.
    """
    indices = np.asarray(frame_indices, dtype=int)
    indices = np.unique(indices[indices >= 0])
    if indices.size == 0 or not fps:
        return []
    runs = np.split(indices, np.nonzero(np.diff(indices) != 1)[0] + 1)
    return [[float((run[0] - 0.5) / fps), float((run[-1] + 0.5) / fps)] for run in runs]


def save_water_timestamps(payload: dict, path: Optional[PathLike] = None) -> Path:
    """Write the water-experiment ``payload`` to the reusable JSON file and return the path."""
    out_path = Path(path) if path is not None else get_water_timestamps_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as jf:
        json.dump(payload, jf, ensure_ascii=False, indent=2)
    return out_path


def load_water_timestamps(path: Optional[PathLike] = None) -> Optional[dict]:
    """Load the reusable water-experiment payload, or ``None`` if the file is missing."""
    in_path = Path(path) if path is not None else get_water_timestamps_path()
    if not in_path.is_file():
        return None
    with open(in_path, "r", encoding="utf-8") as jf:
        return json.load(jf)


def load_high_temp_bands(
    path: Optional[PathLike] = None, *, fps: Optional[float] = None
) -> list[list[float]]:
    """Return the merged ``[start_s, end_s]`` high-temperature bands from the saved file.

    Prefers the precomputed ``high_temperature_bands_s``; if those are absent it falls
    back to recomputing them from ``high_temperature_frame_indices`` (using the file's
    stored ``fps`` or the ``fps`` argument). Returns an empty list when no file or no
    timestamps are available.
    """
    payload = load_water_timestamps(path)
    if not payload:
        return []
    bands = payload.get("high_temperature_bands_s")
    if bands:
        return [[float(start_s), float(end_s)] for start_s, end_s in bands]
    indices = payload.get("high_temperature_frame_indices") or []
    fps_used = payload.get("fps") or fps
    if not indices or not fps_used:
        return []
    return compute_high_temp_bands(indices, float(fps_used))


def shade_high_temp_bands(
    ax,
    bands: Optional[Sequence[Band]],
    *,
    color: str = WATER_HIGHLIGHT_COLOR,
    alpha: float = WATER_HIGHLIGHT_ALPHA,
    zorder: int = 0,
) -> None:
    """Shade pre-merged ``[start_s, end_s]`` time bands as translucent vertical spans.

    The line(s) should already be plotted on ``ax`` before calling this so the bands
    sit behind the signal (``zorder=0``). The current x-limits are preserved so the
    bands - which may start/end slightly outside the data - do not rescale the axis.
    """
    if not bands:
        return
    xlim = ax.get_xlim()
    for band in bands:
        start_s, end_s = band
        ax.axvspan(
            float(start_s),
            float(end_s),
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=zorder,
        )
    ax.set_xlim(xlim)
