"""Configurable per-recording metrics tables for the thesis.

This module reads the aggregate metrics file written by ``metric_pipeline.py``
(``saved_results/all_npz_metrics.csv``, one row per *recording x ROI*) and produces a
single presentable table whose content is driven by the hard-coded :data:`CONFIG` block at
the top of the file. Edit those constants and run the module to regenerate the table; the
CSV and the fully-bordered LaTeX ``.tex`` are written to ``saved_results/`` under an
auto-generated, config-describing filename.

Configuration knobs (see :data:`CONFIG`)
----------------------------------------
``APPROACHES``  Which approaches to include. A list drawn from ``"own"`` (== new_approach),
                ``"nair"`` (== nair_et_al), ``"yang"`` (== yang_et_al), or simply
                ``["all"]`` for every approach.
``HALVED``      ``True`` also considers the every-second-peak ("halved") values. In ``peaks``
                modus the halved variant of an ROI competes for the best/ranked slot and a
                ``Halved`` column (yes/no) is shown; in ``mean`` modus a parallel block of
                "halved" aggregate columns is appended.
``VIEW``        ``"side"`` or ``"front"`` -- restricts the candidate ROIs to the side-view
                (the manual temple / forehead&background / temp rectangles) or the
                front-view (the automatically detected face regions plus the front-view
                manual rectangle) ones.
``MANUAL``      ``True`` keeps ROIs whose token contains ``"manual"``; ``False`` drops them.
                (Note: every *side*-view ROI is a manual rectangle, so ``VIEW="side"`` with
                ``MANUAL=False`` yields no candidates.)
``ROIS``        Optional explicit allow-list of ROI tokens (e.g. ``["forehead", "glabella"]``).
                When non-empty only those ROIs are considered (still intersected with the
                other filters). When empty the ``whole_face`` ROI is additionally dropped for
                Nair et al. / Yang et al. (it is not a meaningful region there), matching
                ``summarize_metrics.py``.
``MODUS``       ``"peaks"`` (the HR-peak agreement / error / Bland-Altman metrics, as before)
                or ``"mean"`` (the across-ROIs spread of the per-ROI mean signal-HR bias).
``RANK``        ``False`` -> one row per recording (its best ROI in ``peaks`` modus).
                ``True``  -> ``peaks`` only: every candidate ROI of each recording is listed,
                ordered best -> worst by the composite criterion *within that recording*,
                with a ``Rank`` column; halved variants (when ``HALVED``) are added as extra
                ranked rows.

The ``forehead_no_ema`` ROI is **always** ignored.

``peaks`` modus columns
-----------------------
``Approach``, ``Camera`` (ImageIR / VarioCam / Lepton), ``Participant`` (1..3),
``Experiment`` (before / after sport), ``ground truth mean HR (bpm)``, ``ROI``, optionally
``Rank`` (``RANK=True``), optionally ``Halved`` (``HALVED=True``) and then the HR-agreement
fraction, rmse, mae and the four Bland-Altman statistics. r² is intentionally omitted.

``mean`` modus columns
----------------------
``Approach``, ``Camera``, ``Participant``, ``Experiment``, ``Perspective`` (side / front
view), ``ground truth mean HR (bpm)`` and the across-ROIs spread of the per-ROI
``signal_hr_mean_abs_diff_to_gt_mean`` -- its Mean, Median, RMSE, Range, Max and Min over
the recording's candidate ROIs. With ``HALVED=True`` the same six statistics computed from
``signal_hr_mean_halved_abs_diff_to_gt_mean`` (available only for the manual rectangles) are
appended as a parallel "halved" block. "Across ROIs" means: for one recording take each
candidate ROI's ``signal_hr_mean_abs_diff_to_gt_mean`` and summarise those per-ROI values.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:  # normal package import (python -m src.metrics.create_tables_for_paper)
    from src.metrics.create_table import (
        REPO_ROOT,
        _APPROACH_LABELS,
        _APPROACH_ORDER,
        _CAMERA_DISPLAY,
        _CAMERA_ORDER,
        _COLUMN_SOURCES,
        _TIME_SEGMENT_RE,
        _approach_label,
        _detect_approach,
        _detect_camera,
        _finalize_table,
        _parse_participant_experiment,
        _variant_has_data,
        _variant_score_row,
    )
    from src.metrics.summarize_metrics import IGNORED_APPROACH_ROIS, group_scores
except ModuleNotFoundError:  # allow running this file directly
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.metrics.create_table import (
        REPO_ROOT,
        _APPROACH_LABELS,
        _APPROACH_ORDER,
        _CAMERA_DISPLAY,
        _CAMERA_ORDER,
        _COLUMN_SOURCES,
        _TIME_SEGMENT_RE,
        _approach_label,
        _detect_approach,
        _detect_camera,
        _finalize_table,
        _parse_participant_experiment,
        _variant_has_data,
        _variant_score_row,
    )
    from src.metrics.summarize_metrics import IGNORED_APPROACH_ROIS, group_scores

# Override the approach sort order imported from create_table.py: own approach on top,
# then Yang et al., then Nair et al.  Every other function in this module uses
# _APPROACH_ORDER for the "_approach_order" sort key, so changing it here is sufficient.
_APPROACH_ORDER: dict[str, int] = {
    "new_approach": 0,
    "yang_et_al":   1,
    "nair_et_al":   2,
}

# ======================================================================================
# CONFIG -- edit these hard-coded values, then run the module to (re)generate the table.
# ======================================================================================
CONFIG: dict[str, object] = {
    # Which approaches to include: any subset of {"own", "nair", "yang"} or ["all"].
    "APPROACHES": ["own"],
    # Also consider the every-second-peak ("halved") values (True/False).
    "HALVED": True,
    # Which ROI perspective to consider: "side" or "front".
    "VIEW": "side",
    # Whether ROIs whose token contains "manual" are considered (True/False).
    "MANUAL": True,
    # Optional explicit ROI allow-list (e.g. ["forehead", "glabella"]). [] = all that pass
    # the VIEW / MANUAL filters.
    "ROIS": ["manual_rectangle_temp"],
    # "peaks" (HR-peak agreement / error / Bland-Altman metrics) or "mean" (across-ROIs
    # spread of the per-ROI mean signal-HR bias).
    "MODUS": "peaks",
    # peaks only: False = best ROI per recording; True = all ROIs per recording, ranked.
    "RANK": False,
}

DEFAULT_INPUT_CSV = REPO_ROOT / "saved_results" / "all_npz_metrics.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "saved_results"

# Short alias -> approach folder token. The folder tokens themselves are also accepted.
_APPROACH_ALIASES = {
    "own": "new_approach",
    "nair": "nair_et_al",
    "yang": "yang_et_al",
}

# ROI tokens recognised at the end of a plots_folder leaf, longest / most specific first so
# that a compound token wins the ``endswith`` match before its suffix (e.g.
# ``manual_rectangle_forehead`` before ``forehead``, ``left_forehead`` before ``forehead``).
_ROI_VOCABULARY = (
    "manual_rectangle_front_view",
    "manual_rectangle_forehead",
    "manual_rectangle_temp",
    "manual_rectangle",
    "left_forehead",
    "right_forehead",
    "forehead_no_ema",
    "whole_face",
    "forehead",
    "glabella",
    "dlib",
    "tracker",
)

# ROIs that are always ignored, regardless of configuration.
_IGNORED_ROIS = frozenset({"forehead_no_ema"})#, "manual_rectangle_temp"})

# The three side-view manual ROIs; every other recognised ROI is treated as front view.
_SIDE_VIEW_ROIS = frozenset(
    {"manual_rectangle", "manual_rectangle_forehead", "manual_rectangle_temp"})


# Display name overrides for ROI tokens in all tables. When a token is absent the
# underscore-replaced original (e.g. "left forehead") is used as a fallback.
# Edit these entries to change how a particular ROI label appears in the output.
_ROI_DISPLAY: dict[str, str] = {
    "manual_rectangle": "temple (static box)",
    "manual_rectangle_forehead": "forehead & background (static box)",
    "manual_rectangle_temp": "forehead & background (static box)",
    "manual_rectangle_front_view": "face (static box)",
    "forehead": "forehead (tracked)",
    "left_forehead": "left forehead (tracked)",
    "right_forehead": "right forehead (tracked)",
    "glabella": "glabella (tracked)",
    "whole_face": "whole face (tracked)",
    "dlib": "face (original Nair et al. tracker)",
    "tracker": "face (original Yang et al. tracker)",
}


# Display-name overrides for every possible output column.  Keys are the internal
# column names used during table assembly; values are what appears in the CSV header and
# the LaTeX table.  Columns absent from this dict are passed through unchanged.
# Edit these entries to rename a column without touching any other code.
_COLUMN_DISPLAY: dict[str, str] = {
    # ---- descriptive / identifier columns ----
    "Approach":                        "Approach",
    "Camera":                          "Camera",
    "Participant":                     "ID",
    "Experiment":                      "Experiment",
    "Perspective":                     "Perspective",
    "ground truth mean HR (bpm)":      "Reference mean HR (bpm)",
    "ROI":                             "ROI",
    "Rank":                            "Rank",
    "Halved":                          "Every 2nd peak used only",
    # ---- peaks modus metric columns (from _COLUMN_SOURCES) ----
    "fraction where signal +- GT <= 5 bpm": "Within-5-BPM accuracy (%)",
    "rmse":                            "RMSE",
    "mae":                             "MAE",
    "Bland-Altman mean difference":    "Mean diff",
    "Bland-Altman upper limit of agreement": "Upper LOA",
    "Bland-Altman lower limit of agreement": "Lower LOA",
    "Bland-Altman SD of differences":  "SD diff",
    # ---- mean modus metric columns (regular) ----
    "Mean abs. signal-GT HR diff across ROIs (bpm)":   "Mean abs. signal-GT HR diff across ROIs (bpm)",
    "Median abs. signal-GT HR diff across ROIs (bpm)": "Median abs. signal-GT HR diff across ROIs (bpm)",
    "RMSE abs. signal-GT HR diff across ROIs (bpm)":   "RMSE abs. signal-GT HR diff across ROIs (bpm)",
    "Range abs. signal-GT HR diff across ROIs (bpm)":  "Range abs. signal-GT HR diff across ROIs (bpm)",
    "Max abs. signal-GT HR diff across ROIs (bpm)":    "Max abs. signal-GT HR diff across ROIs (bpm)",
    "Min abs. signal-GT HR diff across ROIs (bpm)":    "Min abs. signal-GT HR diff across ROIs (bpm)",
    # ---- mean modus metric columns (halved) ----
    "Mean abs. signal-GT HR diff across ROIs, halved (bpm)":   "Mean abs. signal-GT HR diff across ROIs, halved (bpm)",
    "Median abs. signal-GT HR diff across ROIs, halved (bpm)": "Median abs. signal-GT HR diff across ROIs, halved (bpm)",
    "RMSE abs. signal-GT HR diff across ROIs, halved (bpm)":   "RMSE abs. signal-GT HR diff across ROIs, halved (bpm)",
    "Range abs. signal-GT HR diff across ROIs, halved (bpm)":  "Range abs. signal-GT HR diff across ROIs, halved (bpm)",
    "Max abs. signal-GT HR diff across ROIs, halved (bpm)":    "Max abs. signal-GT HR diff across ROIs, halved (bpm)",
    "Min abs. signal-GT HR diff across ROIs, halved (bpm)":    "Min abs. signal-GT HR diff across ROIs, halved (bpm)",
}


# mean_range modus: HR bounds that define the "resting" recording group.
_MEAN_RANGE_HR_LOW: float = 55.0
_MEAN_RANGE_HR_HIGH: float = 85.0


def _mean_range_resting_mask(table: pd.DataFrame) -> "pd.Series[bool]":
    """Return a boolean mask selecting the 'resting' recording group.

    A row belongs to the resting group when **all** three conditions hold:

    * Ground-truth mean HR (the column after ``_COLUMN_DISPLAY`` renaming) falls in
      [``_MEAN_RANGE_HR_LOW``, ``_MEAN_RANGE_HR_HIGH``].
    * ``Camera`` is not Lepton.
    * ``Perspective`` equals ``"front view"``.

    All other rows are placed in the *other* group.
    """
    gt_col = _COLUMN_DISPLAY.get("ground truth mean HR (bpm)", "ground truth mean HR (bpm)")
    lepton_display = _CAMERA_DISPLAY.get("lepton", "Lepton")
    hr = pd.to_numeric(table[gt_col], errors="coerce")
    return (
        hr.between(_MEAN_RANGE_HR_LOW, _MEAN_RANGE_HR_HIGH)
        & (table["Camera"] != lepton_display)
        & (table["Perspective"] == "front view")
    )


# mean-modus output column names (regular + the appended halved block).
_MEAN_STAT_LABELS = {
    "mean": "Mean abs. signal-GT HR diff across ROIs (bpm)",
    "median": "Median abs. signal-GT HR diff across ROIs (bpm)",
    "rmse": "RMSE abs. signal-GT HR diff across ROIs (bpm)",
    "range": "Range abs. signal-GT HR diff across ROIs (bpm)",
    "max": "Max abs. signal-GT HR diff across ROIs (bpm)",
    "min": "Min abs. signal-GT HR diff across ROIs (bpm)",
}
_MEAN_STAT_LABELS_HALVED = {
    key: label.replace(" (bpm)", ", halved (bpm)")
    for key, label in _MEAN_STAT_LABELS.items()
}
_MEAN_SIGNAL_COL = "signal_hr_mean_abs_diff_to_gt_mean"
_MEAN_SIGNAL_COL_HALVED = "signal_hr_mean_halved_abs_diff_to_gt_mean"


# --------------------------------------------------------------------------------------
# ROI / recording parsing
# --------------------------------------------------------------------------------------
def _strip_time_segment(plots_folder: str) -> str:
    """Return the plots_folder leaf with the trailing ``_30_00-50_00s`` segment removed."""
    leaf = str(plots_folder).replace("\\", "/").rstrip("/").split("/")[-1]
    match = _TIME_SEGMENT_RE.search(leaf)
    return leaf[: match.start()] if match else leaf


def _extract_roi(plots_folder: str) -> Optional[str]:
    """Return the ROI token of a plots_folder leaf, or ``None`` if unrecognised."""
    base = _strip_time_segment(plots_folder)
    for roi in _ROI_VOCABULARY:
        if base.endswith("_" + roi):
            return roi
    return None


def _roi_view(roi: str) -> str:
    """Return ``"side"`` for the three side-view manual ROIs, else ``"front"``."""
    return "side" if roi in _SIDE_VIEW_ROIS else "front"


def _roi_is_manual(roi: str) -> bool:
    """Return ``True`` when the ROI token contains ``"manual"``."""
    return "manual" in roi


def _recording_key(plots_folder: str) -> tuple[str, str, str]:
    """Return an ``(approach, camera, recording_stem)`` key shared by a recording's ROIs.

    The ROI token is stripped from the leaf so every ROI of the *same* recording maps to the
    same key and therefore is grouped (and, in ``peaks`` modus, competes) together.
    """
    base = _strip_time_segment(plots_folder)
    roi = _extract_roi(plots_folder)
    recording = base[: -(len(roi) + 1)] if roi and base.endswith("_" + roi) else base
    return _detect_approach(plots_folder), _detect_camera(plots_folder), recording


def _selected_approach_tokens(approaches: Sequence[str] | str) -> list[str]:
    """Resolve the configured approaches to a deterministic list of folder tokens."""
    if isinstance(approaches, str):
        approaches = [approaches]
    normalised = [str(a).strip().lower() for a in approaches]
    if "all" in normalised:
        return list(_APPROACH_ORDER.keys())
    tokens: list[str] = []
    for alias in normalised:
        if alias in _APPROACH_ALIASES:
            tokens.append(_APPROACH_ALIASES[alias])
        elif alias in _APPROACH_LABELS:  # a folder token was given directly
            tokens.append(alias)
        else:
            raise ValueError(
                f"Unknown approach {alias!r}; expected one of "
                f"{sorted(_APPROACH_ALIASES) + ['all']} or a folder token "
                f"{sorted(_APPROACH_LABELS)}."
            )
    return list(dict.fromkeys(tokens))  # de-duplicate, preserve order


# --------------------------------------------------------------------------------------
# Loading / filtering
# --------------------------------------------------------------------------------------
def _load_and_filter(
    csv_path: str | Path,
    *,
    approach_tokens: list[str],
    view: str,
    manual: bool,
    rois: Sequence[str],
) -> pd.DataFrame:
    """Read the aggregate CSV and apply the (ANDed) candidate-ROI filters.

    Returns the surviving rows with helper columns ``_roi``, ``_approach``, ``_camera`` and
    ``_rec_key`` attached. Raises ``ValueError`` when no candidate rows remain.
    """
    df = pd.read_csv(csv_path)
    if "plots_folder" not in df.columns:
        raise ValueError(f"'plots_folder' column not found in {csv_path}")

    pf = df["plots_folder"].astype(str)
    df = df.assign(
        _roi=pf.apply(_extract_roi),
        _approach=pf.apply(_detect_approach),
        _camera=pf.apply(_detect_camera),
    )

    # Always drop unrecognised and explicitly ignored ROIs (e.g. forehead_no_ema).
    df = df[df["_roi"].notna() & ~df["_roi"].isin(_IGNORED_ROIS)]
    # Approach filter.
    df = df[df["_approach"].isin(approach_tokens)]
    # View filter (side / front).
    df = df[df["_roi"].apply(_roi_view) == view]
    # Manual filter.
    if not manual:
        df = df[~df["_roi"].apply(_roi_is_manual)]
    # Explicit ROI allow-list, or -- when none is given -- the pipeline's whole_face ignore
    # for Nair et al. / Yang et al.
    wanted = {r for r in rois if r not in _IGNORED_ROIS}
    if wanted:
        df = df[df["_roi"].isin(wanted)]
    else:
        df = df[
            ~df.apply(
                lambda r: (r["_approach"], r["_roi"]) in IGNORED_APPROACH_ROIS, axis=1
            )
        ]

    df = df.copy()
    if df.empty:
        raise ValueError(
            "No candidate ROI rows left after filtering "
            f"(approaches={approach_tokens}, view={view!r}, manual={manual}, "
            f"rois={list(rois)}). Note: every side-view ROI is a manual rectangle, so "
            "VIEW='side' with MANUAL=False selects nothing."
        )
    df["_rec_key"] = df["plots_folder"].astype(str).apply(_recording_key)
    return df


# --------------------------------------------------------------------------------------
# peaks modus
# --------------------------------------------------------------------------------------
def _variant_candidates(
    group: pd.DataFrame, *, halved: bool
) -> list[tuple[pd.Series, bool]]:
    """Return the ``(row, halved)`` candidates of a recording group with usable metrics.

    Every ROI contributes its regular variant; when ``halved`` is enabled the ROIs that also
    have every-second-peak metrics (the manual rectangles) additionally contribute a halved
    variant.
    """
    candidates: list[tuple[pd.Series, bool]] = []
    for _index, row in group.iterrows():
        if _variant_has_data(row, False):
            candidates.append((row, False))
        if halved and _variant_has_data(row, True):
            candidates.append((row, True))
    return candidates


def _score_and_order(
    candidates: list[tuple[pd.Series, bool]]
) -> tuple[list[int], list[Optional[float]]]:
    """Order candidate indices best -> worst by the composite criterion.

    Ties prefer the regular variant, then the lower MAE, then the ``plots_folder`` (so the
    order is deterministic). Returns ``(order, scores)`` where ``order[0]`` is the winner.
    """
    scores = group_scores(
        [_variant_score_row(row, halved) for row, halved in candidates], "composite"
    )

    def _key(i: int) -> tuple:
        row, halved = candidates[i]
        mae_col = "metrics_closest_halved_mae" if halved else "metrics_closest_mae"
        mae = row.get(mae_col)
        return (
            scores[i] if scores[i] is not None else math.inf,
            1 if halved else 0,  # prefer the regular variant on ties
            float(mae) if pd.notna(mae) else math.inf,
            str(row.get("plots_folder", "")),
        )

    return sorted(range(len(candidates)), key=_key), scores


def _peaks_record(
    row: pd.Series, halved: bool, *, rank: Optional[int] = None
) -> dict[str, object]:
    """Build one peaks-modus output record (descriptive columns + the metric columns)."""
    pf = str(row["plots_folder"])
    camera = _detect_camera(pf)
    parsed = _parse_participant_experiment(pf)
    experiment = parsed[1] if parsed else None

    record: dict[str, object] = {
        "Approach": _approach_label(pf),
        "Camera": _CAMERA_DISPLAY.get(camera, camera),
        "Participant": parsed[0] if parsed else pd.NA,
        "Experiment": (
            None
            if experiment is None
            else ("before sport" if experiment in (1, 2) else "after sport")
        ),
        "ground truth mean HR (bpm)": row.get("gt_hr_mean"),
        "ROI": _ROI_DISPLAY.get(str(row["_roi"]), str(row["_roi"]).replace("_", " ")),
        "Halved": "yes" if halved else "no",
        "_approach_order": _APPROACH_ORDER.get(_detect_approach(pf), 99),
        "_camera_order": _CAMERA_ORDER.get(camera, 99),
        "_participant": parsed[0] if parsed else 99,
        "_experiment": experiment if experiment is not None else 99,
    }
    if rank is not None:
        record["Rank"] = rank
        record["_rank"] = rank
    for out_col, (closest_col, halved_col) in _COLUMN_SOURCES.items():
        record[out_col] = row.get(halved_col if halved else closest_col)
    return record


def _build_peaks_table(df: pd.DataFrame, *, halved: bool, rank: bool) -> pd.DataFrame:
    """Assemble the peaks-modus table (best ROI per recording, or all ROIs ranked)."""
    records: list[dict[str, object]] = []
    for _key, group in df.groupby("_rec_key", sort=False):
        candidates = _variant_candidates(group, halved=halved)
        if not candidates:
            continue
        order, _scores = _score_and_order(candidates)
        if rank:
            for rank_number, idx in enumerate(order, start=1):
                row, is_halved = candidates[idx]
                records.append(_peaks_record(row, is_halved, rank=rank_number))
        else:
            row, is_halved = candidates[order[0]]
            records.append(_peaks_record(row, is_halved))

    if not records:
        raise ValueError("No recording had usable peak metrics to report.")

    table = pd.DataFrame.from_records(records)
    sort_cols = ["_approach_order", "_camera_order", "_participant", "_experiment"]
    if rank:
        sort_cols.append("_rank")
    table = table.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    columns = [
        "Approach",
        "Camera",
        "Participant",
        "Experiment",
        "ground truth mean HR (bpm)",
        "ROI",
    ]
    if rank:
        columns.append("Rank")
    if halved:
        columns.append("Halved")
    columns += list(_COLUMN_SOURCES.keys())

    table = table[columns]
    table["Participant"] = table["Participant"].astype("Int64")
    if rank:
        table["Rank"] = table["Rank"].astype("Int64")
    return table


# --------------------------------------------------------------------------------------
# mean modus
# --------------------------------------------------------------------------------------
def _spread_stats(values: np.ndarray) -> dict[str, float]:
    """Return the across-ROIs spread (mean / median / rmse / range / max / min) of ``values``."""
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "range": float(np.max(values) - np.min(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


def _build_mean_table(df: pd.DataFrame, *, halved: bool, view: str) -> pd.DataFrame:
    """Assemble the mean-modus table (across-ROIs spread of the per-ROI signal-HR bias)."""
    perspective = "front view" if view == "front" else "side view"
    records: list[dict[str, object]] = []
    for (approach_token, camera_token, _stem), group in df.groupby(
        "_rec_key", sort=False
    ):
        values = (
            pd.to_numeric(group[_MEAN_SIGNAL_COL], errors="coerce").dropna().to_numpy()
        )
        if len(values) == 0:
            continue

        parsed = None
        for pf in group["plots_folder"].astype(str):
            parsed = _parse_participant_experiment(pf)
            if parsed:
                break
        experiment = parsed[1] if parsed else None
        gt_values = pd.to_numeric(group["gt_hr_mean"], errors="coerce").dropna()

        stats = _spread_stats(values)
        record: dict[str, object] = {
            "Approach": _APPROACH_LABELS.get(approach_token, approach_token),
            "Camera": _CAMERA_DISPLAY.get(camera_token, camera_token),
            "Participant": parsed[0] if parsed else pd.NA,
            "Experiment": (
                None
                if experiment is None
                else ("before sport" if experiment in (1, 2) else "after sport")
            ),
            "Perspective": perspective,
            "ground truth mean HR (bpm)": (
                float(gt_values.iloc[0]) if len(gt_values) else pd.NA
            ),
            "_approach_order": _APPROACH_ORDER.get(approach_token, 99),
            "_camera_order": _CAMERA_ORDER.get(camera_token, 99),
            "_participant": parsed[0] if parsed else 99,
            "_experiment": experiment if experiment is not None else 99,
        }
        for key, label in _MEAN_STAT_LABELS.items():
            record[label] = stats[key]
        if halved:
            halved_values = (
                pd.to_numeric(group[_MEAN_SIGNAL_COL_HALVED], errors="coerce")
                .dropna()
                .to_numpy()
            )
            halved_stats = (
                _spread_stats(halved_values) if len(halved_values) else None
            )
            for key, label in _MEAN_STAT_LABELS_HALVED.items():
                record[label] = (
                    halved_stats[key] if halved_stats is not None else pd.NA
                )
        records.append(record)

    if not records:
        raise ValueError(
            f"No recording had usable '{_MEAN_SIGNAL_COL}' data to report."
        )

    table = pd.DataFrame.from_records(records)
    table = table.sort_values(
        ["_approach_order", "_camera_order", "_participant", "_experiment"],
        kind="stable",
    ).reset_index(drop=True)

    columns = [
        "Approach",
        "Camera",
        "Participant",
        "Experiment",
        "Perspective",
        "ground truth mean HR (bpm)",
    ]
    columns += list(_MEAN_STAT_LABELS.values())
    if halved:
        columns += list(_MEAN_STAT_LABELS_HALVED.values())

    table = table[columns]
    table["Participant"] = table["Participant"].astype("Int64")
    return table


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------
def create_configurable_table(
    *,
    approaches: Sequence[str] | str = None,  # type: ignore[assignment]
    halved: Optional[bool] = None,
    view: Optional[str] = None,
    manual: Optional[bool] = None,
    rois: Optional[Sequence[str]] = None,
    modus: Optional[str] = None,
    rank: Optional[bool] = None,
    csv_path: str | Path = DEFAULT_INPUT_CSV,
    output_path: Optional[str | Path] = None,
    round_decimals: Optional[int] = 2,
    latex_output_path: Optional[str | Path] = None,
    latex_caption: Optional[str] = None,
    latex_label: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """Build the table described by the given settings (defaulting to :data:`CONFIG`).

    Every keyword that is left ``None`` falls back to its :data:`CONFIG` value, so calling
    this with no arguments reproduces exactly what running the module does. Returns the
    assembled table and its fully-bordered LaTeX representation, and -- when the respective
    paths are given -- writes the CSV and ``.tex`` files.
    """
    approaches = CONFIG["APPROACHES"] if approaches is None else approaches  # type: ignore[assignment]
    halved = bool(CONFIG["HALVED"]) if halved is None else halved
    view = str(CONFIG["VIEW"]) if view is None else view
    manual = bool(CONFIG["MANUAL"]) if manual is None else manual
    rois = list(CONFIG["ROIS"]) if rois is None else list(rois)  # type: ignore[arg-type]
    modus = str(CONFIG["MODUS"]) if modus is None else modus
    rank = bool(CONFIG["RANK"]) if rank is None else rank

    view = view.strip().lower()
    modus = modus.strip().lower()
    if view not in ("side", "front"):
        raise ValueError(f"VIEW must be 'side' or 'front', got {view!r}.")
    if modus not in ("peaks", "mean"):
        raise ValueError(f"MODUS must be 'peaks' or 'mean', got {modus!r}.")
    if rank and modus == "mean":
        raise ValueError("RANK=True is only supported for MODUS='peaks'.")

    approach_tokens = _selected_approach_tokens(approaches)
    df = _load_and_filter(
        csv_path, approach_tokens=approach_tokens, view=view, manual=manual, rois=rois
    )

    if modus == "mean":
        needed = [_MEAN_SIGNAL_COL, "gt_hr_mean"]
        if halved:
            needed.append(_MEAN_SIGNAL_COL_HALVED)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(
                f"{csv_path} is missing expected columns for mean modus: {missing}. "
                "Re-run metric_pipeline.py to regenerate the aggregate CSV."
            )
        table = _build_mean_table(df, halved=halved, view=view)
    else:
        needed = [col for pair in _COLUMN_SOURCES.values() for col in pair] + [
            "gt_hr_mean"
        ]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(
                f"{csv_path} is missing expected metric columns for peaks modus: "
                f"{missing}. Re-run metric_pipeline.py to regenerate the aggregate CSV."
            )
        table = _build_peaks_table(df, halved=halved, rank=rank)

    # Rename columns according to _COLUMN_DISPLAY (only entries that differ from their
    # current name have any effect; columns absent from the dict are left unchanged).
    table = table.rename(columns=_COLUMN_DISPLAY)

    return _finalize_table(
        table,
        output_path=output_path,
        round_decimals=round_decimals,
        latex_output_path=latex_output_path,
        latex_caption=latex_caption,
        latex_label=latex_label,
    )


# --------------------------------------------------------------------------------------
# mean_range modus: two mean tables split by resting / other partition
# --------------------------------------------------------------------------------------
def create_mean_range_tables(
    *,
    approaches: Sequence[str] | str = None,  # type: ignore[assignment]
    halved: Optional[bool] = None,
    view: Optional[str] = None,
    manual: Optional[bool] = None,
    rois: Optional[Sequence[str]] = None,
    csv_path: str | Path = DEFAULT_INPUT_CSV,
    round_decimals: Optional[int] = 2,
    output_path_resting: Optional[str | Path] = None,
    output_path_other: Optional[str | Path] = None,
    latex_output_path_resting: Optional[str | Path] = None,
    latex_output_path_other: Optional[str | Path] = None,
    latex_caption_resting: Optional[str] = None,
    latex_caption_other: Optional[str] = None,
    latex_label_resting: Optional[str] = None,
    latex_label_other: Optional[str] = None,
) -> tuple[tuple[pd.DataFrame, str], tuple[pd.DataFrame, str]]:
    """Build two mean-modus tables split by the resting / other partition.

    All filtering and table-assembly steps are identical to
    :func:`create_configurable_table` with ``modus="mean"``.  The fully assembled
    table is then partitioned into:

    * **Resting group** -- recordings where *all* of the following hold:
      ground-truth mean HR \u2208 [:data:`_MEAN_RANGE_HR_LOW`,
      :data:`_MEAN_RANGE_HR_HIGH`] bpm, camera \u2260 Lepton, and
      ``Perspective == "front view"`` (use ``VIEW="front"`` in :data:`CONFIG`).
    * **Other group** -- every remaining recording.

    Returns ``((table_resting, latex_resting), (table_other, latex_other))``.
    """
    # Build the complete mean table (no file output yet; rounding applied here).
    full_table, _ = create_configurable_table(
        approaches=approaches,
        halved=halved,
        view=view,
        manual=manual,
        rois=rois,
        modus="mean",
        rank=False,
        csv_path=csv_path,
        output_path=None,
        round_decimals=round_decimals,
        latex_output_path=None,
    )

    mask = _mean_range_resting_mask(full_table)
    table_resting = full_table[mask].reset_index(drop=True)
    table_other = full_table[~mask].reset_index(drop=True)

    # Pass round_decimals so the LaTeX float_format matches the original table.
    # Re-rounding already-rounded values is a no-op.
    result_resting = _finalize_table(
        table_resting,
        output_path=output_path_resting,
        round_decimals=round_decimals,
        latex_output_path=latex_output_path_resting,
        latex_caption=latex_caption_resting,
        latex_label=latex_label_resting,
    )
    result_other = _finalize_table(
        table_other,
        output_path=output_path_other,
        round_decimals=round_decimals,
        latex_output_path=latex_output_path_other,
        latex_caption=latex_caption_other,
        latex_label=latex_label_other,
    )
    return result_resting, result_other


# --------------------------------------------------------------------------------------
# Config-driven filename / caption helpers and CLI entry point
# --------------------------------------------------------------------------------------
def _config_slug(config: dict[str, object] = CONFIG) -> str:
    """Build a descriptive, filesystem-friendly slug from a config dict."""
    approaches = config["APPROACHES"]
    if isinstance(approaches, str):
        approaches = [approaches]
    normalised = [str(a).strip().lower() for a in approaches]
    approach_part = "all" if "all" in normalised else "-".join(normalised)

    parts = [str(config["MODUS"]).lower(), str(config["VIEW"]).lower(), approach_part]
    parts.append("manual" if config["MANUAL"] else "nomanual")
    rois = list(config["ROIS"])  # type: ignore[arg-type]
    if rois:
        parts.append("rois-" + "-".join(str(r) for r in rois))
    if config["HALVED"]:
        parts.append("halved")
    if config["RANK"]:
        parts.append("ranked")
    return "table_paper_" + "_".join(parts)


def _config_caption(config: dict[str, object] = CONFIG) -> str:
    """Build a human-readable LaTeX caption describing the configured table."""
    approaches = config["APPROACHES"]
    if isinstance(approaches, str):
        approaches = [approaches]
    normalised = [str(a).strip().lower() for a in approaches]
    approach_text = (
        "all approaches" if "all" in normalised else ", ".join(normalised) + " approach(es)"
    )
    view_text = f"{config['VIEW']}-view"
    if config["MODUS"] in ("mean", "mean_range"):
        body = (
            f"Across-ROIs spread of the per-ROI mean signal-HR bias for the {view_text} "
            f"ROIs, {approach_text}"
        )
    elif config["RANK"]:
        body = (
            f"All {view_text} ROIs per recording ranked best to worst by the composite "
            f"criterion, {approach_text}"
        )
    else:
        body = f"Best {view_text} ROI per recording, {approach_text}"
    if config["HALVED"]:
        body += " (every-second-peak values included)"
    return body + "."


if __name__ == "__main__":
    slug = _config_slug()

    def _print_summary(tbl: pd.DataFrame, ltx: str, header: str = "") -> None:
        """Print a table summary block to stdout."""
        if header:
            print(f"\n-- {header} ({len(tbl)} rows) --")
        else:
            print(f"[create_tables_for_paper] {len(tbl)} rows; per camera:")
        if "Camera" in tbl.columns:
            print(tbl["Camera"].value_counts().to_string())
        if "Halved" in tbl.columns:
            print(
                "[create_tables_for_paper] halved rows: "
                f"{int((tbl['Halved'] == 'yes').sum())} of {len(tbl)}"
            )
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 260,
            "display.float_format", lambda v: f"{v:.2f}",
        ):
            print(tbl.to_string(index=False))
        print("\n[create_tables_for_paper] LaTeX table:\n")
        print(ltx)

    print(f"\n[create_tables_for_paper] config: {CONFIG}")

    if str(CONFIG.get("MODUS", "")).lower() == "mean_range":
        caption_base = _config_caption().rstrip(".")
        (table_r, latex_r), (table_o, latex_o) = create_mean_range_tables(
            output_path_resting=DEFAULT_OUTPUT_DIR / f"{slug}_resting.csv",
            output_path_other=DEFAULT_OUTPUT_DIR / f"{slug}_other.csv",
            latex_output_path_resting=DEFAULT_OUTPUT_DIR / f"{slug}_resting.tex",
            latex_output_path_other=DEFAULT_OUTPUT_DIR / f"{slug}_other.tex",
            latex_caption_resting=(
                caption_base
                + f" (resting: HR {_MEAN_RANGE_HR_LOW:.0f}"
                f"--{_MEAN_RANGE_HR_HIGH:.0f}\\,bpm, non-Lepton, front view)."
            ),
            latex_caption_other=caption_base + " (other recordings).",
            latex_label_resting=f"tab:{slug}_resting",
            latex_label_other=f"tab:{slug}_other",
        )
        _print_summary(
            table_r, latex_r,
            f"resting: HR {_MEAN_RANGE_HR_LOW:.0f}\u2013{_MEAN_RANGE_HR_HIGH:.0f} bpm, "
            "non-Lepton, front view",
        )
        _print_summary(table_o, latex_o, "other recordings")
    else:
        table, latex_table = create_configurable_table(
            output_path=DEFAULT_OUTPUT_DIR / f"{slug}.csv",
            latex_output_path=DEFAULT_OUTPUT_DIR / f"{slug}.tex",
            latex_caption=_config_caption(),
            latex_label=f"tab:{slug}",
        )
        _print_summary(table, latex_table)
