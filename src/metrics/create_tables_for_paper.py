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
                ranked rows.  Superseded by ``RANK_OPTION`` when that is non-``None``.
``RANK_OPTION`` ``None``               -> honours ``RANK`` (see above).
                ``"inside_rois"``      -> like ``RANK=True``: all ROIs of each recording
                ranked by the composite criterion within that recording.
                ``"inside_approach"``  -> all ROIs of a given approach ranked globally
                (across every recording of that approach); the ``Rank`` column counts from
                1 over the whole approach.

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
    from src.metrics.create_tables_helpers import (
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
    from src.metrics.summarize_metrics import (
        COMPOSITE_METRICS,
        IGNORED_APPROACH_ROIS,
        group_scores,
        selection_score,
    )
except ModuleNotFoundError:  # allow running this file directly
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.metrics.create_tables_helpers import (
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
    from src.metrics.summarize_metrics import (
        COMPOSITE_METRICS,
        IGNORED_APPROACH_ROIS,
        group_scores,
        selection_score,
    )

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
    "APPROACHES": ["yang"],#, "nair", "own "],
    # Also consider the every-second-peak ("halved") values (True/False).
    "HALVED": True,
    # Which ROI perspective to consider: "side" or "front".
    "VIEW": "front",
    # Whether ROIs whose token contains "manual" are considered (True/False).
    "MANUAL": True,
    # Optional explicit ROI allow-list (e.g. ["forehead", "glabella"]). [] = all that pass
    # the VIEW / MANUAL filters.
    "ROIS": [],
    # "peaks" (HR-peak agreement / error / Bland-Altman metrics) or "mean" (across-ROIs
    # spread of the per-ROI mean signal-HR bias).
    # mean_range modus
    "MODUS": "peaks",
    # peaks only: False = best ROI per recording; True = all ROIs per recording, ranked.
    "RANK": True,
    # Overrides RANK when non-None. "inside_rois" = rank within each recording (like
    # RANK=True); "inside_approach" = rank all ROIs of an approach globally across all
    # recordings of that approach.
    "RANK_OPTION": "inside_rois", #"inside_approach",
    # peaks only: extra weight for "within-5-BPM accuracy" (hr_agreement_fraction) in
    # the composite ROI-selection criterion.  ``None`` = equal weight for all 6
    # sub-metrics (default).  A float in (0, 1) gives that fraction of the total weight
    # to hr_agreement_fraction; the remaining (1 - weight) is split equally among the
    # other 5 sub-metrics.  E.g. 0.3 means accuracy counts 30 %, each other metric 14 %.
    "ACCURACY_WEIGHT": None,#0.6,
    # peaks only: when True, three additional comparison tables are written alongside
    # the main table (see _build_best_vs_second_tables):
    #   *_vs2nd.csv/.tex                -- best vs second-best ROI per recording (3 rows each)
    #   *_vs2nd_by_cam_roi.csv/.tex     -- mean abs diff grouped by (approach, camera, best ROI)
    #   *_vs2nd_by_approach.csv/.tex    -- mean abs diff grouped by (approach, best ROI)
    #   *_vs2nd_by_cam_roi_med.csv/.tex -- median abs diff grouped by (approach, camera, best ROI)
    "COMPARISON_TO_SECOND_BEST": False,
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
    "within-5-BPM accuracy":           "Within-5-BPM accuracy (%)",
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
    )


# mean-modus output column names (regular + the appended halved block).
# All six stats are included; the mean_range modus drops Median / RMSE / Range
# in create_mean_range_tables so only Mean / Max / Min appear in those tables.
_MEAN_STAT_LABELS = {
    "mean":   "Mean abs. signal-GT HR diff across ROIs (bpm)",
    "median": "Median abs. signal-GT HR diff across ROIs (bpm)",
    "rmse":   "RMSE abs. signal-GT HR diff across ROIs (bpm)",
    "range":  "Range abs. signal-GT HR diff across ROIs (bpm)",
    "max":    "Max abs. signal-GT HR diff across ROIs (bpm)",
    "min":    "Min abs. signal-GT HR diff across ROIs (bpm)",
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


def _weighted_composite_scores(
    group_rows: list[dict],
    *,
    accuracy_weight: float,
) -> list[Optional[float]]:
    """Weighted composite score where ``hr_agreement_fraction`` gets ``accuracy_weight``.

    The remaining ``(1 - accuracy_weight)`` is distributed equally among the other
    sub-metrics in :data:`~src.metrics.summarize_metrics.COMPOSITE_METRICS`.  Returns a
    list of "lower is better" scores (the same convention as
    :func:`~src.metrics.summarize_metrics.group_scores`).
    """
    other_metrics = [m for m in COMPOSITE_METRICS if m != "hr_agreement_fraction"]
    n_other = len(other_metrics)  # 5 when COMPOSITE_METRICS has 6 entries
    w_acc = float(accuracy_weight)
    w_other = (1.0 - w_acc) / n_other if n_other else 0.0

    def _avg_ranks(vals: list[Optional[float]]) -> list[float]:
        n = len(vals)
        order = sorted(
            range(n),
            key=lambda i: (vals[i] is None, vals[i] if vals[i] is not None else 0.0),
        )
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = ((i + 1) + (j + 1)) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    acc_ranks = _avg_ranks([selection_score(r, "hr_agreement_fraction") for r in group_rows])
    other_rank_lists = [
        _avg_ranks([selection_score(r, m) for r in group_rows])
        for m in other_metrics
    ]
    n = len(group_rows)
    return [
        w_acc * acc_ranks[i]
        + sum(w_other * other_rank_lists[mi][i] for mi in range(n_other))
        for i in range(n)
    ]


def _score_and_order(
    candidates: list[tuple[pd.Series, bool]],
    *,
    accuracy_weight: Optional[float] = None,
) -> tuple[list[int], list[Optional[float]]]:
    """Order candidate indices best -> worst by the composite criterion.

    When ``accuracy_weight`` is given the within-5-BPM accuracy sub-metric receives that
    fraction of the total composite weight; the remaining weight is split equally among
    the other sub-metrics.  Ties prefer the regular variant, then the lower MAE, then the
    ``plots_folder`` (so the order is deterministic).  Returns ``(order, scores)`` where
    ``order[0]`` is the winner.
    """
    score_rows = [_variant_score_row(row, halved) for row, halved in candidates]
    if accuracy_weight is not None:
        scores = _weighted_composite_scores(score_rows, accuracy_weight=accuracy_weight)
    else:
        scores = group_scores(score_rows, "composite")

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


def _build_peaks_table(
    df: pd.DataFrame,
    *,
    halved: bool,
    rank_option: Optional[str],
    accuracy_weight: Optional[float] = None,
) -> pd.DataFrame:
    """Assemble the peaks-modus table.

    rank_option:
      ``None``               → one row per recording (its best ROI).
      ``"inside_rois"``      → all ROIs ranked within each recording.
      ``"inside_approach"``  → all ROIs ranked globally within each approach.

    accuracy_weight:
      When not ``None``, the within-5-BPM accuracy sub-metric is given this fraction of
      the total composite weight; the remaining weight is distributed equally among the
      other sub-metrics.  ``None`` uses the standard equal-weight composite.
    """
    records: list[dict[str, object]] = []

    if rank_option == "inside_approach":
        # Rank all candidates of an approach together across every recording.
        for _approach_token, approach_df in df.groupby("_approach", sort=False):
            all_candidates: list[tuple[pd.Series, bool]] = []
            for _key, group in approach_df.groupby("_rec_key", sort=False):
                all_candidates.extend(_variant_candidates(group, halved=halved))
            if not all_candidates:
                continue
            order, _scores = _score_and_order(all_candidates, accuracy_weight=accuracy_weight)
            for global_rank, idx in enumerate(order, start=1):
                row, is_halved = all_candidates[idx]
                records.append(_peaks_record(row, is_halved, rank=global_rank))
    else:
        for _key, group in df.groupby("_rec_key", sort=False):
            candidates = _variant_candidates(group, halved=halved)
            if not candidates:
                continue
            order, _scores = _score_and_order(candidates, accuracy_weight=accuracy_weight)
            if rank_option == "inside_rois":
                for rank_number, idx in enumerate(order, start=1):
                    row, is_halved = candidates[idx]
                    records.append(_peaks_record(row, is_halved, rank=rank_number))
            else:
                row, is_halved = candidates[order[0]]
                records.append(_peaks_record(row, is_halved))

    if not records:
        raise ValueError("No recording had usable peak metrics to report.")

    show_rank = rank_option in ("inside_rois", "inside_approach")
    table = pd.DataFrame.from_records(records)
    if rank_option == "inside_approach":
        sort_cols = ["_approach_order", "_rank"]
    elif rank_option == "inside_rois":
        sort_cols = ["_approach_order", "_camera_order", "_participant", "_experiment", "_rank"]
    else:
        sort_cols = ["_approach_order", "_camera_order", "_participant", "_experiment"]
    table = table.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    columns = [
        "Approach",
        "Camera",
        "Participant",
        "Experiment",
        "ground truth mean HR (bpm)",
        "ROI",
    ]
    if show_rank:
        columns.append("Rank")
    if halved:
        columns.append("Halved")
    columns += list(_COLUMN_SOURCES.keys())

    table = table[columns]
    table["Participant"] = table["Participant"].astype("Int64")
    if show_rank:
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
# Peaks summary (RANK=False only)
# --------------------------------------------------------------------------------------
def _generate_peaks_summary(
    table: pd.DataFrame,
    *,
    latex_label: Optional[str] = None,
) -> str:
    """Generate a textual summary for a RANK=False peaks-modus table.

    Produces two sections per approach:

    1. Per-camera ranges of within-5-BPM accuracy and RMSE, split by before-sport /
       after-sport recordings (one sentence per camera).
    2. Full metric details for the best and second-best recording results, ranked by
       within-5-BPM accuracy (descending) then RMSE (ascending).

    ``table`` must already have columns renamed according to :data:`_COLUMN_DISPLAY`.
    """
    def _col(internal: str) -> str:
        return _COLUMN_DISPLAY.get(internal, internal)

    ACC_COL   = _col("within-5-BPM accuracy")
    RMSE_COL  = _col("rmse")
    MAE_COL   = _col("mae")
    BIAS_COL  = _col("Bland-Altman mean difference")
    U_LOA_COL = _col("Bland-Altman upper limit of agreement")
    L_LOA_COL = _col("Bland-Altman lower limit of agreement")
    SD_COL    = _col("Bland-Altman SD of differences")
    HALVED_COL = _col("Halved")
    EXP_COL   = _col("Experiment")
    PART_COL  = _col("Participant")
    CAM_COL   = _col("Camera")
    APP_COL   = _col("Approach")

    required = [ACC_COL, RMSE_COL, MAE_COL, EXP_COL, CAM_COL, APP_COL]
    missing = [c for c in required if c not in table.columns]
    if missing:
        return f"[Summary unavailable: missing columns {missing}]"

    cam_disp_to_token = {v: k for k, v in _CAMERA_DISPLAY.items()}
    app_disp_to_token = {v: k for k, v in _APPROACH_LABELS.items()}

    def _range_str(df: pd.DataFrame, col: str) -> str:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            return "N/A"
        lo, hi = vals.min(), vals.max()
        return f"{lo:.2f}\u2013{hi:.2f}" if abs(hi - lo) > 1e-9 else f"{lo:.2f}"

    def _fv(row: pd.Series, col: str) -> str:
        if col not in row.index:
            return "N/A"
        v = pd.to_numeric(row[col], errors="coerce")
        return f"{v:.2f}" if pd.notna(v) else "N/A"

    approaches_ordered = sorted(
        table[APP_COL].dropna().unique(),
        key=lambda a: _APPROACH_ORDER.get(app_disp_to_token.get(str(a), str(a)), 99),
    )

    paragraphs: list[str] = []

    for approach in approaches_ordered:
        app_df = table[table[APP_COL] == approach].copy()
        if app_df.empty:
            continue

        # --- Part 1: Per-camera accuracy / RMSE ranges ---
        cameras = sorted(
            app_df[CAM_COL].dropna().unique(),
            key=lambda c: _CAMERA_ORDER.get(cam_disp_to_token.get(str(c), str(c)), 99),
        )
        cam_texts: list[str] = []
        for cam in cameras:
            cam_df = app_df[app_df[CAM_COL] == cam]
            parts: list[tuple[str, str, str, str]] = []
            for exp in ("before sport", "after sport"):
                exp_df = cam_df[cam_df[EXP_COL] == exp]
                if not exp_df.empty:
                    parts.append((
                        exp,
                        _range_str(exp_df, ACC_COL),
                        _range_str(exp_df, RMSE_COL),
                        _range_str(exp_df, MAE_COL),
                    ))
            if not parts:
                continue
            if len(parts) == 2:
                (e1, a1, r1, mae1), (e2, a2, r2, mae2) = parts
                cam_texts.append(
                    f"{cam} achieves within-5-BPM accuracies of {a1} {e1} "
                    f"and {a2} {e2}, with RMSE values of {r1}\u202fbpm "
                    f"and {r2}\u202fbpm and MAE values of {mae1}\u202fbpm "
                    f"and {mae2}\u202fbpm, respectively"
                )
            else:
                e1, a1, r1, mae1 = parts[0]
                cam_texts.append(
                    f"{cam} achieves within-5-BPM accuracies of {a1} {e1}, "
                    f"with RMSE values of {r1}\u202fbpm and MAE values of {mae1}\u202fbpm"
                )
        if cam_texts:
            cam_summary = ". ".join(cam_texts) + "."
            paragraphs.append(
                f"Across all recordings, for the {approach} the {cam_summary}"
            )

        # --- Part 2: Best and second-best results ---
        app_df["_acc_s"] = pd.to_numeric(app_df[ACC_COL], errors="coerce")
        app_df["_rmse_s"] = pd.to_numeric(app_df[RMSE_COL], errors="coerce")
        top2 = (
            app_df
            .sort_values(["_acc_s", "_rmse_s"], ascending=[False, True], kind="stable")
            .head(2)
            .reset_index(drop=True)
        )
        if top2.empty:
            continue
        best = top2.iloc[0]
        second = top2.iloc[1] if len(top2) > 1 else None

        best_cam = str(best.get(CAM_COL, ""))
        best_exp = str(best.get(EXP_COL, ""))
        shared = (
            second is not None
            and str(second.get(CAM_COL, "")) == best_cam
            and str(second.get(EXP_COL, "")) == best_exp
            and bool(best_cam) and bool(best_exp)
        )
        if shared:
            intro = (
                f"For the {approach}, the best results are all with the {best_cam} camera, "
                f"{best_exp} recordings:"
            )
        else:
            intro = f"For the {approach}, the best results are:"

        def _row_text(row: pd.Series, rank: int, _shared: bool = shared) -> str:
            part = row.get(PART_COL, pd.NA)
            part_str = f"Participant {part}" if pd.notna(part) else "a participant"
            halved = (
                HALVED_COL in row.index
                and str(row.get(HALVED_COL, "no")).strip().lower() == "yes"
            )
            context = ""
            if not _shared:
                rc, re = row.get(CAM_COL, ""), row.get(EXP_COL, "")
                if rc and re:
                    context = f" ({rc}, {re})"
                elif rc:
                    context = f" ({rc})"
            if rank == 1:
                h_clause = ", where only every second peak is detected," if halved else ""
                ref = f" (\\ref{{{latex_label}}})" if latex_label else ""
                return (
                    f"{part_str}{context}{h_clause} achieves the best performance with "
                    f"a within-5-BPM accuracy of {_fv(row, ACC_COL)}, "
                    f"an RMSE of {_fv(row, RMSE_COL)}\u202fbpm, "
                    f"an MAE of {_fv(row, MAE_COL)}\u202fbpm, "
                    f"a mean difference of {_fv(row, BIAS_COL)}\u202fbpm, "
                    f"limits of agreement (LOA) from {_fv(row, L_LOA_COL)} "
                    f"to {_fv(row, U_LOA_COL)}\u202fbpm, "
                    f"and an SD of the differences of {_fv(row, SD_COL)}\u202fbpm{ref}."
                )
            else:
                h_clause = ", also with only every second peak detected," if halved else ""
                return (
                    f"{part_str}{context}{h_clause} ranks second with "
                    f"a within-5-BPM accuracy of {_fv(row, ACC_COL)}, "
                    f"an RMSE of {_fv(row, RMSE_COL)}\u202fbpm, "
                    f"an MAE of {_fv(row, MAE_COL)}\u202fbpm, "
                    f"a mean difference of {_fv(row, BIAS_COL)}\u202fbpm, "
                    f"LOA from {_fv(row, L_LOA_COL)} to {_fv(row, U_LOA_COL)}\u202fbpm, "
                    f"and an SD of {_fv(row, SD_COL)}\u202fbpm."
                )

        para = f"{intro}\n{_row_text(best, 1)}"
        if second is not None:
            para += f" {_row_text(second, 2)}"
        paragraphs.append(para)

    return "\n\n".join(paragraphs)


# --------------------------------------------------------------------------------------
# Best-vs-second comparison tables
# --------------------------------------------------------------------------------------

def _build_best_vs_second_tables(
    df: pd.DataFrame,
    *,
    halved: bool,
    accuracy_weight: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build three best-vs-second-best ROI comparison tables.

    For each recording the best ROI is identified, then a *distinct* second-best is
    selected: when the second-ranked candidate is simply the halved (or un-halved)
    sibling of the best ROI it is skipped and the third-ranked candidate is used
    instead; the "Second ROI" entry is then marked with ``*``.

    Returns
    -------
    table1
        Per-recording – three rows per recording:
        *1st (best)*, *2nd* (or *3rd* when halved sibling skipped), and *Abs. diff* showing
        the absolute difference ``|best_metric − second_metric|`` for every metric column.
    table2
        Per *(Approach, Camera, Best ROI)* – mean of ``|best − second|`` absolute metric
        differences across recordings sharing the same best ROI and camera.
        Lists all observed second-best ROIs with counts.
    table3
        Per *(Approach, Best ROI)* – same means aggregated over all cameras,
        with one row per distinct best ROI within each approach.
        Lists all observed second-best ROIs with counts.
    """
    metric_cols = list(_COLUMN_SOURCES.keys())

    def _roi_label(row: pd.Series) -> str:
        return _ROI_DISPLAY.get(str(row["_roi"]), str(row["_roi"]).replace("_", " "))

    def _metric_val(row: pd.Series, out_col: str, h: bool) -> Optional[float]:
        closest_col, halved_col = _COLUMN_SOURCES[out_col]
        v = row.get(halved_col if h else closest_col)
        try:
            fv = float(v)  # type: ignore[arg-type]
            return fv if math.isfinite(fv) else None
        except (TypeError, ValueError):
            return None

    # ---- collect one comparison record per recording ----
    comp_records: list[dict[str, object]] = []

    for _key, group in df.groupby("_rec_key", sort=False):
        candidates = _variant_candidates(group, halved=halved)
        if len(candidates) < 2:
            continue
        order, _scores = _score_and_order(candidates, accuracy_weight=accuracy_weight)

        row_best, halved_best = candidates[order[0]]
        best_roi_label = _roi_label(row_best)

        # Find second: skip the halved/un-halved sibling of best.
        second_cand_idx: Optional[int] = None
        third_used = False
        for rank_pos, cand_idx in enumerate(order[1:], start=2):
            row_cand, halved_cand = candidates[cand_idx]
            is_sibling = (
                str(row_cand["_roi"]) == str(row_best["_roi"])
                and halved_cand != halved_best
            )
            if not is_sibling:
                second_cand_idx = cand_idx
                third_used = rank_pos > 2
                break

        if second_cand_idx is None:
            continue  # only one distinct ROI in this recording

        row_second, halved_second = candidates[second_cand_idx]
        second_roi_label = _roi_label(row_second)

        pf = str(row_best["plots_folder"])
        camera = _detect_camera(pf)
        parsed = _parse_participant_experiment(pf)
        experiment = parsed[1] if parsed else None

        rec: dict[str, object] = {
            "Approach": _approach_label(pf),
            "Camera": _CAMERA_DISPLAY.get(camera, camera),
            "Participant": parsed[0] if parsed else pd.NA,
            "Experiment": (
                None if experiment is None
                else ("before sport" if experiment in (1, 2) else "after sport")
            ),
            "Best ROI": best_roi_label,
            "Best Halved": "yes" if halved_best else "no",
            "Second ROI": second_roi_label + (" *" if third_used else ""),
            "Second Halved": "yes" if halved_second else "no",
            "_approach_order": _APPROACH_ORDER.get(_detect_approach(pf), 99),
            "_camera_order": _CAMERA_ORDER.get(camera, 99),
            "_participant": parsed[0] if parsed else 99,
            "_experiment": experiment if experiment is not None else 99,
            "_best_roi_key": best_roi_label,
            "_second_roi_key": second_roi_label,
        }
        for m in metric_cols:
            b = _metric_val(row_best, m, halved_best)
            s = _metric_val(row_second, m, halved_second)
            rec[f"_b_{m}"] = b if b is not None else pd.NA
            rec[f"_s_{m}"] = s if s is not None else pd.NA
            rec[f"_d_{m}"] = abs(b - s) if (b is not None and s is not None) else pd.NA
        comp_records.append(rec)

    if not comp_records:
        raise ValueError(
            "No recording had \u22652 usable distinct-ROI candidates for "
            "the best-vs-second comparison."
        )

    comp_df = (
        pd.DataFrame.from_records(comp_records)
        .sort_values(
            ["_approach_order", "_camera_order", "_participant", "_experiment"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # ---- Table 1: three rows per recording ----
    t1_rows: list[dict[str, object]] = []
    for _, r in comp_df.iterrows():
        base = {
            "Approach": r["Approach"],
            "Camera": r["Camera"],
            "Participant": r["Participant"],
            "Experiment": r["Experiment"],
        }
        row_b = {**base, "Position": "1st (best)",
                 "ROI": r["Best ROI"], "Halved": r["Best Halved"]}
        row_b.update({m: r[f"_b_{m}"] for m in metric_cols})
        t1_rows.append(row_b)

        second_label = (
            "3rd (halved sibling skipped *)" if str(r["Second ROI"]).endswith(" *")
            else "2nd"
        )
        row_s = {**base, "Position": second_label,
                 "ROI": r["Second ROI"], "Halved": r["Second Halved"]}
        row_s.update({m: r[f"_s_{m}"] for m in metric_cols})
        t1_rows.append(row_s)

        row_d = {**base, "Position": "Abs. diff (1st vs 2nd)", "ROI": "", "Halved": ""}
        row_d.update({m: r[f"_d_{m}"] for m in metric_cols})
        t1_rows.append(row_d)

    t1_id = ["Approach", "Camera", "Participant", "Experiment",
              "Position", "ROI", "Halved"]
    table1 = pd.DataFrame.from_records(t1_rows)[t1_id + metric_cols]
    table1["Participant"] = table1["Participant"].astype("Int64")

    def _roi_counts(series: pd.Series) -> str:
        counts = series.value_counts()
        return "; ".join(f"{roi} ({cnt})" for roi, cnt in counts.items())

    # ---- Table 2: per (Approach, Camera, Best ROI) ----
    t2_rows: list[dict[str, object]] = []
    for (approach, camera, best_roi), grp in comp_df.groupby(
        ["Approach", "Camera", "_best_roi_key"], sort=False
    ):
        r2: dict[str, object] = {
            "Approach": approach,
            "Camera": camera,
            "Best ROI": best_roi,
            "N recordings": len(grp),
            "Second ROI(s)": _roi_counts(grp["_second_roi_key"]),
            "_approach_order": int(grp["_approach_order"].iloc[0]),
            "_camera_order": int(grp["_camera_order"].iloc[0]),
        }
        for m in metric_cols:
            vals = pd.to_numeric(grp[f"_d_{m}"], errors="coerce").dropna()
            r2[m] = float(vals.mean()) if not vals.empty else pd.NA
        t2_rows.append(r2)

    t2_id = ["Approach", "Camera", "Best ROI", "N recordings", "Second ROI(s)"]
    table2 = (
        pd.DataFrame.from_records(t2_rows)
        .sort_values(["_approach_order", "_camera_order", "Best ROI"], kind="stable")
        .reset_index(drop=True)
    )[t2_id + metric_cols]

    # ---- Table 4: per (Approach, Camera, Best ROI) – median ----
    t4_rows: list[dict[str, object]] = []
    for (approach, camera, best_roi), grp in comp_df.groupby(
        ["Approach", "Camera", "_best_roi_key"], sort=False
    ):
        r4: dict[str, object] = {
            "Approach": approach,
            "Camera": camera,
            "Best ROI": best_roi,
            "N recordings": len(grp),
            "Second ROI(s)": _roi_counts(grp["_second_roi_key"]),
            "_approach_order": int(grp["_approach_order"].iloc[0]),
            "_camera_order": int(grp["_camera_order"].iloc[0]),
        }
        for m in metric_cols:
            vals = pd.to_numeric(grp[f"_d_{m}"], errors="coerce").dropna()
            r4[m] = float(vals.median()) if not vals.empty else pd.NA
        t4_rows.append(r4)

    t4_id = ["Approach", "Camera", "Best ROI", "N recordings", "Second ROI(s)"]
    table4 = (
        pd.DataFrame.from_records(t4_rows)
        .sort_values(["_approach_order", "_camera_order", "Best ROI"], kind="stable")
        .reset_index(drop=True)
    )[t4_id + metric_cols]

    # ---- Table 3: per (Approach, Best ROI) ----
    t3_rows: list[dict[str, object]] = []
    for (approach, best_roi), grp in comp_df.groupby(
        ["Approach", "_best_roi_key"], sort=False
    ):
        r3: dict[str, object] = {
            "Approach": approach,
            "Best ROI": best_roi,
            "N recordings": len(grp),
            "Second ROI(s)": _roi_counts(grp["_second_roi_key"]),
            "_approach_order": int(grp["_approach_order"].iloc[0]),
        }
        for m in metric_cols:
            vals = pd.to_numeric(grp[f"_d_{m}"], errors="coerce").dropna()
            r3[m] = float(vals.mean()) if not vals.empty else pd.NA
        t3_rows.append(r3)

    t3_id = ["Approach", "Best ROI", "N recordings", "Second ROI(s)"]
    table3 = (
        pd.DataFrame.from_records(t3_rows)
        .sort_values(["_approach_order", "Best ROI"], kind="stable")
        .reset_index(drop=True)
    )[t3_id + metric_cols]

    return table1, table2, table3, table4


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
    rank_option: Optional[str] = None,
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

    # Resolve accuracy_weight from CONFIG (None = equal weights).
    _cfg_acc_weight = CONFIG.get("ACCURACY_WEIGHT")
    accuracy_weight: Optional[float] = (
        float(_cfg_acc_weight)
        if _cfg_acc_weight is not None
        else None
    )
    if accuracy_weight is not None and not (0.0 < accuracy_weight < 1.0):
        raise ValueError(
            f"ACCURACY_WEIGHT must be None or a float strictly between 0 and 1, "
            f"got {accuracy_weight!r}."
        )

    # Resolve effective rank_option (kwarg > CONFIG["RANK_OPTION"] > CONFIG["RANK"]).
    if rank_option is None:
        cfg_rank_option = CONFIG.get("RANK_OPTION")
        if cfg_rank_option is not None and str(cfg_rank_option).strip():
            rank_option = str(cfg_rank_option).strip().lower()
        else:
            # Legacy RANK bool fallback.
            _rank = bool(CONFIG["RANK"]) if rank is None else rank
            rank_option = "inside_rois" if _rank else None
    elif isinstance(rank_option, bool):
        rank_option = "inside_rois" if rank_option else None
    if rank_option is not None:
        rank_option = rank_option.strip().lower()
        if rank_option not in ("inside_rois", "inside_approach"):
            raise ValueError(
                f"RANK_OPTION must be None, 'inside_rois' or 'inside_approach', "
                f"got {rank_option!r}."
            )

    view = view.strip().lower()
    modus = modus.strip().lower()
    if view not in ("side", "front"):
        raise ValueError(f"VIEW must be 'side' or 'front', got {view!r}.")
    if modus not in ("peaks", "mean"):
        raise ValueError(f"MODUS must be 'peaks' or 'mean', got {modus!r}.")
    if rank_option and modus == "mean":
        raise ValueError("RANK_OPTION is only supported for MODUS='peaks'.")

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
        table = _build_peaks_table(df, halved=halved, rank_option=rank_option, accuracy_weight=accuracy_weight)

    # Rename columns according to _COLUMN_DISPLAY (only entries that differ from their
    # current name have any effect; columns absent from the dict are left unchanged).
    table = table.rename(columns=_COLUMN_DISPLAY)

    # For peaks tables generate a per-approach textual summary, print it,
    # and save it alongside the CSV (same stem, .summary.txt extension).
    if modus == "peaks":
        _peaks_summary = _generate_peaks_summary(table, latex_label=latex_label)
        print("\n[create_tables_for_paper] Peaks summary:\n")
        print(_peaks_summary)
        if output_path is not None:
            _sum_path = Path(str(output_path)).with_suffix(".summary.txt")
            _sum_path.parent.mkdir(parents=True, exist_ok=True)
            _sum_path.write_text(_peaks_summary + "\n", encoding="utf-8")
            print(f"[create_table] Wrote peaks summary to {_sum_path}")

    # Best-vs-second comparison tables (peaks modus only).
    if modus == "peaks" and bool(CONFIG.get("COMPARISON_TO_SECOND_BEST", False)):
        try:
            t1, t2, t3, t4 = _build_best_vs_second_tables(
                df, halved=halved, accuracy_weight=accuracy_weight
            )
            # Apply the same column-display renames as the main table.
            t1 = t1.rename(columns=_COLUMN_DISPLAY)
            t2 = t2.rename(columns=_COLUMN_DISPLAY)
            t3 = t3.rename(columns=_COLUMN_DISPLAY)
            t4 = t4.rename(columns=_COLUMN_DISPLAY)
            if output_path is not None:
                _out = Path(str(output_path))
                _stem = _out.stem
                _dir = _out.parent
                _capt_base = (latex_caption or "").rstrip(".")
                _lbl_base = latex_label or ""
                _subtables = [
                    (
                        "_vs2nd", t1,
                        (_capt_base + ": best vs second-best ROI per recording "
                         "(Abs. diff = |best − second|).") if _capt_base else None,
                        f"{_lbl_base}_vs2nd" if _lbl_base else None,
                    ),
                    (
                        "_vs2nd_by_cam_roi", t2,
                        (_capt_base + ": mean absolute metric diff (|best − second|) "
                         "grouped by approach, camera and best ROI.") if _capt_base else None,
                        f"{_lbl_base}_vs2nd_by_cam_roi" if _lbl_base else None,
                    ),
                    (
                        "_vs2nd_by_approach", t3,
                        (_capt_base + ": mean absolute metric diff (|best − second|) "
                         "grouped by approach and best ROI.") if _capt_base else None,
                        f"{_lbl_base}_vs2nd_by_approach" if _lbl_base else None,
                    ),
                    (
                        "_vs2nd_by_cam_roi_med", t4,
                        (_capt_base + ": median absolute metric diff (|best \u2212 second|) "
                         "grouped by approach, camera and best ROI.") if _capt_base else None,
                        f"{_lbl_base}_vs2nd_by_cam_roi_med" if _lbl_base else None,
                    ),
                ]
                for _sfx, _tbl, _cap, _lbl in _subtables:
                    _finalize_table(
                        _tbl,
                        output_path=_dir / f"{_stem}{_sfx}.csv",
                        round_decimals=round_decimals,
                        latex_output_path=_dir / f"{_stem}{_sfx}.tex",
                        latex_caption=_cap,
                        latex_label=_lbl,
                    )
        except ValueError as _e:
            print(f"[create_tables_for_paper] Warning: comparison-to-second skipped: {_e}")

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

# Columns dropped from the mean table when building mean_range tables
# (Median / RMSE / Range are excluded; only Mean / Max / Min are kept).
_MEAN_RANGE_DROP_COLS: tuple[str, ...] = (
    "Median abs. signal-GT HR diff across ROIs (bpm)",
    "RMSE abs. signal-GT HR diff across ROIs (bpm)",
    "Range abs. signal-GT HR diff across ROIs (bpm)",
    "Median abs. signal-GT HR diff across ROIs, halved (bpm)",
    "RMSE abs. signal-GT HR diff across ROIs, halved (bpm)",
    "Range abs. signal-GT HR diff across ROIs, halved (bpm)",
)


def _build_combined_mean_range_latex(
    latex_resting: str,
    latex_other: str,
    *,
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """Combine the two mean-range table LaTeX strings into one ``table`` float.

    Both groups share a single ``tabular`` environment with a unified two-row
    header (multicolumn grouping + short column names defined once).  A full-width
    separator row labels each group inside the table.
    """

    def _extract_tabular_spec(s: str) -> str:
        """Return the column spec string from ``\\begin{tabular}{...}``."""
        marker = r"\begin{tabular}{"
        start = s.find(marker)
        if start == -1:
            return "|l|l|r|l|l|r|r|r|r|"
        start += len(marker)
        end = s.find("}", start)
        return s[start:end] if end != -1 else "|l|l|r|l|l|r|r|r|r|"

    def _extract_data_rows(s: str) -> str:
        """Return only the data rows (after the column-header hline) of a tabular."""
        start = s.find(r"\begin{tabular}")
        end = s.rfind(r"\end{tabular}")
        if start == -1 or end == -1:
            return ""
        content = s[start: end + len(r"\end{tabular}")]
        # Line layout: [0]=\begin{tabular}{...}, [1]=\hline, [2]=header \\,
        # [3]=\hline, [4..n-1]=data rows + \hline, [-1]=\end{tabular}
        lines = content.split("\n")
        if len(lines) <= 4:
            return ""
        return "\n".join(lines[4:-1])

    col_spec = _extract_tabular_spec(latex_resting)
    # Count column types (l / r / c / p) to determine total column count.
    n_cols = sum(1 for c in col_spec if c in "lrcp")
    # 6 descriptive columns (Approach, Camera, ID, Experiment, Perspective, Ref-HR);
    # remaining columns are metrics.
    n_metrics = n_cols - 6

    data_r = _extract_data_rows(latex_resting)
    data_o = _extract_data_rows(latex_other)

    hr_lo = f"{_MEAN_RANGE_HR_LOW:.0f}"
    hr_hi = f"{_MEAN_RANGE_HR_HIGH:.0f}"

    # ---- Two-row header (multicolumn grouping + short column names) ----
    if n_metrics > 3:
        # Halved columns present (Mean/Max/Min + Mean_h/Max_h/Min_h).
        n_halved = n_metrics - 3
        header_mc = (
            r"\multicolumn{6}{|c|}{Experimental Details} &" + "\n"
            r"\multicolumn{3}{c|}{Absolute signal-reference HR difference metrics (bpm)} &" + "\n"
            rf"\multicolumn{{{n_halved}}}{{c|}}{{Halved (bpm)}} \\"
        )
        header_cols = (
            r"Approach & Camera & ID & Experiment & Perspective & "
            r"Reference mean HR (bpm) & Mean & Max & Min"
            + " & Mean & Max & Min" * (n_halved // 3)
            + r" \\"
        )
    else:
        header_mc = (
            r"\multicolumn{6}{|c|}{Experimental Details} &" + "\n"
            r"\multicolumn{3}{c|}{Absolute signal-reference HR difference metrics (bpm)} \\"
        )
        header_cols = (
            r"Approach & Camera & ID & Experiment & Perspective & "
            r"Reference mean HR (bpm) & Mean & Max & Min \\"
        )

    # ---- Full-width group separator rows ----
    sep_r = (
        rf"\multicolumn{{{n_cols}}}{{|l|}}{{\textbf{{(a) Resting group "
        rf"(HR {hr_lo}--{hr_hi}\,bpm, non-Lepton, front view)}}}} \\"
    )
    sep_o = rf"\multicolumn{{{n_cols}}}{{|l|}}{{\textbf{{(b) Other recordings}}}} \\"

    # data_r already ends with \hline (one after each data row), so sep_o is
    # preceded by that final \hline without needing an extra one.
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{" + col_spec + "}",
        r"\hline",
        header_mc,
        r"\hline",
        header_cols,
        r"\hline",
        sep_r,
        r"\hline",
        data_r,
        sep_o,
        r"\hline",
        data_o,
        r"\end{tabular}",
    ]
    if caption is not None:
        lines.append(r"\caption{" + caption + "}")
    if label is not None:
        lines.append(r"\label{" + str(label) + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


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
    latex_output_path_combined: Optional[str | Path] = None,
    latex_caption_combined: Optional[str] = None,
    latex_label_combined: Optional[str] = None,
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

    # Drop Median / RMSE / Range columns: mean_range tables only show Mean / Max / Min.
    full_table = full_table.drop(
        columns=[c for c in _MEAN_RANGE_DROP_COLS if c in full_table.columns]
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

    # Build and optionally write the combined two-part LaTeX file.
    (_, latex_r), (_, latex_o) = result_resting, result_other
    latex_combined = _build_combined_mean_range_latex(
        latex_r,
        latex_o,
        caption=latex_caption_combined,
        label=latex_label_combined,
    )
    if latex_output_path_combined is not None:
        latex_output_path_combined = Path(str(latex_output_path_combined))
        latex_output_path_combined.parent.mkdir(parents=True, exist_ok=True)
        latex_output_path_combined.write_text(latex_combined + "\n", encoding="utf-8")
        print(f"[create_table] Wrote combined mean-range LaTeX table to {latex_output_path_combined}")

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
    rank_option_cfg = str(config.get("RANK_OPTION") or "").strip().lower()
    if rank_option_cfg == "inside_approach":
        parts.append("ranked_global")
    elif rank_option_cfg == "inside_rois" or (not rank_option_cfg and config.get("RANK")):
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
    rank_option_cfg = str(config.get("RANK_OPTION") or "").strip().lower()
    _rank_active = rank_option_cfg in ("inside_rois", "inside_approach") or (
        not rank_option_cfg and bool(config.get("RANK"))
    )
    if config["MODUS"] in ("mean", "mean_range"):
        body = (
            f"Across-ROIs spread of the per-ROI mean signal-HR bias for the {view_text} "
            f"ROIs, {approach_text}"
        )
    elif rank_option_cfg == "inside_approach":
        body = (
            f"All {view_text} ROIs ranked globally within each approach by the composite "
            f"criterion, {approach_text}"
        )
    elif _rank_active:
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
            latex_output_path_combined=DEFAULT_OUTPUT_DIR / f"{slug}_combined.tex",
            latex_caption_combined=caption_base + ".",
            latex_label_combined=f"tab:{slug}_combined",
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
