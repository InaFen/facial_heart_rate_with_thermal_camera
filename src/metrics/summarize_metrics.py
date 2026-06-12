"""Reduce the aggregate metrics CSV to the single best ROI per recording.

Input
-----
``saved_results/all_npz_metrics.csv`` (one row per *recording x ROI*, produced by
``metric_pipeline.py``). Several ROIs (forehead, glabella, left/right forehead, whole
face, dlib, tracker, manual rectangle, ...) are evaluated for the *same* recording. Two
rows belong to the same recording when their ``plots_folder`` is identical except for
the ROI token, e.g.::

    .../lepton/metric_results/Ina1_interpolated_forehead_30_00-50_00s
    .../lepton/metric_results/Ina1_interpolated_dlib_30_00-50_00s

Both describe individual *Ina1* (nair_et_al, lepton) - only the ROI differs.

Output
------
``saved_results/all_npz_metrics_best_roi.csv``: one row per recording, namely the row
of the ROI that scored best on the *closest*-matching metrics. The original columns are
preserved and four bookkeeping columns are prepended:

    roi                 the selected ROI token
    num_rois_in_group   how many ROIs competed for this recording
    selection_metric    which criterion decided the winner
    selection_score     the winner's score (always "lower is better"; for the
                        ``composite`` criterion this is the mean rank across sub-metrics)

How is the "best" ROI decided?
------------------------------
The ``closest`` metrics compare the predicted instantaneous heart rate against the
ground truth after matching every predicted beat to its nearest ground-truth beat. Each
metric captures a *different* aspect of quality, so there is no single universally
correct one:

    mae       Mean absolute error [bpm]. Average magnitude of the per-beat error.
              Most directly interpretable and folds accuracy + scatter into one
              number. Lower is better. -> good default.
    rmse      Root-mean-square error [bpm]. Like MAE but squares the errors first, so
              it punishes the occasional large blunder harder (rmse >= mae always; a
              big gap signals outliers). Lower is better.
    r2        Coefficient of determination. Fraction of ground-truth-HR variance the
              prediction explains. Higher is better - BUT over a short 20 s window the
              true HR barely varies, so r2 is numerically unstable here (often negative)
              and is a poor primary criterion.
    abs_bias  |Bland-Altman mean difference| [bpm]. Systematic over/under-estimation.
              Near zero is better, but a method can have ~0 bias yet huge scatter
              (over- and under-shoots cancelling). Lower is better.
    sd_diff   Bland-Altman SD of the differences [bpm] == precision/repeatability.
              loa_width Limits-of-agreement width = 2 * 1.96 * sd_diff. Lower is better.
    ks_pvalue Two-sample KS p-value (distribution similarity). Higher = more similar.
              Weak ranking signal.

The selection criterion is hard-coded to ``composite`` (it is not user selectable):
within each recording the candidate ROIs are *ranked* on each of {mae, rmse, abs_bias,
sd_diff} (rank 1 = best), the ranks are averaged, and the lowest mean rank wins. Ranking
is scale-free, so no single metric dominates just because it has larger units, which
avoids relying on any one noisy metric. (``mae`` alone is the most interpretable single
criterion and the composite is built around it, so the two usually agree.)

Usage
-----
    python -m src.metrics.summarize_metrics             # always uses the composite criterion
    python -m src.metrics.summarize_metrics --verbose   # also print the per-recording ranking
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

DEFAULT_INPUT_CSV = Path("saved_results/all_npz_metrics.csv")
DEFAULT_OUTPUT_CSV = Path("saved_results/all_npz_metrics_best_roi.csv")

# Known ROI tokens, longest/compound first so e.g. "left_forehead" is not shortened to
# "forehead". The leaf of a plots_folder looks like "<recording>_<roi>_<time-segment>".
ROI_VOCABULARY: Tuple[str, ...] = (
    "left_forehead",
    "right_forehead",
    "whole_face",
    "manual_rectangle",
    "forehead",
    "glabella",
    "dlib",
    "tracker",
)

# Trailing time-segment token, e.g. "_30_00-50_00s".
_TIME_SEGMENT_RE = re.compile(r"_(\d+_\d+-\d+_\d+s)$")

# Criterion used to pick the winner. Hard-coded to "composite" (not user selectable):
# within each recording the candidate ROIs are ranked on each sub-metric and the lowest
# mean rank wins. "lower is better" is enforced internally (metrics where higher is
# better are negated), so selection is always a simple minimisation.
SELECTION_METRIC = "composite"
AVAILABLE_METRICS: Tuple[str, ...] = (
    "mae",
    "rmse",
    "r2",
    "abs_bias",
    "sd_diff",
    "loa_width",
    #"ks_pvalue",
    "composite",
)
# Sub-metrics averaged (by rank) for the scale-free "composite" criterion. r2 is left
# out on purpose - it is unstable for near-constant HR over a 20 s window.
COMPOSITE_METRICS: Tuple[str, ...] = ("mae", "rmse", "abs_bias", "sd_diff")

# Columns prepended to every output row.
HELPER_FIELDNAMES: Tuple[str, ...] = (
    "roi",
    "num_rois_in_group",
    "selection_metric",
    "selection_score",
)


# --------------------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------------------


def _to_float(value: object) -> Optional[float]:
    """Parse ``value`` to a finite float, or ``None`` if blank/invalid/non-finite."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def split_plots_folder(plots_folder: str) -> Tuple[str, str, str]:
    """Split a ``plots_folder`` into ``(group_key, roi, recording)``.

    ``group_key`` is the folder with the ROI token removed, so all ROIs of one recording
    share it. If the ROI cannot be identified the whole folder becomes its own group and
    the ROI is reported as ``"unknown"`` (so no row is silently dropped).
    """
    head, _, leaf = plots_folder.rpartition("/")
    match = _TIME_SEGMENT_RE.search(leaf)
    if match is None:
        return plots_folder, "unknown", leaf
    time_segment = match.group(1)
    base = leaf[: match.start()]
    for roi in ROI_VOCABULARY:
        if base.endswith("_" + roi):
            recording = base[: -(len(roi) + 1)]
            group_key = f"{head}/{recording}_{time_segment}" if head else f"{recording}_{time_segment}"
            return group_key, roi, recording
    # Time segment recognised but ROI vocabulary did not match -> keep as own group.
    return plots_folder, "unknown", base


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def selection_score(row: Dict[str, str], metric: str) -> Optional[float]:
    """Return a "lower is better" score for ``row`` under ``metric`` (``None`` if N/A).

    Criteria where higher is better (``r2``, ``ks_pvalue``) are negated so callers can
    always minimise. The ``composite`` criterion is handled separately (it needs the
    whole group) and is rejected here.
    """
    if metric == "mae":
        return _to_float(row.get("metrics_closest_mae"))
    if metric == "rmse":
        return _to_float(row.get("metrics_closest_rmse"))
    if metric == "abs_bias":
        bias = _to_float(row.get("ba_stats_closest_mean_diff"))
        return abs(bias) if bias is not None else None
    if metric in ("sd_diff", "loa_width"):
        sd = _to_float(row.get("ba_stats_closest_sd_diff"))
        if sd is None:
            return None
        # loa width = 2 * 1.96 * sd_diff is monotonic in sd_diff (same ranking).
        return (2.0 * 1.96 * sd) if metric == "loa_width" else sd
    if metric == "r2":
        r2 = _to_float(row.get("metrics_closest_r2"))
        return -r2 if r2 is not None else None  # maximise -> negate
    if metric == "ks_pvalue":
        pvalue = _to_float(row.get("metrics_closest_ks_pvalue"))
        return -pvalue if pvalue is not None else None  # maximise -> negate
    raise ValueError(f"Unknown selection metric: {metric!r}")


def _average_ranks(values: List[Optional[float]]) -> List[float]:
    """Average competition ranks (1 = best/lowest); ``None`` shares the worst rank."""
    n = len(values)
    order = sorted(
        range(n),
        key=lambda i: (values[i] is None, values[i] if values[i] is not None else 0.0),
    )
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = ((i + 1) + (j + 1)) / 2.0  # mean of the 1-based positions i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _sort_key(score: Optional[float]) -> float:
    """Map a "lower is better" score to a sortable float (``None`` -> +inf = worst)."""
    return score if score is not None else math.inf


def group_scores(group_rows: List[Dict[str, str]], metric: str) -> List[Optional[float]]:
    """Per-ROI "lower is better" score for every row in a recording group.

    For the ``composite`` criterion this is the mean competition rank across
    ``COMPOSITE_METRICS`` (scale-free); for every other criterion it is simply
    ``selection_score`` evaluated on each row. Routing the composite case through here
    keeps callers (selection and verbose display) from calling ``selection_score`` with
    ``"composite"`` directly, which it intentionally rejects.
    """
    if metric == "composite":
        per_metric_ranks = [
            _average_ranks([selection_score(r, sub) for r in group_rows])
            for sub in COMPOSITE_METRICS
        ]
        return [
            sum(per_metric_ranks[m][i] for m in range(len(COMPOSITE_METRICS)))
            / len(COMPOSITE_METRICS)
            for i in range(len(group_rows))
        ]
    return [selection_score(row, metric) for row in group_rows]


def select_best_index(group_rows: List[Dict[str, str]], metric: str) -> Tuple[int, Optional[float]]:
    """Index of the winning ROI row in ``group_rows`` and its reported score.

    Ties are broken deterministically by mae, then rmse, then ``plots_folder`` so the
    result is reproducible.
    """
    scores = group_scores(group_rows, metric)

    def key(i: int) -> tuple:
        return (
            _sort_key(scores[i]),
            _sort_key(selection_score(group_rows[i], "mae")),
            _sort_key(selection_score(group_rows[i], "rmse")),
            group_rows[i].get("plots_folder", ""),
        )

    best = min(range(len(group_rows)), key=key)
    return best, scores[best]


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def summarize_best_rois(
    input_csv: str | Path = DEFAULT_INPUT_CSV,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    metric: str = SELECTION_METRIC,
    verbose: bool = False,
) -> Path:
    """Write ``output_csv`` keeping only the best-ROI row per recording.

    Returns the output path. Raises ``ValueError`` for an unknown ``metric`` and
    ``FileNotFoundError`` if the input CSV is missing.
    """
    if metric not in AVAILABLE_METRICS:
        raise ValueError(
            f"metric must be one of {AVAILABLE_METRICS}, got {metric!r}"
        )
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input metrics CSV not found: {input_csv}")

    with open(input_csv, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        original_fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader]

    if "plots_folder" not in original_fieldnames:
        raise ValueError(
            "Input CSV has no 'plots_folder' column - cannot group ROIs by recording."
        )

    # Group rows by recording (plots_folder with the ROI token removed). Preserve first
    # appearance order so the output is stable / diff-friendly.
    groups: "defaultdict[str, List[Dict[str, str]]]" = defaultdict(list)
    group_order: List[str] = []
    roi_of: Dict[int, str] = {}
    for row in rows:
        group_key, roi, _recording = split_plots_folder(row.get("plots_folder", ""))
        if group_key not in groups:
            group_order.append(group_key)
        groups[group_key].append(row)
        roi_of[id(row)] = roi

    selected: List[Dict[str, object]] = []
    win_counter: "Counter[Tuple[str, str, str]]" = Counter()  # (approach, camera, roi)
    multi_win_counter: "Counter[Tuple[str, str, str]]" = Counter()
    for group_key in group_order:
        group_rows = groups[group_key]
        best_index, score = select_best_index(group_rows, metric)
        best_row = group_rows[best_index]
        best_roi = roi_of[id(best_row)]

        out_row: Dict[str, object] = {
            "roi": best_roi,
            "num_rois_in_group": len(group_rows),
            "selection_metric": metric,
            "selection_score": "" if score is None else round(score, 6),
        }
        out_row.update(best_row)
        selected.append(out_row)

        approach, camera = _approach_camera(best_row.get("plots_folder", ""))
        win_counter[(approach, camera, best_roi)] += 1
        if len(group_rows) > 1:
            multi_win_counter[(approach, camera, best_roi)] += 1

        if verbose and len(group_rows) > 1:
            scores = group_scores(group_rows, metric)
            ranked = sorted(
                ((roi_of[id(group_rows[i])], scores[i]) for i in range(len(group_rows))),
                key=lambda t: _sort_key(t[1]),
            )
            label = "mean rank" if metric == "composite" else metric
            shown = ", ".join(
                f"{roi}={'n/a' if sc is None else round(sc, 3)}" for roi, sc in ranked
            )
            print(f"  {group_key}\n    winner={best_roi} | {label}: {shown}")

    fieldnames = list(HELPER_FIELDNAMES) + original_fieldnames
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames, extrasaction="ignore", restval=""
        )
        writer.writeheader()
        writer.writerows(selected)

    _print_summary(
        input_rows=len(rows),
        groups=len(group_order),
        metric=metric,
        win_counter=win_counter,
        multi_win_counter=multi_win_counter,
        output_csv=output_csv,
    )
    return output_csv


def _approach_camera(plots_folder: str) -> Tuple[str, str]:
    """Best-effort ``(approach, camera)`` from a ``saved_results/<app>/results/<cam>/...`` path."""
    parts = plots_folder.split("/")
    approach = parts[1] if len(parts) > 1 else "?"
    camera = parts[3] if len(parts) > 3 else "?"
    return approach, camera


def _print_summary(
    input_rows: int,
    groups: int,
    metric: str,
    win_counter: "Counter[Tuple[str, str, str]]",
    multi_win_counter: "Counter[Tuple[str, str, str]]",
    output_csv: Path,
) -> None:
    """Print how many rows were reduced and which ROI won most often per approach/camera."""
    print(
        f"[summarize_metrics] {input_rows} rows -> {groups} recordings "
        f"(best ROI by '{metric}') -> {output_csv}"
    )

    def render(counter: "Counter[Tuple[str, str, str]]", title: str) -> None:
        if not counter:
            print(f"  {title}: (none)")
            return
        print(f"  {title}:")
        per_app_cam: "defaultdict[Tuple[str, str], Counter]" = defaultdict(Counter)
        for (approach, camera, roi), count in counter.items():
            per_app_cam[(approach, camera)][roi] += count
        for (approach, camera) in sorted(per_app_cam):
            tally = per_app_cam[(approach, camera)]
            ordered = ", ".join(
                f"{roi}:{cnt}" for roi, cnt in tally.most_common()
            )
            print(f"    {approach}/{camera}: {ordered}")

    render(multi_win_counter, "ROI wins among recordings with >1 ROI (the real contest)")
    render(win_counter, "ROI wins across all recordings (incl. single-ROI recordings)")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Keep only the best-ROI row per recording from the aggregate metrics CSV.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV, help="Input metrics CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the per-recording ROI ranking for multi-ROI recordings.",
    )
    args = parser.parse_args(argv)
    # The selection criterion is intentionally hard-coded (see SELECTION_METRIC); it is
    # not exposed as a CLI option.
    summarize_best_rois(
        input_csv=args.input,
        output_csv=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
