"""Build presentable metrics tables from the aggregate per-recording CSV.

This module reads the aggregate metrics file written by ``metric_pipeline.py``
(``saved_results/all_npz_metrics.csv``) and produces compact, human-readable tables for
the thesis.

``create_table_side_view_own_approach`` builds the table for the "own approach"
(``new_approach``) side-view / temple / manual ROI recordings and
``create_table_manual_front_view_own_approach`` the equivalent table for the front-view /
face / manual ROI recordings. In the side-view table the MWIR camera reports the *halved*
variant metrics (every second predicted peak dropped, ``*_closest_halved_*``), because
that variant is the meaningful one for those recordings, while Vario and Lepton report the
regular ``*_closest_*`` metrics. The front-view table uses the regular ``*_closest_*``
metrics for every camera (including MWIR).

``create_table_manual_all_approaches`` builds a single combined table that, unlike the two
own-approach builders above, covers the manual-ROI recordings of *every* approach (Nair et
al., Yang et al. and the own approach) and reports both perspectives at once (a
``Perspective`` column tells the side-view temple and front-view face recordings apart). As
in the side-view table the MWIR side-view rows use the *halved* metrics where those exist
(only the own approach's MWIR manual recordings have a halved variant); every other row --
including the Nair et al./Yang et al. MWIR rows -- uses the regular ``*_closest_*`` metrics.
``create_table_side_view_all_approaches`` and ``create_table_front_view_all_approaches``
produce the same content split into two single-perspective tables (one with all side-view
temple recordings, one with all front-view face recordings) instead of one combined table.

``create_table_best_roi_all_approaches`` builds a table with the same metric columns (always
from the regular ``*_closest_*`` columns) for *every* recording across all approaches (Nair
et al., Yang et al. and the own approach), reading the best-ROI summary CSV
(``saved_results/all_npz_metrics_best_roi.csv``) and reporting the ROI selected there.
Recordings whose selected ROI is the side-view ``manual_rectangle`` are dropped (those
side-view temple experiments are reported separately in the own-approach tables); the
front-view ``manual_rectangle_front_view`` recordings are kept.

None of these tables include r² (it is unstable over the short comparison window and is not
one of the metrics used to select the best ROI).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:  # normal package import (python -m src.metrics.create_table)
    from src.metrics.summarize_metrics import group_scores
except ModuleNotFoundError:  # allow running this file directly
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.metrics.summarize_metrics import group_scores

# Repository root: src/metrics/create_table.py -> parents[2] == project root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = REPO_ROOT / "saved_results" / "all_npz_metrics.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "saved_results" / "table_side_view_own_approach.csv"
DEFAULT_OUTPUT_TEX = REPO_ROOT / "saved_results" / "table_side_view_own_approach.tex"
DEFAULT_OUTPUT_CSV_FRONT = (
    REPO_ROOT / "saved_results" / "table_manual_front_view_own_approach.csv"
)
DEFAULT_OUTPUT_TEX_FRONT = (
    REPO_ROOT / "saved_results" / "table_manual_front_view_own_approach.tex"
)
DEFAULT_OUTPUT_CSV_MANUAL_ALL = (
    REPO_ROOT / "saved_results" / "table_manual_all_approaches.csv"
)
DEFAULT_OUTPUT_TEX_MANUAL_ALL = (
    REPO_ROOT / "saved_results" / "table_manual_all_approaches.tex"
)
DEFAULT_OUTPUT_CSV_SIDE_ALL = (
    REPO_ROOT / "saved_results" / "table_side_view_all_approaches.csv"
)
DEFAULT_OUTPUT_TEX_SIDE_ALL = (
    REPO_ROOT / "saved_results" / "table_side_view_all_approaches.tex"
)
DEFAULT_OUTPUT_CSV_FRONT_ALL = (
    REPO_ROOT / "saved_results" / "table_front_view_all_approaches.csv"
)
DEFAULT_OUTPUT_TEX_FRONT_ALL = (
    REPO_ROOT / "saved_results" / "table_front_view_all_approaches.tex"
)
DEFAULT_OUTPUT_CSV_SIDE_TEMPLE = (
    REPO_ROOT / "saved_results" / "table_side_view_temple_all_approaches.csv"
)
DEFAULT_OUTPUT_TEX_SIDE_TEMPLE = (
    REPO_ROOT / "saved_results" / "table_side_view_temple_all_approaches.tex"
)
DEFAULT_OUTPUT_CSV_SIDE_BEST = (
    REPO_ROOT / "saved_results" / "table_side_view_best_all_approaches.csv"
)
DEFAULT_OUTPUT_TEX_SIDE_BEST = (
    REPO_ROOT / "saved_results" / "table_side_view_best_all_approaches.tex"
)
DEFAULT_INPUT_BEST_ROI_CSV = (
    REPO_ROOT / "saved_results" / "all_npz_metrics_best_roi.csv"
)
DEFAULT_OUTPUT_CSV_ALL = (
    REPO_ROOT / "saved_results" / "table_best_roi_all_approaches.csv"
)
DEFAULT_OUTPUT_TEX_ALL = (
    REPO_ROOT / "saved_results" / "table_best_roi_all_approaches.tex"
)

# Cameras whose metrics come from the regular "closest" columns. Every other camera
# (i.e. mwir) uses the every-second-peak-dropped "closest_halved" columns instead.
_CLOSEST_ONLY_CAMERAS = ("vario", "lepton")
_CAMERA_ORDER = {"mwir": 0, "vario": 1, "lepton": 2}
# Camera folder token -> display name shown in the tables.
_CAMERA_DISPLAY = {"mwir": "ImageIR", "vario": "VarioCam", "lepton": "Lepton"}

# Approach folder name -> display label and sort order ("own approach" == new_approach).
_APPROACH_LABELS = {
    "nair_et_al": "Nair et al.",
    "yang_et_al": "Yang et al.",
    "new_approach": "own approach",
}
_APPROACH_ORDER = {"nair_et_al": 0, "yang_et_al": 1, "new_approach": 2}

# Output column -> (closest column, closest_halved column). Column order: the per-moment
# HR-agreement fraction first, then the error metrics (metrics_*) and finally the
# Bland-Altman stats (ba_stats_*). r² is intentionally omitted from every table (it is not
# one of the COMPOSITE_METRICS used to pick the best ROI and is unstable over the short
# comparison window).
_COLUMN_SOURCES: dict[str, tuple[str, str]] = {
    "fraction where signal +- GT <= 5 bpm": (
        "hr_agreement_fraction",
        "hr_agreement_fraction_halved",
    ),
    "rmse": ("metrics_closest_rmse", "metrics_closest_halved_rmse"),
    "mae": ("metrics_closest_mae", "metrics_closest_halved_mae"),
    "Bland-Altman mean difference": (
        "ba_stats_closest_mean_diff",
        "ba_stats_closest_halved_mean_diff",
    ),
    "Bland-Altman upper limit of agreement": (
        "ba_stats_closest_loa_high",
        "ba_stats_closest_halved_loa_high",
    ),
    "Bland-Altman lower limit of agreement": (
        "ba_stats_closest_loa_low",
        "ba_stats_closest_halved_loa_low",
    ),
    "Bland-Altman SD of differences": (
        "ba_stats_closest_sd_diff",
        "ba_stats_closest_halved_sd_diff",
    ),
}


def _detect_camera(plots_folder: str) -> str:
    """Return 'mwir' / 'vario' / 'lepton' parsed from a plots_folder path ('unknown' otherwise)."""
    path = str(plots_folder).replace("\\", "/")
    for camera in ("mwir", "vario", "lepton"):
        if f"/{camera}/" in path:
            return camera
    for camera in ("mwir", "vario", "lepton"):
        if camera in path:
            return camera
    return "unknown"


def _detect_approach(plots_folder: str) -> str:
    """Return 'nair_et_al' / 'yang_et_al' / 'new_approach' from a path ('unknown' otherwise)."""
    path = str(plots_folder).replace("\\", "/")
    for approach in _APPROACH_LABELS:
        if f"/{approach}/" in path or path.startswith(f"{approach}/"):
            return approach
    for approach in _APPROACH_LABELS:
        if approach in path:
            return approach
    return "unknown"


def _approach_label(plots_folder: str) -> str:
    """Display label ('Nair et al.' / 'Yang et al.' / 'own approach') for a plots_folder."""
    return _APPROACH_LABELS.get(_detect_approach(plots_folder), "unknown")


def _parse_participant_experiment(plots_folder: str) -> Optional[tuple[int, int]]:
    """Parse ``(participant_number, experiment_number)`` from a plots_folder path.

    Participant: ``Ina`` -> 1, ``Julian`` -> 2, ``Paulo`` -> 3 (Paulo also appears as
    ``P1``..``P4`` for the Vario camera). The experiment number (1..4) is taken from the
    4th underscore-separated token for MWIR stems (``Ina_20_4_2_...``,
    ``Julian_31_03_1_Ascii_...``) and from the trailing digit of the leading token for the
    Vario/Lepton stems (``Ina2_...``, ``paulo3_...``, ``P3_...``). Returns ``None`` when the
    stem cannot be parsed.
    """
    path = str(plots_folder).replace("\\", "/")
    stem = path.rstrip("/").split("/")[-1].lower()

    if "ina" in stem:
        participant = 1
    elif "julian" in stem:
        participant = 2
    elif "paulo" in stem or re.match(r"^p\d", stem):
        participant = 3
    else:
        return None

    try:
        if _detect_camera(path) == "mwir":
            experiment = int(stem.split("_")[3])
        else:
            match = re.match(r"^[a-z]+(\d+)", stem)
            experiment = int(match.group(1))  # type: ignore[union-attr]
    except (IndexError, ValueError, AttributeError):
        return None

    if experiment not in (1, 2, 3, 4):
        return None
    return participant, experiment


def _experiment_label(
    plots_folder: str, view: str = "side view", region: str = "temple"
) -> Optional[str]:
    """Return 'Participant x, {view}, {region}, manual, before/after sport' (or ``None``).

    ``view``/``region`` describe the ROI: "side view"/"temple" for the side-view temple
    recordings and "front view"/"face" for the front-view face ones. Experiments 1 and 2
    are *before sport*, experiments 3 and 4 are *after sport*.
    """
    parsed = _parse_participant_experiment(plots_folder)
    if parsed is None:
        return None
    participant, experiment = parsed
    phase = "before" if experiment in (1, 2) else "after"
    return f"Participant {participant}, {view}, {region}, manual, {phase} sport"


def _participant_sport_label(plots_folder: str) -> Optional[str]:
    """Return 'Participant x, experiment y, before/after sport' (or ``None``).

    Used for the all-approaches table. Unlike :func:`_experiment_label` this omits the
    view / region / manual tokens, because that table reports the ROI in its own column
    and spans many ROIs. The experiment number (1..4) is included so each recording gets a
    unique label: two experiments share a phase (experiments 1 and 2 are *before sport*,
    experiments 3 and 4 are *after sport*), so the phase alone would label two rows alike.
    """
    parsed = _parse_participant_experiment(plots_folder)
    if parsed is None:
        return None
    participant, experiment = parsed
    phase = "before" if experiment in (1, 2) else "after"
    return f"Participant {participant}, experiment {experiment}, {phase} sport"


# Trailing time-segment token of a plots_folder leaf, e.g. "_30_00-50_00s".
_TIME_SEGMENT_RE = re.compile(r"_\d+_\d+-\d+_\d+s$")


def _manual_roi_kind(plots_folder: str) -> Optional[str]:
    """Classify a manual_rectangle plots_folder leaf precisely.

    Returns ``"temple"`` for the plain ``manual_rectangle`` ROI, ``"forehead_bg"`` for the
    forehead & background ``manual_rectangle_forehead`` ROI and ``"front_view"`` for
    ``manual_rectangle_front_view``. Every other ROI -- including ``manual_rectangle_temp``
    -- yields ``None``. Detection is exact (it matches the ROI token right before the
    trailing time segment), so ``manual_rectangle_forehead`` is never mistaken for the
    temple ``manual_rectangle``.
    """
    leaf = str(plots_folder).replace("\\", "/").rstrip("/").split("/")[-1]
    match = _TIME_SEGMENT_RE.search(leaf)
    base = leaf[: match.start()] if match else leaf
    if base.endswith("_manual_rectangle_front_view"):
        return "front_view"
    if base.endswith("_manual_rectangle_forehead"):
        return "forehead_bg"
    if base.endswith("_manual_rectangle"):
        return "temple"
    return None


def _recording_key(plots_folder: str) -> tuple[str, str, str]:
    """Return an ``(approach, camera, recording_stem)`` key shared by a recording's ROIs.

    The ROI token is stripped from the leaf so the temple (``manual_rectangle``) and the
    forehead & background (``manual_rectangle_forehead``) rows of the *same* side-view
    recording map to the same key and therefore compete against each other.
    """
    leaf = str(plots_folder).replace("\\", "/").rstrip("/").split("/")[-1]
    match = _TIME_SEGMENT_RE.search(leaf)
    base = leaf[: match.start()] if match else leaf
    for roi in (
        "manual_rectangle_front_view",
        "manual_rectangle_forehead",
        "manual_rectangle",
    ):
        if base.endswith("_" + roi):
            base = base[: -(len(roi) + 1)]
            break
    return _detect_approach(plots_folder), _detect_camera(plots_folder), base


# LaTeX escaping for table cells/headers. The per-character map escapes the LaTeX
# special characters; the symbol list then converts the few math-like tokens that appear
# in the column names (``r²``, ``+-``, ``<=``) into proper LaTeX *after* escaping, so the
# braces / dollar signs they introduce are kept literally instead of being re-escaped.
_LATEX_SPECIAL_CHARS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_LATEX_SYMBOL_TOKENS = (
    ("²", r"\textsuperscript{2}"),
    ("±", r"$\pm$"),
    ("+-", r"$\pm$"),
    ("<=", r"$\leq$"),
    (">=", r"$\geq$"),
)


def _latex_escape(value: object) -> str:
    """Escape a cell/header for LaTeX (and turn ``r²``, ``+-``, ``<=`` into proper symbols)."""
    escaped = "".join(_LATEX_SPECIAL_CHARS.get(ch, ch) for ch in str(value))
    for token, replacement in _LATEX_SYMBOL_TOKENS:
        escaped = escaped.replace(token, replacement)
    return escaped


def _dataframe_to_latex_full_borders(
    df: pd.DataFrame,
    *,
    float_format: str = "%.2f",
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """Render ``df`` as a fully-bordered LaTeX ``tabular`` (vertical + horizontal lines).

    Every column is separated by a vertical rule and every row -- including the header --
    is followed by an ``\\hline``, i.e. a complete grid ("all borders"). Text columns are
    left-aligned, numeric columns right-aligned. When ``caption`` or ``label`` is given the
    ``tabular`` is wrapped in a ``table`` float with ``\\centering``.
    """
    alignments = [
        "r" if pd.api.types.is_numeric_dtype(dtype) else "l" for dtype in df.dtypes
    ]
    col_format = "|" + "|".join(alignments) + "|"

    lines = [r"\begin{tabular}{" + col_format + "}", r"\hline"]
    lines.append(" & ".join(_latex_escape(c) for c in df.columns) + r" \\")
    lines.append(r"\hline")
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, (float, np.floating)):
                cells.append(float_format % float(value))
            elif isinstance(value, (int, np.integer)):
                cells.append(str(int(value)))
            else:
                cells.append(_latex_escape(value))
        lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    tabular = "\n".join(lines)

    if caption is None and label is None:
        return tabular

    wrapped = [r"\begin{table}[ht]", r"\centering", tabular]
    if caption is not None:
        wrapped.append(r"\caption{" + _latex_escape(caption) + "}")
    if label is not None:
        wrapped.append(r"\label{" + str(label) + "}")
    wrapped.append(r"\end{table}")
    return "\n".join(wrapped)


def _finalize_table(
    table: pd.DataFrame,
    *,
    output_path: Optional[str | Path],
    round_decimals: Optional[int],
    latex_output_path: Optional[str | Path],
    latex_caption: Optional[str],
    latex_label: Optional[str],
) -> tuple[pd.DataFrame, str]:
    """Round, optionally write the CSV, and build / optionally write the bordered LaTeX.

    Shared tail for every table builder: rounds the numeric columns, writes ``output_path``
    (CSV) and ``latex_output_path`` (``.tex``) when given, and returns
    ``(table, latex_table)``.
    """
    if round_decimals is not None:
        numeric_cols = table.select_dtypes(include="number").columns
        table[numeric_cols] = table[numeric_cols].round(round_decimals)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
        print(f"[create_table] Wrote {len(table)} rows to {output_path}")

    float_format = f"%.{int(round_decimals)}f" if round_decimals is not None else "%g"
    latex_table = _dataframe_to_latex_full_borders(
        table, float_format=float_format, caption=latex_caption, label=latex_label
    )
    if latex_output_path is not None:
        latex_output_path = Path(latex_output_path)
        latex_output_path.parent.mkdir(parents=True, exist_ok=True)
        latex_output_path.write_text(latex_table + "\n", encoding="utf-8")
        print(f"[create_table] Wrote LaTeX table to {latex_output_path}")

    return table, latex_table


def _create_own_approach_manual_table(
    csv_path: str | Path,
    output_path: Optional[str | Path],
    *,
    front_view: bool,
    view_label: str,
    region_label: str,
    use_halved_for_mwir: bool,
    include_camera: bool,
    round_decimals: Optional[int],
    latex_output_path: Optional[str | Path],
    latex_caption: Optional[str],
    latex_label: Optional[str],
) -> tuple[pd.DataFrame, str]:
    """Shared builder for the new_approach manual-ROI metrics tables.

    Keeps only ``new_approach`` ``manual_rectangle`` rows; ``front_view`` selects the
    ``manual_rectangle_front_view`` recordings (front view / face) when True and the plain
    ``manual_rectangle`` side-view temple recordings when False. When ``use_halved_for_mwir``
    is True the reported metrics come from the ``closest_halved`` columns for the MWIR camera
    (and from ``closest`` for Vario/Lepton); when False every camera uses the regular
    ``closest`` columns.
    """
    df_full = pd.read_csv(csv_path)
    if "plots_folder" not in df_full.columns:
        raise ValueError(f"'plots_folder' column not found in {csv_path}")

    needed = [col for pair in _COLUMN_SOURCES.values() for col in pair] + ["gt_hr_mean"]
    missing = [col for col in needed if col not in df_full.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing expected metric columns: {missing}. "
            "Re-run metric_pipeline.py to regenerate the aggregate CSV."
        )

    pf_full = df_full["plots_folder"].astype(str)
    is_front_view = pf_full.str.contains("manual_rectangle_front_view", na=False)
    # The new forehead & background ROI (manual_rectangle_forehead) is reported by the
    # dedicated side-view tables, so keep it out of the temple side-view / face front-view
    # own-approach tables here.
    is_forehead_bg = pf_full.str.contains("manual_rectangle_forehead", na=False)
    keep = (
        pf_full.str.contains("new_approach", na=False)
        & pf_full.str.contains("manual_rectangle", na=False)
        & ~is_forehead_bg
        & (is_front_view if front_view else ~is_front_view)
    )
    df = df_full.loc[keep].copy()
    if df.empty:
        kind = (
            "manual_rectangle_front_view (front-view)"
            if front_view
            else "manual_rectangle (side-view)"
        )
        raise ValueError(f"No new_approach {kind} rows found in the CSV.")

    pf = df["plots_folder"].astype(str)
    camera = pf.apply(_detect_camera)
    # mwir -> halved columns, but only when requested; otherwise every camera uses closest.
    use_halved = (
        (~camera.isin(_CLOSEST_ONLY_CAMERAS)).to_numpy()
        if use_halved_for_mwir
        else np.zeros(len(camera), dtype=bool)
    )

    table = pd.DataFrame(index=df.index)
    table["Camera"] = camera.map(_CAMERA_DISPLAY).fillna(camera)
    table["Experiment"] = pf.apply(
        lambda p: _experiment_label(p, view_label, region_label)
    )
    table["ground truth mean HR (bpm)"] = df["gt_hr_mean"].to_numpy()
    for out_col, (closest_col, halved_col) in _COLUMN_SOURCES.items():
        table[out_col] = np.where(
            use_halved, df[halved_col].to_numpy(), df[closest_col].to_numpy()
        )

    # Stable, readable ordering: camera, then participant, then experiment.
    parsed = pf.apply(_parse_participant_experiment)
    table["_camera_order"] = camera.map(_CAMERA_ORDER).fillna(99)
    table["_participant"] = parsed.apply(lambda p: p[0] if p else 99)
    table["_experiment"] = parsed.apply(lambda p: p[1] if p else 99)
    table = (
        table.sort_values(["_camera_order", "_participant", "_experiment"])
        .drop(columns=["_camera_order", "_participant", "_experiment"])
        .reset_index(drop=True)
    )

    if not include_camera:
        table = table.drop(columns=["Camera"])

    return _finalize_table(
        table,
        output_path=output_path,
        round_decimals=round_decimals,
        latex_output_path=latex_output_path,
        latex_caption=latex_caption,
        latex_label=latex_label,
    )


def create_table_side_view_own_approach(
    csv_path: str | Path = DEFAULT_INPUT_CSV,
    output_path: Optional[str | Path] = None,
    *,
    include_camera: bool = True,
    round_decimals: Optional[int] = 2,
    latex_output_path: Optional[str | Path] = None,
    latex_caption: Optional[str] = None,
    latex_label: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """Build the side-view own-approach (``new_approach``) metrics table.

    Only ``new_approach`` rows for the manual side-view temple ROI (``manual_rectangle``,
    excluding the ``manual_rectangle_front_view`` variant) are kept, because the reported
    label is "side view, temple, manual". The reported metrics come from the
    ``closest_halved`` columns for the MWIR camera and from the ``closest`` columns for the
    Vario/Lepton cameras (so ``closest_halved`` is only used for MWIR).

    Parameters
    ----------
    csv_path:
        Path to the aggregate metrics CSV (default ``saved_results/all_npz_metrics.csv``).
    output_path:
        If given, the resulting table is also written to this CSV path.
    include_camera:
        If ``True`` (default) a leading ``Camera`` column is added. It is needed to tell
        apart the three cameras, which otherwise share identical ``Experiment`` labels.
        Set to ``False`` to keep only the requested 9 columns.
    round_decimals:
        Number of decimals to round the numeric columns to for readability (``None`` keeps
        full precision).
    latex_output_path:
        If given, the fully-bordered LaTeX table is also written to this ``.tex`` file.
    latex_caption / latex_label:
        Optional caption / label; when either is given the LaTeX ``tabular`` is wrapped in
        a ``table`` float with ``\\centering``.

    Returns
    -------
    (pandas.DataFrame, str)
        The assembled table (one row per camera/participant/experiment) and its
        fully-bordered LaTeX representation.
    """
    return _create_own_approach_manual_table(
        csv_path,
        output_path,
        front_view=False,
        view_label="side view",
        region_label="temple",
        use_halved_for_mwir=True,
        include_camera=include_camera,
        round_decimals=round_decimals,
        latex_output_path=latex_output_path,
        latex_caption=latex_caption,
        latex_label=latex_label,
    )


def create_table_manual_front_view_own_approach(
    csv_path: str | Path = DEFAULT_INPUT_CSV,
    output_path: Optional[str | Path] = None,
    *,
    include_camera: bool = True,
    round_decimals: Optional[int] = 2,
    latex_output_path: Optional[str | Path] = None,
    latex_caption: Optional[str] = None,
    latex_label: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """Build the front-view own-approach (``new_approach``) metrics table.

    Same as :func:`create_table_side_view_own_approach`, but only ``new_approach`` rows for
    the manual front-view face ROI (``manual_rectangle_front_view``) are kept and the
    ``Experiment`` label reads "front view, face" instead of "side view, temple". Unlike the
    side-view table, every camera (including MWIR) uses the regular ``closest`` columns here
    -- the ``closest_halved`` variant is not used for the front view. Parameters and return
    value match :func:`create_table_side_view_own_approach`.
    """
    return _create_own_approach_manual_table(
        csv_path,
        output_path,
        front_view=True,
        view_label="front view",
        region_label="face",
        use_halved_for_mwir=False,
        include_camera=include_camera,
        round_decimals=round_decimals,
        latex_output_path=latex_output_path,
        latex_caption=latex_caption,
        latex_label=latex_label,
    )


def _create_manual_all_approaches_table(
    csv_path: str | Path,
    output_path: Optional[str | Path],
    *,
    perspective: str,
    round_decimals: Optional[int],
    latex_output_path: Optional[str | Path],
    latex_caption: Optional[str],
    latex_label: Optional[str],
) -> tuple[pd.DataFrame, str]:
    """Shared builder for the all-approaches manual-ROI metrics tables.

    ``perspective`` selects which manual recordings are kept and how they are labelled:

    * ``"both"``  -- keep the side-view temple (``manual_rectangle``) and the front-view
      face (``manual_rectangle_front_view``) recordings and add a ``Perspective`` column to
      tell them apart (used by :func:`create_table_manual_all_approaches`).
    * ``"side"``  -- keep only the side-view temple recordings (no ``Perspective`` column,
      it would be constant).
    * ``"front"`` -- keep only the front-view face recordings (no ``Perspective`` column).

    Columns: ``Approach``, ``Camera`` (display name), ``Participant``, ``Experiment``
    (before / after sport), ``Perspective`` (only when ``perspective == "both"``), ``ROI``
    ("temple (manual detection)" / "face (manual detection)") and then the metric columns.
    The MWIR (ImageIR) *side-view* rows report the ``closest_halved`` metrics when available
    (only the own approach's MWIR manual recordings have a halved variant); every other row
    reports the regular ``closest`` metrics.
    """
    if perspective not in ("both", "side", "front"):
        raise ValueError(
            f"perspective must be 'both', 'side' or 'front', got {perspective!r}"
        )

    df_full = pd.read_csv(csv_path)
    if "plots_folder" not in df_full.columns:
        raise ValueError(f"'plots_folder' column not found in {csv_path}")

    needed = [col for pair in _COLUMN_SOURCES.values() for col in pair] + ["gt_hr_mean"]
    missing = [col for col in needed if col not in df_full.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing expected metric columns: {missing}. "
            "Re-run metric_pipeline.py to regenerate the aggregate CSV."
        )

    pf_full = df_full["plots_folder"].astype(str)
    # The new forehead & background ROI (manual_rectangle_forehead) is reported by the
    # dedicated side-view tables; exclude it here so it does not leak into the temple
    # side-view / combined manual tables.
    is_forehead_full = pf_full.str.contains("manual_rectangle_forehead", na=False)
    is_manual_full = (
        pf_full.str.contains("manual_rectangle", na=False) & ~is_forehead_full
    )
    is_front_full = pf_full.str.contains("manual_rectangle_front_view", na=False)
    if perspective == "side":
        keep = is_manual_full & ~is_front_full
    elif perspective == "front":
        keep = is_manual_full & is_front_full
    else:
        keep = is_manual_full
    df = df_full.loc[keep].copy()
    if df.empty:
        raise ValueError(
            f"No manual_rectangle ({perspective}-view) rows found in the CSV."
        )

    pf = df["plots_folder"].astype(str)
    camera = pf.apply(_detect_camera)
    is_front = pf.str.contains("manual_rectangle_front_view", na=False).to_numpy()
    # The "halved" variant (every second predicted peak dropped) is only computed for the
    # own approach's MWIR manual recordings; Nair et al. / Yang et al. have no halved data.
    # Use the halved metrics for MWIR side-view rows that actually have them (the own
    # approach) and fall back to the regular closest metrics everywhere else (all front-view
    # rows, the Vario/Lepton rows, and the Nair/Yang MWIR rows that lack a halved variant).
    halved_source_cols = [halved for (_closest, halved) in _COLUMN_SOURCES.values()]
    halved_available = df[halved_source_cols].notna().all(axis=1).to_numpy()
    use_halved = (camera == "mwir").to_numpy() & ~is_front & halved_available & (pf.str.contains("new_approach", na=False).to_numpy())
    parsed = pf.apply(_parse_participant_experiment)

    table = pd.DataFrame(index=df.index)
    base_approach = pf.apply(_approach_label)

    table["Approach"] = np.where(
    use_halved,
    "own approach (using every second detected HR peak)",
    base_approach
    )
    table["Camera"] = camera.map(_CAMERA_DISPLAY).fillna(camera)
    table["Participant"] = parsed.apply(lambda p: p[0] if p else pd.NA).astype("Int64")
    table["Experiment"] = parsed.apply(
        lambda p: None
        if p is None
        else ("before sport" if p[1] in (1, 2) else "after sport")
    )
    table["ground truth mean HR (bpm)"] = df["gt_hr_mean"].to_numpy()
    if perspective == "both":
        table["Perspective"] = np.where(is_front, "front view", "side view")
    table["ROI"] = np.where(
        is_front, "face (manual detection)", "temple (manual detection)"
    )
    for out_col, (closest_col, halved_col) in _COLUMN_SOURCES.items():
        table[out_col] = np.where(
            use_halved, df[halved_col].to_numpy(), df[closest_col].to_numpy()
        )

    # Stable ordering: approach, camera, perspective (side then front), participant, experiment.
    table["_approach_order"] = pf.apply(
        lambda p: _APPROACH_ORDER.get(_detect_approach(p), 99)
    )
    table["_camera_order"] = camera.map(_CAMERA_ORDER).fillna(99).to_numpy()
    table["_perspective_order"] = np.where(is_front, 1, 0)
    table["_participant"] = parsed.apply(lambda p: p[0] if p else 99)
    table["_experiment"] = parsed.apply(lambda p: p[1] if p else 99)
    table = (
        table.sort_values(
            [
                "_approach_order",
                "_camera_order",
                "_perspective_order",
                "_participant",
                "_experiment",
            ]
        )
        .drop(
            columns=[
                "_approach_order",
                "_camera_order",
                "_perspective_order",
                "_participant",
                "_experiment",
            ]
        )
        .reset_index(drop=True)
    )

    return _finalize_table(
        table,
        output_path=output_path,
        round_decimals=round_decimals,
        latex_output_path=latex_output_path,
        latex_caption=latex_caption,
        latex_label=latex_label,
    )


def create_table_manual_all_approaches(
    csv_path: str | Path = DEFAULT_INPUT_CSV,
    output_path: Optional[str | Path] = None,
    *,
    round_decimals: Optional[int] = 2,
    latex_output_path: Optional[str | Path] = None,
    latex_caption: Optional[str] = None,
    latex_label: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """Build the manual-ROI metrics table for *all* approaches and both perspectives.

    Unlike :func:`create_table_side_view_own_approach` /
    :func:`create_table_manual_front_view_own_approach` (own approach only), this keeps the
    manual-ROI recordings of every approach -- Nair et al., Yang et al. and the own approach
    -- and reports both the side-view temple (``manual_rectangle``) and the front-view face
    (``manual_rectangle_front_view``) recordings in one table, told apart by a
    ``Perspective`` column. Use :func:`create_table_side_view_all_approaches` /
    :func:`create_table_front_view_all_approaches` to get the same content split into two
    separate single-perspective tables instead.

    Columns: ``Approach`` (Nair et al. / Yang et al. / own approach), ``Camera`` (display
    name ImageIR / VarioCam / Lepton), ``Participant`` (1..3), ``Experiment`` (before / after
    sport), ``Perspective`` (side view / front view), ``ROI`` ("temple (manual detection)" /
    "face (manual detection)") and then the metric columns. The MWIR (ImageIR) *side-view*
    rows report the ``closest_halved`` metrics **when available** -- which is only the case
    for the own approach's MWIR manual recordings (Nair et al. / Yang et al. have no halved
    variant). Every other row -- all front-view rows, the Vario/Lepton rows, and the
    Nair/Yang MWIR rows -- reports the regular ``closest`` metrics.

    Parameters and return value match the other table builders (see e.g.
    :func:`create_table_best_roi_all_approaches`).
    """
    return _create_manual_all_approaches_table(
        csv_path,
        output_path,
        perspective="both",
        round_decimals=round_decimals,
        latex_output_path=latex_output_path,
        latex_caption=latex_caption,
        latex_label=latex_label,
    )


def create_table_side_view_all_approaches(
    csv_path: str | Path = DEFAULT_INPUT_CSV,
    output_path: Optional[str | Path] = None,
    *,
    round_decimals: Optional[int] = 2,
    latex_output_path: Optional[str | Path] = None,
    latex_caption: Optional[str] = None,
    latex_label: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """Build the side-view manual-ROI metrics table for *all* approaches.

    Like :func:`create_table_manual_all_approaches` but keeps only the side-view temple
    (``manual_rectangle``) recordings of every approach (Nair et al., Yang et al. and the
    own approach). The constant ``Perspective`` column is omitted; the ``ROI`` column reads
    "temple (manual detection)" throughout. As in the combined table the MWIR (ImageIR) rows
    report the ``closest_halved`` metrics when available (only the own approach's MWIR manual
    recordings have a halved variant). Parameters and return value match
    :func:`create_table_manual_all_approaches`.
    """
    return _create_manual_all_approaches_table(
        csv_path,
        output_path,
        perspective="side",
        round_decimals=round_decimals,
        latex_output_path=latex_output_path,
        latex_caption=latex_caption,
        latex_label=latex_label,
    )


def create_table_front_view_all_approaches(
    csv_path: str | Path = DEFAULT_INPUT_CSV,
    output_path: Optional[str | Path] = None,
    *,
    round_decimals: Optional[int] = 2,
    latex_output_path: Optional[str | Path] = None,
    latex_caption: Optional[str] = None,
    latex_label: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """Build the front-view manual-ROI metrics table for *all* approaches.

    Like :func:`create_table_manual_all_approaches` but keeps only the front-view face
    (``manual_rectangle_front_view``) recordings of every approach. The constant
    ``Perspective`` column is omitted; the ``ROI`` column reads "face (manual detection)"
    throughout. Every camera (including MWIR/ImageIR) uses the regular ``closest`` metrics
    here -- the ``closest_halved`` variant is never used for the front view. Parameters and
    return value match :func:`create_table_manual_all_approaches`.
    """
    return _create_manual_all_approaches_table(
        csv_path,
        output_path,
        perspective="front",
        round_decimals=round_decimals,
        latex_output_path=latex_output_path,
        latex_caption=latex_caption,
        latex_label=latex_label,
    )


# --- Side-view "best variant" tables ---------------------------------------------------
# These two tables (temple-only and best-of-side-view) pick, per recording, between a
# candidate ROI's regular metrics and its every-second-peak ("halved") metrics using the
# composite criterion from summarize_metrics (the same notion of "better" used to choose the
# best ROI). The chosen variant decides both the reported metric columns and whether the
# Approach cell is annotated "(using only every second detected HR peak)".

# Canonical "closest" score column -> (regular, halved) source columns. group_scores reads
# the *closest* names, so a halved candidate is scored by copying its halved columns into
# these canonical names.
_VARIANT_SCORE_SOURCES: dict[str, tuple[str, str]] = {
    "metrics_closest_mae": ("metrics_closest_mae", "metrics_closest_halved_mae"),
    "metrics_closest_rmse": ("metrics_closest_rmse", "metrics_closest_halved_rmse"),
    "ba_stats_closest_mean_diff": (
        "ba_stats_closest_mean_diff",
        "ba_stats_closest_halved_mean_diff",
    ),
    "ba_stats_closest_sd_diff": (
        "ba_stats_closest_sd_diff",
        "ba_stats_closest_halved_sd_diff",
    ),
    "hr_agreement_fraction": ("hr_agreement_fraction", "hr_agreement_fraction_halved"),
}


def _variant_has_data(row: pd.Series, halved: bool) -> bool:
    """True when the regular/halved closest metrics exist for ``row`` (gauged by MAE)."""
    column = "metrics_closest_halved_mae" if halved else "metrics_closest_mae"
    return bool(pd.notna(row.get(column)))


def _variant_score_row(row: pd.Series, halved: bool) -> dict:
    """Canonical 'closest' score dict (regular or halved) fed to group_scores."""
    source = 1 if halved else 0
    scored: dict[str, object] = {
        canonical: row.get(columns[source])
        for canonical, columns in _VARIANT_SCORE_SOURCES.items()
    }
    scored["plots_folder"] = row.get("plots_folder", "")  # group_scores tie-break key
    return scored


def _select_best_side_view_variant(
    group: pd.DataFrame, *, include_forehead: bool
) -> Optional[tuple[pd.Series, bool]]:
    """Return the best ``(row, halved)`` candidate for one side-view recording group.

    Candidates are the regular and every-second-peak ("halved") variants of the temple
    (and, when ``include_forehead``, the forehead & background) rows that actually have
    metric data. The winner minimises the composite criterion (mean competition rank across
    COMPOSITE_METRICS, the same "better" notion used to pick the best ROI); ties prefer the
    regular variant, then temple over forehead & background, then the lower MAE. Returns
    ``None`` when no candidate has data.
    """
    candidates: list[tuple[pd.Series, bool]] = []
    for _index, row in group.iterrows():
        kind = row["_roi_kind"]
        if kind == "temple" or (include_forehead and kind == "forehead_bg"):
            for halved in (False, True):
                if _variant_has_data(row, halved):
                    candidates.append((row, halved))
    if not candidates:
        return None

    scores = group_scores(
        [_variant_score_row(row, halved) for row, halved in candidates], "composite"
    )

    def _tie_key(i: int) -> tuple:
        row, halved = candidates[i]
        mae_column = "metrics_closest_halved_mae" if halved else "metrics_closest_mae"
        mae_value = row.get(mae_column)
        return (
            scores[i] if scores[i] is not None else math.inf,
            1 if halved else 0,  # prefer the regular variant on ties
            0 if row["_roi_kind"] == "temple" else 1,  # then temple over forehead & bg
            float(mae_value) if pd.notna(mae_value) else math.inf,
        )

    best = min(range(len(candidates)), key=_tie_key)
    return candidates[best][0], candidates[best][1]


def _create_side_view_best_table(
    csv_path: str | Path,
    output_path: Optional[str | Path],
    *,
    include_forehead: bool,
    round_decimals: Optional[int],
    latex_output_path: Optional[str | Path],
    latex_caption: Optional[str],
    latex_label: Optional[str],
) -> tuple[pd.DataFrame, str]:
    """Shared builder for the two side-view all-approaches tables.

    For every side-view recording (Nair et al., Yang et al. and the own approach) the temple
    ``manual_rectangle`` ROI -- and, when ``include_forehead``, also the forehead & background
    ``manual_rectangle_forehead`` ROI -- competes in BOTH its regular and its every-second-peak
    ("halved") variant, and the variant scoring best on the composite criterion (mean
    competition rank across mae, rmse, |bias|, SD of differences, LoA width and the
    HR-agreement fraction -- the same "better" notion used to pick the best ROI) is reported.
    The winning row uses the matching ``closest`` / ``closest_halved`` metric columns; when a
    halved variant wins, its ``Approach`` cell is annotated
    ``"... (using only every second detected HR peak)"``.

    ``include_forehead=False`` -> temple only (the ``ROI`` column is the constant "temple
    (manual detection)"); ``include_forehead=True`` -> the best of temple / forehead &
    background, with the winning ROI shown as "temple (manual detection)" /
    "forehead & background (manual detection)". The side-view ``manual_rectangle_temp`` and the
    front-view ``manual_rectangle_front_view`` recordings are never considered here.
    """
    df_full = pd.read_csv(csv_path)
    if "plots_folder" not in df_full.columns:
        raise ValueError(f"'plots_folder' column not found in {csv_path}")

    needed = [col for pair in _COLUMN_SOURCES.values() for col in pair] + ["gt_hr_mean"]
    missing = [col for col in needed if col not in df_full.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing expected metric columns: {missing}. "
            "Re-run metric_pipeline.py to regenerate the aggregate CSV."
        )

    roi_kind = df_full["plots_folder"].astype(str).apply(_manual_roi_kind)
    wanted = ["temple", "forehead_bg"] if include_forehead else ["temple"]
    df = df_full.loc[roi_kind.isin(wanted)].copy()
    if df.empty:
        kinds = "temple / forehead & background" if include_forehead else "temple"
        raise ValueError(f"No side-view ({kinds}) manual_rectangle rows found in {csv_path}.")
    df["_roi_kind"] = df["plots_folder"].astype(str).apply(_manual_roi_kind)
    df["_rec_key"] = df["plots_folder"].astype(str).apply(_recording_key)

    # Per recording (temple + forehead & background share a _rec_key), pick the best
    # (row, halved) candidate by the composite criterion.
    winners: list[tuple[pd.Series, bool]] = []
    for _rec_key, group in df.groupby("_rec_key", sort=False):
        picked = _select_best_side_view_variant(group, include_forehead=include_forehead)
        if picked is not None:
            winners.append(picked)
    if not winners:
        raise ValueError(f"No side-view recording in {csv_path} had usable metrics to rank.")

    records: list[dict[str, object]] = []
    for row, halved in winners:
        pf = str(row["plots_folder"])
        camera_token = _detect_camera(pf)
        parsed = _parse_participant_experiment(pf)
        experiment_num = parsed[1] if parsed else None
        base_label = _approach_label(pf)
        record: dict[str, object] = {
            "Approach": (
                f"{base_label} (using only every second detected HR peak)"
                if halved
                else base_label
            ),
            "Camera": _CAMERA_DISPLAY.get(camera_token, camera_token),
            "Participant": parsed[0] if parsed else pd.NA,
            "Experiment": (
                None
                if experiment_num is None
                else ("before sport" if experiment_num in (1, 2) else "after sport")
            ),
            "ground truth mean HR (bpm)": row.get("gt_hr_mean"),
            "ROI": (
                "temple (manual detection)"
                if row["_roi_kind"] == "temple"
                else "forehead & background (manual detection)"
            ),
            "_approach_order": _APPROACH_ORDER.get(_detect_approach(pf), 99),
            "_camera_order": _CAMERA_ORDER.get(camera_token, 99),
            "_participant": parsed[0] if parsed else 99,
            "_experiment": experiment_num if experiment_num is not None else 99,
        }
        for out_col, (closest_col, halved_col) in _COLUMN_SOURCES.items():
            record[out_col] = row.get(halved_col if halved else closest_col)
        records.append(record)

    table = pd.DataFrame.from_records(records)
    table = table.sort_values(
        ["_approach_order", "_camera_order", "_participant", "_experiment"]
    ).reset_index(drop=True)
    final_columns = [
        "Approach",
        "Camera",
        "Participant",
        "Experiment",
        "ground truth mean HR (bpm)",
        "ROI",
    ] + list(_COLUMN_SOURCES.keys())
    table = table[final_columns]
    table["Participant"] = table["Participant"].astype("Int64")

    return _finalize_table(
        table,
        output_path=output_path,
        round_decimals=round_decimals,
        latex_output_path=latex_output_path,
        latex_caption=latex_caption,
        latex_label=latex_label,
    )


def create_table_side_view_temple_all_approaches(
    csv_path: str | Path = DEFAULT_INPUT_CSV,
    output_path: Optional[str | Path] = None,
    *,
    round_decimals: Optional[int] = 2,
    latex_output_path: Optional[str | Path] = None,
    latex_caption: Optional[str] = None,
    latex_label: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """Side-view temple (manual_rectangle) metrics table across all approaches.

    One row per side-view recording (Nair et al., Yang et al. and the own approach): for each
    recording the regular and the every-second-peak ("halved") temple variant compete and the
    one scoring better on the composite criterion is reported (when the halved variant wins,
    the ``Approach`` cell reads "... (using only every second detected HR peak)"). The forehead
    & background ROI is NOT considered here -- use
    :func:`create_table_side_view_best_all_approaches` for that. Parameters and return value
    match the other table builders.
    """
    return _create_side_view_best_table(
        csv_path,
        output_path,
        include_forehead=False,
        round_decimals=round_decimals,
        latex_output_path=latex_output_path,
        latex_caption=latex_caption,
        latex_label=latex_label,
    )


def create_table_side_view_best_all_approaches(
    csv_path: str | Path = DEFAULT_INPUT_CSV,
    output_path: Optional[str | Path] = None,
    *,
    round_decimals: Optional[int] = 2,
    latex_output_path: Optional[str | Path] = None,
    latex_caption: Optional[str] = None,
    latex_label: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """Best side-view result per recording across all approaches.

    Like :func:`create_table_side_view_temple_all_approaches`, but for each side-view recording
    the best of up to FOUR candidates is reported: the temple (``manual_rectangle``) and the
    forehead & background (``manual_rectangle_forehead``) ROI, each in its regular and its
    every-second-peak ("halved") variant. The winning ROI is shown in the ``ROI`` column
    ("temple (manual detection)" / "forehead & background (manual detection)") and a halved win
    annotates the ``Approach`` cell. Parameters and return value match the other table builders.
    """
    return _create_side_view_best_table(
        csv_path,
        output_path,
        include_forehead=True,
        round_decimals=round_decimals,
        latex_output_path=latex_output_path,
        latex_caption=latex_caption,
        latex_label=latex_label,
    )


def create_table_best_roi_all_approaches(
    csv_path: str | Path = DEFAULT_INPUT_BEST_ROI_CSV,
    output_path: Optional[str | Path] = None,
    *,
    round_decimals: Optional[int] = 2,
    latex_output_path: Optional[str | Path] = None,
    latex_caption: Optional[str] = None,
    latex_label: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """Build the best-ROI metrics table across all approaches.

    Reads the best-ROI summary CSV (default ``saved_results/all_npz_metrics_best_roi.csv``,
    produced by ``summarize_metrics.py`` -- one row per recording, already reduced to its
    winning ROI) and keeps every row except those whose selected ROI is the side-view
    ``manual_rectangle`` (dropped because those side-view temple experiments are reported
    separately in the own-approach tables; the front-view ``manual_rectangle_front_view``
    recordings are kept), across all approaches (Nair et al., Yang et al. and the own
    approach). The reported metric columns are the same as the manual-ROI tables but always
    taken from the regular ``closest`` columns (the ``closest_halved`` variant is not used
    here) and **without r²**, which is not one of the COMPOSITE_METRICS used to select the
    best ROI. Four descriptive columns lead: Approach, Camera, Experiment (Participant x,
    experiment y, before/after sport) and ROI (the ROI selected in the CSV).

    Parameters
    ----------
    csv_path:
        Path to the best-ROI CSV (default ``saved_results/all_npz_metrics_best_roi.csv``).
    output_path:
        If given, the resulting table is also written to this CSV path.
    round_decimals:
        Number of decimals to round the numeric columns to (``None`` keeps full precision).
    latex_output_path:
        If given, the fully-bordered LaTeX table is also written to this ``.tex`` file.
    latex_caption / latex_label:
        Optional caption / label; when either is given the LaTeX ``tabular`` is wrapped in
        a ``table`` float with ``\\centering``.

    Returns
    -------
    (pandas.DataFrame, str)
        The assembled table (one row per recording) and its fully-bordered LaTeX
        representation.
    """
    df = pd.read_csv(csv_path)
    for required in ("plots_folder", "roi"):
        if required not in df.columns:
            raise ValueError(
                f"'{required}' column not found in {csv_path}. "
                "Run summarize_metrics.py to (re)generate the best-ROI CSV."
            )

    closest_cols = [closest for (closest, _halved) in _COLUMN_SOURCES.values()]
    missing = [col for col in closest_cols + ["gt_hr_mean"] if col not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing expected metric columns: {missing}. "
            "Re-run metric_pipeline.py / summarize_metrics.py to regenerate it."
        )
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}.")

    # Drop the side-view temple recordings (roi == "manual_rectangle"); they are reported
    # separately in the own-approach side-view table. Exact match keeps the front-view
    # "manual_rectangle_front_view" recordings (a different ROI token).
    df = df.loc[~df["roi"].astype(str).isin(["manual_rectangle", "manual_rectangle_forehead"])].copy()
    if df.empty:
        raise ValueError(
            f"No rows left in {csv_path} after dropping side-view 'manual_rectangle' rows."
        )

    pf = df["plots_folder"].astype(str)
    camera = pf.apply(_detect_camera)
    table = pd.DataFrame(index=df.index)
    table["Approach"] = pf.apply(_approach_label)
    table["Camera"] = camera.map(_CAMERA_DISPLAY).fillna(camera)
    parsed = pf.apply(_parse_participant_experiment)
    table["Participant"] = parsed.apply(lambda p: p[0] if p else pd.NA).astype("Int64")
    table["Experiment"] = parsed.apply(
        lambda p: None
        if p is None
        else ("before sport" if p[1] in (1, 2) else "after sport")
    )
    table["ground truth mean HR (bpm)"] = df["gt_hr_mean"].to_numpy()
    # Make the ROI token readable: drop underscores (e.g. "left_forehead" -> "left forehead").
    table["ROI"] = df["roi"].astype(str).str.replace("_", " ", regex=False)
    # Always the regular closest columns (first element of each _COLUMN_SOURCES pair).
    # (r² is intentionally not part of _COLUMN_SOURCES, see its definition above.)
    for out_col, (closest_col, _halved_col) in _COLUMN_SOURCES.items():
        table[out_col] = df[closest_col].to_numpy()

    # Stable, readable ordering: approach, camera, participant, experiment.
    parsed = pf.apply(_parse_participant_experiment)
    table["_approach_order"] = pf.apply(
        lambda p: _APPROACH_ORDER.get(_detect_approach(p), 99)
    )
    table["_camera_order"] = camera.map(_CAMERA_ORDER).fillna(99)
    table["_participant"] = parsed.apply(lambda p: p[0] if p else 99)
    table["_experiment"] = parsed.apply(lambda p: p[1] if p else 99)
    table = (
        table.sort_values(
            ["_approach_order", "_camera_order", "_participant", "_experiment"]
        )
        .drop(
            columns=["_approach_order", "_camera_order", "_participant", "_experiment"]
        )
        .reset_index(drop=True)
    )

    return _finalize_table(
        table,
        output_path=output_path,
        round_decimals=round_decimals,
        latex_output_path=latex_output_path,
        latex_caption=latex_caption,
        latex_label=latex_label,
    )


def create_table_ranked_recordings(
    approach: str,
    csv_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    *,
    round_decimals: Optional[int] = 2,
    latex_output_path: Optional[str | Path] = None,
    latex_caption: Optional[str] = None,
    latex_label: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """Build the ranked-recordings table for a single approach.

    Reads the per-approach ranked CSV written by
    ``summarize_metrics.rank_recordings_per_approach`` (default
    ``saved_results/all_npz_metrics_best_roi_ranked_<approach>.csv``) -- one row per
    recording, already ordered best -> worst by the same ``composite`` criterion that picked
    each recording's best ROI -- and renders it while **keeping that ranking order** (it is
    deliberately *not* re-sorted by camera/participant/experiment like the other tables).

    A leading ``Rank`` column (1 = the approach's best recording) is added; the remaining
    descriptive columns mirror :func:`create_table_best_roi_all_approaches` minus the (here
    constant) ``Approach`` column: ``Camera`` (ImageIR / VarioCam / Lepton), ``Participant``
    (1..3), ``Experiment`` (experiment number + before / after sport, so each row is uniquely
    identifiable even though the order follows the rank), ``ground truth mean HR (bpm)`` and
    ``ROI`` (the ROI selected for that recording), followed by the regular ``closest`` metric
    columns (the ``closest_halved`` variant is never used here, and r² is omitted, exactly as
    in the best-ROI table).

    ``approach`` must be one of ``"nair_et_al"``, ``"yang_et_al"`` or ``"new_approach"``.
    When ``csv_path`` is ``None`` it defaults to the approach's ranked CSV in
    ``saved_results``. Parameters and return value otherwise match
    :func:`create_table_best_roi_all_approaches`.
    """
    if approach not in _APPROACH_LABELS:
        raise ValueError(
            f"approach must be one of {tuple(_APPROACH_LABELS)}, got {approach!r}"
        )
    if csv_path is None:
        csv_path = (
            REPO_ROOT
            / "saved_results"
            / f"all_npz_metrics_best_roi_ranked_{approach}.csv"
        )

    df = pd.read_csv(csv_path)
    for required in ("plots_folder", "roi", "recording_rank"):
        if required not in df.columns:
            raise ValueError(
                f"'{required}' column not found in {csv_path}. "
                "Run summarize_metrics.py to (re)generate the ranked CSVs."
            )

    closest_cols = [closest for (closest, _halved) in _COLUMN_SOURCES.values()]
    missing = [col for col in closest_cols + ["gt_hr_mean"] if col not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing expected metric columns: {missing}. "
            "Re-run metric_pipeline.py / summarize_metrics.py to regenerate it."
        )
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}.")

    # Keep the ranking order: order by the recording_rank written into the CSV (1 = best)
    # rather than the camera/participant/experiment ordering used by the other tables.
    df = df.copy()
    df["_rank"] = pd.to_numeric(df["recording_rank"], errors="coerce")
    df = df.sort_values("_rank", kind="stable").reset_index(drop=True)

    pf = df["plots_folder"].astype(str)
    camera = pf.apply(_detect_camera)
    parsed = pf.apply(_parse_participant_experiment)

    table = pd.DataFrame(index=df.index)
    table["Rank"] = df["_rank"].astype("Int64")
    table["Camera"] = camera.map(_CAMERA_DISPLAY).fillna(camera)
    table["Participant"] = parsed.apply(lambda p: p[0] if p else pd.NA).astype("Int64")
    table["Experiment"] = parsed.apply(
        lambda p: None
        if p is None
        else (
            f"experiment {p[1]}, "
            + ("before sport" if p[1] in (1, 2) else "after sport")
        )
    )
    table["ground truth mean HR (bpm)"] = df["gt_hr_mean"].to_numpy()
    # Make the ROI token readable: drop underscores (e.g. "left_forehead" -> "left forehead").
    table["ROI"] = df["roi"].astype(str).str.replace("_", " ", regex=False)
    # Always the regular closest columns (first element of each _COLUMN_SOURCES pair);
    # r² is intentionally not part of _COLUMN_SOURCES (see its definition above).
    for out_col, (closest_col, _halved_col) in _COLUMN_SOURCES.items():
        table[out_col] = df[closest_col].to_numpy()

    # NB: deliberately no re-sorting -- the table already follows the best -> worst ranking.
    return _finalize_table(
        table,
        output_path=output_path,
        round_decimals=round_decimals,
        latex_output_path=latex_output_path,
        latex_caption=latex_caption,
        latex_label=latex_label,
    )


# ── Add these default path constants alongside the other DEFAULT_OUTPUT_* constants ──
DEFAULT_OUTPUT_CSV_ROI_CONSISTENCY = (
    REPO_ROOT / "saved_results" / "table_roi_signal_consistency.csv"
)
DEFAULT_OUTPUT_TEX_ROI_CONSISTENCY = (
    REPO_ROOT / "saved_results" / "table_roi_signal_consistency.tex"
)


def create_table_roi_signal_consistency(
    csv_path: str | Path = DEFAULT_INPUT_CSV,
    output_path: Optional[str | Path] = None,
    *,
    round_decimals: Optional[int] = 2,
    latex_output_path: Optional[str | Path] = None,
    latex_caption: Optional[str] = None,
    latex_label: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """Build the ROI-consistency table: per-recording spread of the signal-HR bias across ROIs.

    Reads the aggregate metrics CSV and, for every recording (identified by approach, camera,
    participant and experiment number), collects the ``signal_hr_mean_abs_diff_to_gt_mean``
    values across all of that recording's ROIs. The mean, median, RMSE and range of those
    per-ROI values are reported, yielding one row per recording.

    Columns: ``Approach``, ``Camera`` (ImageIR / VarioCam / Lepton), ``Participant``
    (1..3), ``Experiment`` (before / after sport), ``Perspective`` (side view / front
    view), ``ground truth mean HR (bpm)``, ``Mean abs. signal-GT HR diff across ROIs
    (bpm)``, ``Median abs. signal-GT HR diff across ROIs (bpm)``, ``RMSE abs. signal-GT
    HR diff across ROIs (bpm)`` and ``Range abs. signal-GT HR diff across ROIs (bpm)``.

    The perspective (side view / front view) is inferred from the recording's path: if any
    ROI path contains ``"front_view"`` the recording is labelled "front view", otherwise
    "side view". Parameters and return value match the other table builders (see e.g.
    :func:`create_table_best_roi_all_approaches`).
    """
    df_full = pd.read_csv(csv_path)
    if "plots_folder" not in df_full.columns:
        raise ValueError(f"'plots_folder' column not found in {csv_path}")

    target_col = "signal_hr_mean_abs_diff_to_gt_mean"
    required = [target_col, "gt_hr_mean"]
    missing = [c for c in required if c not in df_full.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing expected columns: {missing}. "
            "Re-run metric_pipeline.py to regenerate the aggregate CSV."
        )

    pf_full = df_full["plots_folder"].astype(str)
    camera_full = pf_full.apply(_detect_camera)
    approach_full = pf_full.apply(_detect_approach)
    parsed_full = pf_full.apply(_parse_participant_experiment)

    # Keep only rows whose recording can be identified.
    valid = parsed_full.apply(lambda p: p is not None)
    df = df_full.loc[valid].copy()
    df["_approach"] = approach_full.loc[valid].values
    df["_camera"] = camera_full.loc[valid].values
    df["_participant"] = parsed_full.loc[valid].apply(lambda p: p[0]).values
    df["_exp_num"] = parsed_full.loc[valid].apply(lambda p: p[1]).values

    records: list[dict[str, object]] = []
    for (app, cam, part, exp), group in df.groupby(
        ["_approach", "_camera", "_participant", "_exp_num"], sort=False
    ):
        values = group[target_col].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue

        pf_group = group["plots_folder"].astype(str)
        front = pf_group.str.contains("front_view", na=False).any()
        gt_vals = group["gt_hr_mean"].dropna()
        gt_hr = gt_vals.iloc[0] if len(gt_vals) > 0 else pd.NA

        records.append(
            {
                "Approach": _APPROACH_LABELS.get(app, app),
                "Camera": _CAMERA_DISPLAY.get(cam, cam),
                "Participant": int(part),
                "Experiment": "before sport"
                if int(exp) in (1, 2)
                else "after sport",
                "Perspective": "front view" if front else "side view",
                "ground truth mean HR (bpm)": gt_hr,
                "Mean abs. signal-GT HR diff across ROIs (bpm)": float(
                    np.mean(values)
                ),
                "Median abs. signal-GT HR diff across ROIs (bpm)": float(
                    np.median(values)
                ),
                "RMSE abs. signal-GT HR diff across ROIs (bpm)": float(
                    np.sqrt(np.mean(values ** 2))
                ),
                "Range abs. signal-GT HR diff across ROIs (bpm)": float(
                    np.max(values) - np.min(values)
                ),
                "Max abs. signal-GT HR diff across ROIs (bpm)": float(np.max(values)),
                "Min abs. signal-GT HR diff across ROIs (bpm)": float(np.min(values)),
            }
        )

    if not records:
        raise ValueError(
            f"No recordings with valid '{target_col}' data found in {csv_path}."
        )

    table = pd.DataFrame.from_records(records)

    # Stable, readable ordering: approach, camera, participant, experiment.
    _approach_rev = {v: k for k, v in _APPROACH_LABELS.items()}
    _camera_rev = {v: k for k, v in _CAMERA_DISPLAY.items()}
    table["_approach_order"] = (
        table["Approach"].map(_approach_rev).map(_APPROACH_ORDER).fillna(99)
    )
    table["_camera_order"] = (
        table["Camera"].map(_camera_rev).map(_CAMERA_ORDER).fillna(99)
    )
    table["_participant"] = table["Participant"]
    table["_experiment"] = table["Experiment"].map({"before sport": 0, "after sport": 1})
    table = (
        table.sort_values(
            ["_approach_order", "_camera_order", "_participant", "_experiment"]
        )
        .drop(
            columns=[
                "_approach_order",
                "_camera_order",
                "_participant",
                "_experiment",
            ]
        )
        .reset_index(drop=True)
    )
    table["Participant"] = table["Participant"].astype("Int64")

    return _finalize_table(
        table,
        output_path=output_path,
        round_decimals=round_decimals,
        latex_output_path=latex_output_path,
        latex_caption=latex_caption,
        latex_label=latex_label,
    )

if __name__ == "__main__":
    builders = [
        (
            "side view temple (all approaches)",
            lambda: create_table_side_view_temple_all_approaches(
                output_path=DEFAULT_OUTPUT_CSV_SIDE_TEMPLE,
                latex_output_path=DEFAULT_OUTPUT_TEX_SIDE_TEMPLE,
                latex_caption=(
                    "Side-view temple (manual) results across all approaches; per "
                    "recording the regular or every-second-peak variant is shown, "
                    "whichever scores better."
                ),
                latex_label="tab:side_view_temple_all_approaches",
            ),
        ),
        (
            "side view best (all approaches)",
            lambda: create_table_side_view_best_all_approaches(
                output_path=DEFAULT_OUTPUT_CSV_SIDE_BEST,
                latex_output_path=DEFAULT_OUTPUT_TEX_SIDE_BEST,
                latex_caption=(
                    "Best side-view result per recording across all approaches "
                    "(temple or forehead & background, regular or every-second-peak "
                    "variant, whichever scores best)."
                ),
                latex_label="tab:side_view_best_all_approaches",
            ),
        ),
        (
            "front view (own approach)",
            lambda: create_table_manual_front_view_own_approach(
                output_path=DEFAULT_OUTPUT_CSV_FRONT,
                latex_output_path=DEFAULT_OUTPUT_TEX_FRONT,
                latex_caption="Front-view (face, manual) results for the own approach.",
                latex_label="tab:front_view_own_approach",
            ),
        ),
        (
            "manual all approaches",
            lambda: create_table_manual_all_approaches(
                output_path=DEFAULT_OUTPUT_CSV_MANUAL_ALL,
                latex_output_path=DEFAULT_OUTPUT_TEX_MANUAL_ALL,
                latex_caption="Manual-ROI (temple / face) results across all approaches.",
                latex_label="tab:manual_all_approaches",
            ),
        ),
        (
            "front view all approaches",
            lambda: create_table_front_view_all_approaches(
                output_path=DEFAULT_OUTPUT_CSV_FRONT_ALL,
                latex_output_path=DEFAULT_OUTPUT_TEX_FRONT_ALL,
                latex_caption="Front-view (face, manual) results across all approaches.",
                latex_label="tab:front_view_all_approaches",
            ),
        ),
        (
            "all approaches (best ROI)",
            lambda: create_table_best_roi_all_approaches(
                output_path=DEFAULT_OUTPUT_CSV_ALL,
                latex_output_path=DEFAULT_OUTPUT_TEX_ALL,
                latex_caption="Best-ROI results per recording across all approaches.",
                latex_label="tab:best_roi_all_approaches",
            ),
        ),
        (
            "ROI signal consistency (all approaches)",
            lambda: create_table_roi_signal_consistency(
                output_path=DEFAULT_OUTPUT_CSV_ROI_CONSISTENCY,
                latex_output_path=DEFAULT_OUTPUT_TEX_ROI_CONSISTENCY,
                latex_caption=(
                    "Per-recording spread of the absolute signal-HR difference "
                    "across all ROIs (mean, median, RMSE, range)."
                ),
                latex_label="tab:roi_signal_mean",
            ),
        ),
    ]
    # Per-approach ranked-recordings tables (one CSV/.tex per approach). Unlike the tables
    # above these keep the best -> worst ranking order from the ranked CSVs instead of being
    # re-sorted by camera/participant/experiment. The default-argument binding
    # (``_approach=_approach``) captures each approach in its own lambda.
    builders += [
        (
            f"ranked recordings ({_APPROACH_LABELS[_approach]})",
            (
                lambda _approach=_approach: create_table_ranked_recordings(
                    _approach,
                    output_path=REPO_ROOT
                    / "saved_results"
                    / f"table_ranked_recordings_{_approach}.csv",
                    latex_output_path=REPO_ROOT
                    / "saved_results"
                    / f"table_ranked_recordings_{_approach}.tex",
                    latex_caption=(
                        f"{_APPROACH_LABELS[_approach]} recordings (best ROI per recording) "
                        "ranked best to worst by the composite criterion."
                    ),
                    latex_label=f"tab:ranked_recordings_{_approach}",
                )
            ),
        )
        for _approach in ("nair_et_al", "yang_et_al", "new_approach")
    ]
    for title, builder in builders:
        try:
            res, tex = builder()
        except Exception as exc:  # report and continue with the other tables
            print(f"\n[create_table] {title}: SKIPPED ({type(exc).__name__}: {exc})")
            continue
        print(f"\n[create_table] {title}: {len(res)} rows; per camera:")
        if "Camera" in res.columns:
            print(res["Camera"].value_counts().to_string())
        with pd.option_context(
            "display.max_columns", None, "display.width", 220, "display.float_format", lambda v: f"{v:.2f}"
        ):
            print(res.to_string(index=False))
        print(f"\n[create_table] {title} LaTeX table:\n")
        print(tex)
