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

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

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

    needed = [col for pair in _COLUMN_SOURCES.values() for col in pair]
    missing = [col for col in needed if col not in df_full.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing expected metric columns: {missing}. "
            "Re-run metric_pipeline.py to regenerate the aggregate CSV."
        )

    pf_full = df_full["plots_folder"].astype(str)
    is_front_view = pf_full.str.contains("manual_rectangle_front_view", na=False)
    keep = (
        pf_full.str.contains("new_approach", na=False)
        & pf_full.str.contains("manual_rectangle", na=False)
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
    table["Camera"] = camera
    table["Experiment"] = pf.apply(
        lambda p: _experiment_label(p, view_label, region_label)
    )
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
    ``Perspective`` column.

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
    df_full = pd.read_csv(csv_path)
    if "plots_folder" not in df_full.columns:
        raise ValueError(f"'plots_folder' column not found in {csv_path}")

    needed = [col for pair in _COLUMN_SOURCES.values() for col in pair]
    missing = [col for col in needed if col not in df_full.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing expected metric columns: {missing}. "
            "Re-run metric_pipeline.py to regenerate the aggregate CSV."
        )

    pf_full = df_full["plots_folder"].astype(str)
    df = df_full.loc[pf_full.str.contains("manual_rectangle", na=False)].copy()
    if df.empty:
        raise ValueError("No manual_rectangle rows found in the CSV.")

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
    use_halved = (camera == "mwir").to_numpy() & ~is_front & halved_available
    parsed = pf.apply(_parse_participant_experiment)

    table = pd.DataFrame(index=df.index)
    table["Approach"] = pf.apply(_approach_label)
    table["Camera"] = camera.map(_CAMERA_DISPLAY).fillna(camera)
    table["Participant"] = parsed.apply(lambda p: p[0] if p else pd.NA).astype("Int64")
    table["Experiment"] = parsed.apply(
        lambda p: None
        if p is None
        else ("before sport" if p[1] in (1, 2) else "after sport")
    )
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
    missing = [col for col in closest_cols if col not in df.columns]
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
    df = df.loc[df["roi"].astype(str) != "manual_rectangle"].copy()
    if df.empty:
        raise ValueError(
            f"No rows left in {csv_path} after dropping side-view 'manual_rectangle' rows."
        )

    pf = df["plots_folder"].astype(str)
    table = pd.DataFrame(index=df.index)
    table["Approach"] = pf.apply(_approach_label)
    table["Camera"] = pf.apply(_detect_camera)
    #table["Experiment"] = pf.apply(_participant_sport_label)
    table["ROI"] = df["roi"].astype(str)
    # Always the regular closest columns (first element of each _COLUMN_SOURCES pair).
    # (r² is intentionally not part of _COLUMN_SOURCES, see its definition above.)
    for out_col, (closest_col, _halved_col) in _COLUMN_SOURCES.items():
        table[out_col] = df[closest_col].to_numpy()

    # Stable, readable ordering: approach, camera, participant, experiment.
    parsed = pf.apply(_parse_participant_experiment)
    table["_approach_order"] = pf.apply(
        lambda p: _APPROACH_ORDER.get(_detect_approach(p), 99)
    )
    table["_camera_order"] = table["Camera"].map(_CAMERA_ORDER).fillna(99)
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


if __name__ == "__main__":
    builders = [
        (
            "side view (own approach)",
            lambda: create_table_side_view_own_approach(
                output_path=DEFAULT_OUTPUT_CSV,
                latex_output_path=DEFAULT_OUTPUT_TEX,
                latex_caption="Side-view (temple, manual) results for the own approach.",
                latex_label="tab:side_view_own_approach",
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
            "all approaches (best ROI)",
            lambda: create_table_best_roi_all_approaches(
                output_path=DEFAULT_OUTPUT_CSV_ALL,
                latex_output_path=DEFAULT_OUTPUT_TEX_ALL,
                latex_caption="Best-ROI results per recording across all approaches.",
                latex_label="tab:best_roi_all_approaches",
            ),
        ),
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
