
from typing import Tuple, Dict, Optional
import csv
import os
from datetime import datetime, timezone
from pathlib import Path
import re

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
    #f1 halved
    "tp_distance_based_halved",
    "fp_distance_based_halved",
    "fn_distance_based_halved",
    "precision_distance_based_halved",
    "recall_distance_based_halved",
    "f1_distance_based_halved",
    "mean_pat_distance_based_halved",
    "std_pat_distance_based_halved",
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