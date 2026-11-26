# Metrics

This package evaluates predicted heart-rate signals against ECG/OpenSignals
ground truth and converts per-recording results into aggregate CSV and LaTeX
tables.

## Pipeline overview

The normal workflow is:

1. Each approach writes a predicted signal and ground truth to an `.npz` file
   under `saved_results/<approach>/results/<camera>/`.
2. `metric_pipeline.py` evaluates those `.npz` files and writes plots, JSON
   metrics, and the aggregate `saved_results/all_npz_metrics.csv`.
3. `summarize_metrics.py` selects the best ROI for each recording and ranks
   recordings within each approach.
4. `create_tables_for_paper.py` creates configurable thesis tables and LaTeX
   output.
5. `plot_csv_data.py` and `visualize_results_npy.py` provide additional visual
   inspection.

## Running metric evaluation

Run the batch evaluation from the repository root:

```bash
python -m src.metrics.metric_pipeline
```

The hard-coded configuration for the evaluation is in the `__main__` block of
`metric_pipeline.py`. Important settings include:

- `RUN_MODE`: `structure` for the full result tree or `single` for one file.
- `EXISTING_ENTRY_MODE`: `skip` or `overwrite`.
- `APPROACHES_TO_RUN`: `new`, `yang`, `nair`, or their full folder names.
- `CAMERAS_TO_RUN`: for example `("mwir", "vario", "lepton")`, or `None`.
- `INDIVIDUALS_TO_RUN`: an individual name, tuple of names, or `None`.
- `GT_PEAK_METHOD`: `find_peaks` or `staple`.
- `HARD_CODED_PATTERN`: usually `*.npz`.
- `HARD_CODED_TOLERANCE_MS`: peak-matching tolerance.
- `PAT_TIME_WINDOW_LIMITS`: optional bounds for pulse-arrival-time analysis.

The structure batch expects this layout:

```text
saved_results/
  new_approach/results/<camera>/<recording>.npz
  yang_et_al/results/<camera>/<recording>.npz
  nair_et_al/results/<camera>/<recording>.npz
```

For each `.npz`, the pipeline can produce:

- per-run plots and Bland-Altman plots;
- `00_metrics.json`;
- a peak-matching or distance-based peak evaluation;
- mean and instantaneous heart-rate comparisons;
- aggregate rows in `saved_results/all_npz_metrics.csv`.

## Selecting and ranking the best ROIs

After `all_npz_metrics.csv` has been generated, run:

```bash
python -m src.metrics.summarize_metrics --verbose
```

Outputs:

- `saved_results/all_npz_metrics_best_roi.csv`
- `saved_results/all_npz_metrics_best_roi_ranked_nair_et_al.csv`
- `saved_results/all_npz_metrics_best_roi_ranked_yang_et_al.csv`
- `saved_results/all_npz_metrics_best_roi_ranked_new_approach.csv`

Useful options:

- `--exclude-manual-rectangle` excludes all manual-rectangle ROIs.
- `--include-side-view-in-ranking` includes side-view manual recordings in
  per-approach rankings.
- `--input` selects a different aggregate metrics CSV.
- `--output` selects a different best-ROI CSV.
- `--rank-output-dir` selects the ranking output directory.

ROI selection uses a scale-free composite score from MAE, RMSE, absolute
Bland-Altman bias, difference SD, limits-of-agreement width, and heart-rate
agreement. Lower mean rank is better.

## Creating paper tables

The configurable table generator is:

```bash
python -m src.metrics.create_tables_for_paper
```

Edit the `CONFIG` dictionary in
`src/metrics/create_tables_for_paper.py` to choose:

- `APPROACHES`: `own`, `nair`, `yang`, or `["all"]`;
- `HALVED`: include every-second-peak variants;
- `VIEW`: `front` or `side`;
- `MANUAL`: include or exclude manual ROIs;
- `ROIS`: an optional explicit ROI allow-list;
- `MODUS`: `peaks` or `mean`;
- `RANK` and `RANK_OPTION`: rank ROIs within recordings or approaches;
- `ACCURACY_WEIGHT`: optionally weight heart-rate agreement more heavily;
- `COMPARISON_TO_SECOND_BEST`: add comparison tables.

Generated CSV and LaTeX files are written below `saved_results/`.

The older/fixed table builder can also be run with:

```bash
python -m src.metrics.create_tables_helpers
```

It contains predefined side-view, front-view, manual-ROI, ranked-recording,
and best-ROI table builders.

## Additional utilities

### Plot aggregate metrics

`plot_csv_data.py` creates boxplots of metric values by camera. Its `__main__`
block currently reads `saved_results/all_npz_metrics.csv` and writes F1-score
plots below `saved_results/new_approach/results/evaluation_csv/`.

The reusable plotting function is `plot_metric_boxplots_per_camera()`.

### Inspect `.npy` and `.npz` files

Update the `PATH` constant in `visualize_results_npy.py`, then run:

```bash
python -m src.metrics.visualize_results_npy
```

The script prints stored keys, shapes, types, heart-rate-like values, and
adaptive plots for 1D, 2D, and higher-dimensional arrays.

### Test against time-shuffled signals

`metric_pipeline_on_random_signal.py` randomly shuffles the time order of a
predicted signal and evaluates the shuffled baseline.

Update `input_folder` and `output_csv` in its `__main__` block, then run:

```bash
python -m src.metrics.metric_pipeline_on_random_signal
```

By default, it reads
`saved_results/new_approach/for_random_experiment/` and writes
`saved_results/time_shuffled_baseline.csv`.


## Helper modules

- `helpers_metric_pipeline.py`: result-folder and plot helpers.
- `utils/csv_helpers.py`: aggregate CSV schema, migration, and row appending.
- `utils/f1_score_functions.py`: distance-based peak matching and F1 metrics.
- `create_tables_helpers.py`: camera, participant, ROI, sorting, and table
  formatting helpers.
