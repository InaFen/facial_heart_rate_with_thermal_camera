# Batch runner 

The batch runner reads one CSV per camera:

```text
experiments/experiments_info/lepton_experiments.csv
```

Each CSV row should contain at least:

- `recording_path`: directory or recording used as input
- `camera`: currently `lepton`, `vario`, or `mwir`
- `person`: participant identifier
- `before_after_sport`: recording condition
- `area`: `face` or `temple`
- `timeframe_use`: signal interval as `start,end`
- `ground_truth_path`: extracted ground-truth `.npz` file
- `offset_ground_truth`: synchronization offset in seconds

Paths in the checked-in YAML files and experiment CSVs are local machine
paths. Adapt them to your data locations before running a pipeline.

## Running a batch

The recommended entry point is the unified batch runner. Edit the
`CONFIGURATION` block in `scripts/run_batch.py` to select pipelines, cameras,
participants, ROIs, row limits, and visualization mode:

```bash
python scripts/run_batch.py
```

Supported pipeline names are:

- `new`
- `yang`
- `nair`

Supported camera names are `lepton`, `vario`, and `mwir`.

Batch output is written below `saved_results/<pipeline>/batch_runs/<camera>/`
or `saved_results/<pipeline>/visualization_runs/<camera>/`. It can include
generated configs, plots, ROI overlays, videos, metrics, and per-run batch
reports.

For manual ROIs, set `manual_roi_backend`, enable reuse/overwrite behavior,
and provide the relevant ROI name in `roi_configs`. Cached masks are stored
below `saved_results/manual_roi_masks/`.
