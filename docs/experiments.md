# Experiments and Metrics

This doc complements `README.md` with more detail on evaluation metrics and how to run
consistent, serial Monte Carlo replicas and collate results afterward.

## Metrics and aggregation

### CO2/H2O (eddy-covariance)
Per-window fluxes are computed and then aggregated per (site, theta, rotation_mode).

Window-level fluxes (scaled):
- `F_CO2_ref`, `F_CO2_deg` in umol m-2 s-1
- `F_LE_ref`, `F_LE_deg` in W m-2
- `F_H_ref`,  `F_H_deg` in W m-2

Aggregation (see `minspecs_simulation/results.py`):
- Regression slope/intercept/R2 on 30-min window pairs (ref vs deg).
- Regression slope/intercept/R2 on daily means (paired windows grouped by date).
- Cumulative bias over the full period (sum over windows x 1800 s), and mean monthly bias.
- Relative bias is computed against the paired reference totals.
- Gaps are harmonized by using only paired, finite ref/deg values.
- Outlier filtering removes windows with residuals outside percentile bounds (defaults 5-95).

Optional per-window logs:
- If you run with `--window-logs`, per-window CSVs are written for post-hoc analysis.
- See `minspecs_simulation/site_runner.py` for the log format.

### CH4 (methane-only track)
Per-window methane flux is computed from `F_CH4_ref` and `F_CH4_deg`.
These are aggregated using the same regression and cumulative-bias logic as CO2/H2O.

Additional per-window QC metrics are computed in `ch4_window_processor.py`
(e.g., RMSE, variance ratio, high-frequency energy ratio, fraction NaN), but
they are not aggregated into the final CSV by default.

## Running multiple Monte Carlo replicas (serial)

The simulation is already parallel within a run (per-window processing).
For multiple independent replicas, run them serially and vary `theta_seed`.

Recommended pattern:
1) Keep all other settings identical (sites, theta ranges, N_theta, rotation modes).
2) Change only `theta_seed` per run.
3) Write outputs to separate files or directories.

Examples:
- CO2/H2O Monte Carlo: edit `scripts/run_co2_montecarlo.py` and set `theta_seed`.
- CH4 Monte Carlo: edit `scripts/run_ch4_montecarlo.py` and set `theta_seed`.
- General runner: use `scripts/run_experiment.py --results-dir runs/run_001`
  and set `theta_seed` in the script.

### Batch script (serial)

Use the batch runner to execute multiple CO2/H2O Monte Carlo runs in sequence,
each with its own output directory and optional collation.

Example:
```bash
python -m scripts.run_co2_batch_serial --data-root D:\data\ec\raw\ICOS_npz --n-runs 6 --n-theta 50
```

Key options:
- `--results-root`: base directory for per-run outputs.
- `--n-runs`, `--n-theta`: number of runs and thetas per run.
- `--seed-start`, `--seed-step`: control seeds across runs.
- `--no-collate` or `--collate-path`: control collation output.
- `--no-checkpoints`: disable per-site checkpoints.

## Per-site checkpoints

For long or flaky I/O runs, you can write per-site checkpoint CSVs. This keeps
completed sites on disk even if the run stalls later.

- `run_experiment()` supports `checkpoint_dir` (see `minspecs_simulation/main.py`).
- The batch script enables this by default and writes checkpoints under
  `<run_dir>/checkpoints/`.
- Disable with `--no-checkpoints` if you do not want per-site files.

## Collating results after the fact

Use a small pandas helper to concatenate CSVs and tag each run.

```python
import glob
import pandas as pd
from pathlib import Path

dfs = []
for path in glob.glob(r"runs\\run_*\\results*.csv"):
    run_id = Path(path).parent.name
    df = pd.read_csv(path)
    df["run_id"] = run_id
    dfs.append(df)

out = pd.concat(dfs, ignore_index=True)
out.to_csv("results_all_runs.csv", index=False)
```

If you use per-window logs (`--window-logs`), you can collate those similarly
by concatenating per-site log CSVs and tagging run_id.
