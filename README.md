# minspecs_eddy

Power/precision trade-off simulations for eddy-covariance systems. Three experiment tracks:
- CO2/H2O degradation (Monte Carlo sampling or univariate sweeps).
- Subsampling/scheduling strategies (power-saving modes, no sensor degradation by default).
- Methane-only degradation (Monte Carlo sampling or univariate sweeps).

## Data expectations
- ICOS-style cached windows in `.npz` under an ecosystem/site hierarchy (e.g., `igbp_ENF/CH-Dav/*.npz`).
- Point `data_root` in the scripts to your cache locations (`D:\data\ec\raw\ICOS_npz` for CO2/H2O, `D:\data\ec\raw\CH4` for methane in the examples).

## How to run

### CO2/H2O degradation (Monte Carlo)
- Entry: `scripts/run_co2_montecarlo.py` (or `scripts/run_experiment.py`)
- Edit `sites`, `theta_ranges`, `N_theta`, and `data_root` as needed.
- Run:
  ```bash
  python scripts/run_co2_montecarlo.py
  ```
- Output: `results_co2_montecarlo.csv`

### CO2/H2O degradation (univariate sweeps)
- Entry: `scripts/run_co2_sweep.py`
- Edit `baseline_theta`, `sweep_map`, and `data_root` as needed.
- Run:
  ```bash
  python scripts/run_co2_sweep.py
  ```
- Output: `results_co2_sweep.csv`

### Subsampling/scheduling experiment
- Entry: `scripts/run_subsampling.py`
- Strategies included: decimate (/2, /5, /10), ogive early-stop, burst modes (5/25, 10/50, 10/110 on/off), diurnal variants (day high / night low for burst and decimate).
- Adjust `sites`, `data_root`, or `build_subsample_specs()` to change schedules.
- Run:
  ```bash
  python scripts/run_subsampling.py
  ```
- Output: `results_subsampling.csv`

### Methane (CH4) degradation (Monte Carlo)
- Entry: `scripts/run_ch4_montecarlo.py`
- Configure `site_list`, `theta_ranges`, and `data_root` for your CH4 cache.
- Run:
  ```bash
  python scripts/run_ch4_montecarlo.py
  ```
- Output: `results_ch4_montecarlo.csv`

### Methane (CH4) degradation (univariate sweeps)
- Entry: `scripts/run_ch4_sweep.py` (or `scripts/run_ch4_perf.py`)
- Configure `site_list`, `baseline_theta`, `sweep_map`, and `data_root` for your CH4 cache.
- Run:
  ```bash
  python scripts/run_ch4_sweep.py
  ```
- Output: `results_ch4_sweep.csv`

### Serial batch runner (CO2/H2O)
- Entry: `scripts/run_co2_batch_serial.py`
- Runs multiple Monte Carlo replicas serially with per-run outputs and optional collation.
- Example:
  ```bash
  python -m scripts.run_co2_batch_serial --data-root D:\data\ec\raw\ICOS_npz --n-runs 6 --n-theta 50
  ```

## Result structure
- CSV rows are per-site per-configuration.
- Metadata columns include ecosystem/site, theta/subsample identifiers, and rotation mode.
- Metrics include regression slope/intercept/R2 on 30-min windows and daily means, plus cumulative bias (monthly mean and full-period totals). Gaps are harmonized by requiring paired finite ref/deg values.

## Metrics & aggregation (summary)
- Fluxes evaluated for CO2/H2O: F_CO2 (umol m-2 s-1), F_LE (W m-2), F_H (W m-2).
- Aggregated metrics: regression slope/intercept/R2 on 30-min window pairs and on daily means.
- Cumulative bias: full-period totals and mean monthly bias, with relative bias.
- Outlier filtering is applied to residuals per flux (default 5th-95th percentiles; configurable).
- Methane (CH4): window-level F_CH4_ref/deg are computed and aggregated with the same regression/bias logic.

## Reproducible runs & collation
- For independent Monte Carlo replicas, keep configuration identical and vary `theta_seed` in the script.
- Run serially (one after the other) and write results to separate files or directories.
- Collate results afterward (e.g., concatenate CSVs and add a run_id).
- See `docs/experiments.md` for detailed metrics and multi-run workflow.

## Checkpoints (per-site)
- Long runs can write per-site CSV checkpoints so completed sites are not lost.
- The batch runner supports this by default (see `scripts/run_co2_batch_serial.py`).
- Checkpoints are written under each run directory in `checkpoints/` and can be disabled with `--no-checkpoints`.

## Key modules
- `minspecs_simulation/main.py`: orchestrators (`run_experiment`, `run_subsampling_experiment`).
- `minspecs_simulation/site_runner.py`: per-site fan-out over windows and configurations.
- `minspecs_simulation/window_processor.py`: per-window CO2/H2O engine (with subsampling hook).
- `minspecs_simulation/ch4_runner.py`, `minspecs_simulation/ch4_window_processor.py`: methane path.
- `minspecs_simulation/types.py`: theta and subsampling spec definitions.
- `minspecs_simulation/results.py`: aggregation logic (regressions and cumulative biases).
- `minspecs_simulation/writer.py`: flatten to CSV.

## Notes
- Subsampling is applied before any degradation to mimic scheduled acquisition; default theta for that path is ideal (no noise/lag).
- Rotation modes default to `double` and `none` for CO2/H2O; methane path is independent.
- For quick tests, use `max_files_per_site`/`max_files` in the scripts to limit windows.
