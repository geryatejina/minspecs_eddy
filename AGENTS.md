# AGENTS.md

Project-specific guidance for the Codex CLI agent working in this repo.

## Scope
- Repo: `minspecs_eddy`
- Purpose: simulate power/precision trade-offs for eddy-covariance systems (CO2/H2O/CH4).

## Ground rules
- Avoid running long simulations unless explicitly requested.
- Prefer reading existing scripts/README for run configs; do not assume data paths.
- Keep edits minimal and ASCII-only unless the file already contains non-ASCII.

## Data expectations
- ICOS-style cached windows in `.npz` under ecosystem/site folders.
- Typical example roots (may differ on your machine):
  - CO2/H2O: `D:\data\ec\raw\ICOS_npz`
  - CH4: `D:\data\ec\raw\CH4`
- If data is missing, ask before changing any paths.

## How to run (entry points)
- CO2/H2O Monte Carlo: `python scripts/run_co2_montecarlo.py`
- CO2/H2O sweeps: `python scripts/run_co2_sweep.py`
- Subsampling: `python scripts/run_subsampling.py`
- CH4 Monte Carlo: `python scripts/run_ch4_montecarlo.py`
- CH4 sweeps: `python scripts/run_ch4_sweep.py`
- General runner (CO2/H2O, optional window logs): `python scripts/run_experiment.py --results-dir <dir> --window-logs`

## Metrics (high level)
- Aggregation happens in `minspecs_simulation/results.py`.
- Regression stats on 30-min windows and daily means: slope/intercept/R2.
- Cumulative bias over full period + mean monthly bias (and relative bias).
- CH4 adds per-window RMSE, variance ratio, HF ratio, fraction NaN.

## Reproducibility / batching
- Monte Carlo sampling is controlled by `theta_seed` in the scripts.
- For independent replicas, vary `theta_seed` and write outputs to separate dirs.
- Collate runs after-the-fact rather than merging mid-run.

## Performance / parallelism
- Per-run parallelism uses `ProcessPoolExecutor` in `site_runner.py` and `ch4_runner.py`.
- If running multiple runs concurrently, reduce `max_workers` per run to avoid oversubscription.

## Logging
- Optional per-window logs via `--window-logs` (see `site_runner.py` for format).

