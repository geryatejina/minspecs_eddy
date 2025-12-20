# minspecs_eddy

Power/precision trade-off simulations for eddy-covariance systems. Three experiment tracks:
- CO₂/H₂O degradation sweep (baseline minspecs).
- Subsampling/scheduling strategies (power-saving modes, no sensor degradation by default).
- Methane-only degradation sweep (separate CH₄ pipeline).

## Data expectations
- ICOS-style cached windows in `.npz` under an ecosystem/site hierarchy (e.g., `igbp_ENF/CH-Dav/*.npz`).
- Point `data_root` in the scripts to your cache locations (`D:\data\ec\raw\ICOS_npz` for CO₂/H₂O, `D:\data\ec\raw\CH4` for methane in the examples).

## How to run

### CO₂/H₂O degradation sweep
- Entry: `scripts/run_experiment.py`
- Edit `sites`, `theta_ranges`, `N_theta`, and `data_root` as needed.
- Run:
  ```bash
  python scripts/run_experiment.py
  ```
- Output: `results.csv`

### Subsampling/scheduling experiment
- Entry: `scripts/run_subsampling.py`
- Strategies included: decimate (/2, /5, /10), ogive early-stop, burst modes (5/25, 10/50, 10/110 on/off), diurnal variants (day high / night low for burst and decimate).
- Adjust `sites`, `data_root`, or `build_subsample_specs()` to change schedules.
- Run:
  ```bash
  python scripts/run_subsampling.py
  ```
- Output: `results_subsampling.csv`

### Methane (CH₄) degradation sweep
- Entry: `scripts/run_ch4_perf.py`
- Configure `site_list`, `baseline_theta`, `sweep_map`, and `data_root` for your CH₄ cache.
- Run:
  ```bash
  python scripts/run_ch4_perf.py
  ```
- Output: `results_ch4.csv`

## Result structure
- CSV rows are per-site per-configuration.
- Metadata columns include ecosystem/site, theta/subsample identifiers, rotation mode, and window counts.
- Metrics include flux biases, random errors, cumulative day/week/month stats (from `minspecs_simulation/results.py`).

## Key modules
- `minspecs_simulation/main.py`: orchestrators (`run_experiment`, `run_subsampling_experiment`).
- `minspecs_simulation/site_runner.py`: per-site fan-out over windows and configurations.
- `minspecs_simulation/window_processor.py`: per-window CO₂/H₂O engine (with subsampling hook).
- `minspecs_simulation/ch4_runner.py`, `minspecs_simulation/ch4_window_processor.py`: methane path.
- `minspecs_simulation/types.py`: theta and subsampling spec definitions.
- `minspecs_simulation/results.py`: aggregation logic (bias, random error, cumulative day/week/month).
- `minspecs_simulation/writer.py`: flatten to CSV.

## Notes
- Subsampling is applied before any degradation to mimic scheduled acquisition; default theta for that path is ideal (no noise/lag).
- Rotation modes default to `double` and `none` for CO₂/H₂O; methane path is independent.
- For quick tests, use `max_files_per_site`/`max_files` in the scripts to limit windows.
