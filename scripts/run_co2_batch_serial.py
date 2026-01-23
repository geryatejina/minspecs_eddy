import argparse
import os
from pathlib import Path

import pandas as pd

from minspecs_simulation.main import run_experiment
from minspecs_simulation.writer import results_to_dataframe
from minspecs_simulation.window_processor import set_empty_log


def build_theta_ranges():
    return {
        "fs_sonic": (5, 20),
        "tau_sonic": (0.01, 1.0),
        "sigma_w_noise": (0.0, 0.2),
        "sigma_Ts_noise": (0.0, 0.2),
        "fs_irga": (5, 20),
        "tau_irga": (0.01, 1.0),
        "sigma_CO2dens_noise": (0.0, 0.1),
        "sigma_H2Odens_noise": (0.0, 0.1),
        "sigma_Tcell_noise": (0.0, 1.0),
        "k_CO2_Tsens": (-0.01, 0.01),
        "k_H2O_Tsens": (-0.01, 0.01),
        "sigma_lag_jitter": (0.0, 0.02),
    }


def parse_rotation_modes(value):
    modes = [m.strip() for m in value.split(",")]
    return [m for m in modes if m]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run serial CO2/H2O Monte Carlo batches over all sites."
    )
    parser.add_argument(
        "--data-root",
        default=r"D:\data\ec\raw\ICOS_npz",
        help="Root folder containing ecosystem/site window caches.",
    )
    parser.add_argument(
        "--results-root",
        default="runs/serial_co2_batch",
        help="Root folder to store per-run outputs.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=6,
        help="Number of serial runs to execute.",
    )
    parser.add_argument(
        "--n-theta",
        type=int,
        default=50,
        help="Number of theta samples per run.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="Seed for the first run; subsequent runs add seed-step.",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Seed increment between runs.",
    )
    parser.add_argument(
        "--rotation-modes",
        default="double,none",
        help="Comma-separated rotation modes to evaluate.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Worker processes per run.",
    )
    parser.add_argument(
        "--file-pattern",
        default="*.npz",
        help="Glob for window files.",
    )
    parser.add_argument(
        "--window-logs",
        action="store_true",
        help="Write per-window logs into each run directory.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Skip per-site checkpoint CSVs.",
    )
    parser.add_argument(
        "--no-collate",
        action="store_true",
        help="Skip writing a collated CSV across all runs.",
    )
    parser.add_argument(
        "--collate-path",
        default=None,
        help="Output path for collated CSV (default: <results-root>/results_all_runs.csv).",
    )
    parser.add_argument(
        "--empty-log",
        help="Log empty/NaN arrays: 'stderr', 'stdout', or a file path.",
    )
    args = parser.parse_args()

    if args.empty_log:
        os.environ["MINSPECS_EMPTY_LOG"] = args.empty_log
        set_empty_log(args.empty_log)

    if args.n_runs < 1:
        raise ValueError("n-runs must be >= 1")
    if args.n_theta < 1:
        raise ValueError("n-theta must be >= 1")

    rotation_modes = parse_rotation_modes(args.rotation_modes)
    theta_ranges = build_theta_ranges()

    base_dir = Path(args.results_root)
    base_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for run_idx in range(args.n_runs):
        seed = args.seed_start + run_idx * args.seed_step
        run_id = f"run_{run_idx + 1:03d}"
        run_dir = base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        window_log_dir = run_dir if args.window_logs else None

        print(f"[batch] Starting {run_id} with theta_seed={seed} ...")
        checkpoint_dir = None if args.no_checkpoints else (run_dir / "checkpoints")
        results = run_experiment(
            ecosystem_site_list=None,  # auto-discover all sites under data_root
            theta_ranges=theta_ranges,
            N_theta=args.n_theta,
            rotation_modes=rotation_modes,
            data_root=Path(args.data_root),
            file_pattern=args.file_pattern,
            max_workers=args.max_workers,
            max_files_per_site=336,
            theta_seed=seed,
            skip_map=None,
            window_log_dir=window_log_dir,
            checkpoint_dir=checkpoint_dir,
        )

        df = results_to_dataframe(results)
        if not df.empty:
            df["run_id"] = run_id
            df["theta_seed"] = seed
            dfs.append(df)

        out_path = run_dir / "results.csv"
        df.to_csv(out_path, index=False)
        print(f"[batch] Saved {out_path}")

    if not args.no_collate and dfs:
        collated = dfs[0].copy() if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)
        collate_path = Path(args.collate_path) if args.collate_path else (base_dir / "results_all_runs.csv")
        collated.to_csv(collate_path, index=False)
        print(f"[batch] Collated results saved to {collate_path}")
