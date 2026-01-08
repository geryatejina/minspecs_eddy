import argparse
import os
from pathlib import Path
import pandas as pd
from minspecs_simulation.main import run_experiment
from minspecs_simulation.writer import write_results_to_csv, results_to_dataframe
from minspecs_simulation.window_processor import set_empty_log

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CO2/H2O Monte Carlo experiment.")
    parser.add_argument(
        "--empty-log",
        help="Log empty/NaN arrays: 'stderr', 'stdout', or a file path.",
    )
    args = parser.parse_args()
    if args.empty_log:
        os.environ["MINSPECS_EMPTY_LOG"] = args.empty_log
        set_empty_log(args.empty_log)

    sites = [
        ("igbp_ENF", "CH-Dav"),
        ("igbp_GRA", "BE-Dor"),
        ("igbp_WET", "FI-Sii"),
    ]

    theta_ranges = {
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

    results = run_experiment(
        ecosystem_site_list=sites,
        theta_ranges=theta_ranges,
        N_theta=10,
        rotation_modes=("double", "none"),
        data_root=Path(r"D:\data\ec\raw\ICOS_npz"),
        file_pattern="*.npz",
        max_workers=8,
        max_files_per_site=1440,  # ~1 month (30 days * 48 half-hour windows)
        theta_seed=42,  # deterministic theta sampling for resume
        skip_map=None,
    )

    write_results_to_csv(results, "results.csv")
