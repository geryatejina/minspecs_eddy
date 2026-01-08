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
    parser.add_argument(
        "--results-dir",
        help="If set, write results.csv into this directory.",
    )
    parser.add_argument(
        "--window-logs",
        action="store_true",
        help="Write per-window logs into the results directory.",
    )
    args = parser.parse_args()
    if args.empty_log:
        os.environ["MINSPECS_EMPTY_LOG"] = args.empty_log
        set_empty_log(args.empty_log)

    base_dir = Path(args.results_dir) if args.results_dir else Path(".")
    if args.results_dir:
        base_dir.mkdir(parents=True, exist_ok=True)
    window_log_dir = base_dir if args.window_logs else None

    sites = [
        ("igbp_CRO", "BE-Lon"),
        # ("igbp_CRO", "DE-Geb"),
        # ("igbp_CSH", "BE-Maa"),
        # ("igbp_DBF", "CZ-Lnz"),
        # ("igbp_DBF", "DE-HoH"),
        # ("igbp_EBF", "FR-Pue"),
        # ("igbp_ENF", "Be-Bra"),
        # ("igbp_ENF", "CH-Dav"),
        ("igbp_GRA", "BE-Dor"),
        # ("igbp_GRA", "FR-Lqu"),
        # ("igbp_MF",  "BE-Vie"),
        # ("igbp_MF",  "IT-Cp2"),
        # ("igbp_WET", "FI-Sii"),
        # ("igbp_WET", "GL-ZaF"),
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
        N_theta=100,
        rotation_modes=("double", "none"),
        data_root=Path(r"D:\data\ec\raw\ICOS_npz"),
        file_pattern="*.npz",
        max_workers=8,
        max_files_per_site=10,  # 1440 is ~1 month (30 days * 48 half-hour windows)
        theta_seed=42,  # deterministic theta sampling for resume
        skip_map=None,
        window_log_dir=window_log_dir,
    )

    out_path = base_dir / "results.csv"
    write_results_to_csv(results, out_path)
