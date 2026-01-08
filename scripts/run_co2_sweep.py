import argparse
import os
from pathlib import Path

from minspecs_simulation.main import run_experiment
from minspecs_simulation.types import Theta
from minspecs_simulation.writer import write_results_to_csv
from minspecs_simulation.window_processor import set_empty_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CO2/H2O univariate sweep.")
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
    ]

    baseline_theta = Theta(
        fs_sonic=20.0,
        tau_sonic=0.0,
        sigma_w_noise=0.0,
        sigma_Ts_noise=0.0,
        fs_irga=20.0,
        tau_irga=0.0,
        sigma_CO2dens_noise=0.0,
        sigma_H2Odens_noise=0.0,
        sigma_Tcell_noise=0.0,
        k_CO2_Tsens=0.0,
        k_H2O_Tsens=0.0,
        sigma_lag_jitter=0.0,
    )

    sweep_map = {
        "fs_sonic": [5, 10, 20],
        "tau_irga": [0.02, 0.1, 0.5],
        "sigma_CO2dens_noise": [0.0, 0.05, 0.1],
    }

    results = run_experiment(
        ecosystem_site_list=sites,
        baseline_theta=baseline_theta,
        sweep_map=sweep_map,
        rotation_modes=("double", "none"),
        data_root=Path(r"D:\data\ec\raw\ICOS_npz"),
        file_pattern="*.npz",
        max_workers=8,
        max_files_per_site=1440,
    )

    write_results_to_csv(results, "results_co2_sweep.csv")
