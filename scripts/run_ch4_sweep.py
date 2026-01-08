import argparse
import os
from pathlib import Path

from minspecs_simulation.ch4_types import MethaneTheta
from minspecs_simulation.ch4_runner import run_ch4_experiment
from minspecs_simulation.writer import write_results_to_csv
from minspecs_simulation.window_processor import set_empty_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CH4 univariate sweep.")
    parser.add_argument(
        "--empty-log",
        help="Log empty/NaN arrays: 'stderr', 'stdout', or a file path.",
    )
    args = parser.parse_args()
    if args.empty_log:
        os.environ["MINSPECS_EMPTY_LOG"] = args.empty_log
        set_empty_log(args.empty_log)

    site_list = ["BE-Lon"]

    baseline_theta = MethaneTheta(
        f_eff=20.0,
        tau=0.02,
        sigma_rho=0.0,
        sigma_gain=0.0,
        sigma_drift=0.0,
        sigma_lag=0.0,
    )

    sweep_map = {
        "f_eff": [1.0, 5.0, 10.0, 20.0],
        "tau": [0.02, 0.1, 0.5, 1.0],
        "sigma_rho": [0.3, 2.0, 10.0, 20.0],
        "sigma_gain": [0.001, 0.01, 0.03, 0.05],
        "sigma_drift": [0.0, 10.0, 50.0, 100.0],
        "sigma_lag": [0.0, 0.02, 0.05, 0.10],
    }

    results = run_ch4_experiment(
        site_list=site_list,
        baseline_theta=baseline_theta,
        sweep_map=sweep_map,
        data_root=Path(r"D:\data\ec\raw\CH4"),
        file_pattern="*.npz",
        max_files_per_site=None,
        max_workers=8,
        f_raw=20.0,
        ecosystem_label="CH4",
    )

    write_results_to_csv(results, "results_ch4_sweep.csv")
