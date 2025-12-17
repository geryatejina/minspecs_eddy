from pathlib import Path
import pandas as pd
from minspecs_simulation.main import run_experiment
from minspecs_simulation.writer import write_results_to_csv, results_to_dataframe

if __name__ == "__main__":

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
        max_files_per_site=336,  # first week (7 days * 48 half-hour windows)
        theta_seed=42,  # deterministic theta sampling for resume
        skip_map=None,
    )

    write_results_to_csv(results, "results.csv")
