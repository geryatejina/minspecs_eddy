# scripts/run_sites.py

from minspecs_simulation.main import run_experiment
from minspecs_simulation.sampling import ThetaRanges

if __name__ == "__main__":
    sites = {
        "ENF": "/data/ENF/",
        "GRA": "/data/GRA/",
    }

    ranges = ThetaRanges(
        fs_sonic=(5, 20),
        tau_sonic=(0.01, 1.0),
        sigma_w_noise=(0.0, 0.2),
        sigma_Ts_noise=(0.0, 0.2),
        fs_irga=(5, 20),
        tau_irga=(0.01, 1.0),
        sigma_CO2dens_noise=(0.0, 0.1),
        sigma_H2Odens_noise=(0.0, 0.1),
        sigma_Tcell_noise=(0.0, 1.0),
        k_CO2_Tsens=(-0.01, 0.01),
        k_H2O_Tsens=(-0.01, 0.01),
        sigma_lag_jitter=(0.0, 0.02),
    )

    results = run_experiment(
        sites=sites,
        theta_ranges=ranges,
        N_theta=50,
    )

    print("DONE:", results)
