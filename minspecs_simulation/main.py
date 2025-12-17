"""
main.py
-------

High-level orchestration of the minspecs_eddy simulation.

- Loops over sites
- Samples theta values
- Calls run_site() for each site
- Returns a nested dict keyed by (ecosystem, site)
"""

from __future__ import annotations
from pathlib import Path

from .sampling import sample_thetas
from .site_runner import run_site
from .io_icos import ecosystem_sites

def run_experiment(
    ecosystem_site_list=None,
    theta_ranges: dict = None,
    N_theta: int = 10,
    rotation_modes: list[str] | tuple[str, ...] = ("double", "none"),
    data_root: Path = None,
    max_workers: int | None = None,
    max_files_per_site: int | None = None,
    file_pattern: str = "*.npz",
    theta_seed: int | None = None,
    skip_map: dict | None = None,
):    
    """
    Run the entire experiment across all sites.

    Parameters
    ----------
    ecosystem_site_list : list[tuple(str, str)]
        Example: [("igbp_ENF", "CH-Dav"), ("igbp_GRA", "DE-Gri")]
    theta_ranges : dict
        dict of parameter ranges, e.g. {"fs_sonic": (5,20), ...}
    N_theta : int
        number of theta samples to generate
    rotation_modes : list[str] or tuple[str, ...]
        Discrete processing modes (e.g., ["double", "none"] for rotated vs non-rotated with correction)
    data_root : Path
        ICOS root directory containing ecosystem/site folders
    max_workers : int or None
        number of processes for file-level parallelism
    max_files_per_site : int or None
        optional cap on number of 30-min windows per site (for test runs)
    file_pattern : str
        glob pattern for window files (e.g., '*.npz' for cached arrays)
    theta_seed : int or None
        RNG seed for theta sampling (for reproducibility/resume)
    skip_map : dict or None
        optional mapping {(ecosystem, site): set(theta_index)} to skip already processed combos

    Returns
    -------
    dict : {
        (ecosystem, site): {
            theta_index: aggregated_metrics_dict,
            ...
        }
    }
    """
    # --- automatic site discovery ---
    if ecosystem_site_list is None:
        if data_root is None:
            raise ValueError("If ecosystem_site_list is not provided, data_root must be set.")

        print(f"[main] Discovering ICOS sites in {data_root} ...")
        ecosystem_site_list = list(ecosystem_sites(data_root))

        print(f"[main] Found {len(ecosystem_site_list)} sites:")
        for eco, site in ecosystem_site_list:
            print(f"   - {eco}/{site}")


    # --- sample theta values ---
    print(f"[main] Sampling {N_theta} theta values...")
    theta_list = sample_thetas(N_theta, theta_ranges, seed=theta_seed)

    experiment_results = {}
    rotation_modes = list(rotation_modes)

    # --- loop over sites ---
    for ecosystem, site in ecosystem_site_list:
        print(f"\n[main] === Processing site: {ecosystem}/{site} ===")

        site_results = run_site(
            site_id=site,
            ecosystem=ecosystem,
            theta_list=theta_list,
            rotation_modes=rotation_modes,
            data_root=data_root,
            max_workers=max_workers,
            max_files=max_files_per_site,
            file_pattern=file_pattern,
            skip_set=(skip_map or {}).get((ecosystem, site)),
        )

        experiment_results[(ecosystem, site)] = site_results
        print(f"[main] Completed site: {ecosystem}/{site}")

    print("\n[main] All sites processed.")
    return experiment_results
