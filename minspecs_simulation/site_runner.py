"""
site_runner.py
--------------

Run the minspecs_eddy simulation for a SINGLE ICOS site.

For each θ:
    For each D:
        Parallel over all 30-min ICOS files:
            - load raw file
            - compute mixing ratios
            - convert DataFrame → arrays
            - run per-window engine (process_single_window)
        Sort window results chronologically
        Aggregate all windows into one summary row
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .io_icos import (
    iter_site_files,
    read_raw_file,
    add_mixing_ratios,
    df_to_arrays,
)
from .window_processor import process_single_window
from .results import aggregate_window_results


def _process_file_all(
    path: Path,
    theta_list,
    D_values,
    site_id: str,
):
    """
    Worker helper: read one raw file, then compute window results for all
    (theta, D) combinations. Returns a list of window_result dicts.
    """
    df = read_raw_file(path)
    df = add_mixing_ratios(df)
    arrays = df_to_arrays(df)

    results = []
    for theta_index, theta in enumerate(theta_list):
        for D in D_values:
            results.append(process_single_window(
                str(path),
                arrays,
                theta,
                D,
                site_id,
                theta_index,
            ))
    return results


def run_site(
    site_id: str,
    ecosystem: str,
    theta_list: List,
    D_values: List[int],
    data_root: Path,
    max_workers: int | None = None,
    max_files: Optional[int] = None,
) -> Dict[Tuple[int, int], Dict]:
    """
    Run simulation for one site.

    Parameters
    ----------
    site_id : str
        e.g. "CH-Dav"
    ecosystem : str
        e.g. "igbp_ENF"
    theta_list : list[Theta]
        list of sampled theta parameter sets
    D_values : list[int]
        decimation factors (e.g., [1,2,3,5,10])
    data_root : Path
        root folder containing ICOS ecosystem/site directories
    max_workers : int or None
        number of worker processes for file-level parallelism
    max_files : int or None
        optional cap on number of files (windows) per site, for testing

    Returns
    -------
    dict:
        Keys: (theta_index, D)
        Values: aggregated metrics dict
    """

    site_aggregated: Dict[Tuple[int, int], Dict] = {}

    # Collect all file paths once per site to avoid repeated discovery
    file_paths = list(iter_site_files(
        site=site_id,
        ecosystem=ecosystem,
        data_root=data_root,
        pattern="*.csv",
        max_files=max_files,
    ))

    # Map (theta_index, D) -> list of window results
    window_results_by_combo: Dict[Tuple[int, int], List[Dict]] = {
        (ti, D): [] for ti in range(len(theta_list)) for D in D_values
    }

    # Parallel over files; each worker reads once and fans out over all combos
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_process_file_all, path, theta_list, D_values, site_id): path
            for path in file_paths
        }
        for f in as_completed(futures):
            for result in f.result():
                key = (result["theta_index"], result["D"])
                window_results_by_combo[key].append(result)

    # Aggregate per (theta, D)
    for (theta_index, D), window_results in window_results_by_combo.items():
        if not window_results:
            continue

        window_results.sort(key=lambda r: r["window_start"])
        aggregated = aggregate_window_results(window_results)

        theta_dict = theta_list[theta_index].__dict__
        for key, value in theta_dict.items():
            aggregated[key] = value

        aggregated["site_id"] = site_id
        aggregated["ecosystem"] = ecosystem
        aggregated["theta_index"] = theta_index
        aggregated["D"] = D

        site_aggregated[(theta_index, D)] = aggregated

    return site_aggregated
