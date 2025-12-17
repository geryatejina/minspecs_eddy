"""
site_runner.py
--------------

Run the minspecs_eddy simulation for a SINGLE ICOS site.

For each theta:
    Parallel over all 30-min ICOS files:
        - load raw file
        - compute mixing ratios
        - convert DataFrame to arrays
        - run per-window engine (process_single_window)
    Sort window results chronologically
    Aggregate all windows into one summary row
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from .io_icos import (
    iter_site_files,
    load_window_arrays,
)
from .window_processor import process_single_window
from .io_icos import extract_window_timestamp_from_filename
from .window_processor import find_optimal_lag
from .results import aggregate_window_results
from .timelag import get_site_lag_samples


def _process_file_all(
    path: Path,
    theta_list,
    rotation_modes,
    site_id: str,
    lag_samples: int,
):
    """
    Worker helper: read one raw file, then compute window results for all
    thetas. Returns a list of window_result dicts.
    """
    arrays = load_window_arrays(path)

    results = []
    for theta_index, theta in enumerate(theta_list):
        for rotation_mode in rotation_modes:
            results.append(process_single_window(
                str(path),
                arrays,
                theta,
                site_id,
                theta_index,
                rotation_mode,
                lag_samples,
            ))
    return results


def run_site(
    site_id: str,
    ecosystem: str,
    theta_list: List,
    rotation_modes: List[str],
    data_root: Path,
    max_workers: int | None = None,
    file_pattern: str = "*.npz",
    max_files: Optional[int] = None,
    skip_set: Optional[set[int]] = None,
) -> Dict[int, Dict]:
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
    rotation_modes : list[str]
        Discrete processing modes to evaluate (e.g., ["double", "none"])
    data_root : Path
        root folder containing ICOS ecosystem/site directories
    max_workers : int or None
        number of worker processes for file-level parallelism
    max_files : int or None
        optional cap on number of files (windows) per site, for testing
    skip_set : set[int] or None
        optional set of theta indices to skip (for resume)

    Returns
    -------
    dict:
        Keys: (theta_index, rotation_mode)
        Values: aggregated metrics dict
    """

    site_aggregated: Dict[tuple, Dict] = {}

    # Collect all file paths once per site to avoid repeated discovery
    file_paths = list(iter_site_files(
        site=site_id,
        ecosystem=ecosystem,
        data_root=data_root,
        pattern=file_pattern,
        max_files=max_files,
    ))

    # Map (theta_index, rotation_mode) -> list of window results
    combo_keys = [
        (ti, rm) for ti in range(len(theta_list)) for rm in rotation_modes
    ]
    window_results_by_combo: Dict[tuple, List[Dict]] = {
        key: [] for key in combo_keys
    }

    # ------------------------------------------------------------
    # Nominal lag: use hard-coded map if available; else fallback to quick estimate
    # ------------------------------------------------------------
    lag_samples = get_site_lag_samples(ecosystem, site_id)
    if lag_samples is None:
        def _is_daytime(ts):
            return 10 <= ts.hour < 17

        def _within_first_days(ts, start_date):
            return (ts.date() - start_date).days <= 2

        file_paths_sorted = sorted(file_paths)
        if not file_paths_sorted:
            raise FileNotFoundError("No files to process for site.")

        start_ts = extract_window_timestamp_from_filename(file_paths_sorted[0])
        start_date = start_ts.date()

        lag_samples_list = []
        for path in file_paths_sorted:
            ts = extract_window_timestamp_from_filename(path)
            if not _is_daytime(ts):
                continue
            if not _within_first_days(ts, start_date):
                continue
            arrays = load_window_arrays(path)
            w = arrays["w"]
            rho_co2 = arrays["rho_CO2"]
            lag = find_optimal_lag(w, rho_co2, max_lag_samples=15, decimate=2)
            if np.isfinite(lag):
                lag_samples_list.append(int(lag))

        lag_samples = int(np.median(lag_samples_list)) if lag_samples_list else 0
        print(f"[site_runner] {site_id}: using estimated nominal lag {lag_samples} samples (based on {len(lag_samples_list)} windows)")
    else:
        print(f"[site_runner] {site_id}: using hard-coded nominal lag {lag_samples} samples")

    # Parallel over files; each worker reads once and fans out over all combos
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_process_file_all, path, theta_list, rotation_modes, site_id, lag_samples): path
            for path in file_paths
        }
        for f in as_completed(futures):
            for result in f.result():
                key = (result["theta_index"], result["rotation_mode"])
                window_results_by_combo[key].append(result)

    # Aggregate per theta
    for (theta_index, rotation_mode), window_results in window_results_by_combo.items():
        if skip_set and theta_index in skip_set:
            continue
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
        aggregated["rotation_mode"] = rotation_mode

        site_aggregated[(theta_index, rotation_mode)] = aggregated

    return site_aggregated
