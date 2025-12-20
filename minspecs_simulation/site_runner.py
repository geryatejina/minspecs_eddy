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
from .types import SubsampleSpec, Theta


def _process_file_all(
    path: Path,
    theta_list,
    rotation_modes,
    site_id: str,
    ecosystem: str,
    lag_samples: int,
    subsample_specs: Optional[List[SubsampleSpec]] = None,
):
    """
    Worker helper: read one raw file, then compute window results for all
    thetas. Returns a list of window_result dicts.
    """
    arrays = load_window_arrays(path)

    results = []

    if subsample_specs:
        base_theta = theta_list[0] if theta_list else None
        if base_theta is None:
            raise ValueError("subsample_specs provided but no theta available for processing.")
        for subsample_index, spec in enumerate(subsample_specs):
            for rotation_mode in rotation_modes:
                results.append(process_single_window(
                    str(path),
                    arrays,
                    base_theta,
                    site_id,
                    subsample_index,
                    rotation_mode,
                    lag_samples,
                    ecosystem,
                    subsample_spec=spec,
                    subsample_index=subsample_index,
                ))
    else:
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
                    ecosystem,
                ))
    return results


def run_site(
    site_id: str,
    ecosystem: str,
    theta_list: Optional[List] = None,
    rotation_modes: List[str] = None,
    data_root: Path = None,
    max_workers: int | None = None,
    file_pattern: str = "*.npz",
    max_files: Optional[int] = None,
    skip_set: Optional[set[int]] = None,
    subsample_specs: Optional[List[SubsampleSpec]] = None,
) -> Dict[int, Dict]:
    """
    Run simulation for one site.

    Parameters
    ----------
    site_id : str
        e.g. "CH-Dav"
    ecosystem : str
        e.g. "igbp_ENF"
    theta_list : list[Theta] or None
        list of sampled theta parameter sets (if None, a no-degradation theta is used)
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
    subsample_specs : list[SubsampleSpec] or None
        If provided, run one subsampling strategy per spec (not combined)

    Returns
    -------
    dict:
        Keys: (theta_index, rotation_mode)
        Values: aggregated metrics dict
    """

    site_aggregated: Dict[tuple, Dict] = {}

    if rotation_modes is None:
        rotation_modes = ["double", "none"]

    if theta_list is None:
        theta_list = [Theta(
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
        )]

    # Collect all file paths once per site to avoid repeated discovery
    file_paths = list(iter_site_files(
        site=site_id,
        ecosystem=ecosystem,
        data_root=data_root,
        pattern=file_pattern,
        max_files=max_files,
    ))

    # Map (theta_index, rotation_mode) -> list of window results
    if subsample_specs:
        combo_keys = [
            (si, rm) for si in range(len(subsample_specs)) for rm in rotation_modes
        ]
    else:
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
            ex.submit(_process_file_all, path, theta_list, rotation_modes, site_id, ecosystem, lag_samples, subsample_specs): path
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

        if subsample_specs:
            theta_dict = theta_list[0].__dict__ if theta_list else {}
        else:
            theta_dict = theta_list[theta_index].__dict__
        for key, value in theta_dict.items():
            aggregated[key] = value

        aggregated["site_id"] = site_id
        aggregated["ecosystem"] = ecosystem
        aggregated["theta_index"] = theta_index
        aggregated["rotation_mode"] = rotation_mode

        if subsample_specs:
            spec = subsample_specs[theta_index]
            aggregated["subsample_mode"] = spec.mode.value
            aggregated["subsample_label"] = spec.label()
            aggregated["subsample_index"] = theta_index

            first = window_results[0]
            for meta_key in [
                "target_fs",
                "effective_fs",
                "lag_samples_effective",
                "kept_fraction",
                "decimate_factor",
                "burst_on_sec",
                "burst_off_sec",
                "ogive_stop_time_sec",
                "ogive_threshold",
                "ogive_trailing_sec",
                "ogive_min_dwell_sec",
                "diurnal_phase",
            ]:
                if meta_key in first:
                    aggregated[meta_key] = first[meta_key]

            kept_vals = [w.get("kept_fraction") for w in window_results if w.get("kept_fraction") is not None]
            if kept_vals:
                aggregated["kept_fraction_mean"] = float(np.nanmean(kept_vals))

            stop_vals = [w.get("ogive_stop_time_sec") for w in window_results if w.get("ogive_stop_time_sec") is not None]
            if stop_vals:
                aggregated["ogive_stop_time_sec_mean"] = float(np.nanmean(stop_vals))

            eff_vals = [w.get("effective_fs") for w in window_results if w.get("effective_fs") is not None]
            if eff_vals:
                aggregated["effective_fs_mean"] = float(np.nanmean(eff_vals))

            target_vals = [w.get("target_fs") for w in window_results if w.get("target_fs") is not None]
            if target_vals:
                aggregated["target_fs_mean"] = float(np.nanmean(target_vals))

        site_aggregated[(theta_index, rotation_mode)] = aggregated

    return site_aggregated
