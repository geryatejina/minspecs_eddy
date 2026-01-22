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

from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import os
import time
import pandas as pd
import numpy as np

from .io_icos import (
    iter_site_files,
    load_window_arrays,
)
from .window_processor import process_single_window, process_single_window_multi, compute_reference_metrics
from .io_icos import extract_window_timestamp_from_filename
from .window_processor import find_optimal_lag
from .results import aggregate_window_results
from .timelag import get_site_lag_samples
from .types import SubsampleSpec, Theta


IO_RETRY_COUNT = int(os.getenv("MINSPECS_IO_RETRIES", "3"))
IO_RETRY_BACKOFF_SEC = float(os.getenv("MINSPECS_IO_RETRY_BACKOFF", "0.5"))
STALL_TIMEOUT_SEC = float(os.getenv("MINSPECS_STALL_TIMEOUT_SEC", "0"))
MAX_POOL_RESTARTS = int(os.getenv("MINSPECS_MAX_POOL_RESTARTS", "2"))


def _progress_bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[?] 0/0"
    frac = min(max(done / total, 0.0), 1.0)
    filled = int(round(width * frac))
    bar = "#" * filled + "-" * (width - filled)
    pct = int(round(frac * 100))
    return f"[{bar}] {done}/{total} ({pct}%)"


def _load_window_arrays_with_retry(path: Path, ecosystem: str, site_id: str):
    """
    Retry transient IO errors when loading window arrays.
    Returns arrays dict on success, or None on failure.
    """
    retries = max(IO_RETRY_COUNT, 0)
    backoff = max(IO_RETRY_BACKOFF_SEC, 0.0)
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return load_window_arrays(path)
        except (PermissionError, OSError) as exc:
            last_exc = exc
            if attempt >= retries:
                break
            sleep_sec = backoff * (2 ** attempt)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
    print(
        f"[site_runner] {ecosystem}/{site_id}: skipping unreadable file {path} "
        f"(after {retries + 1} attempts: {last_exc})"
    )
    return None


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
    arrays = _load_window_arrays_with_retry(path, ecosystem, site_id)
    if arrays is None:
        return []
    reference = compute_reference_metrics(arrays, lag_samples)

    results = []

    if subsample_specs:
        base_theta = theta_list[0] if theta_list else None
        if base_theta is None:
            raise ValueError("subsample_specs provided but no theta available for processing.")
        for subsample_index, spec in enumerate(subsample_specs):
            results.extend(process_single_window_multi(
                str(path),
                arrays,
                base_theta,
                site_id,
                subsample_index,
                rotation_modes,
                lag_samples,
                ecosystem,
                subsample_spec=spec,
                subsample_index=subsample_index,
                reference=reference,
            ))
    else:
        for theta_index, theta in enumerate(theta_list):
            results.extend(process_single_window_multi(
                str(path),
                arrays,
                theta,
                site_id,
                theta_index,
                rotation_modes,
                lag_samples,
                ecosystem,
                reference=reference,
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
    outlier_lower_pct: float = 5.0,
    outlier_upper_pct: float = 95.0,
    window_log_dir: Optional[Path] = None,
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
    total_files = len(file_paths)
    combos = (len(subsample_specs) if subsample_specs else len(theta_list)) * len(rotation_modes)
    print(
        f"[site_runner] {ecosystem}/{site_id}: "
        f"{total_files} window(s) x {combos} combo(s); "
        f"max_workers={max_workers if max_workers is not None else 'default'}"
    )

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
    processed = 0
    report_every = max(1, total_files // 20) if total_files else 1
    remaining_paths = list(file_paths)
    restarts = 0

    while remaining_paths:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_process_file_all, path, theta_list, rotation_modes, site_id, ecosystem, lag_samples, subsample_specs): path
                for path in remaining_paths
            }
            pending = set(futures.keys())
            remaining_paths = []
            try:
                while pending:
                    timeout = STALL_TIMEOUT_SEC if STALL_TIMEOUT_SEC > 0 else None
                    done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
                    if not done:
                        restarts += 1
                        print(
                            f"\n[site_runner] {ecosystem}/{site_id}: "
                            f"stall detected (no completions for {STALL_TIMEOUT_SEC}s); "
                            f"restarting pool ({restarts}/{MAX_POOL_RESTARTS})"
                        )
                        remaining_paths = [futures[f] for f in pending]
                        for fut in pending:
                            fut.cancel()
                        ex.shutdown(wait=False, cancel_futures=True)
                        break

                    for f in done:
                        path = futures[f]
                        try:
                            results = f.result()
                        except Exception as exc:
                            print(f"[site_runner] {ecosystem}/{site_id}: worker failed for {path} ({exc})")
                            results = []
                        for result in results:
                            key = (result["theta_index"], result["rotation_mode"])
                            window_results_by_combo[key].append(result)
                        processed += 1
                        if processed % report_every == 0 or processed == total_files:
                            bar = _progress_bar(processed, total_files)
                            print(f"\r[site_runner] {ecosystem}/{site_id}: {bar}", end="", flush=True)
                            if processed == total_files:
                                print()
            except KeyboardInterrupt:
                for fut in futures:
                    fut.cancel()
                ex.shutdown(wait=False, cancel_futures=True)
                raise

        if remaining_paths:
            if restarts > MAX_POOL_RESTARTS:
                print(
                    f"\n[site_runner] {ecosystem}/{site_id}: max pool restarts exceeded; "
                    f"skipping {len(remaining_paths)} remaining window(s)"
                )
                processed += len(remaining_paths)
                remaining_paths = []
                bar = _progress_bar(processed, total_files)
                print(f"\r[site_runner] {ecosystem}/{site_id}: {bar}", end="", flush=True)
                print()

    # Optional: log per-window metrics. For subsampling, write a wide format:
    # window_start, is_day, ogive_stop_time_sec, then F_{flux}_ref and F_{flux}_{label} per subsample label.
    if window_log_dir:
        log_dir = Path(window_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if subsample_specs:
            labels = [spec.label() for spec in subsample_specs]
            # Collect rows keyed by timestamp
            rows_by_ts = {}
            for (theta_index, rotation_mode), window_results in window_results_by_combo.items():
                label = subsample_specs[theta_index].label() if theta_index is not None else None
                for w in window_results:
                    ts = w.get("window_start")
                    if ts is None:
                        continue
                    key = ts
                    if key not in rows_by_ts:
                        rows_by_ts[key] = {
                            "window_start": ts.isoformat(),
                            "is_day": w.get("is_day"),
                            "ogive_stop_time_sec": w.get("ogive_stop_time_sec"),
                        }
                    row = rows_by_ts[key]
                    # If an ogive window arrives later, populate the stop time
                    if row.get("ogive_stop_time_sec") is None and w.get("ogive_stop_time_sec") is not None:
                        row["ogive_stop_time_sec"] = w.get("ogive_stop_time_sec")
                    for flux in ["CO2", "LE", "H"]:
                        ref_key = f"F_{flux}_ref"
                        if ref_key not in row:
                            row[ref_key] = w.get(ref_key)
                        if label:
                            row[f"F_{flux}_{label}"] = w.get(f"F_{flux}_deg")

            if rows_by_ts:
                log_cols = ["window_start", "is_day", "ogive_stop_time_sec"]
                for flux in ["CO2", "LE", "H"]:
                    log_cols.append(f"F_{flux}_ref")
                    for label in labels:
                        log_cols.append(f"F_{flux}_{label}")
                df_log = pd.DataFrame(rows_by_ts.values())
                df_log.sort_values("window_start", inplace=True)
                df_log = df_log[log_cols]
                log_path = log_dir / f"{ecosystem}_{site_id}_windows_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
                df_log.to_csv(log_path, index=False)
        else:
            # Wide log for non-subsampling paths (one row per window).
            labels = [f"theta{ti}_{rm}" for ti in range(len(theta_list)) for rm in rotation_modes]
            rows_by_ts = {}
            for (theta_index, rotation_mode), window_results in window_results_by_combo.items():
                label = f"theta{theta_index}_{rotation_mode}"
                for w in window_results:
                    ts = w.get("window_start")
                    if ts is None:
                        continue
                    key = ts
                    if key not in rows_by_ts:
                        rows_by_ts[key] = {
                            "window_start": ts.isoformat(),
                            "is_day": w.get("is_day"),
                        }
                    row = rows_by_ts[key]
                    for flux in ["CO2", "LE", "H"]:
                        ref_key = f"F_{flux}_ref"
                        if ref_key not in row:
                            row[ref_key] = w.get(ref_key)
                        row[f"F_{flux}_{label}"] = w.get(f"F_{flux}_deg")

            if rows_by_ts:
                log_cols = ["window_start", "is_day"]
                for flux in ["CO2", "LE", "H"]:
                    log_cols.append(f"F_{flux}_ref")
                    for label in labels:
                        log_cols.append(f"F_{flux}_{label}")
                df_log = pd.DataFrame(rows_by_ts.values())
                df_log.sort_values("window_start", inplace=True)
                df_log = df_log[log_cols]
                log_path = log_dir / f"{ecosystem}_{site_id}_windows_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
                df_log.to_csv(log_path, index=False)

    # Aggregate per theta
    for (theta_index, rotation_mode), window_results in window_results_by_combo.items():
        if skip_set and theta_index in skip_set:
            continue
        if not window_results:
            continue

        window_results.sort(key=lambda r: r["window_start"])
        aggregated = aggregate_window_results(window_results, lower_pct=outlier_lower_pct, upper_pct=outlier_upper_pct)

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

        site_aggregated[(theta_index, rotation_mode)] = aggregated

    print(f"[site_runner] {ecosystem}/{site_id}: aggregation complete for {len(window_results_by_combo)} combos")
    return site_aggregated
