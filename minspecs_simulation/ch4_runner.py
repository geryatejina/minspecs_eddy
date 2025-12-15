"""
ch4_runner.py
-------------

Site- and experiment-level orchestration for the methane-only simulation
track. This mirrors the existing CO2/H2O runner but keeps all methane logic
isolated so the original pipeline stays untouched.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .ch4_types import MethaneTheta
from .ch4_io import iter_ch4_files, load_ch4_arrays
from .ch4_window_processor import process_ch4_window
from .results import aggregate_window_results


ThetaPlanEntry = Tuple[MethaneTheta, str | None, float | None]


def build_theta_plan(
    baseline: MethaneTheta,
    sweeps: dict[str, Sequence[float]] | None = None,
) -> List[ThetaPlanEntry]:
    """
    Construct a list of MethaneTheta variants for univariate sweeps.

    The first entry is the baseline (no sweep). Each subsequent entry
    changes exactly one parameter to a provided value.
    """
    plan: List[ThetaPlanEntry] = [(baseline, None, None)]
    if not sweeps:
        return plan

    base_dict = baseline.__dict__
    for param, values in sweeps.items():
        if param not in base_dict:
            raise KeyError(f"Unknown MethaneTheta parameter: {param}")
        for val in values:
            theta_kwargs = dict(base_dict)
            theta_kwargs[param] = val
            plan.append((MethaneTheta(**theta_kwargs), param, val))
    return plan


def _process_ch4_file(path: Path, theta_plan: List[ThetaPlanEntry], site_id: str, f_raw: float):
    arrays = load_ch4_arrays(path)

    results = []
    for idx, (theta, sweep_param, sweep_value) in enumerate(theta_plan):
        res = process_ch4_window(
            path=path,
            arrays=arrays,
            theta=theta,
            theta_index=idx,
            site_id=site_id,
            f_raw=f_raw,
        )
        res["sweep_param"] = sweep_param
        res["sweep_value"] = sweep_value
        results.append(res)
    return results


def run_ch4_site(
    site_id: str,
    theta_plan: List[ThetaPlanEntry],
    data_root: Path,
    file_pattern: str = "*.npz",
    max_files: Optional[int] = None,
    max_workers: Optional[int] = None,
    f_raw: float = 20.0,
) -> Dict[Tuple[int, int], Dict]:
    """
    Run methane simulation for one site. Returns structure compatible with
    writer/results: keys are (theta_index, D) where D is fixed to 1.
    """
    file_paths = list(iter_ch4_files(
        site=site_id,
        data_root=data_root,
        pattern=file_pattern,
        max_files=max_files,
    ))

    window_results_by_theta: Dict[int, List[Dict]] = {idx: [] for idx in range(len(theta_plan))}

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_process_ch4_file, path, theta_plan, site_id, f_raw): path
            for path in file_paths
        }
        for f in as_completed(futures):
            for res in f.result():
                window_results_by_theta[res["theta_index"]].append(res)

    aggregated: Dict[Tuple[int, int], Dict] = {}
    for theta_index, window_results in window_results_by_theta.items():
        if not window_results:
            continue
        window_results.sort(key=lambda r: r["window_start"])
        agg = aggregate_window_results(window_results)

        theta, sweep_param, sweep_value = theta_plan[theta_index]
        agg.update(theta.__dict__)
        agg["sweep_param"] = sweep_param
        agg["sweep_value"] = sweep_value
        agg["site_id"] = site_id
        agg["theta_index"] = theta_index
        agg["D"] = 1

        aggregated[(theta_index, 1)] = agg

    return aggregated


def run_ch4_experiment(
    site_list: Sequence[str],
    baseline_theta: MethaneTheta,
    sweep_map: dict[str, Sequence[float]] | None,
    data_root: Path,
    file_pattern: str = "*.npz",
    max_files_per_site: Optional[int] = None,
    max_workers: Optional[int] = None,
    f_raw: float = 20.0,
    ecosystem_label: str = "CH4",
):
    """
    Run methane-only experiment across multiple sites.

    Returns
    -------
    dict: {(ecosystem_label, site): {(theta_index, 1): aggregated_metrics}}
    """
    theta_plan = build_theta_plan(baseline_theta, sweep_map)

    experiment_results: Dict[Tuple[str, str], Dict] = {}
    for site in site_list:
        print(f"[ch4] Processing site {site} with {len(theta_plan)} theta variants...")
        site_results = run_ch4_site(
            site_id=site,
            theta_plan=theta_plan,
            data_root=data_root,
            file_pattern=file_pattern,
            max_files=max_files_per_site,
            max_workers=max_workers,
            f_raw=f_raw,
        )
        experiment_results[(ecosystem_label, site)] = site_results
        print(f"[ch4] Completed site {site}")

    return experiment_results


__all__ = [
    "build_theta_plan",
    "run_ch4_site",
    "run_ch4_experiment",
]
