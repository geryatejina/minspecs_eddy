"""
results.py
----------

Aggregates window-level QC + flux metrics into a single summary
record for (site, θ, D).

Input: list of dictionaries, each from process_window_for_theta_D()
Output: one dictionary with aggregated metrics
"""

from __future__ import annotations
from typing import List, Dict
import numpy as np


def _mean(values):
    """Safe mean ignoring NaN."""
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))


def _fraction(values):
    """Fraction of True or 1 values."""
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))


def aggregate_window_results(window_results: List[Dict]) -> Dict:
    """
    Given a list of window-level metric dicts (one per 30-min file),
    return a single aggregated record.

    Each window_result dict contains:
        site_id, theta_index, D,
        F_CO2_ref, F_CO2_deg, bias_CO2, ...
        + 20+ QC metrics

    We compute mean, fraction, etc.
    """
    if len(window_results) == 0:
        raise ValueError("No window results to aggregate.")

    # Extract keys that are numeric metrics
    numeric_keys = [
        k for k in window_results[0].keys()
        if isinstance(window_results[0][k], (int, float, np.floating))
    ]

    # Metadata fields (same for all windows)
    site_id = window_results[0]["site_id"]
    theta_index = window_results[0]["theta_index"]
    D = window_results[0]["D"]

    # Accumulation structure
    agg = {
        "site_id": site_id,
        "theta_index": theta_index,
        "D": D,
        "n_windows": len(window_results),
    }

    # Collect lists per metric
    metric_lists = {key: [] for key in numeric_keys}

    for w in window_results:
        for key in numeric_keys:
            metric_lists[key].append(w[key])

    # Compute aggregations
    for key, values in metric_lists.items():

        # Boolean-like metrics detected by naming convention
        if key.startswith("sign_flip_"):
            agg[key] = _fraction(values)
        elif key.startswith("frac_nan_"):
            agg[key] = _mean(values)
        elif key.startswith("hf_ratio_"):
            agg[key] = _mean(values)
        else:
            # default: mean
            agg[key] = _mean(values)

    return agg
