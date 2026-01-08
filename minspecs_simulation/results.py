"""
results.py
----------

Aggregation focused on:
    - Regression stats (slope/intercept/r2) on 30-min window fluxes
    - Regression stats on daily-mean fluxes
    - Cumulative bias at month-end and full-period totals
"""

from __future__ import annotations

from typing import List, Dict, Iterable, Tuple
import numpy as np

WINDOW_SECONDS = 1800.0  # 30-minute windows


def _discover_fluxes(window_results: List[Dict]) -> List[str]:
    fluxes = set()
    for w in window_results:
        for key in w.keys():
            if key.startswith("F_") and key.endswith("_ref"):
                flux = key[2:-4]
                if f"F_{flux}_deg" in w:
                    fluxes.add(flux)
    return sorted(fluxes)


def _filter_outliers(
    window_results: List[Dict],
    fluxes: Iterable[str],
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
) -> List[Dict]:
    """
    Drop windows with extreme residuals to reduce skew in aggregation.
    A window is kept only if each flux residual is within [lower_pct, upper_pct] for that flux.
    """
    if len(window_results) < 4:  # too few to percentile-filter meaningfully
        return window_results

    def pct_bounds(vals):
        arr = np.array(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 4:
            return None
        lo = np.nanpercentile(arr, lower_pct)
        hi = np.nanpercentile(arr, upper_pct)
        return lo, hi

    bounds = {}
    for flux in fluxes:
        residuals = []
        ref_key = f"F_{flux}_ref"
        deg_key = f"F_{flux}_deg"
        for w in window_results:
            ref = w.get(ref_key)
            deg = w.get(deg_key)
            if ref is None or deg is None:
                continue
            if np.isfinite(ref) and np.isfinite(deg):
                residuals.append(deg - ref)
        bounds[flux] = pct_bounds(residuals)

    def within(val, bound):
        if bound is None or val is None or not np.isfinite(val):
            return True
        lo, hi = bound
        return lo <= val <= hi

    filtered = []
    for w in window_results:
        keep = True
        for flux in fluxes:
            ref = w.get(f"F_{flux}_ref")
            deg = w.get(f"F_{flux}_deg")
            residual = None
            if ref is not None and deg is not None:
                residual = deg - ref
            if not within(residual, bounds.get(flux)):
                keep = False
                break
        if keep:
            filtered.append(w)

    # Avoid dropping everything; fall back if filtering was too aggressive
    return filtered if filtered else window_results


def _mean(values):
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))


def _safe_rel_bias(bias, ref):
    if ref == 0 or not np.isfinite(ref) or not np.isfinite(bias):
        return np.nan
    return bias / abs(ref)


def _regression_stats(x_vals, y_vals):
    """
    Simple linear regression y = slope * x + intercept.
    Returns (slope, intercept, r2); NaN if not enough finite points.
    """
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan
    x = x[mask]
    y = y[mask]
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    return float(slope), float(intercept), float(r2)


def _pair_series(window_results: List[Dict], flux: str) -> List[Tuple]:
    pairs = []
    ref_key = f"F_{flux}_ref"
    deg_key = f"F_{flux}_deg"
    for w in window_results:
        ts = w.get("window_start")
        ref = w.get(ref_key)
        deg = w.get(deg_key)
        if ts is None or ref is None or deg is None:
            continue
        # Keep only paired finite points to enforce aligned gaps.
        if np.isfinite(ref) and np.isfinite(deg):
            pairs.append((ts, float(ref), float(deg)))
    return pairs


def _daily_means(pairs: List[Tuple]) -> Tuple[List[float], List[float]]:
    groups: Dict = {}
    for ts, ref, deg in pairs:
        if not hasattr(ts, "date"):
            continue
        day = ts.date()
        if day not in groups:
            groups[day] = {"ref": [], "deg": []}
        groups[day]["ref"].append(ref)
        groups[day]["deg"].append(deg)
    daily_ref = []
    daily_deg = []
    for g in groups.values():
        if g["ref"]:
            daily_ref.append(_mean(g["ref"]))
            daily_deg.append(_mean(g["deg"]))
    return daily_ref, daily_deg


def _cum_bias(pairs: List[Tuple]) -> Tuple[float, float]:
    if not pairs:
        return np.nan, np.nan
    ref_sum = sum(ref * WINDOW_SECONDS for _, ref, _ in pairs)
    deg_sum = sum(deg * WINDOW_SECONDS for _, _, deg in pairs)
    bias = deg_sum - ref_sum
    return bias, _safe_rel_bias(bias, ref_sum)


def _monthly_bias_means(pairs: List[Tuple]) -> Tuple[float, float]:
    if not pairs:
        return np.nan, np.nan
    groups: Dict = {}
    for ts, ref, deg in pairs:
        if not hasattr(ts, "year") or not hasattr(ts, "month"):
            continue
        key = (ts.year, ts.month)
        if key not in groups:
            groups[key] = {"ref": 0.0, "deg": 0.0}
        groups[key]["ref"] += ref * WINDOW_SECONDS
        groups[key]["deg"] += deg * WINDOW_SECONDS
    biases = []
    rel_biases = []
    for g in groups.values():
        bias = g["deg"] - g["ref"]
        biases.append(bias)
        rel_biases.append(_safe_rel_bias(bias, g["ref"]))
    return _mean(biases), _mean(rel_biases)


def aggregate_window_results(window_results: List[Dict], lower_pct: float = 5.0, upper_pct: float = 95.0) -> Dict:
    """
    Aggregate metrics for one (site, theta, rotation_mode):
        - Regression stats on window fluxes and daily means
        - Cumulative bias over months and full period
    """
    if len(window_results) == 0:
        raise ValueError("No window results to aggregate.")

    fluxes = _discover_fluxes(window_results)
    if not fluxes:
        raise ValueError("No flux pairs found in window results.")

    window_results = _filter_outliers(window_results, fluxes, lower_pct=lower_pct, upper_pct=upper_pct)

    site_id = window_results[0]["site_id"]
    theta_index = window_results[0]["theta_index"]
    rotation_mode = window_results[0].get("rotation_mode")

    agg = {
        "site_id": site_id,
        "theta_index": theta_index,
    }
    if rotation_mode is not None:
        agg["rotation_mode"] = rotation_mode

    for flux in fluxes:
        pairs = _pair_series(window_results, flux)
        ref_vals = [p[1] for p in pairs]
        deg_vals = [p[2] for p in pairs]

        slope, intercept, r2 = _regression_stats(ref_vals, deg_vals)
        daily_ref, daily_deg = _daily_means(pairs)
        slope_daily, intercept_daily, r2_daily = _regression_stats(daily_ref, daily_deg)

        period_bias, period_rel_bias = _cum_bias(pairs)
        month_bias_mean, month_rel_bias_mean = _monthly_bias_means(pairs)

        agg[f"reg_slope_{flux}"] = slope
        agg[f"reg_intercept_{flux}"] = intercept
        agg[f"reg_r2_{flux}"] = r2

        agg[f"reg_slope_{flux}_daily"] = slope_daily
        agg[f"reg_intercept_{flux}_daily"] = intercept_daily
        agg[f"reg_r2_{flux}_daily"] = r2_daily

        agg[f"cum_bias_{flux}_period"] = period_bias
        agg[f"cum_rel_bias_{flux}_period"] = period_rel_bias
        agg[f"cum_bias_{flux}_month_mean"] = month_bias_mean
        agg[f"cum_rel_bias_{flux}_month_mean"] = month_rel_bias_mean

    return agg
