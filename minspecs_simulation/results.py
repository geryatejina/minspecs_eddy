"""
results.py
----------

Flux-focused aggregation:
    - Bias (mean residual) and random error (std residual) for CO2, LE, H
    - Day/night splits
    - Cumulative bias over days, weeks, and full period
"""

from __future__ import annotations

from typing import List, Dict
import numpy as np

WINDOW_SECONDS = 1800.0  # 30-minute windows


def _mean(values):
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))


def _std(values):
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.nanstd(arr, ddof=1))


def _safe_rel_bias(bias, ref):
    if ref == 0 or not np.isfinite(ref) or not np.isfinite(bias):
        return np.nan
    return bias / abs(ref)


def aggregate_window_results(window_results: List[Dict]) -> Dict:
    """
    Aggregate flux metrics for one (site, theta, rotation_mode):
        - Bias and random error (overall, day, night)
        - Cumulative bias over days/weeks/full period
    """
    if len(window_results) == 0:
        raise ValueError("No window results to aggregate.")

    site_id = window_results[0]["site_id"]
    theta_index = window_results[0]["theta_index"]
    rotation_mode = window_results[0].get("rotation_mode")

    agg = {
        "site_id": site_id,
        "theta_index": theta_index,
        "n_windows": len(window_results),
        "n_windows_day": sum(1 for w in window_results if bool(w.get("is_day"))),
        "n_windows_night": sum(1 for w in window_results if w.get("is_day") is not None and not bool(w.get("is_day"))),
    }
    if rotation_mode is not None:
        agg["rotation_mode"] = rotation_mode

    is_day = [bool(w.get("is_day")) for w in window_results]
    night_mask = [not d for d in is_day]

    def subset(vals, mask):
        return [v for v, m in zip(vals, mask) if m]

    # Residuals and refs
    res = {
        "CO2": [w["res_CO2"] for w in window_results],
        "LE": [w["res_LE"] for w in window_results],
        "H": [w["res_H"] for w in window_results],
    }
    ref = {
        "CO2": [w["F_CO2_ref"] for w in window_results],
        "LE": [w["F_LE_ref"] for w in window_results],
        "H": [w["F_H_ref"] for w in window_results],
    }

    # Bias/random error overall and day/night
    for flux in ["CO2", "LE", "H"]:
        bias_all = _mean(res[flux])
        rnd_all = _std(res[flux])
        rel_bias_all = _safe_rel_bias(bias_all, _mean(ref[flux]))

        bias_day = _mean(subset(res[flux], is_day))
        rnd_day = _std(subset(res[flux], is_day))
        rel_bias_day = _safe_rel_bias(bias_day, _mean(subset(ref[flux], is_day)))

        bias_night = _mean(subset(res[flux], night_mask))
        rnd_night = _std(subset(res[flux], night_mask))
        rel_bias_night = _safe_rel_bias(bias_night, _mean(subset(ref[flux], night_mask)))

        agg[f"bias_{flux}"] = bias_all
        agg[f"random_error_{flux}"] = rnd_all
        agg[f"rel_bias_{flux}"] = rel_bias_all

        agg[f"bias_{flux}_day"] = bias_day
        agg[f"random_error_{flux}_day"] = rnd_day
        agg[f"rel_bias_{flux}_day"] = rel_bias_day

        agg[f"bias_{flux}_night"] = bias_night
        agg[f"random_error_{flux}_night"] = rnd_night
        agg[f"rel_bias_{flux}_night"] = rel_bias_night

    # Cumulative sums and biases
    def accumulate(ref_key, deg_key, res_key, grouper):
        groups = {}
        for w in window_results:
            g = grouper(w)
            if g not in groups:
                groups[g] = {"ref": 0.0, "deg": 0.0, "res2": 0.0}
            groups[g]["ref"] += w[ref_key] * WINDOW_SECONDS
            groups[g]["deg"] += w[deg_key] * WINDOW_SECONDS
            groups[g]["res2"] += (w[res_key] * WINDOW_SECONDS) ** 2
        return groups

    def cum_bias_stats(groups):
        biases = []
        rel_biases = []
        for g in groups.values():
            bias = g["deg"] - g["ref"]
            biases.append(bias)
            rel_biases.append(_safe_rel_bias(bias, g["ref"]))
        return _mean(biases), _mean(rel_biases)

    for flux, ref_key, deg_key, res_key in [
        ("CO2", "F_CO2_ref", "F_CO2_deg", "res_CO2"),
        ("LE", "F_LE_ref", "F_LE_deg", "res_LE"),
        ("H", "F_H_ref", "F_H_deg", "res_H"),
    ]:
        period = accumulate(ref_key, deg_key, res_key, lambda w: "all")["all"]
        daily = accumulate(ref_key, deg_key, res_key, lambda w: w["window_start"].date())
        weekly = accumulate(
            ref_key,
            deg_key,
            res_key,
            lambda w: (w["window_start"].isocalendar().year, w["window_start"].isocalendar().week),
        )
        monthly = accumulate(
            ref_key,
            deg_key,
            res_key,
            lambda w: (w["window_start"].year, w["window_start"].month),
        )

        pbias = period["deg"] - period["ref"]
        prel = _safe_rel_bias(pbias, period["ref"])
        prand = np.sqrt(period["res2"])

        dbias_mean, drel_mean = cum_bias_stats(daily)
        wbias_mean, wrel_mean = cum_bias_stats(weekly)
        mbias_mean, mrel_mean = cum_bias_stats(monthly)

        def avg_random(groups):
            vals = []
            for g in groups.values():
                vals.append(np.sqrt(g["res2"]))
            return _mean(vals)

        drand_mean = avg_random(daily)
        wrand_mean = avg_random(weekly)
        mrand_mean = avg_random(monthly)

        agg[f"cum_bias_{flux}_period"] = pbias
        agg[f"cum_rel_bias_{flux}_period"] = prel
        agg[f"cum_random_{flux}_period"] = prand
        agg[f"cum_bias_{flux}_day_mean"] = dbias_mean
        agg[f"cum_rel_bias_{flux}_day_mean"] = drel_mean
        agg[f"cum_random_{flux}_day_mean"] = drand_mean
        agg[f"cum_bias_{flux}_week_mean"] = wbias_mean
        agg[f"cum_rel_bias_{flux}_week_mean"] = wrel_mean
        agg[f"cum_random_{flux}_week_mean"] = wrand_mean
        agg[f"cum_bias_{flux}_month_mean"] = mbias_mean
        agg[f"cum_rel_bias_{flux}_month_mean"] = mrel_mean
        agg[f"cum_random_{flux}_month_mean"] = mrand_mean

    return agg
