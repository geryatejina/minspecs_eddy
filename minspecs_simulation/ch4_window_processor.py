"""
ch4_window_processor.py
-----------------------

Per-window methane degradation and flux computation. This is a parallel
path to the CO2/H2O engine and leaves that code untouched.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .ch4_degradation import degrade_methane_density, resample_reference, resample_uniform
from .ch4_types import MethaneTheta
from .ch4_io import window_id_from_path
from .window_processor import covariance, detrend
from .qc import high_freq_energy, fraction_nan, safe_var


def _align_ref(w, rho):
    min_len = min(len(w), len(rho))
    return w[:min_len], rho[:min_len]


def process_ch4_window(
    path,
    arrays,
    theta: MethaneTheta,
    theta_index: int,
    site_id: str,
    f_raw: float,
    seed=None,
):
    """
    Process a single methane window:
        - compute reference flux at raw rate
        - apply methane-only degradations (D_theta)
        - resample w to f_eff for covariance
        - compute degraded flux and QC metrics
    """
    rng = np.random.default_rng(seed)
    window_start = window_id_from_path(Path(path))

    w_raw = np.asarray(arrays["w"], dtype=float)
    rho_raw = np.asarray(arrays["rho_CH4"], dtype=float)

    if w_raw.size == 0 or rho_raw.size == 0:
        return {
            "site_id": site_id,
            "theta_index": theta_index,
            "window_start": window_start,
            "F_CH4_ref": np.nan,
            "F_CH4_deg": np.nan,
            "bias_CH4": np.nan,
            "rel_bias_CH4": np.nan,
            "rmse_CH4": np.nan,
            "var_ratio_CH4": np.nan,
            "hf_ratio_CH4": np.nan,
            "frac_nan_ch4": np.nan,
            "n_samples_raw": int(w_raw.size),
            "n_samples_eff": 0,
        }

    w_ref, rho_ref = _align_ref(w_raw, rho_raw)

    # Reference flux at raw rate (ideal sonic + ideal methane)
    F_CH4_ref = covariance(w_ref, rho_ref)

    # Reference resampled to f_eff for like-with-like comparisons
    rho_ref_eff = resample_reference(rho_ref, f_raw, theta.f_eff)
    w_eff = resample_uniform(w_ref, f_raw, theta.f_eff)

    # Degraded methane density at f_eff
    rho_deg = degrade_methane_density(rho_ref, theta, f_raw, rng)

    # Degraded flux using synchronous w at f_eff
    F_CH4_deg = covariance(w_eff, rho_deg)

    bias = F_CH4_deg - F_CH4_ref
    rel_bias = bias / abs(F_CH4_ref) if F_CH4_ref != 0 else np.nan

    rmse = float(np.sqrt(np.nanmean((rho_deg - rho_ref_eff) ** 2)))

    var_ref = safe_var(rho_ref_eff)
    var_deg = safe_var(rho_deg)
    var_ratio = var_deg / var_ref if var_ref > 0 else np.nan

    hf_ref = high_freq_energy(rho_ref_eff) if rho_ref_eff.size >= 200 else np.nan
    hf_deg = high_freq_energy(rho_deg) if rho_deg.size >= 200 else np.nan
    hf_ratio = hf_deg / hf_ref if (not np.isnan(hf_ref)) and hf_ref != 0 else np.nan

    return {
        "site_id": site_id,
        "theta_index": theta_index,
        "window_start": window_start,
        "F_CH4_ref": F_CH4_ref,
        "F_CH4_deg": F_CH4_deg,
        "bias_CH4": bias,
        "rel_bias_CH4": rel_bias,
        "rmse_CH4": rmse,
        "var_ratio_CH4": var_ratio,
        "hf_ratio_CH4": hf_ratio,
        "frac_nan_ch4": fraction_nan(rho_deg),
        "n_samples_raw": int(w_ref.size),
        "n_samples_eff": int(rho_deg.size),
    }


__all__ = ["process_ch4_window"]
