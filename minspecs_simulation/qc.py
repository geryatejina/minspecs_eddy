"""
qc.py
------

Quality-control and degradation diagnostics for a single 30-min window.

Metrics included:
    - Variance ratios (σ²_deg / σ²_ref)
    - Covariance ratios (cov_deg / cov_ref)
    - Flux RMSE
    - Relative bias
    - Sign change indicator
    - Normalized MAD
    - Fraction of NaN/invalid samples
    - Energy-loss proxy metric (ratio of high-frequency variance)
"""

import numpy as np
from .window_processor import detrend, covariance


# --------------------------------------------------------------
# Utility statistics
# --------------------------------------------------------------

def safe_var(x):
    return float(np.nanvar(detrend(x)))


def safe_cov(w, x):
    return float(np.nanmean(detrend(w) * detrend(x)))


def safe_mean(x):
    return float(np.nanmean(x))


def fraction_nan(x):
    return float(np.isnan(x).sum() / len(x))


# --------------------------------------------------------------
# Energy-loss proxy (spectral)
# --------------------------------------------------------------

def high_freq_energy(x, cutoff_index=50):
    """
    Compute an approximate proxy for high-frequency energy.
    cutoff_index = number of highest-frequency bins to average.

    This is NOT a PSD integration, but a fast diagnostic:
        - FFT
        - take magnitude spectrum
        - average N bins at top end

    You cannot compute full cospectra cheaply here, but this
    proxy is excellent at detecting low-pass attenuation.
    """
    x = detrend(x)
    n = len(x)
    if n < 200:
        return np.nan

    X = np.fft.rfft(x)
    mag = np.abs(X)
    return float(np.nanmean(mag[-cutoff_index:]))


# --------------------------------------------------------------
# Main QC function
# --------------------------------------------------------------

def compute_qc_metrics(
    w_ref, Ts_ref, mrCO2_ref, mrH2O_ref,
    w_deg, Ts_deg, mrCO2_deg, mrH2O_deg,
    F_CO2_ref, F_CO2_deg,
    F_LE_ref, F_LE_deg,
    F_H_ref, F_H_deg,
):
    """
    Produce a dict of QC metrics for one 30-min window.
    """

    # Align reference and degraded series to the same length
    min_len = min(
        len(w_ref), len(Ts_ref), len(mrCO2_ref), len(mrH2O_ref),
        len(w_deg), len(Ts_deg), len(mrCO2_deg), len(mrH2O_deg),
    )
    if min_len == 0:
        return {  # all metrics NaN if no overlapping samples
            "var_ratio_w": np.nan,
            "var_ratio_T": np.nan,
            "var_ratio_CO2": np.nan,
            "var_ratio_H2O": np.nan,
            "cov_ratio_T": np.nan,
            "cov_ratio_CO2": np.nan,
            "cov_ratio_H2O": np.nan,
            "rel_bias_CO2": np.nan,
            "rel_bias_LE": np.nan,
            "rel_bias_H": np.nan,
            "rmse_CO2": np.nan,
            "rmse_H2O": np.nan,
            "rmse_T": np.nan,
            "rmse_w": np.nan,
            "mad_CO2": np.nan,
            "mad_H2O": np.nan,
            "mad_T": np.nan,
            "mad_w": np.nan,
            "sign_flip_CO2": np.nan,
            "sign_flip_LE": np.nan,
            "sign_flip_H": np.nan,
            "frac_nan_deg": np.nan,
            "hf_ratio_CO2": np.nan,
            "hf_ratio_w": np.nan,
        }

    w_ref = w_ref[:min_len]
    Ts_ref = Ts_ref[:min_len]
    mrCO2_ref = mrCO2_ref[:min_len]
    mrH2O_ref = mrH2O_ref[:min_len]

    w_deg = w_deg[:min_len]
    Ts_deg = Ts_deg[:min_len]
    mrCO2_deg = mrCO2_deg[:min_len]
    mrH2O_deg = mrH2O_deg[:min_len]

    # ---------------------------------------------
    # Variances
    # ---------------------------------------------
    var_w_ref  = safe_var(w_ref)
    var_w_deg  = safe_var(w_deg)
    var_T_ref  = safe_var(Ts_ref)
    var_T_deg  = safe_var(Ts_deg)

    var_CO2_ref = safe_var(mrCO2_ref)
    var_CO2_deg = safe_var(mrCO2_deg)

    var_H2O_ref = safe_var(mrH2O_ref)
    var_H2O_deg = safe_var(mrH2O_deg)

    # Variance ratios
    vr_w   = var_w_deg   / var_w_ref   if var_w_ref > 0 else np.nan
    vr_T   = var_T_deg   / var_T_ref   if var_T_ref > 0 else np.nan
    vr_CO2 = var_CO2_deg / var_CO2_ref if var_CO2_ref > 0 else np.nan
    vr_H2O = var_H2O_deg / var_H2O_ref if var_H2O_ref > 0 else np.nan

    # ---------------------------------------------
    # Covariances (other than fluxes)
    # ---------------------------------------------
    cov_wT_ref  = safe_cov(w_ref, Ts_ref)
    cov_wT_deg  = safe_cov(w_deg, Ts_deg)

    cov_wCO2_ref = safe_cov(w_ref, mrCO2_ref)
    cov_wCO2_deg = safe_cov(w_deg, mrCO2_deg)

    cov_wH2O_ref = safe_cov(w_ref, mrH2O_ref)
    cov_wH2O_deg = safe_cov(w_deg, mrH2O_deg)

    cov_ratio_T   = cov_wT_deg   / cov_wT_ref   if cov_wT_ref   != 0 else np.nan
    cov_ratio_CO2 = cov_wCO2_deg / cov_wCO2_ref if cov_wCO2_ref != 0 else np.nan
    cov_ratio_H2O = cov_wH2O_deg / cov_wH2O_ref if cov_wH2O_ref != 0 else np.nan

    # ---------------------------------------------
    # Flux biases (already known)
    # ---------------------------------------------
    bias_CO2 = F_CO2_deg - F_CO2_ref
    bias_LE  = F_LE_deg  - F_LE_ref
    bias_H   = F_H_deg   - F_H_ref

    # Relative biases
    rel_bias_CO2 = bias_CO2 / abs(F_CO2_ref) if F_CO2_ref != 0 else np.nan
    rel_bias_LE  = bias_LE  / abs(F_LE_ref)  if F_LE_ref  != 0 else np.nan
    rel_bias_H   = bias_H   / abs(F_H_ref)   if F_H_ref   != 0 else np.nan

    # RMSE
    rmse_CO2 = np.sqrt((mrCO2_deg - mrCO2_ref)**2).mean()
    rmse_H2O = np.sqrt((mrH2O_deg - mrH2O_ref)**2).mean()
    rmse_T   = np.sqrt((Ts_deg    - Ts_ref)**2).mean()
    rmse_w   = np.sqrt((w_deg     - w_ref)**2).mean()

    # MAD
    mad_CO2 = np.nanmean(np.abs(mrCO2_deg - mrCO2_ref))
    mad_H2O = np.nanmean(np.abs(mrH2O_deg - mrH2O_ref))
    mad_T   = np.nanmean(np.abs(Ts_deg    - Ts_ref))
    mad_w   = np.nanmean(np.abs(w_deg     - w_ref))

    # ---------------------------------------------
    # Sign-change indicator
    # ---------------------------------------------
    sign_flip_CO2 = int(np.sign(F_CO2_ref) != np.sign(F_CO2_deg))
    sign_flip_LE  = int(np.sign(F_LE_ref)  != np.sign(F_LE_deg))
    sign_flip_H   = int(np.sign(F_H_ref)   != np.sign(F_H_deg))

    # ---------------------------------------------
    # Simple "fraction of invalid samples"
    # ---------------------------------------------
    frac_nan_deg = float(
        np.isnan(w_deg).sum()
        + np.isnan(mrCO2_deg).sum()
        + np.isnan(mrH2O_deg).sum()
    ) / (len(w_deg) * 3)

    # ---------------------------------------------
    # Collect everything into a dict
    # ---------------------------------------------
    return dict(
        var_ratio_w    = vr_w,
        var_ratio_T    = vr_T,
        var_ratio_CO2  = vr_CO2,
        var_ratio_H2O  = vr_H2O,

        cov_ratio_T    = cov_ratio_T,
        cov_ratio_CO2  = cov_ratio_CO2,
        cov_ratio_H2O  = cov_ratio_H2O,

        rel_bias_CO2   = rel_bias_CO2,
        rel_bias_LE    = rel_bias_LE,
        rel_bias_H     = rel_bias_H,

        rmse_CO2       = rmse_CO2,
        rmse_H2O       = rmse_H2O,
        rmse_T         = rmse_T,
        rmse_w         = rmse_w,

        mad_CO2        = mad_CO2,
        mad_H2O        = mad_H2O,
        mad_T          = mad_T,
        mad_w          = mad_w,

        sign_flip_CO2  = sign_flip_CO2,
        sign_flip_LE   = sign_flip_LE,
        sign_flip_H    = sign_flip_H,

        frac_nan_deg   = frac_nan_deg,

        # High-frequency energy proxy disabled (FFT-heavy); keep keys as NaN
        hf_ratio_CO2   = np.nan,
        hf_ratio_w     = np.nan,
    )
