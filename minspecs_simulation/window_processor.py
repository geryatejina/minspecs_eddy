"""
window_processor.py
-------------------

Per-window simulation engine for minspecs_eddy.

Pipeline:
    - load arrays (u, v, w, Ts, rho_CO2, rho_H2O, T_cell, P_cell)
    - compute reference fluxes with double-rotation tilt correction
    - apply sensor degradations
    - compute degraded fluxes for requested rotation mode
    - return metrics + QC diagnostics
"""

from __future__ import annotations

from pathlib import Path
from math import hypot

import numpy as np

from .types import Theta
from .fracdelay import fractional_delay
from .io_icos import extract_window_timestamp_from_filename

try:
    import numba as _nb  # optional acceleration
except ImportError:  # pragma: no cover - optional dependency
    _nb = None

# numba-accelerated first-order IIR if available
if _nb is not None:
    @_nb.njit(cache=True)
    def _lowpass_first_order_numba(x, a):
        y = np.empty_like(x, dtype=np.float64)
        y[0] = x[0]
        one_minus_a = 1.0 - a
        for i in range(1, len(x)):
            y[i] = a * y[i - 1] + one_minus_a * x[i]
        return y
else:
    _lowpass_first_order_numba = None


# =============================================================
# Helper: detrend and covariance
# =============================================================

def detrend(x):
    """Remove mean for turbulence covariance calculations; safe to all-NaN."""
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    if not mask.any():
        return np.full_like(x, np.nan)
    m = x[mask].mean()
    out = x - m
    out[~mask] = np.nan
    return out


def covariance(w, s):
    """Compute covariance cov(w, s). No rotation, no WPL, pure turbulence."""
    w_d = detrend(w)
    s_d = detrend(s)
    mask = np.isfinite(w_d) & np.isfinite(s_d)
    if not mask.any():
        return np.nan
    return float(np.mean(w_d[mask] * s_d[mask]))


# =============================================================
# Double rotation (tilt correction)
# =============================================================

def double_rotate(u, v, w):
    """
    Two-step rotation to enforce mean(v)=0 and mean(w)=0.
    Returns rotated components and the rotation angles (alpha, beta).
    """
    u0 = float(np.nanmean(u))
    v0 = float(np.nanmean(v))
    w0 = float(np.nanmean(w))

    alpha = np.arctan2(v0, u0) if hypot(u0, v0) > 0 else 0.0
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    u1 =  u * cos_a + v * sin_a
    v1 = -u * sin_a + v * cos_a
    w1 = w

    u1_mean = float(np.nanmean(u1))
    w1_mean = float(np.nanmean(w1))

    beta = np.arctan2(w1_mean, u1_mean) if hypot(u1_mean, w1_mean) > 0 else 0.0
    cos_b, sin_b = np.cos(beta), np.sin(beta)
    u2 =  u1 * cos_b + w1 * sin_b
    v2 =  v1
    w2 = -u1 * sin_b + w1 * cos_b

    return u2, v2, w2, alpha, beta


# =============================================================
# Timelag search (cheap, discrete)
# =============================================================

def _cov_at_lag(w, x, lag):
    """
    Covariance between w and x shifted by integer lag (samples).
    Positive lag means x lags w (x is shifted forward).
    """
    if lag > 0:
        w_s = w[lag:]
        x_s = x[:-lag]
    elif lag < 0:
        w_s = w[:lag]
        x_s = x[-lag:]
    else:
        w_s = w
        x_s = x
    cov = covariance(w_s, x_s)
    return cov


def find_optimal_lag(w, x, max_lag_samples=15, decimate=1):
    """
    Find integer lag (in samples) maximizing absolute covariance.
    Search is bounded to keep compute cheap.
    decimate: optional integer downsampling for lag search only.
    """
    best_lag = 0
    best_cov = -np.inf

    if decimate and decimate > 1:
        w = w[::decimate]
        x = x[::decimate]
        lag_step = decimate
    else:
        lag_step = 1

    search_range = range(-max_lag_samples, max_lag_samples + 1)
    for lag in search_range:
        cov = _cov_at_lag(w, x, lag)
        if not np.isfinite(cov):
            continue
        cov_abs = abs(cov)
        if cov_abs > best_cov:
            best_cov = cov_abs
            best_lag = lag
    return best_lag * lag_step


def apply_integer_lag(x, lag):
    """
    Shift signal by integer samples; fill boundaries with NaN.
    Positive lag means x lags w (content moves forward).
    """
    if lag == 0:
        return x.copy()
    y = np.full_like(x, np.nan, dtype=float)
    if lag > 0:
        y[lag:] = x[:-lag]
    else:
        y[:lag] = x[-lag:]
    return y


# =============================================================
# Low-pass filter (first-order IIR)
# =============================================================

def lowpass_first_order(x, tau, dt):
    if tau <= 0:
        return x.copy()

    a = np.exp(-dt / tau)
    if _lowpass_first_order_numba is not None:
        # Ensure contiguous float64 for numba fast path
        x_f64 = np.ascontiguousarray(x, dtype=np.float64)
        return _lowpass_first_order_numba(x_f64, a)

    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    one_minus_a = 1.0 - a

    for i in range(1, len(x)):
        y[i] = a * y[i-1] + one_minus_a * x[i]

    return y


# =============================================================
# Noise
# =============================================================

def add_gaussian_noise(x, sigma, rng):
    if sigma <= 0:
        return x.copy()
    return x + rng.normal(0, sigma, size=x.shape)


# =============================================================
# Lag jitter (placeholder: global shift)
# =============================================================

def apply_lag_jitter(x, f_s, sigma_lag, rng, delay_samples=None):
    """
    Apply lag jitter using a *fractional-delay* model.

    We draw a single random lag offset (Gaussian, std = sigma_lag [s]),
    convert it to samples, and apply a fractional delay filter.

    This models an uncertain but constant lag over the 30-min window,
    with sub-sample precision.
    """
    if sigma_lag <= 0:
        return x

    if delay_samples is None:
        # Draw jitter in seconds and convert to samples
        jitter_seconds = rng.normal(loc=0.0, scale=sigma_lag)
        delay_samples = jitter_seconds * f_s

    return fractional_delay(x, delay_samples)


# =============================================================
# Density and mixing ratio conversion
# =============================================================

def density_to_mr(rho, T_cell, P_cell):
    """
    rho: mol/m3
    T_cell: degC
    P_cell: kPa

    output: mixing ratio mol/mol
    """
    R = 8.314462618  # J / (mol K)

    T_K = T_cell + 273.15
    P_Pa = P_cell * 1000.0

    return rho * R * T_K / P_Pa


# =============================================================
# Reference flux computation
# =============================================================

def compute_fluxes(w, Ts, mr_CO2, mr_H2O):
    """
    Raw turbulence fluxes (no rotation, no corrections).
    """
    F_CO2 = covariance(w, mr_CO2)
    F_LE  = covariance(w, mr_H2O)
    F_H   = covariance(w, Ts)
    return F_CO2, F_LE, F_H


# =============================================================
# Full engine: process_window_for_theta
# =============================================================

def process_window_for_theta(arrays, theta, site_id, theta_index,
                             window_start, rotation_mode, lag_samples,
                             seed=None):
    """
        arrays = df_to_arrays(df), containing numpy arrays:
            u, v, w, Ts, rho_CO2, rho_H2O, T_cell, P_cell

        This function:
            1) computes reference fluxes (undegraded, double-rotated)
            2) applies sensor degradations according to theta
            3) recomputes mixing ratios
            4) computes degraded fluxes in the specified rotation mode
            5) returns bias metrics
        """
    from .qc import compute_qc_metrics

    rng = np.random.default_rng(seed)
    f_raw = 20.0               # ICOS raw sampling frequency
    dt = 1.0 / f_raw

    # ------------------------------------------------------------
    # Unpack input arrays
    # ------------------------------------------------------------

    u       = arrays["u"]
    v       = arrays["v"]
    w       = arrays["w"]
    Ts      = arrays["Ts"]
    rho_CO2 = arrays["rho_CO2"]
    rho_H2O = arrays["rho_H2O"]
    T_cell  = arrays["T_cell"]
    P_cell  = arrays["P_cell"]

    # ============================================================
    # -1. Apply precomputed nominal timelag to analyzer channels
    # ============================================================

    rho_CO2_lag = apply_integer_lag(rho_CO2, lag_samples)
    rho_H2O_lag = apply_integer_lag(rho_H2O, lag_samples)
    T_cell_lag  = apply_integer_lag(T_cell, lag_samples)
    P_cell_lag  = apply_integer_lag(P_cell, lag_samples)

    # ============================================================
    # 0. Double rotation on reference wind (tilt correction)
    # ============================================================

    u_ref_rot, v_ref_rot, w_ref_rot, alpha_ref, beta_ref = double_rotate(u, v, w)

    # ============================================================
    # 1. Reference mixing ratios (undegraded)
    # ============================================================

    mr_CO2_ref = density_to_mr(rho_CO2_lag * 1e-3, T_cell_lag, P_cell_lag)  # mmol/m3 -> mol/m3
    mr_H2O_ref = density_to_mr(rho_H2O_lag * 1e-3, T_cell_lag, P_cell_lag)

    # ============================================================
    # 2. Reference fluxes (20 Hz, rotated)
    # ============================================================

    F_CO2_ref, F_LE_ref, F_H_ref = compute_fluxes(
        w_ref_rot, Ts, mr_CO2_ref, mr_H2O_ref
    )

    # ============================================================
    # 3. Sonic degradations
    # ============================================================

    u_f  = lowpass_first_order(u,  theta.tau_sonic, dt)
    v_f  = lowpass_first_order(v,  theta.tau_sonic, dt)
    w_f  = lowpass_first_order(w,  theta.tau_sonic, dt)
    Ts_f = lowpass_first_order(Ts, theta.tau_sonic, dt)

    w_n  = add_gaussian_noise(w_f,  theta.sigma_w_noise,  rng)
    Ts_n = add_gaussian_noise(Ts_f, theta.sigma_Ts_noise, rng)

    u_d  = u_f
    v_d  = v_f
    w_d  = w_n
    Ts_d = Ts_n

    # ============================================================
    # 4. IRGA degradations (densities + T_cell)
    # ============================================================

    rhoC_f = lowpass_first_order(rho_CO2_lag, theta.tau_irga, dt)
    rhoW_f = lowpass_first_order(rho_H2O_lag, theta.tau_irga, dt)

    rhoC_n = add_gaussian_noise(rhoC_f, theta.sigma_CO2dens_noise, rng)
    rhoW_n = add_gaussian_noise(rhoW_f, theta.sigma_H2Odens_noise, rng)

    Tcell_n = add_gaussian_noise(T_cell_lag, theta.sigma_Tcell_noise, rng)

    dT = Tcell_n - np.nanmean(Tcell_n)
    rhoC_g = rhoC_n * (1 + theta.k_CO2_Tsens * dT)
    rhoW_g = rhoW_n * (1 + theta.k_H2O_Tsens * dT)

    # Common lag jitter applied to analyzer channels (densities and T_cell)
    if theta.sigma_lag_jitter > 0:
        jitter_seconds = rng.normal(loc=0.0, scale=theta.sigma_lag_jitter)
        delay_samples = jitter_seconds * f_raw
        rhoC_j = apply_lag_jitter(rhoC_g, f_raw, theta.sigma_lag_jitter, rng, delay_samples=delay_samples)
        rhoW_j = apply_lag_jitter(rhoW_g, f_raw, theta.sigma_lag_jitter, rng, delay_samples=delay_samples)
        Tcell_j = apply_lag_jitter(Tcell_n, f_raw, theta.sigma_lag_jitter, rng, delay_samples=delay_samples)
        Pcell_j = apply_lag_jitter(P_cell, f_raw, theta.sigma_lag_jitter, rng, delay_samples=delay_samples)
    else:
        rhoC_j = rhoC_g
        rhoW_j = rhoW_g
        Tcell_j = Tcell_n
        Pcell_j = P_cell

    rhoC_d = rhoC_j
    rhoW_d = rhoW_j
    Tcell_d = Tcell_j
    Pcell_d = Pcell_j

    # ============================================================
    # 5. Recompute mixing ratios AFTER degradation
    # ============================================================

    mr_CO2_deg = density_to_mr(rhoC_d * 1e-3, Tcell_d, Pcell_d)
    mr_H2O_deg = density_to_mr(rhoW_d * 1e-3, Tcell_d, Pcell_d)

    # ============================================================
    # 6. Rotation mode handling for degraded wind
    # ============================================================

    if rotation_mode not in ("double", "none"):
        raise ValueError(f"Unknown rotation_mode: {rotation_mode}")

    if rotation_mode == "double":
        u_deg_rot, v_deg_rot, w_deg_for_flux, alpha_deg, beta_deg = double_rotate(u_d, v_d, w_d)
    else:
        w_deg_for_flux = w_d
        alpha_deg = beta_deg = 0.0

    w_mean_deg = float(np.nanmean(w_deg_for_flux))

    # ============================================================
    # 7. Degraded fluxes
    # ============================================================

    F_CO2_raw, F_LE_raw, F_H_raw = compute_fluxes(
        w_d, Ts_d, mr_CO2_deg, mr_H2O_deg
    )

    if rotation_mode == "double":
        F_CO2_deg, F_LE_deg, F_H_deg = compute_fluxes(
            w_deg_for_flux, Ts_d, mr_CO2_deg, mr_H2O_deg
        )
    else:
        correction_factor = 0.7
        F_CO2_corr = F_CO2_raw + w_mean_deg * correction_factor * F_CO2_raw
        F_LE_corr  = F_LE_raw  + w_mean_deg * correction_factor * F_LE_raw
        F_CO2_deg, F_LE_deg, F_H_deg = F_CO2_corr, F_LE_corr, F_H_raw

    # ============================================================
    # 8. QC metrics (using rotated reference, mode-dependent degraded)
    # ============================================================

    qc = compute_qc_metrics(
        w_ref=w_ref_rot,
        Ts_ref=Ts,
        mrCO2_ref=mr_CO2_ref,
        mrH2O_ref=mr_H2O_ref,

        w_deg=w_deg_for_flux,
        Ts_deg=Ts_d,
        mrCO2_deg=mr_CO2_deg,
        mrH2O_deg=mr_H2O_deg,

        F_CO2_ref=F_CO2_ref,
        F_CO2_deg=F_CO2_deg,
        F_LE_ref=F_LE_ref,
        F_LE_deg=F_LE_deg,
        F_H_ref=F_H_ref,
        F_H_deg=F_H_deg,
    )

    # ============================================================
    # 9. Package all metrics together
    # ============================================================

    return {
        "site_id": site_id,
        "theta_index": theta_index,
        "rotation_mode": rotation_mode,

        "window_start": window_start,

        "F_CO2_ref": F_CO2_ref,
        "F_CO2_deg": F_CO2_deg,
        "F_LE_ref": F_LE_ref,
        "F_LE_deg": F_LE_deg,
        "F_H_ref": F_H_ref,
        "F_H_deg": F_H_deg,

        "F_CO2_raw": F_CO2_raw,
        "F_LE_raw": F_LE_raw,
        "F_H_raw": F_H_raw,
        "w_mean_deg": w_mean_deg,

        "bias_CO2": F_CO2_deg - F_CO2_ref,
        "bias_LE":  F_LE_deg  - F_LE_ref,
        "bias_H":   F_H_deg   - F_H_ref,
    } | qc


# =============================================================
# Wrapper called from site_runner
# =============================================================

def process_single_window(path, arrays, theta, site_id, theta_index, rotation_mode, lag_samples):
    """
    Wrapper that extracts the window timestamp from the file name,
    then calls the physics engine.
    """
    window_start = extract_window_timestamp_from_filename(Path(path))

    return process_window_for_theta(
        arrays=arrays,
        theta=theta,
        site_id=site_id,
        theta_index=theta_index,
        rotation_mode=rotation_mode,
        lag_samples=lag_samples,
        window_start=window_start,   # <-- pass timestamp here
    )
