"""
window_processor.py
--------------------

This module contains the full per-window simulation engine for minspecs_eddy.

It takes:
    - numpy arrays from io_icos.df_to_arrays()
    - a Theta object
    - a decimation factor D
and computes:
    - reference undegraded fluxes
    - degraded fluxes
    - bias metrics

This is the physics core of the entire project.
"""

import numpy as np
from .types import Theta, WindowMetrics
from .fracdelay import fractional_delay
from pathlib import Path
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
    """Remove mean for turbulence covariance calculations."""
    m = np.nanmean(x)
    return x - m

def covariance(w, s):
    """Compute covariance cov(w, s). No rotation, no WPL, pure turbulence."""
    return float(np.nanmean(detrend(w) * detrend(s)))


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

def apply_lag_jitter(x, f_s, sigma_lag, rng):
    """
    Apply lag jitter using a *fractional-delay* model.

    We draw a single random lag offset (Gaussian, std = sigma_lag [s]),
    convert it to samples, and apply a fractional delay filter.

    This models an uncertain but constant lag over the 30-min window,
    with sub-sample precision.
    """
    if sigma_lag <= 0:
        return x

    # Draw jitter in seconds and convert to samples
    jitter_seconds = rng.normal(loc=0.0, scale=sigma_lag)
    delay_samples = jitter_seconds * f_s

    return fractional_delay(x, delay_samples)


# =============================================================
# Decimation
# =============================================================

def decimate(x, D):
    if D <= 1:
        return x
    return x[::D]


# =============================================================
# Density → mixing ratio conversion
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
# Full engine: process_window_for_theta_D
# =============================================================

def process_window_for_theta_D(arrays, theta, D, site_id, theta_index,
                               window_start,
                               seed=None):
    """
        arrays = df_to_arrays(df), containing numpy arrays:
            u, v, w, Ts, rho_CO2, rho_H2O, T_cell, P_cell

        This function:
            1) computes reference fluxes (undegraded)
            2) applies sensor degradations according to θ
            3) recomputes mixing ratios
            4) computes degraded fluxes at f_eff
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
    # 1. Reference mixing ratios (undegraded)
    # ============================================================

    mr_CO2_ref = density_to_mr(rho_CO2 * 1e-3, T_cell, P_cell)  # assuming mmol/m3 → mol/m3
    mr_H2O_ref = density_to_mr(rho_H2O * 1e-3, T_cell, P_cell)

    # ============================================================
    # 2. Reference fluxes (20 Hz)
    # ============================================================

    F_CO2_ref, F_LE_ref, F_H_ref = compute_fluxes(
        w, Ts, mr_CO2_ref, mr_H2O_ref
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

    u_d  = decimate(u_f,  D)
    v_d  = decimate(v_f,  D)
    w_d  = decimate(w_n,  D)
    Ts_d = decimate(Ts_n, D)

    # ============================================================
    # 4. IRGA degradations (densities + T_cell)
    # ============================================================

    # Filter
    rhoC_f = lowpass_first_order(rho_CO2, theta.tau_irga, dt)
    rhoW_f = lowpass_first_order(rho_H2O, theta.tau_irga, dt)

    # Noise
    rhoC_n = add_gaussian_noise(rhoC_f, theta.sigma_CO2dens_noise, rng)
    rhoW_n = add_gaussian_noise(rhoW_f, theta.sigma_H2Odens_noise, rng)

    # Tcell noise
    Tcell_n = add_gaussian_noise(T_cell, theta.sigma_Tcell_noise, rng)

    # Gain drift
    dT = Tcell_n - np.nanmean(Tcell_n)
    rhoC_g = rhoC_n * (1 + theta.k_CO2_Tsens * dT)
    rhoW_g = rhoW_n * (1 + theta.k_H2O_Tsens * dT)

    # Jitter
    rhoC_j = apply_lag_jitter(rhoC_g, f_raw, theta.sigma_lag_jitter, rng)
    rhoW_j = apply_lag_jitter(rhoW_g, f_raw, theta.sigma_lag_jitter, rng)

    # Decimate all
    rhoC_d = decimate(rhoC_j, D)
    rhoW_d = decimate(rhoW_j, D)
    Tcell_d = decimate(Tcell_n, D)
    Pcell_d = decimate(P_cell, D)

    # ============================================================
    # 5. Recompute mixing ratios AFTER degradation
    # ============================================================

    mr_CO2_deg = density_to_mr(rhoC_d * 1e-3, Tcell_d, Pcell_d)
    mr_H2O_deg = density_to_mr(rhoW_d * 1e-3, Tcell_d, Pcell_d)

    # ============================================================
    # 6. Degraded fluxes
    # ============================================================

    F_CO2_deg, F_LE_deg, F_H_deg = compute_fluxes(
        w_d, Ts_d, mr_CO2_deg, mr_H2O_deg
    )

    # ============================================================
    # 7. QC metrics
    # ============================================================

    qc = compute_qc_metrics(
        w_ref=w,
        Ts_ref=Ts,
        mrCO2_ref=mr_CO2_ref,
        mrH2O_ref=mr_H2O_ref,

        w_deg=w_d,
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
    # 8. Package all metrics together
    # ============================================================

    return {
        "site_id": site_id,
        "theta_index": theta_index,
        "D": D,

        "window_start": window_start,

        "F_CO2_ref": F_CO2_ref,
        "F_CO2_deg": F_CO2_deg,
        "F_LE_ref": F_LE_ref,
        "F_LE_deg": F_LE_deg,
        "F_H_ref": F_H_ref,
        "F_H_deg": F_H_deg,

        "bias_CO2": F_CO2_deg - F_CO2_ref,
        "bias_LE":  F_LE_deg  - F_LE_ref,
        "bias_H":   F_H_deg   - F_H_ref,
    } | qc


# =============================================================
# Wrapper called from site_runner
# =============================================================

def process_single_window(path, arrays, theta, D, site_id, theta_index):
    """
    Wrapper that extracts the window timestamp from the file name,
    then calls the physics engine.
    """
    window_start = extract_window_timestamp_from_filename(Path(path))

    return process_window_for_theta_D(
        arrays=arrays,
        theta=theta,
        D=D,
        site_id=site_id,
        theta_index=theta_index,
        window_start=window_start,   # <-- pass timestamp here
    )
