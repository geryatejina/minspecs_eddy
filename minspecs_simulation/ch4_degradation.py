"""
ch4_degradation.py
------------------

Methane-only degradation operator for open-path QCL simulation.

The operator is causal and applies degradations in the prescribed order:
    1) First-order low-pass (tau)
    2) Resample to effective output rate (f_eff)
    3) Lag jitter (sigma_lag)
    4) Additive noise (sigma_rho)
    5) Multiplicative gain noise (sigma_gain)
    6) Low-frequency drift (sigma_drift)

This module is intentionally separate from the CO2/H2O path to avoid
coupling the two simulations.
"""
from __future__ import annotations

import numpy as np

from .window_processor import lowpass_first_order, add_gaussian_noise, apply_lag_jitter
from .ch4_types import MethaneTheta


def _validate_frequencies(f_raw: float, f_eff: float):
    if f_eff <= 0:
        raise ValueError("f_eff must be > 0")
    if f_raw <= 0:
        raise ValueError("f_raw must be > 0")


def resample_uniform(x: np.ndarray, f_in: float, f_out: float) -> np.ndarray:
    """
    Resample a uniformly-sampled 1D signal to a new uniform rate using
    linear interpolation. Works for non-integer ratios.
    """
    _validate_frequencies(f_in, f_out)

    n = x.shape[0]
    if n == 0:
        return x
    if n == 1:
        return np.full(1, x[0], dtype=float)

    t_in = np.arange(n, dtype=float) / f_in
    duration = t_in[-1]

    t_out = np.arange(0.0, duration + 1e-9, 1.0 / f_out)
    return np.interp(t_out, t_in, x)


def apply_gain_noise(x: np.ndarray, sigma_gain: float, rng: np.random.Generator) -> np.ndarray:
    if sigma_gain <= 0:
        return x.copy()
    gains = rng.normal(loc=1.0, scale=sigma_gain, size=x.shape)
    return x * gains


def generate_low_freq_drift(n: int, sigma_drift: float, rng: np.random.Generator) -> np.ndarray:
    """
    Simple constrained random-walk drift whose standard deviation over the
    window matches sigma_drift. This is a pragmatic low-frequency drift
    surrogate; it stays causal and bounded to the specified amplitude.
    """
    if sigma_drift <= 0 or n == 0:
        return np.zeros(n, dtype=float)

    steps = rng.normal(0.0, 1.0, size=n)
    drift = np.cumsum(steps)
    drift -= np.nanmean(drift)

    std = np.nanstd(drift)
    if std > 0:
        drift *= sigma_drift / std
    return drift


def degrade_methane_density(
    rho_true: np.ndarray,
    theta: MethaneTheta,
    f_raw: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply the ordered methane degradation operator D_theta to the true
    methane density time series.

    Parameters
    ----------
    rho_true : np.ndarray
        High-quality methane density [ug m-3] at raw rate f_raw.
    theta : MethaneTheta
        Parameter set controlling each degradation component.
    f_raw : float
        Raw sampling rate of the methane/sonic data [Hz].
    rng : np.random.Generator
        RNG for all stochastic components.

    Returns
    -------
    np.ndarray
        Degraded methane density at the effective rate theta.f_eff.
    """
    _validate_frequencies(f_raw, theta.f_eff)

    dt_raw = 1.0 / f_raw

    # 1) Bandwidth limitation
    rho_lp = lowpass_first_order(rho_true, theta.tau, dt_raw)

    # 2) Resample / decimate to effective analyzer output rate
    rho_resamp = resample_uniform(rho_lp, f_raw, theta.f_eff)

    # 3) Lag jitter (fractional delay at f_eff)
    rho_lag = apply_lag_jitter(rho_resamp, theta.f_eff, theta.sigma_lag, rng)

    # 4) Additive noise
    rho_add = add_gaussian_noise(rho_lag, theta.sigma_rho, rng)

    # 5) Multiplicative (gain/baseline) noise
    rho_gain = apply_gain_noise(rho_add, theta.sigma_gain, rng)

    # 6) Slow drift (random walk scaled to sigma_drift over the window)
    drift = generate_low_freq_drift(len(rho_gain), theta.sigma_drift, rng)

    return rho_gain + drift


def resample_reference(rho_true: np.ndarray, f_raw: float, f_eff: float) -> np.ndarray:
    """
    Reference resample helper: resample the *undegraded* methane density to
    the effective output rate. This is used for like-with-like comparisons
    (RMSE, variance ratios) without injecting degradation effects.
    """
    return resample_uniform(rho_true, f_raw, f_eff)


__all__ = [
    "degrade_methane_density",
    "resample_reference",
    "resample_uniform",
    "generate_low_freq_drift",
    "apply_gain_noise",
]
