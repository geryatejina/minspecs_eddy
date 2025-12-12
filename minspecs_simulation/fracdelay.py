"""
fracdelay.py
------------

Simple fractional-delay implementation based on linear interpolation.

Given a continuous delay d (in samples), we approximate:
    y[n] ≈ x[n - d]

This is already a *true* fractional delay (not just integer shifting)
and is perfectly adequate for simulating realistic sub-sample lag jitter.
"""

from __future__ import annotations

import numpy as np


def fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """
    Apply a constant fractional delay to a 1D signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal, shape (N,).
    delay_samples : float
        Delay in samples. Positive delay means the output lags
        the input: y[n] ~ x[n - delay_samples].

    Returns
    -------
    y : np.ndarray
        Delayed signal, same length as x. Boundary regions where the
        interpolation stencil goes out of bounds are filled with NaN.
    """
    N = x.shape[0]
    y = np.full_like(x, np.nan, dtype=float)

    # Index at which we evaluate input x:
    #   t_in[n] = n - delay
    n = np.arange(N, dtype=float)
    t_in = n - delay_samples

    i0 = np.floor(t_in).astype(int)          # left index
    frac = t_in - i0                         # fractional part in [0, 1)

    # Valid indices: we need i0 and i0+1 inside [0, N-1]
    mask = (i0 >= 0) & (i0 + 1 < N)

    i0_valid = i0[mask]
    frac_valid = frac[mask]

    # Linear interpolation: x(t) ≈ (1 - frac) * x[i0] + frac * x[i0+1]
    y[mask] = (1.0 - frac_valid) * x[i0_valid] + frac_valid * x[i0_valid + 1]

    return y
