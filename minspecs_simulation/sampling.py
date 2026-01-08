import numpy as np
from typing import Sequence
from .types import Theta

def sample_thetas(N: int, ranges: dict, seed: int | None = None, cls=Theta):
    """
    Sobol-like pseudo-random sampler (simple uniform sampler).
    ranges = dict(name → (low, high))
    """
    names = list(ranges.keys())
    dim = len(names)

    rnd = np.random.default_rng(seed)
    M = rnd.random((N, dim))

    thetas = []
    for i in range(N):
        vals = {}
        for j, name in enumerate(names):
            low, high = ranges[name]
            vals[name] = low + (high - low) * M[i, j]
        thetas.append(cls(**vals))

    return thetas


def build_theta_plan(baseline, sweeps: dict[str, Sequence[float]] | None = None):
    """
    Construct a list of (theta, sweep_param, sweep_value) entries for
    univariate sweeps. The first entry is the baseline.
    """
    plan = [(baseline, None, None)]
    if not sweeps:
        return plan

    base_dict = baseline.__dict__
    for param, values in sweeps.items():
        if param not in base_dict:
            raise KeyError(f"Unknown theta parameter: {param}")
        for val in values:
            theta_kwargs = dict(base_dict)
            theta_kwargs[param] = val
            plan.append((baseline.__class__(**theta_kwargs), param, val))
    return plan
