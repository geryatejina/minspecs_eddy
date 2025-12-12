import numpy as np
from .types import Theta

def sample_thetas(N: int, ranges: dict, seed: int | None = None):
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
        thetas.append(Theta(**vals))

    return thetas
