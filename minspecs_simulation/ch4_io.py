"""
ch4_io.py
---------

Lightweight IO helpers for the methane-only simulation path. This keeps
non-ICOS data handling isolated from the CO2/H2O pipeline.

Expected data layout (flexible):
    data_root/
        <site>/
            *.npz or *.csv

Each file must provide, at minimum:
    - w : vertical velocity [m/s]
    - rho_CH4 : methane density [ug m-3]
Optional extras (kept if present):
    - Ts : sonic temperature [degC]
    - time : optional timestamp vector (not required)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd

DEFAULT_CH4_ROOT = Path(os.getenv("CH4_DATA_ROOT", r"D:\data\ec\raw\CH4"))


def iter_ch4_sites(data_root: Path = DEFAULT_CH4_ROOT) -> Generator[str, None, None]:
    for entry in Path(data_root).iterdir():
        if entry.is_dir():
            yield entry.name


def iter_ch4_files(
    site: str,
    data_root: Path = DEFAULT_CH4_ROOT,
    pattern: str = "*.npz",
    max_files: Optional[int] = None,
):
    site_path = Path(data_root) / site
    if not site_path.exists():
        raise FileNotFoundError(f"Site directory not found: {site_path}")

    files = sorted(site_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} under {site_path}")

    for idx, f in enumerate(files):
        if max_files is not None and idx >= max_files:
            break
        yield f


def _pick_key(options, mapping):
    for key in options:
        if key in mapping:
            return key
    return None


def load_ch4_arrays(path: Path):
    """
    Load methane window arrays from NPZ or CSV.

    Required arrays: w, rho_CH4. Optional: Ts.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        mapping = {k: data[k] for k in data.files}
    else:
        df = pd.read_csv(path)
        mapping = {c: df[c].to_numpy() for c in df.columns}

    w_key = _pick_key(["w", "W"], mapping)
    rho_key = _pick_key(["rho_CH4", "rho_ch4", "ch4_density", "CH4_dens"], mapping)
    Ts_key = _pick_key(["Ts", "ts", "T_SONIC"], mapping)

    if w_key is None or rho_key is None:
        raise KeyError(f"File {path} must contain w and rho_CH4 columns/arrays")

    arrays = {
        "w": np.asarray(mapping[w_key]),
        "rho_CH4": np.asarray(mapping[rho_key]),
    }
    if Ts_key is not None:
        arrays["Ts"] = np.asarray(mapping[Ts_key])
    return arrays


def window_id_from_path(path: Path):
    """
    Window identifier for sorting/logging.
    Attempts to parse a datetime from the file stem; falls back to the stem.
    """
    stem = Path(path).stem
    ts = pd.to_datetime(stem, errors="coerce")
    if not pd.isna(ts):
        return ts.to_pydatetime()
    return stem


__all__ = [
    "iter_ch4_sites",
    "iter_ch4_files",
    "load_ch4_arrays",
    "window_id_from_path",
]
