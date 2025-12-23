"""
io_icos.py
----------

Unified ICOS IO utilities for minspecs_eddy:

- Traverse ICOS root folder: ecosystem → site → files
- Load raw Level-0 CSV file
- Compute mixing ratios (CO2_MR, H2O_MR)
- Convert DataFrame → numpy arrays for the simulation engine

This file consolidates **all functionality** that previously lived in
minspecs_eddy.py, so we do NOT lose site/ecosystem traversal.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Optional, Tuple

import pandas as pd
import numpy as np
from datetime import datetime



DEFAULT_DATA_ROOT = Path(os.getenv("ICOS_DATA_ROOT", r"D:\data\ec\raw\ICOS"))
DEFAULT_CACHE_ROOT = Path(os.getenv("ICOS_NPY_ROOT", r"D:\data\ec\raw\ICOS_npy"))

# Variables needed for simulation
SELECT_COLUMNS = [
    "U", "V", "W", "T_SONIC",
    "CO2_CONC", "H2O_CONC",
    "T_CELL", "PRESS_CELL",
]

SANITY_LIMITS = {
    # Wind (m/s)
    "U": (-15.0, 15.0),
    "V": (-15.0, 15.0),
    "W": (-3.0, 3.0),
    # Sonic temperature is stored in Kelvin in the cached NPZ files
    "T_SONIC": (250.0, 320.0),  # K
    # Cell temperature is in degC
    "T_CELL": (-10.0, 40.0),
    # Gas densities: CO2 ~12–40 mmol/m3 for 300–1000 ppm at typical air density
    "CO2_CONC": (7.0, 30.0),    # mmol/m3
    # Water vapor density: generous envelope (typical < 1000 mmol/m3)
    "H2O_CONC": (0.0, 1000.0),  # mmol/m3
    # Cell pressure (kPa)
    "PRESS_CELL": (90.0, 105.0),
}

SANITY_LIMITS_ARRAY = {
    "u": SANITY_LIMITS["U"],            # m/s
    "v": SANITY_LIMITS["V"],            # m/s
    "w": SANITY_LIMITS["W"],            # m/s
    "Ts": SANITY_LIMITS["T_SONIC"],     # K (sonic temperature in cached NPZ)
    "T_cell": SANITY_LIMITS["T_CELL"],  # degC
    "rho_CO2": SANITY_LIMITS["CO2_CONC"],  # mmol/m3
    "rho_H2O": SANITY_LIMITS["H2O_CONC"],  # mmol/m3
    "P_cell": SANITY_LIMITS["PRESS_CELL"], # kPa
}

IDEAL_GAS_CONSTANT = 8.314462618  # J/(mol K)


def extract_window_timestamp_from_filename(path: Path) -> datetime:
    """
    Extract YYYYMMDDHHMM timestamp from ICOS filenames of the form:
        BE-Lon_EC_202306190900_L05_F01.csv

    - Timestamp is always the 3rd underscore-separated field.
    - Always 12 digits: YYYYMMDDHHMM.
    """
    fname = path.name
    parts = fname.split("_")
    ts = parts[2]  # '202306190900'

    return datetime.strptime(ts, "%Y%m%d%H%M")

#=====================================================================
# Ecosystem / site traversal  (RESTORED FROM ORIGINAL CODE)
#=====================================================================

def ecosystem_sites(data_root: Path = DEFAULT_DATA_ROOT) -> Generator[Tuple[str, str], None, None]:
    """
    Discover all (ecosystem, site) folders in the ICOS directory.

    Yields:
        (ecosystem_name, site_name)
    """
    for eco_dir in Path(data_root).iterdir():
        if not eco_dir.is_dir():
            continue
        ecosystem = eco_dir.name
        # inside ecosystem, look for sites
        for site_dir in eco_dir.iterdir():
            if site_dir.is_dir():
                yield ecosystem, site_dir.name


def build_site_path(ecosystem: str, site: str, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    site_path = Path(data_root) / ecosystem / site
    if not site_path.exists():
        raise FileNotFoundError(f"Site directory not found: {site_path}")
    return site_path


def iter_site_files(
    site: str,
    ecosystem: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    pattern: str = "*.csv",
    max_files: Optional[int] = None,
) -> Generator[Path, None, None]:
    site_path = build_site_path(ecosystem, site, data_root)
    files = sorted(site_path.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files under {site_path}")

    for idx, f in enumerate(files):
        if max_files is not None and idx >= max_files:
            break
        yield f


#=====================================================================
# File loading
#=====================================================================

def read_raw_file(file_path: Path) -> pd.DataFrame:
    """
    Load ICOS Level-0 CSV file, skipping timestamp parsing (unused downstream).
    """
    # Only pull needed measurement columns; timestamp column is ignored.
    df = pd.read_csv(
        file_path,
        usecols=lambda c: c in SELECT_COLUMNS,
        low_memory=False,
    )

    # select only available needed columns
    cols = [c for c in SELECT_COLUMNS if c in df.columns]
    df = df[cols]

    # Apply simple physical sanity filters; values outside are set to NaN.
    for col, (lo, hi) in SANITY_LIMITS.items():
        if col not in df.columns:
            continue
        mask = df[col].between(lo, hi)
        df.loc[~mask, col] = np.nan

    return df


def _sanitize_arrays(arrays: dict) -> dict:
    """
    Apply physical sanity limits to array data (works for CSV- or NPZ-loaded inputs).
    Units are assumed to match ICOS Level-0: wind m/s, temps degC, CO2/H2O mmol/m3, P kPa.
    """
    out = {}
    for key, arr in arrays.items():
        if key not in SANITY_LIMITS_ARRAY:
            out[key] = arr
            continue
        lo, hi = SANITY_LIMITS_ARRAY[key]
        a = np.asarray(arr, dtype=float)
        mask = np.isfinite(a) & (a >= lo) & (a <= hi)
        if not mask.all():
            a = a.copy()
            a[~mask] = np.nan
        out[key] = a
    return out


def cache_csv_to_npz(csv_path: Path, raw_root: Path = DEFAULT_DATA_ROOT, cache_root: Path = DEFAULT_CACHE_ROOT, overwrite: bool = False) -> Path:
    """
    Convert one CSV window to NPZ in the mirrored cache root.
    """
    rel = csv_path.relative_to(raw_root)
    target_path = (cache_root / rel).with_suffix(".npz")

    if target_path.exists() and not overwrite:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    df = read_raw_file(csv_path)
    df = add_mixing_ratios(df)
    arrays = df_to_arrays(df)
    np.savez(target_path, **arrays)
    return target_path


def load_window_arrays(path: Path):
    """
    Load window arrays from NPZ if present, otherwise parse CSV.
    Applies sanity limits regardless of source.
    """
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        arrays = {k: data[k] for k in data.files}
    else:
        df = read_raw_file(path)
        df = add_mixing_ratios(df)
        arrays = df_to_arrays(df)

    return _sanitize_arrays(arrays)


#=====================================================================
# Mixing ratio computation (kept exactly as original)
#=====================================================================

def add_mixing_ratios(df: pd.DataFrame) -> pd.DataFrame:
    T = df["T_CELL"] + 273.15
    P = df["PRESS_CELL"] * 1000

    # assume CO2_CONC / H2O_CONC are in mmol/m3 → convert to mol/m3
    c_co2 = df["CO2_CONC"] * 1e-3
    c_h2o = df["H2O_CONC"] * 1e-3

    P_h2o = c_h2o * IDEAL_GAS_CONSTANT * T
    P_dry = (P - P_h2o).clip(lower=1e-6)
    n_dry = P_dry / (IDEAL_GAS_CONSTANT * T)

    out = df.copy()
    out["CO2_MR"] = c_co2 / n_dry * 1e6  # µmol/mol
    out["H2O_MR"] = c_h2o / n_dry * 1e3  # mmol/mol
    return out


#=====================================================================
# Conversion to arrays
#=====================================================================

def df_to_arrays(df: pd.DataFrame):
    return dict(
        u=df["U"].to_numpy(),
        v=df["V"].to_numpy(),
        w=df["W"].to_numpy(),
        Ts=df["T_SONIC"].to_numpy(),
        rho_CO2=df["CO2_CONC"].to_numpy(),
        rho_H2O=df["H2O_CONC"].to_numpy(),
        T_cell=df["T_CELL"].to_numpy(),
        P_cell=df["PRESS_CELL"].to_numpy(),
    )
