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

# Variables needed for simulation
SELECT_COLUMNS = [
    "U", "V", "W", "T_SONIC",
    "CO2_CONC", "H2O_CONC",
    "T_CELL", "PRESS_CELL",
]

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
    Load ICOS Level-0 CSV file.
    """
    df = pd.read_csv(file_path, low_memory=False)

    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col].astype(str), format="%Y%m%d%H%M%S.%f")
    df = df.set_index(ts_col)
    df.index.name = "TIMESTAMP"

    # select only available needed columns
    cols = [c for c in SELECT_COLUMNS if c in df.columns]
    return df[cols]


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
