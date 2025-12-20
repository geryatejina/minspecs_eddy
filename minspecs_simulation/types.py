from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SubsampleMode(str, Enum):
    DECIMATE = "decimate"
    OGIVE_STOP = "ogive_stop"
    BURST = "burst"
    DIURNAL = "diurnal"


@dataclass
class Theta:
    fs_sonic: float
    tau_sonic: float
    sigma_w_noise: float
    sigma_Ts_noise: float
    fs_irga: float
    tau_irga: float
    sigma_CO2dens_noise: float
    sigma_H2Odens_noise: float
    sigma_Tcell_noise: float
    k_CO2_Tsens: float
    k_H2O_Tsens: float
    sigma_lag_jitter: float


@dataclass
class SubsampleSpec:
    """
    Configuration for a downsampling strategy applied before any processing.

    Each SubsampleSpec stands alone; runs are not combined.
    """
    mode: SubsampleMode
    name: Optional[str] = None

    # Decimate
    decimate_factor: Optional[int] = None

    # Ogive early stop
    ogive_threshold: Optional[float] = None
    ogive_trailing_sec: Optional[float] = None
    ogive_min_dwell_sec: Optional[float] = None

    # Burst (also used by diurnal when child mode is burst)
    burst_on_sec: Optional[float] = None
    burst_off_sec: Optional[float] = None

    # Diurnal: choose day/night child specs
    day_spec: Optional["SubsampleSpec"] = None
    night_spec: Optional["SubsampleSpec"] = None

    def label(self) -> str:
        if self.name:
            return self.name
        return self.mode.value


@dataclass
class WindowMetrics:
    site_id: str
    theta_index: int

    F_CO2_ref: float
    F_CO2_deg: float
    bias_CO2: float

    F_LE_ref: float
    F_LE_deg: float
    bias_LE: float

    F_H_ref: float
    F_H_deg: float
    bias_H: float
