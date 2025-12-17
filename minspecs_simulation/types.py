from dataclasses import dataclass

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
