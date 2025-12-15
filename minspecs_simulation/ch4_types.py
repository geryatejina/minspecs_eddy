from dataclasses import dataclass


@dataclass
class MethaneTheta:
    """
    Parameter set for the methane-only QCL degradation operator.

    All parameters map directly to QCL design relaxations and are kept
    separate from the CO2/H2O simulation to avoid coupling the two paths.
    """
    f_eff: float       # effective analyzer output rate [Hz]
    tau: float         # effective first-order time constant [s]
    sigma_rho: float   # additive CH4 density noise [ug m-3, 1-sigma]
    sigma_gain: float  # multiplicative gain noise [fraction, 1-sigma]
    sigma_drift: float # low-frequency drift amplitude over 30 min [ug m-3]
    sigma_lag: float   # lag jitter vs sonic [s, 1-sigma]


__all__ = ["MethaneTheta"]
