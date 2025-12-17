"""
timelag.py
----------

Fixed, hard-coded nominal timelags (in samples at 20 Hz) by ecosystem/site.
These come from the last-month daytime median lag estimates (10–17h, decimate=2, ±15 samples).
"""

from __future__ import annotations

LAG_SAMPLES_MAP = {
    ("igbp_CRO", "BE-Lon"): -4,
    ("igbp_CRO", "DE-Geb"): -2,
    ("igbp_CSH", "BE-Maa"): -4,
    ("igbp_DBF", "CZ-Lnz"): -4,
    ("igbp_DBF", "DE-HoH"): -2,
    ("igbp_EBF", "FR-Pue"): -4,
    ("igbp_ENF", "BE-Bra"): -2,
    ("igbp_ENF", "CH-Dav"): -24,
    ("igbp_GRA", "BE-Dor"): -4,
    ("igbp_GRA", "FR-Lqu"): -2,
    ("igbp_MF", "BE-Vie"): -4,
    ("igbp_MF", "IT-Cp2"): 0,
    ("igbp_WET", "FI-Sii"): -2,
    ("igbp_WET", "GL-ZaF"): -2,
}


def get_site_lag_samples(ecosystem: str, site: str) -> int | None:
    """
    Return fixed nominal lag in samples for the given (ecosystem, site),
    or None if not defined.
    """
    return LAG_SAMPLES_MAP.get((ecosystem, site))
