"""
site_meta.py
------------

Lightweight site metadata: latitude/longitude (degrees) by (ecosystem, site).

Timestamps are assumed to be local standard time (no DST), as per ICOS naming.
"""

from __future__ import annotations

# Populate with (ecosystem, site): (lat_deg, lon_deg)
LAT_LON = {
    ("igbp_ENF", "BE-Bra"): (51.30761, 4.51984),
    ("igbp_GRA", "BE-Dor"): (50.311874, 4.968113),
    ("igbp_CRO", "BE-Lon"): (50.55162, 4.746234),
    ("igbp_CSH", "BE-Maa"): (50.97987, 5.631851),
    ("igbp_MF", "BE-Vie"): (50.304962, 5.998099),
    ("igbp_ENF", "CH-Dav"): (46.81533, 9.85591),
    ("igbp_DBF", "CZ-Lnz"): (48.68155, 16.946331),
    ("igbp_CRO", "DE-Geb"): (51.09973, 10.91463),
    ("igbp_DBF", "DE-HoH"): (52.08656, 11.22235),
    ("igbp_WET", "FI-Sii"): (61.83265, 24.19285),
    ("igbp_GRA", "FR-Lqu"): (45.6444, 2.7349),
    ("igbp_EBF", "FR-Pue"): (43.7413, 3.5957),
    ("igbp_WET", "GL-ZaF"): (74.48152, -20.555773),
    ("igbp_MF", "IT-Cp2"): (41.704266, 12.357293),
}


def get_lat_lon(ecosystem: str, site: str):
    """
    Return (lat, lon) in degrees if available, else None.
    """
    return LAT_LON.get((ecosystem, site))
