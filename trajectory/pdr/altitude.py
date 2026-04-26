"""Barometric altitude for stair detection.

Atmospheric pressure decreases with altitude predictably enough that a
calibrated barometer can resolve floor-to-floor changes (~3 m → ~0.36 hPa).
This is *much* better than trying to integrate vertical acceleration.

We use the standard barometric formula (international barometric formula
for the troposphere):

    h = 44330 * (1 - (P / P0)^(1/5.255))    [metres]

where P0 is the reference pressure at h = 0 m. For relative altitude
within a single building we pick P0 = the first pressure reading, which
makes h(t=0) = 0 by construction.

Sensor Logger usually exposes ``relativeAltitude`` directly — if present,
prefer that and skip this conversion. ``pressure_to_altitude`` is here
for cases where only the raw pressure was logged.
"""
from __future__ import annotations
import numpy as np


def pressure_to_altitude(pressure_hpa: np.ndarray, p0: float | None = None) -> np.ndarray:
    """Convert pressure (hPa) to altitude (metres) relative to the first sample.

    Parameters
    ----------
    pressure_hpa : (N,) array
        Pressure readings in hectopascals (millibars). Sensor Logger
        reports kPa for some Android versions and hPa for others —
        check the column units before passing! 1 kPa = 10 hPa.
    p0 : float, optional
        Reference pressure in the same units. Default = first sample,
        so the output starts at 0 m.

    Returns
    -------
    altitude_m : (N,) array
    """
    p = np.asarray(pressure_hpa, dtype=float)
    if p0 is None:
        p0 = float(p[0])
    return 44330.0 * (1.0 - (p / p0) ** (1.0 / 5.255))
