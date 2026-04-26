"""Resample sensors to a common time base.

Sensor Logger advertises 100 Hz but actual rates differ — the phone IMU
typically delivers ~54 Hz and the watch ~73 Hz, while the magnetometer can
hit 100 Hz on the phone and ~70 Hz on the watch. To fuse them we need them
on a single regular grid. We use linear interpolation, which is fine for
IMU data sampled well above any physical frequency of human motion (step
cadence is ~1.5–2.5 Hz; arm and torso motion is well below 10 Hz).

For quaternion orientation we interpolate component-wise then renormalise.
This is *not* SLERP and will introduce tiny errors during fast rotations,
but it's robust, fast, and the residual is dominated by sensor noise at
the rates we deal with here.
"""
from __future__ import annotations

from typing import Iterable, Optional, Dict
import numpy as np
import pandas as pd

from .io import DeviceData


# Default columns to interpolate, by sensor name. Listed explicitly so we
# don't accidentally interpolate non-numeric columns or 'time'/'seconds_elapsed'.
_NUMERIC_COLS: Dict[str, list[str]] = {
    "accel":         ["x", "y", "z"],
    "accel_total":   ["x", "y", "z"],
    "gravity":       ["x", "y", "z"],
    "gyro":          ["x", "y", "z"],
    "magnet":        ["x", "y", "z"],
    "compass":       ["magneticBearing", "trueBearing"],  # if present
    "orientation":   ["qw", "qx", "qy", "qz", "roll", "pitch", "yaw"],
    "barometer":     ["pressure", "relativeAltitude"],
}


def _resample_one(df: pd.DataFrame, t_new: np.ndarray, cols: list[str]) -> pd.DataFrame:
    """Linearly interpolate the given columns onto ``t_new`` (seconds_elapsed)."""
    t_old = df["seconds_elapsed"].to_numpy()
    out = {"seconds_elapsed": t_new}
    for c in cols:
        if c in df.columns:
            out[c] = np.interp(t_new, t_old, df[c].to_numpy())
    return pd.DataFrame(out)


def _renorm_quat(df: pd.DataFrame) -> pd.DataFrame:
    """Renormalise quaternion rows in-place (after interpolation drift)."""
    if not all(c in df.columns for c in ("qw", "qx", "qy", "qz")):
        return df
    q = df[["qw", "qx", "qy", "qz"]].to_numpy()
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n[n == 0] = 1.0
    q = q / n
    df = df.copy()
    df[["qw", "qx", "qy", "qz"]] = q
    return df


def resample_to(
    device: DeviceData,
    fs: float = 60.0,
    sensors: Optional[Iterable[str]] = None,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> DeviceData:
    """Return a new DeviceData with each sensor resampled to ``fs`` Hz.

    Parameters
    ----------
    device : DeviceData
        Source data (phone or watch).
    fs : float
        Target sampling rate in Hz. 60 Hz is the MobilePoser default and
        sits comfortably below the phone's actual ~54 Hz floor — for
        general PDR work, 50 Hz is also fine.
    sensors : iterable of str, optional
        Subset of sensor attribute names to resample. Default: all sensors
        present on the device.
    t_start, t_end : float, optional
        Restrict the common time base. Defaults to the overlap window of
        every sensor present.

    Returns
    -------
    DeviceData
        Resampled sensors. Quaternions are renormalised after interpolation.
    """
    available = [s for s in _NUMERIC_COLS if getattr(device, s, None) is not None]
    if sensors is not None:
        available = [s for s in available if s in sensors]
    if not available:
        return DeviceData()

    # Common time window = intersection of all selected sensors' ranges.
    starts, ends = [], []
    for s in available:
        df = getattr(device, s)
        starts.append(float(df["seconds_elapsed"].iloc[0]))
        ends.append(float(df["seconds_elapsed"].iloc[-1]))
    t0 = max(starts) if t_start is None else t_start
    t1 = min(ends) if t_end is None else t_end
    if t1 <= t0:
        raise ValueError(f"Empty time window: t_start={t0}, t_end={t1}")

    n = int(np.floor((t1 - t0) * fs)) + 1
    t_new = t0 + np.arange(n) / fs

    out = DeviceData()
    for s in available:
        df = getattr(device, s)
        cols = [c for c in _NUMERIC_COLS[s] if c in df.columns]
        new_df = _resample_one(df, t_new, cols)
        if s == "orientation":
            new_df = _renorm_quat(new_df)
        setattr(out, s, new_df)
    return out


def align_phone_watch(
    phone: DeviceData,
    watch: DeviceData,
    fs: float = 60.0,
) -> tuple[DeviceData, DeviceData]:
    """Resample phone and watch onto a single shared time grid.

    Both devices use the same ``seconds_elapsed`` epoch (Sensor Logger
    starts both clocks together when the recording begins), so we just
    need to pick the overlap window and resample both at the same rate.
    """
    # Overlap window
    p_starts, p_ends, w_starts, w_ends = [], [], [], []
    for s in _NUMERIC_COLS:
        if getattr(phone, s, None) is not None:
            df = getattr(phone, s)
            p_starts.append(float(df["seconds_elapsed"].iloc[0]))
            p_ends.append(float(df["seconds_elapsed"].iloc[-1]))
        if getattr(watch, s, None) is not None:
            df = getattr(watch, s)
            w_starts.append(float(df["seconds_elapsed"].iloc[0]))
            w_ends.append(float(df["seconds_elapsed"].iloc[-1]))

    if not w_starts:
        # No watch data — just resample the phone.
        return resample_to(phone, fs=fs), DeviceData()

    t0 = max(max(p_starts), max(w_starts))
    t1 = min(min(p_ends), min(w_ends))
    if t1 <= t0:
        raise ValueError("Phone and watch have no time overlap.")

    return (
        resample_to(phone, fs=fs, t_start=t0, t_end=t1),
        resample_to(watch, fs=fs, t_start=t0, t_end=t1),
    )
