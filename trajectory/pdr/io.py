"""Load Sensor Logger CSV files from a recording folder.

Sensor Logger writes one CSV per sensor per device. Phone files have plain
names (Accelerometer.csv, Gyroscope.csv, ...) and watch files are prefixed
with "Watch" (WatchAccelerometer.csv, ...). Every file shares the columns
``time`` (epoch nanoseconds) and ``seconds_elapsed`` (float seconds since
recording start). Per-sensor columns vary:

    - Acceleration / Gravity / Gyroscope / Magnetometer : x, y, z
    - Orientation                                       : qw, qx, qy, qz, roll, pitch, yaw
    - Barometer                                         : pressure, relativeAltitude
    - Location                                          : latitude, longitude, altitude, ...
    - Annotation                                        : text, seconds_elapsed
    - Pedometer                                         : steps (cumulative)

This module is deliberately tolerant: missing files are returned as None,
and column ordering differs between Default and Sensor Zoo recordings (a
quirk noted in the project README), so we always index by column name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict
import warnings

import pandas as pd


# Files we expect from a Sensor Logger recording. Anything missing is fine —
# the loader stores None and downstream code checks for it.
PHONE_FILES = {
    "accel":          "Accelerometer.csv",          # gravity removed (user accel)
    "accel_total":    "TotalAcceleration.csv",      # includes gravity
    "gravity":        "Gravity.csv",
    "gyro":           "Gyroscope.csv",
    "magnet":         "Magnetometer.csv",
    "compass":        "Compass.csv",
    "orientation":    "Orientation.csv",
    "barometer":      "Barometer.csv",
    "location":       "Location.csv",
    "pedometer":      "Pedometer.csv",
    "annotation":     "Annotation.csv",
    "metadata":       "Metadata.csv",
}

WATCH_FILES = {
    "accel":          "WatchAccelerometer.csv",
    "accel_total":    "WatchTotalAcceleration.csv",
    "gravity":        "WatchGravity.csv",
    "gyro":           "WatchGyroscope.csv",
    "magnet":         "WatchMagnetometer.csv",
    "orientation":    "WatchOrientation.csv",
    "barometer":      "WatchBarometer.csv",
    "location":       "WatchLocation.csv",
}


@dataclass
class DeviceData:
    """Bundle of sensor DataFrames for one device (phone or watch)."""
    accel:        Optional[pd.DataFrame] = None  # user acceleration (no gravity)
    accel_total:  Optional[pd.DataFrame] = None  # total acceleration (with gravity)
    gravity:      Optional[pd.DataFrame] = None
    gyro:         Optional[pd.DataFrame] = None
    magnet:       Optional[pd.DataFrame] = None
    compass:      Optional[pd.DataFrame] = None
    orientation:  Optional[pd.DataFrame] = None
    barometer:    Optional[pd.DataFrame] = None
    location:     Optional[pd.DataFrame] = None
    pedometer:    Optional[pd.DataFrame] = None

    def has(self, name: str) -> bool:
        df = getattr(self, name, None)
        return df is not None and len(df) > 0


@dataclass
class Recording:
    """A complete Sensor Logger recording (phone + optional watch + annotations)."""
    folder:      Path
    phone:       DeviceData = field(default_factory=DeviceData)
    watch:       DeviceData = field(default_factory=DeviceData)
    annotations: Optional[pd.DataFrame] = None  # text, seconds_elapsed
    metadata:    Optional[pd.DataFrame] = None

    @property
    def duration(self) -> float:
        """Recording duration in seconds, taken from the longest sensor stream."""
        candidates = []
        for dev in (self.phone, self.watch):
            for fld in ("accel", "accel_total", "gyro", "orientation"):
                df = getattr(dev, fld, None)
                if df is not None and "seconds_elapsed" in df.columns and len(df):
                    candidates.append(float(df["seconds_elapsed"].iloc[-1]))
        return max(candidates) if candidates else 0.0

    def annotation_dict(self) -> Dict[str, float]:
        """Return {label: seconds_elapsed} for plotting or segment selection."""
        if self.annotations is None or self.annotations.empty:
            return {}
        return dict(zip(self.annotations["text"], self.annotations["seconds_elapsed"]))


def _read(folder: Path, fname: str) -> Optional[pd.DataFrame]:
    path = folder / fname
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception as e:
        warnings.warn(f"Could not read {path.name}: {e}")
        return None


def load_recording(folder: str | Path, verbose: bool = True) -> Recording:
    """Load all sensor CSVs from a Sensor Logger recording folder.

    Parameters
    ----------
    folder : path-like
        Directory containing the per-sensor CSV files.
    verbose : bool
        Print a one-line summary of what was loaded.

    Returns
    -------
    Recording
        Phone and watch sensor data, plus annotations.

    Notes
    -----
    Missing files do not raise — they become ``None`` on the returned
    ``DeviceData`` and the rest of the pipeline checks for this.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Recording folder not found: {folder}")

    rec = Recording(folder=folder)

    # Phone sensors
    for attr, fname in PHONE_FILES.items():
        if attr == "annotation":
            rec.annotations = _read(folder, fname)
        elif attr == "metadata":
            rec.metadata = _read(folder, fname)
        else:
            setattr(rec.phone, attr, _read(folder, fname))

    # Watch sensors
    for attr, fname in WATCH_FILES.items():
        setattr(rec.watch, attr, _read(folder, fname))

    if verbose:
        n_phone = sum(1 for f in PHONE_FILES if f not in ("annotation", "metadata")
                      and getattr(rec.phone, f) is not None)
        n_watch = sum(1 for f in WATCH_FILES if getattr(rec.watch, f) is not None)
        n_ann = 0 if rec.annotations is None else len(rec.annotations)
        print(f"Loaded {folder.name}: phone={n_phone} sensors, watch={n_watch} sensors, "
              f"annotations={n_ann}, duration={rec.duration:.1f}s")

    return rec
