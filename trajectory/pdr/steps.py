"""Step detection and step-length estimation for pedestrian dead reckoning.

The standard PDR loop is:

    1. Compute |a| = magnitude of user acceleration (gravity removed).
    2. Low-pass filter at ~3 Hz to remove jitter but keep the ~2 Hz step rhythm.
    3. Find peaks above a threshold with a minimum distance between peaks
       — each peak is one foot strike.
    4. Estimate step length from the acceleration signal in that step
       (Weinberg's heuristic: L = K * (a_max - a_min)^(1/4)).

Why not double-integrate acceleration directly? Because gravity-removal
imperfections plus IMU bias produce drift that grows as t^2 — meters per
second within a few seconds. Step counting is robust to this because each
step resets the integration: we only need a *length per step*, not a
continuous velocity estimate.

Weinberg's K constant: empirically ~0.4–0.5 for adults walking on flat
ground; 0.41 is the most-cited default. It's user- and pace-dependent and
should ideally be calibrated against a known walking distance — for this
project, take a measured walk (e.g., 20 m corridor, count steps) and tune
K so the trajectory length matches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


@dataclass
class StepEvent:
    """A single detected step (one foot strike)."""
    index:    int     # sample index in the resampled stream
    time:     float   # seconds_elapsed
    a_max:    float   # peak acceleration magnitude in this step (m/s²)
    a_min:    float   # trough acceleration magnitude in this step (m/s²)
    length:   float   # estimated step length (metres)


def _butter_lowpass(x: np.ndarray, fs: float, fc: float = 3.0, order: int = 4) -> np.ndarray:
    """Zero-phase low-pass Butterworth (filtfilt = no phase lag, important for peak timing)."""
    nyq = 0.5 * fs
    b, a = butter(order, fc / nyq, btype="low")
    return filtfilt(b, a, x)


def detect_steps(
    accel_xyz: np.ndarray,
    fs: float,
    *,
    seconds_elapsed: Optional[np.ndarray] = None,
    min_peak_height: float = 1.2,
    min_peak_distance_s: float = 0.3,
    lowpass_hz: float = 3.0,
    weinberg_k: float = 0.41,
    use_total_accel: bool = False,
) -> list[StepEvent]:
    """Detect foot strikes from a 3-axis acceleration stream.

    Parameters
    ----------
    accel_xyz : (N, 3) array
        Acceleration in m/s². Pass user acceleration (gravity removed) for
        cleanest peaks; if you only have total acceleration, set
        ``use_total_accel=True`` to subtract the running-mean magnitude.
    fs : float
        Sampling rate in Hz.
    seconds_elapsed : (N,) array, optional
        Timestamps for each sample. If omitted, generated from ``fs``.
    min_peak_height : float
        Minimum peak height in the *low-pass-filtered, mean-subtracted*
        magnitude signal (m/s²). 1.2 works well for a phone in a pants
        pocket; lower it (~0.6) for a hand-held phone where impacts are
        damped, raise it (~2.0) for a phone on the chest.
    min_peak_distance_s : float
        Minimum spacing between peaks in seconds. 0.3 s caps the cadence
        at ~3.3 steps/sec which is faster than running for most people.
    lowpass_hz : float
        Low-pass cutoff. 3 Hz preserves step rhythm (~2 Hz) while removing
        higher-frequency vibration.
    weinberg_k : float
        Step-length scaling constant. Calibrate per user; 0.41 default.
    use_total_accel : bool
        If True, the input is assumed to include gravity; we subtract the
        global mean of the magnitude. Cheap and good enough for PDR — for
        more rigour, pass user acceleration.

    Returns
    -------
    list of StepEvent
    """
    a = np.asarray(accel_xyz, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("accel_xyz must be (N, 3)")
    n = a.shape[0]

    if seconds_elapsed is None:
        seconds_elapsed = np.arange(n) / fs

    # Magnitude, then remove the gravity-ish baseline.
    mag = np.linalg.norm(a, axis=1)
    if use_total_accel:
        mag = mag - np.mean(mag)
    # Low-pass to clean up the rhythm.
    if n > 30:
        mag_lp = _butter_lowpass(mag, fs=fs, fc=lowpass_hz)
    else:
        mag_lp = mag.copy()

    # Peak finding.
    distance_samples = max(1, int(min_peak_distance_s * fs))
    peaks, _ = find_peaks(mag_lp, height=min_peak_height, distance=distance_samples)

    # Build StepEvent objects with Weinberg step length per step.
    events: list[StepEvent] = []
    for i, idx in enumerate(peaks):
        # Window = from previous peak (or 0) to this peak (or next peak halfway).
        lo = peaks[i - 1] if i > 0 else max(0, idx - distance_samples)
        hi = peaks[i + 1] if i + 1 < len(peaks) else min(n, idx + distance_samples)
        window = mag[lo:hi + 1]  # use *unfiltered* magnitude for amplitude
        a_max = float(np.max(window))
        a_min = float(np.min(window))
        length = weinberg_step_length(a_max, a_min, k=weinberg_k)
        events.append(StepEvent(
            index=int(idx),
            time=float(seconds_elapsed[idx]),
            a_max=a_max,
            a_min=a_min,
            length=length,
        ))
    return events


def weinberg_step_length(a_max: float, a_min: float, k: float = 0.41) -> float:
    """Weinberg (2002) step length estimator.

        L = K * (a_max - a_min) ** (1/4)

    Returns metres. Inputs must be in the same units (m/s²). The fourth-
    root form is empirical: the bigger the vertical impact, the longer
    the step, but the relationship saturates.
    """
    diff = max(a_max - a_min, 1e-6)
    return float(k * diff ** 0.25)
