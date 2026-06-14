"""End-to-end PDR pipeline, generalised over recordings.

This module wraps the steps that used to be copy-pasted across notebooks
(``pure_walking_pdr.ipynb`` / ``pdr_NU_walking.ipynb``) into a single
:func:`run_pipeline` call that works on *any* Sensor Logger recording folder
-- walking, upstairs, or downstairs -- as long as it has the standard set of
phone + watch CSV files.

The pipeline:

1.  Loads the recording and fixes the watch clock offset.
2.  Resamples phone + watch onto a common grid (default 60 Hz).
3.  Detects steps on both the watch and phone accelerometer.
4.  Selects a phone forward axis (auto, from the quaternion heading) and
    computes four heading sources:

    - ``quat``       -- Android-fused orientation quaternion, world-frame
                         projection + low-pass filter.
    - ``accel_gyro`` -- self-contained complementary filter using only the
                         accelerometer and gyroscope (no magnetometer
                         anywhere), per the supervisor's suggestion.
    - ``gyro``       -- world-frame gyro rate (via the Android quaternion's
                         rotation matrix), integrated with no correction.
    - ``ekf``        -- 1D Kalman filter fusing the world-frame gyro rate
                         with a tilt-compensated magnetometer compass.

5.  Builds 2D trajectories for every (step source x heading source)
    combination and records the closure error for each.
6.  Extracts the barometric altitude profile (phone + watch, if present) for
    stairs recordings.

Everything is returned in a single :class:`PipelineResult` so a plotting
script can iterate over recordings without repeating any of this logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import warnings

import numpy as np

from .io import Recording, DeviceData, load_recording
from .preprocess import fix_watch_clock, align_phone_watch
from .steps import detect_steps, StepEvent
from .heading import (
    heading_from_quaternion, select_forward_axis, world_yaw_rate,
    integrate_gyro_heading, magnetometer_heading, HeadingEKF,
    heading_from_accel_gyro,
)
from .pdr import compute_trajectory, PDRResult


@dataclass
class PipelineResult:
    """Everything computed for one recording, ready for plotting."""
    name: str
    rec: Recording
    phone: DeviceData          # resampled onto t_grid
    watch: DeviceData          # resampled onto t_grid (empty DeviceData if no watch)
    t_grid: np.ndarray
    fs: float

    fwd_axis_name: str
    fwd_axis: np.ndarray

    phone_steps: list[StepEvent]
    watch_steps: list[StepEvent]

    headings: dict[str, np.ndarray]        # name -> (N,) radians, zeroed at t=0
    trajectories: dict[str, PDRResult]     # "<steps> steps + <heading>" -> result
    closure_errors: dict[str, float]       # same keys, metres

    altitude_phone: Optional[np.ndarray] = None   # (N,) metres, relative
    altitude_watch: Optional[np.ndarray] = None   # (N,) metres, relative

    warnings_raised: list[str] = field(default_factory=list)


def _interp_column(df, col: str, t_grid: np.ndarray) -> Optional[np.ndarray]:
    if df is None or col not in df.columns:
        return None
    t_src = df["seconds_elapsed"].to_numpy()
    return np.interp(t_grid, t_src, df[col].to_numpy())


def run_pipeline(
    data_dir: str | Path,
    fs: float = 60.0,
    weinberg_k: float = 0.41,
    min_peak_height: float = 1.2,
    min_peak_distance_s: float = 0.3,
    heading_lowpass_hz: float = 0.5,
    ekf_Q: float = 1e-4,
    ekf_R: float = 0.25,
    ag_gain: float = 0.02,
    verbose: bool = False,
) -> PipelineResult:
    """Run the full PDR pipeline on one recording folder.

    Parameters
    ----------
    data_dir : path-like
        Folder containing the Sensor Logger CSV files (e.g.
        ``data/NU/Walking``).
    fs : float
        Common resampling rate in Hz.
    weinberg_k : float
        Weinberg step-length constant, passed to :func:`detect_steps`.
    min_peak_height, min_peak_distance_s : float
        Step-detection peak parameters, passed to :func:`detect_steps`.
    heading_lowpass_hz : float
        Low-pass cutoff applied to the quaternion and accel+gyro headings.
    ekf_Q, ekf_R : float
        Process / measurement noise for :class:`HeadingEKF`.
    ag_gain : float
        Complementary-filter gain for :func:`heading_from_accel_gyro`.
    verbose : bool
        Forwarded to :func:`load_recording`.

    Returns
    -------
    PipelineResult
    """
    data_dir = Path(data_dir)
    name = data_dir.name
    msgs: list[str] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rec = load_recording(data_dir, verbose=verbose)
        for w in caught:
            msgs.append(str(w.message))

    # --- clock fix + resample -------------------------------------------------
    if rec.phone.accel_total is None:
        raise ValueError(f"{name}: phone TotalAcceleration.csv missing — cannot proceed.")
    phone_t0_ns = int(rec.phone.accel_total["time"].iloc[0])

    has_watch = rec.watch.accel_total is not None
    if has_watch:
        rec.watch = fix_watch_clock(rec.watch, phone_t0_ns)
        phone, watch = align_phone_watch(rec.phone, rec.watch, fs=fs)
    else:
        from .preprocess import resample_to
        phone, watch = resample_to(rec.phone, fs=fs), DeviceData()
        msgs.append("No watch data — watch-based steps/trajectories skipped.")

    t_grid = phone.accel_total["seconds_elapsed"].to_numpy()

    # --- step detection ---------------------------------------------------------
    p_xyz = phone.accel_total[["x", "y", "z"]].to_numpy()
    phone_steps = detect_steps(
        p_xyz, fs, seconds_elapsed=t_grid,
        min_peak_height=min_peak_height, min_peak_distance_s=min_peak_distance_s,
        weinberg_k=weinberg_k, use_total_accel=True,
    )

    watch_steps: list[StepEvent] = []
    if has_watch and watch.accel_total is not None:
        w_xyz = watch.accel_total[["x", "y", "z"]].to_numpy()
        watch_steps = detect_steps(
            w_xyz, fs, seconds_elapsed=t_grid,
            min_peak_height=min_peak_height, min_peak_distance_s=min_peak_distance_s,
            weinberg_k=weinberg_k, use_total_accel=True,
        )

    # --- forward axis + headings -------------------------------------------------
    fwd_name, fwd_axis = select_forward_axis(
        rec.phone.orientation, t_grid, fs=fs, lowpass_hz=heading_lowpass_hz,
    )

    headings: dict[str, np.ndarray] = {}

    headings["quat"] = heading_from_quaternion(
        rec.phone.orientation, t_grid, forward_axis=fwd_axis,
        lowpass_hz=heading_lowpass_hz, fs=fs,
    )

    gyro_xyz = phone.gyro[["x", "y", "z"]].to_numpy()

    h_ag, _ = heading_from_accel_gyro(
        gyro_xyz, p_xyz, fs=fs, forward_axis=fwd_axis,
        lowpass_hz=heading_lowpass_hz, gain=ag_gain,
    )
    headings["accel_gyro"] = h_ag

    yaw_rate_world = world_yaw_rate(gyro_xyz, rec.phone.orientation, t_grid)
    h_gyro = integrate_gyro_heading(yaw_rate_world, fs=fs)
    h_gyro = np.unwrap(h_gyro)
    h_gyro -= h_gyro[0]
    headings["gyro"] = h_gyro

    if phone.magnet is not None and phone.gravity is not None:
        m = phone.magnet[["x", "y", "z"]].to_numpy()
        g = phone.gravity[["x", "y", "z"]].to_numpy()
        comp = magnetometer_heading(m, gravity_xyz=g)
        h_ekf = HeadingEKF(Q=ekf_Q, R=ekf_R).run(yaw_rate_world, comp, fs=fs)
        h_ekf = np.unwrap(h_ekf)
        h_ekf -= h_ekf[0]
        headings["ekf"] = h_ekf
    else:
        msgs.append("No magnetometer/gravity — EKF heading skipped.")

    # --- trajectories -------------------------------------------------------------
    step_sources = {"phone": phone_steps}
    if watch_steps:
        step_sources["watch"] = watch_steps

    trajectories: dict[str, PDRResult] = {}
    closure_errors: dict[str, float] = {}
    for step_label, steps in step_sources.items():
        if not steps:
            continue
        for h_label, h_arr in headings.items():
            key = f"{step_label} steps + {h_label}"
            res = compute_trajectory(steps, h_arr, t_grid)
            trajectories[key] = res
            closure_errors[key] = float(np.linalg.norm(res.xy[-1]))

    # --- altitude -------------------------------------------------------------------
    altitude_phone = _interp_column(rec.phone.barometer, "relativeAltitude", t_grid)
    altitude_watch = _interp_column(rec.watch.barometer, "relativeAltitude", t_grid)

    return PipelineResult(
        name=name,
        rec=rec,
        phone=phone,
        watch=watch,
        t_grid=t_grid,
        fs=fs,
        fwd_axis_name=fwd_name,
        fwd_axis=fwd_axis,
        phone_steps=phone_steps,
        watch_steps=watch_steps,
        headings=headings,
        trajectories=trajectories,
        closure_errors=closure_errors,
        altitude_phone=altitude_phone,
        altitude_watch=altitude_watch,
        warnings_raised=msgs,
    )