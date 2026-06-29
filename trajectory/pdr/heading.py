"""Heading estimation for pedestrian dead reckoning.

We need a single yaw angle (heading-from-North, in radians, increasing
counter-clockwise in the local horizontal plane) at every sample.

We have three candidate sources, each with its own failure mode:

    1. **Gyro integration**  — integrate the z-axis gyro to get yaw rate
       over time. Smooth and responsive but drifts: any constant bias
       integrates to a linear heading error, e.g. 0.5°/s bias → 30° in
       a minute.
    2. **Magnetometer compass** — atan2(my, mx) gives an absolute heading.
       No drift, but indoor magnetic fields are wrecked by steel beams,
       elevators, laptops; the signal is also tilt-dependent and needs
       to be projected into the horizontal plane.
    3. **Sensor fusion quaternion (yaw)** — Sensor Logger's Orientation.csv
       already provides a fused yaw via Android sensor fusion or Madgwick
       in the Sensor Zoo build. This is usually the best free lunch.

The Kalman filter here fuses (1) and (2) when you don't trust the OS-fused
yaw, or want to compare. State is just the heading angle ψ. The gyro
provides a *prediction* (state advances by ω·dt with high process noise),
the magnetometer provides a *measurement* (angle reading with high
measurement noise indoors). The result tracks fast turns from gyro and
gets pulled back by the magnetometer over seconds, so drift stays bounded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


def _wrap(angle: float) -> float:
    """Wrap angle to (-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def integrate_gyro_heading(
    gyro_z: np.ndarray,
    fs: float,
    initial_heading: float = 0.0,
) -> np.ndarray:
    """Integrate z-axis gyro to get heading over time.

    Parameters
    ----------
    gyro_z : (N,) array
        Gyro z-axis reading in rad/s. Sign convention: positive = CCW
        rotation about the device-z axis. When the device is held flat
        screen-up, this is yaw.
    fs : float
        Sampling rate.
    initial_heading : float
        Starting heading in radians.

    Notes
    -----
    No bias correction. For a real PDR system you'd estimate gyro bias
    during a stationary period at the start of the recording and subtract
    it. The EKF below handles bias drift via the magnetometer correction.
    """
    dt = 1.0 / fs
    psi = np.empty(gyro_z.size)
    psi[0] = initial_heading
    for i in range(1, gyro_z.size):
        psi[i] = _wrap(psi[i - 1] + gyro_z[i] * dt)
    return psi


def magnetometer_heading(
    mag_xyz: np.ndarray,
    gravity_xyz: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute compass heading from magnetometer, optionally tilt-compensated.

    Parameters
    ----------
    mag_xyz : (N, 3) array
        Magnetic field in any consistent unit (μT typical).
    gravity_xyz : (N, 3) array, optional
        Gravity vector in the same body frame. If provided, the
        magnetometer is projected onto the horizontal plane (i.e. tilt
        compensation). Without this, the heading is correct only when
        the device is flat.

    Returns
    -------
    heading : (N,) array of radians, in (-π, π].

    Notes
    -----
    The output is in the device's local magnetic-North frame, *not* true
    North. For navigation that crosses a few hundred kilometres this
    matters; for a single building it does not.
    """
    m = np.asarray(mag_xyz, dtype=float)

    if gravity_xyz is None:
        # Naive: assume device flat, screen-up. Heading from -mx, my.
        # Sign convention chosen so heading increases CCW (math convention).
        return np.arctan2(-m[:, 0], m[:, 1])

    # Tilt-compensated compass.
    # Build an east vector e = m × g  (orthogonal to both magnetic and gravity);
    # then north n = g × e.  Heading = atan2(east, north) projected.
    g = np.asarray(gravity_xyz, dtype=float)
    g_n = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
    east = np.cross(m, g_n)
    east /= (np.linalg.norm(east, axis=1, keepdims=True) + 1e-9)
    north = np.cross(g_n, east)
    # Heading is the angle of the device-y axis in the (east, north) frame.
    # Project device y-axis = (0,1,0) onto the horizontal frame.
    # Equivalent to: -atan2(east_y, north_y) for screen-up devices.
    return np.arctan2(east[:, 1], north[:, 1])


@dataclass
class HeadingEKF:
    """1-D Kalman filter that fuses gyro rate (predict) with compass (update).

    State:        ψ          (heading, rad, wrapped to (-π, π])
    Predict:      ψ_k = ψ_{k-1} + ω_z * dt
                  P_k = P_{k-1} + Q                  (Q = process noise per step)
    Update:       z   = compass_heading
                  innov = wrap(z - ψ_k)              (must wrap, angles!)
                  K   = P_k / (P_k + R)              (R = measurement noise)
                  ψ_k = wrap(ψ_k + K * innov)
                  P_k = (1 - K) * P_k

    The trick with angular Kalman filters is that you must wrap the
    innovation before applying the gain — otherwise crossing ±π will
    produce a 2π jump that the filter interprets as a giant correction.

    Tuning rules of thumb:
        Q ~ (gyro_noise_density * dt)²              ≈ 1e-6 to 1e-4 rad²/step
        R ~ (compass_noise_std)²                    ≈ 0.05–0.5 rad² indoors
    Increase Q to trust the gyro less (faster pull-back to compass).
    Increase R to trust the compass less (smoother, more drift).
    """
    Q: float = 1e-4   # process noise variance per step (rad²)
    R: float = 0.25   # measurement noise variance (rad²)  ~0.5 rad std indoors
    P: float = 1.0    # initial heading variance
    psi: float = 0.0  # current heading estimate (rad)

    def predict(self, gyro_z: float, dt: float) -> None:
        self.psi = _wrap(self.psi + gyro_z * dt)
        self.P = self.P + self.Q

    def update(self, compass_psi: float) -> None:
        innov = _wrap(compass_psi - self.psi)
        K = self.P / (self.P + self.R)
        self.psi = _wrap(self.psi + K * innov)
        self.P = (1 - K) * self.P

    def run(
        self,
        gyro_z: np.ndarray,
        compass_psi: np.ndarray,
        fs: float,
        compass_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run filter over arrays. Returns (N,) array of fused headings.

        Parameters
        ----------
        gyro_z : (N,) array
            Yaw rate (rad/s).
        compass_psi : (N,) array
            Compass heading per sample (rad).
        fs : float
            Sampling rate.
        compass_mask : (N,) boolean array, optional
            True at samples where the compass should be used. If you
            detect magnetic anomalies (e.g., |m| outside expected range
            for Earth's field, ~25–65 μT), set those samples to False
            to skip the update step. Defaults to all True.
        """
        n = len(gyro_z)
        if compass_mask is None:
            compass_mask = np.ones(n, dtype=bool)
        dt = 1.0 / fs
        # Initialise with first valid compass reading if available.
        first_valid = np.argmax(compass_mask) if compass_mask.any() else 0
        self.psi = float(compass_psi[first_valid]) if compass_mask.any() else 0.0
        out = np.empty(n)
        for i in range(n):
            self.predict(gyro_z[i], dt)
            if compass_mask[i]:
                self.update(compass_psi[i])
            out[i] = self.psi
        return out


# ---------------------------------------------------------------------------
# Quaternion-based tilt-compensated heading
# ---------------------------------------------------------------------------

def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternions to rotation matrices.

    Parameters
    ----------
    q : (N, 4) array  [qw, qx, qy, qz]
        Unit quaternions.  Renormalise before passing if interpolated.

    Returns
    -------
    R : (N, 3, 3) array
        Rotation matrices such that ``v_world = R[i] @ v_phone``.
    """
    q = np.atleast_2d(q).astype(float)
    qw, qx, qy, qz = q[:,0], q[:,1], q[:,2], q[:,3]
    N = len(q)
    R = np.zeros((N, 3, 3))
    R[:,0,0] = 1 - 2*(qy**2 + qz**2)
    R[:,0,1] = 2*(qx*qy - qw*qz)
    R[:,0,2] = 2*(qx*qz + qw*qy)
    R[:,1,0] = 2*(qx*qy + qw*qz)
    R[:,1,1] = 1 - 2*(qx**2 + qz**2)
    R[:,1,2] = 2*(qy*qz - qw*qx)
    R[:,2,0] = 2*(qx*qz - qw*qy)
    R[:,2,1] = 2*(qy*qz + qw*qx)
    R[:,2,2] = 1 - 2*(qx**2 + qy**2)
    return R


def heading_from_quaternion(
    orientation_df,
    t_grid: np.ndarray,
    forward_axis: np.ndarray = np.array([0., 0., -1.]),
    lowpass_hz: float = 0.5,
    fs: float = 60.0,
) -> np.ndarray:
    """Tilt-compensated heading from orientation quaternion.

    Works regardless of how the phone is oriented (pocket, hand, chest).
    Converts the quaternion to a rotation matrix, projects the phone's
    forward axis into the world frame, and takes ``atan2(north, east)``.
    Optionally low-pass filters to remove step-frequency oscillations.

    Parameters
    ----------
    orientation_df : DataFrame
        Sensor Logger ``Orientation.csv``.  Must contain
        ``seconds_elapsed`` and quaternion columns (``qw, qx, qy, qz``
        or ``w, x, y, z``).
    t_grid : (N,) array
        Common time grid (seconds_elapsed) to interpolate onto.
    forward_axis : (3,) array
        Phone-frame axis that points in the walking direction.
        Default ``[0, 0, -1]`` (−Z, out of screen for a face-down phone).
        For port-up screen-toward-leg in a right pocket, ``[1, 0, 0]``
        (+X) is typically correct — use ``select_forward_axis`` to
        auto-detect.
    lowpass_hz : float
        Low-pass cutoff for heading smoothing.  0.5 Hz removes step
        oscillations (~1.5 Hz) while preserving turns (~0.5–1 Hz).
        Set to 0 to skip filtering.
    fs : float
        Sampling rate of ``t_grid``.

    Returns
    -------
    heading : (N,) array
        Unwrapped heading in radians, zeroed at the first sample.
    """
    from scipy.signal import butter, filtfilt

    # Normalise column names (Sensor Logger uses qw/qx/qy/qz or w/x/y/z).
    df = orientation_df.copy()
    for full, short in [("qw","w"),("qx","x"),("qy","y"),("qz","z")]:
        if full not in df.columns and short in df.columns:
            df = df.rename(columns={short: full})

    t_ori = df["seconds_elapsed"].to_numpy()
    q = np.stack([np.interp(t_grid, t_ori, df[c].to_numpy())
                  for c in ("qw","qx","qy","qz")], axis=1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    R = quat_to_R(q)
    world_fwd = np.einsum("nij,j->ni", R, forward_axis.astype(float))

    raw = np.arctan2(world_fwd[:,1], world_fwd[:,0])
    h   = np.unwrap(raw)
    h  -= h[0]

    if lowpass_hz > 0 and len(h) > 30:
        b, a = butter(2, lowpass_hz / (fs / 2), btype="low")
        h = filtfilt(b, a, h)
        h -= h[0]

    return h


def select_forward_axis(
    orientation_df,
    t_grid: np.ndarray,
    fs: float = 60.0,
    lowpass_hz: float = 0.5,
) -> tuple[str, np.ndarray]:
    """Auto-select the phone-frame axis that gives a ~±360° heading change.

    For a recording that is a single closed loop (out-and-back or rectangle),
    the correct forward axis will produce a total heading change close to
    ±360°.  Tries all six principal axes and returns the best match.

    Parameters
    ----------
    orientation_df : DataFrame
        Sensor Logger ``Orientation.csv``.
    t_grid : (N,) array
        Common time grid.
    fs : float
        Sampling rate of ``t_grid``.
    lowpass_hz : float
        Passed to ``heading_from_quaternion``.

    Returns
    -------
    name : str
        One of ``'+X'``, ``'-X'``, ``'+Y'``, ``'-Y'``, ``'+Z'``, ``'-Z'``.
    axis : (3,) array
        The selected forward axis vector.
    """
    candidates = {
        "+X": np.array([ 1., 0., 0.]),
        "-X": np.array([-1., 0., 0.]),
        "+Y": np.array([ 0., 1., 0.]),
        "-Y": np.array([ 0.,-1., 0.]),
        "+Z": np.array([ 0., 0., 1.]),
        "-Z": np.array([ 0., 0.,-1.]),
    }
    best_name, best_vec, best_score = "+X", candidates["+X"], np.inf
    for name, axis in candidates.items():
        h = heading_from_quaternion(orientation_df, t_grid,
                                     forward_axis=axis,
                                     lowpass_hz=lowpass_hz, fs=fs)
        total = float(h[-1])
        score = min(abs(total - (-2*np.pi)), abs(total - 2*np.pi))
        if score < best_score:
            best_score, best_name, best_vec = score, name, axis
    return best_name, best_vec


def world_yaw_rate(
    gyro_xyz: np.ndarray,
    orientation_df,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Extract the world-vertical yaw rate from a phone in any orientation.

    The raw gyro z-axis only equals the world yaw rate when the phone is
    held flat (screen up). For any other orientation — pocket, tilted, etc.
    — you must rotate the full gyro vector into the world frame first and
    take the vertical (Z) component.

    Parameters
    ----------
    gyro_xyz : (N, 3) array
        Raw gyro in rad/s, already resampled onto ``t_grid``.
    orientation_df : DataFrame
        Sensor Logger ``Orientation.csv`` with quaternion columns.
    t_grid : (N,) array
        Common time grid (seconds_elapsed).

    Returns
    -------
    yaw_rate : (N,) array
        Rotation rate around the world vertical axis (rad/s).
        Positive = counter-clockwise when viewed from above.
    """
    # Build rotation matrices on t_grid
    df = orientation_df.copy()
    for full, short in [("qw","w"),("qx","x"),("qy","y"),("qz","z")]:
        if full not in df.columns and short in df.columns:
            df = df.rename(columns={short: full})
    t_ori = df["seconds_elapsed"].to_numpy()
    q = np.stack([np.interp(t_grid, t_ori, df[c].to_numpy())
                  for c in ("qw","qx","qy","qz")], axis=1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    R = quat_to_R(q)                                   # (N, 3, 3)

    # Rotate gyro vector to world frame, take Z component
    # omega_world[i] = R[i] @ gyro_xyz[i]
    omega_world = np.einsum("nij,nj->ni", R, gyro_xyz)  # (N, 3)
    return omega_world[:, 2]                             # world yaw rate


# ---------------------------------------------------------------------------
# Accelerometer + gyroscope only heading (no magnetometer anywhere)
# ---------------------------------------------------------------------------
#
# The heading sources above all ultimately depend on Android's fused
# orientation quaternion (Orientation.csv) for the rotation matrix used to
# project the gyro or a body axis into the world frame. That fused quaternion
# is itself partly derived from the magnetometer, so "gyro-only" heading is
# not actually magnetometer-free in the strict sense.
#
# This section implements a self-contained attitude estimator using only the
# gyroscope (integration) and the accelerometer (gravity-direction
# correction for roll/pitch). No magnetometer input is used anywhere. Yaw is
# therefore driven purely by gyro integration and will drift, but roll/pitch
# (and hence the rotation matrix used to project gyro/forward-axis into the
# world frame) come from the device's own accelerometer rather than from
# Android's (possibly magnetometer-tainted) sensor fusion.
#
# This is the classical "AHRS without magnetometer" complementary filter:
# accelerometer correction only ever rotates the estimate about a
# horizontal axis (it corrects the direction of gravity in the body frame),
# so it cannot, even in principle, correct yaw -- a useful property, since
# it means this filter's yaw is *exactly* the gyro-integrated yaw, just with
# correct (accelerometer-derived) roll/pitch for projecting into the world
# frame.


def _quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [qw, qx, qy, qz]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit quaternion that rotates unit vector ``a`` onto unit vector ``b``."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d > 1 - 1e-9:
        return np.array([1., 0., 0., 0.])
    if d < -1 + 1e-9:
        # 180 degree rotation: pick any axis orthogonal to a
        axis = np.cross(a, [1., 0., 0.])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0., 1., 0.])
        axis /= np.linalg.norm(axis)
        return np.array([0., axis[0], axis[1], axis[2]])
    axis = np.cross(a, b)
    q = np.array([1 + d, axis[0], axis[1], axis[2]])
    return q / np.linalg.norm(q)


def complementary_filter_attitude(
    gyro_xyz: np.ndarray,
    accel_total_xyz: np.ndarray,
    fs: float,
    gain: float = 0.02,
    gravity_lowpass_hz: float = 0.3,
) -> np.ndarray:
    """Estimate phone-to-world attitude from gyroscope + accelerometer only.

    Predict step integrates the gyroscope (body-frame angular velocity) using
    standard quaternion kinematics. Correct step nudges roll/pitch toward the
    gravity direction estimated from a low-pass-filtered total acceleration
    signal -- no magnetometer is used at any point.

    Parameters
    ----------
    gyro_xyz : (N, 3) array
        Gyroscope reading in rad/s, phone frame, resampled onto a uniform grid.
    accel_total_xyz : (N, 3) array
        Total acceleration (linear + gravity) in m/s^2, phone frame, same grid.
    fs : float
        Sampling rate of the grid (Hz).
    gain : float
        Complementary filter correction gain. Larger values trust the
        accelerometer more (faster roll/pitch convergence, noisier under
        dynamic acceleration); smaller values trust the gyro more.
    gravity_lowpass_hz : float
        Cutoff for extracting the gravity direction from total acceleration.
        Should be well below step frequency (~1.5 Hz) so walking dynamics
        average out, leaving the gravity DC component.

    Returns
    -------
    q : (N, 4) array
        Unit quaternions [qw, qx, qy, qz], phone-to-world, such that
        ``v_world = quat_to_R(q[i]) @ v_phone``.
    """
    from scipy.signal import butter, filtfilt

    n = len(gyro_xyz)
    dt = 1.0 / fs

    # Gravity direction estimate: low-pass the total acceleration.
    b, a = butter(2, gravity_lowpass_hz / (fs / 2), btype="low")
    g_phone = filtfilt(b, a, accel_total_xyz, axis=0)
    g_hat = g_phone / np.linalg.norm(g_phone, axis=1, keepdims=True)

    world_up = np.array([0., 0., 1.])
    q = np.zeros((n, 4))
    # Initialise attitude so that the measured gravity direction maps to
    # world-up -- gives correct roll/pitch immediately, yaw = 0 (arbitrary).
    q[0] = _quat_from_two_vectors(g_hat[0], world_up)

    for i in range(1, n):
        # Predict: integrate gyro (body-frame angular velocity).
        omega_q = np.array([0., gyro_xyz[i, 0], gyro_xyz[i, 1], gyro_xyz[i, 2]])
        dq = 0.5 * _quat_mult(q[i - 1], omega_q)
        q_pred = q[i - 1] + dq * dt
        q_pred /= np.linalg.norm(q_pred)

        # Correct: rotate gravity-direction estimate toward world-up.
        # This correction is orthogonal to yaw -- it only adjusts roll/pitch.
        R_pred = quat_to_R(q_pred[None, :])[0]
        measured_up_world = R_pred @ g_hat[i]
        error = np.cross(measured_up_world, world_up)
        q_corr = np.array([1., 0.5*gain*error[0], 0.5*gain*error[1], 0.5*gain*error[2]])
        q_corr /= np.linalg.norm(q_corr)
        q_new = _quat_mult(q_corr, q_pred)
        q[i] = q_new / np.linalg.norm(q_new)

    return q


def heading_from_accel_gyro(
    gyro_xyz: np.ndarray,
    accel_total_xyz: np.ndarray,
    fs: float,
    forward_axis: np.ndarray = np.array([1., 0., 0.]),
    lowpass_hz: float = 0.5,
    gain: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Tilt-compensated heading using only the gyroscope and accelerometer.

    Builds a self-contained attitude estimate with
    :func:`complementary_filter_attitude` (no magnetometer involved at all),
    then projects ``forward_axis`` into the world frame and takes
    ``atan2(north, east)``, exactly as :func:`heading_from_quaternion` does
    for the Android-fused quaternion.

    Parameters
    ----------
    gyro_xyz : (N, 3) array
        Gyroscope in rad/s, phone frame, resampled onto a uniform grid.
    accel_total_xyz : (N, 3) array
        Total acceleration in m/s^2, phone frame, same grid.
    fs : float
        Sampling rate (Hz).
    forward_axis : (3,) array
        Phone-frame axis pointing in the walking direction. The same axis
        selected for the Android-quaternion heading
        (:func:`select_forward_axis`) applies here too, since it is a
        property of how the phone is carried, not of which attitude
        estimator is used.
    lowpass_hz : float
        Low-pass cutoff removing step-frequency heading oscillations.
        Set to 0 to skip filtering.
    gain : float
        Complementary filter gain, passed to
        :func:`complementary_filter_attitude`.

    Returns
    -------
    heading : (N,) array
        Unwrapped heading in radians, zeroed at the first sample.
    q : (N, 4) array
        The self-contained attitude quaternions, in case the rotation
        matrices are needed elsewhere (e.g. for a matching world-frame
        gyro rate).
    """
    from scipy.signal import butter, filtfilt

    q = complementary_filter_attitude(gyro_xyz, accel_total_xyz, fs, gain=gain)
    R = quat_to_R(q)
    world_fwd = np.einsum("nij,j->ni", R, forward_axis.astype(float))

    raw = np.arctan2(world_fwd[:, 1], world_fwd[:, 0])
    h = np.unwrap(raw)
    h -= h[0]

    if lowpass_hz > 0 and len(h) > 30:
        b, a = butter(2, lowpass_hz / (fs / 2), btype="low")
        h = filtfilt(b, a, h)
        h -= h[0]

    return h, q


# ---------------------------------------------------------------------------
# Accelerometer + gyroscope Extended Kalman Filter (no magnetometer)
# ---------------------------------------------------------------------------
#
# This is the supervisor-requested method: a Kalman filter that fuses ONLY
# the accelerometer and gyroscope (no magnetometer anywhere). It is built in
# the same family as HeadingEKF above -- a small Extended Kalman Filter with
# an explicit state, covariance P, process noise Q, measurement noise R, and a
# Kalman gain K recomputed every step -- so it is a true Kalman filter, not a
# fixed-gain complementary filter.
#
# How it differs from HeadingEKF (gyro + magnetometer):
#   - HeadingEKF's measurement is an absolute compass heading from the
#     magnetometer. The accelerometer canNOT provide an absolute heading
#     (gravity is vertical, so it says nothing about which way is North).
#   - Instead, this filter uses the accelerometer for what it *can* observe:
#     the gravity direction, which (a) lets us project the gyro into the world
#     frame correctly for the heading prediction, and (b) provides a
#     near-zero-mean "the device is not accelerating on average" constraint
#     that lets the filter observe and remove the slow gyroscope bias -- the
#     main cause of gyro-only heading drift.
#
# State (2-vector):   x = [ psi, b ]
#     psi : heading (rad)
#     b   : gyro yaw-rate bias (rad/s)   -- slowly varying, ~constant
#
# Predict (per step, dt):
#     psi <- wrap(psi + (omega_world_z - b) * dt)
#     b   <- b                                  (bias modelled as constant)
#     F = [[1, -dt], [0, 1]]
#     P <- F P F^T + Q
#
# Measurement update:
#     The accelerometer's low-passed gravity direction gives a stable attitude
#     and hence a tilt-compensated world-frame yaw rate. Integrating that rate
#     over a short window gives an independent heading increment that does not
#     use the (bias-corrupted) state estimate -- this is the "measurement" the
#     filter corrects toward, which makes the bias observable. See
#     heading_from_accel_gyro_ekf for how the measurement series is built.
#     z = psi_accel        (accelerometer-derived heading reference, rad)
#     H = [1, 0]
#     innov = wrap(z - psi)
#     S = H P H^T + R ;  K = P H^T / S
#     x <- x + K * innov ;  P <- (I - K H) P


class HeadingBiasEKF:
    """2-state EKF on [heading, gyro-bias], fusing gyro + accelerometer only.

    Same Kalman machinery as :class:`HeadingEKF` (explicit P, Q, R, and a
    gain recomputed each step), extended to two states so the slowly varying
    gyroscope bias can be estimated and removed -- the dominant cause of
    drift in gyro-only heading. No magnetometer is used.

    Parameters / tuning
    -------------------
    q_psi : float
        Process-noise variance for heading per step (rad^2). Small.
    q_bias : float
        Process-noise variance for the bias random walk per step
        ((rad/s)^2). Smaller = assume the bias drifts more slowly.
    R : float
        Measurement-noise variance of the accelerometer-derived heading
        reference (rad^2). Larger = trust the accel reference less.
    """

    def __init__(self, q_psi: float = 1e-5, q_bias: float = 1e-7,
                 R: float = 0.5):
        self.Q = np.array([[q_psi, 0.0], [0.0, q_bias]])
        self.R = float(R)
        self.x = np.zeros(2)            # [psi, bias]
        self.P = np.diag([1.0, 1e-2])   # initial covariance

    def predict(self, omega_world_z: float, dt: float) -> None:
        psi, b = self.x
        psi = _wrap(psi + (omega_world_z - b) * dt)
        self.x = np.array([psi, b])
        F = np.array([[1.0, -dt], [0.0, 1.0]])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, psi_meas: float) -> None:
        H = np.array([[1.0, 0.0]])
        innov = _wrap(psi_meas - self.x[0])
        S = float((H @ self.P @ H.T).item()) + self.R
        K = (self.P @ H.T).flatten() / S      # (2,)
        self.x = self.x + K * innov
        self.x[0] = _wrap(self.x[0])
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P

    def run(
        self,
        omega_world_z: np.ndarray,
        psi_meas: np.ndarray,
        fs: float,
        meas_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run the filter over arrays. Returns (N,) fused, unwrapped heading.

        Parameters
        ----------
        omega_world_z : (N,) array
            World-frame yaw rate from the gyro (rad/s) -- the prediction
            input. Build this with :func:`world_yaw_rate` using the
            accel+gyro complementary attitude so it is tilt-compensated
            without the magnetometer.
        psi_meas : (N,) array
            Accelerometer-derived heading reference (rad) -- the
            measurement. See :func:`heading_from_accel_gyro_ekf`.
        fs : float
            Sampling rate (Hz).
        meas_mask : (N,) bool array, optional
            Samples at which to apply the measurement update. Defaults to
            all True.
        """
        n = len(omega_world_z)
        if meas_mask is None:
            meas_mask = np.ones(n, dtype=bool)
        dt = 1.0 / fs
        self.x = np.array([float(psi_meas[0]), 0.0])
        self.P = np.diag([1.0, 1e-2])
        out = np.empty(n)
        for i in range(n):
            self.predict(omega_world_z[i], dt)
            if meas_mask[i]:
                self.update(psi_meas[i])
            out[i] = self.x[0]
        return np.unwrap(out)


def heading_from_accel_gyro_ekf(
    gyro_xyz: np.ndarray,
    accel_total_xyz: np.ndarray,
    fs: float,
    forward_axis: np.ndarray = np.array([1., 0., 0.]),
    lowpass_hz: float = 0.5,
    gain: float = 0.02,
    q_psi: float = 1e-5,
    q_bias: float = 1e-7,
    R: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Heading from an accel+gyro Extended Kalman Filter (no magnetometer).

    This is the Kalman-filter counterpart to
    :func:`heading_from_accel_gyro` (which is a fixed-gain *complementary*
    filter). Both use only the accelerometer and gyroscope; the difference
    is the fusion mechanism:

    - :func:`heading_from_accel_gyro`     -> complementary filter (fixed gain)
    - :func:`heading_from_accel_gyro_ekf` -> Extended Kalman Filter
      (:class:`HeadingBiasEKF`, dynamic gain + gyro-bias estimation)

    Pipeline
    --------
    1. Build a magnetometer-free attitude with
       :func:`complementary_filter_attitude` (gyro predict + accelerometer
       gravity correct). This gives a tilt-compensated rotation matrix at
       every sample.
    2. From that attitude, derive:
         - ``omega_world_z`` : the world-frame yaw rate (gyro projected
           through the gravity-aligned rotation matrix) -- the EKF
           *prediction* input.
         - ``psi_meas``      : a heading reference, from projecting
           ``forward_axis`` through the same attitude and low-pass
           filtering -- the EKF *measurement* input.
    3. Run :class:`HeadingBiasEKF`, which fuses the two and estimates the
       gyro bias, returning a drift-reduced heading.

    Returns
    -------
    heading : (N,) array
        Unwrapped EKF heading (rad), zeroed at the first sample.
    q : (N, 4) array
        The accel+gyro attitude quaternions (same as
        :func:`complementary_filter_attitude`), for reuse elsewhere.
    """
    from scipy.signal import butter, filtfilt

    # 1. Magnetometer-free attitude.
    q = complementary_filter_attitude(gyro_xyz, accel_total_xyz, fs, gain=gain)
    Rm = quat_to_R(q)

    # 2a. World-frame yaw rate (prediction input).
    omega_world = np.einsum("nij,nj->ni", Rm, gyro_xyz)
    omega_world_z = omega_world[:, 2]

    # 2b. Accelerometer-derived heading reference (measurement input).
    world_fwd = np.einsum("nij,j->ni", Rm, forward_axis.astype(float))
    psi_meas = np.unwrap(np.arctan2(world_fwd[:, 1], world_fwd[:, 0]))
    psi_meas -= psi_meas[0]
    if lowpass_hz > 0 and len(psi_meas) > 30:
        b, a = butter(2, lowpass_hz / (fs / 2), btype="low")
        psi_meas = filtfilt(b, a, psi_meas)
        psi_meas -= psi_meas[0]

    # 3. Run the EKF.
    ekf = HeadingBiasEKF(q_psi=q_psi, q_bias=q_bias, R=R)
    heading = ekf.run(omega_world_z, psi_meas, fs)
    heading -= heading[0]
    return heading, q
