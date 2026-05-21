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
