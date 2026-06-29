"""Load MobilePoser output and extract a trajectory comparable to the PDR results.

MobilePoser (run via ``poser-test-drive/mobileposer_runner.py``) outputs a
global root translation ``tran`` of shape ``(N, 3)`` at 30 Hz, in SMPL/AMASS
world coordinates: ``tran[:, 1]`` is height (vertical, "up"), and
``tran[:, [0, 2]]`` is the horizontal plane.

This module:

1. Loads ``step0_output.npz``.
2. Extracts the horizontal trajectory and the vertical (height) profile.
3. Resamples both onto a PDR ``t_grid`` (60 Hz, ``seconds_elapsed``).
4. Anchors both at the origin / zero, matching the convention used by the
   PDR trajectories and the barometer altitude profiles.

Coordinate caveats
-------------------
MobilePoser's horizontal axes have **no shared heading reference** with the
PDR trajectories or GPS -- exactly like the different PDR heading methods
don't share a reference with each other. Comparing MobilePoser's horizontal
path shape (and closure error) against PDR is therefore a fair *shape*
comparison, but the absolute rotation/orientation of the path is arbitrary.

The vertical axis comparison against the barometer is more direct: both are
expected to track real-world "up is positive", so no rotation ambiguity
applies there -- but the *sign* convention has not been independently
verified for MobilePoser's output. :func:`describe_translation` prints the
height range so this can be checked against the barometer altitude change
(e.g. Upstairs should show a positive height gain of several metres).

Time alignment
---------------
Both ``t_grid`` (PDR) and MobilePoser's internal time axis
(``np.arange(N) / target_hz``) are assumed to start at the same physical
instant (each device's own recording start). See the note in
``mobileposer_runner.py`` for why this is a reasonable approximation. No
correction for the few-hundred-millisecond to ~2 s button-press offset is
applied.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def load_step0_output(npz_path: str | Path) -> Optional[dict]:
    """Load a ``step0_output.npz`` file, or return ``None`` if it doesn't exist.

    Parameters
    ----------
    npz_path : path-like
        Path to ``step0_output.npz`` (e.g.
        ``data/NU/Walking/processed/step0_output.npz``).

    Returns
    -------
    dict or None
        Keys: ``tran`` (N, 3), ``target_hz`` (scalar), plus ``acc``, ``ori``,
        ``joints``, ``pose``, ``contact``, ``combo``, ``mode`` if present.
        ``None`` if the file does not exist (MobilePoser not yet run for this
        recording).
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def describe_translation(tran: np.ndarray, target_hz: float = 30.0) -> str:
    """Human-readable summary of a MobilePoser translation array.

    Useful as a sanity check after running MobilePoser: horizontal distance
    should be in the right ballpark for the recording (e.g. ~25 m for the
    Walking loop), and the height range/final value should match the sign and
    rough magnitude of the barometer altitude change (e.g. ~+10 m for
    Upstairs, ~-13 m for Downstairs).
    """
    horiz = tran[:, [0, 2]]
    height = tran[:, 1]
    duration = len(tran) / target_hz
    total_dist = float(np.sum(np.linalg.norm(np.diff(horiz, axis=0), axis=1)))
    closure = float(np.linalg.norm(horiz[-1] - horiz[0]))
    return (
        f"MobilePoser translation: {len(tran)} frames @ {target_hz} Hz "
        f"({duration:.1f} s)\n"
        f"  horizontal: total distance {total_dist:.2f} m, "
        f"closure error {closure:.2f} m\n"
        f"  height: start {height[0]:+.2f} m, end {height[-1]:+.2f} m, "
        f"range [{height.min():+.2f}, {height.max():+.2f}] m"
    )


def mobileposer_horizontal_trajectory(
    tran: np.ndarray,
    t_grid: np.ndarray,
    target_hz: float = 30.0,
) -> np.ndarray:
    """Resample MobilePoser's horizontal trajectory onto a PDR time grid.

    Parameters
    ----------
    tran : (N, 3) array
        MobilePoser root translation, SMPL convention (``tran[:,1]`` = up).
    t_grid : (M,) array
        PDR ``seconds_elapsed`` time grid (e.g. 60 Hz).
    target_hz : float
        MobilePoser's frame rate (``step0_output['target_hz']``).

    Returns
    -------
    xy : (M, 2) array
        Horizontal trajectory (``tran[:, [0, 2]]``), resampled onto
        ``t_grid`` and anchored so the first sample is ``(0, 0)`` --
        matching the convention used by the PDR trajectories and GPS.
        Samples beyond MobilePoser's recorded duration hold the last value.
    """
    t_grid_rel = t_grid - t_grid[0]
    t_mp = np.arange(len(tran)) / target_hz

    horiz = tran[:, [0, 2]]
    x = np.interp(t_grid_rel, t_mp, horiz[:, 0])
    y = np.interp(t_grid_rel, t_mp, horiz[:, 1])
    xy = np.stack([x, y], axis=1)
    xy -= xy[0]
    return xy


def mobileposer_altitude(
    tran: np.ndarray,
    t_grid: np.ndarray,
    target_hz: float = 30.0,
) -> np.ndarray:
    """Resample MobilePoser's height (vertical) profile onto a PDR time grid.

    Parameters
    ----------
    tran : (N, 3) array
        MobilePoser root translation, SMPL convention (``tran[:,1]`` = up).
    t_grid : (M,) array
        PDR ``seconds_elapsed`` time grid.
    target_hz : float
        MobilePoser's frame rate.

    Returns
    -------
    altitude : (M,) array
        Height profile, resampled onto ``t_grid`` and zeroed at the first
        sample -- matching the convention used by ``pressure_to_altitude``
        and the barometer ``relativeAltitude`` columns (positive = up).
    """
    t_grid_rel = t_grid - t_grid[0]
    t_mp = np.arange(len(tran)) / target_hz
    h = np.interp(t_grid_rel, t_mp, tran[:, 1])
    h -= h[0]
    return h


# ---------------------------------------------------------------------------
# Input sanity check
# ---------------------------------------------------------------------------

def describe_inputs(acc: np.ndarray, ori: np.ndarray, combo: str = "lw_rp") -> str:
    """Sanity-check the IMU tensors that were actually fed into MobilePoser.

    Useful when the output translation looks implausible (e.g. near-zero
    magnitude, or a path in an unexpected direction): if the *active*
    device slots show near-zero acceleration variance, or rotation
    matrices with determinant far from 1.0, the problem is in how the
    input tensors were built (masking, scale, or time-alignment), not in
    the model itself.

    Parameters
    ----------
    acc : (N, 6, 3) array
        ``step0_output['acc']`` -- per-slot acceleration, m/s^2.
    ori : (N, 6, 3, 3) array
        ``step0_output['ori']`` -- per-slot rotation matrices.
    combo : str
        Combo string used (e.g. ``'lw_rp'``), just for the printed header.

    Returns
    -------
    str
        Multi-line human-readable summary, one line per device slot.
    """
    # Slots actually populated by mobileposer_runner.py's 'lw_rp' combo.
    known_active = {0: "left_wrist", 3: "right_pocket"}
    lines = [f"Input sanity check ({acc.shape[0]} frames, combo={combo}):"]
    for i in range(acc.shape[1]):
        a = acc[:, i, :]
        mag = np.linalg.norm(a, axis=1)
        dets = np.linalg.det(ori[:, i, :, :])
        role = known_active.get(i, "unused/masked")
        flag = ""
        if i in known_active and mag.std() < 0.05:
            flag = "  <-- WARNING: near-zero variance for an active slot, " \
                   "check time alignment / masking / scale"
        lines.append(
            f"  slot {i} ({role}): |acc| mean={mag.mean():6.3f} "
            f"std={mag.std():6.3f}  det(R) mean={dets.mean():.4f}{flag}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Foot-contact step timing
# ---------------------------------------------------------------------------

def mobileposer_foot_contact_events(
    contact: np.ndarray,
    target_hz: float = 30.0,
    threshold: float = 0.5,
    min_gap_s: float = 0.25,
) -> dict[str, np.ndarray]:
    """Extract discrete foot-strike timestamps from MobilePoser's contact output.

    MobilePoser's translation estimate is known to be less reliable than
    its pose/contact estimates, especially with sparse device combos like
    ours (2 of 6 possible slots). Foot-contact timing gives an independent
    way to validate MobilePoser's output against your step detector that
    doesn't depend on translation accuracy at all.

    Parameters
    ----------
    contact : (N, 2) array
        ``step0_output['contact']`` -- per-frame foot contact
        probability/score. Which column is left vs. right foot follows
        MobilePoser's own internal convention and has **not** been
        independently verified here -- treat the two columns as
        "channel 0" / "channel 1" unless you've confirmed the mapping
        (e.g. via the Unity visualisation in ``final-sensor.ipynb``).
    target_hz : float
        MobilePoser's frame rate.
    threshold : float
        Contact probability above which a frame counts as "foot down".
    min_gap_s : float
        Minimum time between consecutive detected strikes on the same
        channel -- merges contact frames belonging to one footfall into a
        single event.

    Returns
    -------
    dict
        ``{"channel_0": (M0,) seconds, "channel_1": (M1,) seconds,
        "all": sorted union of both}``, all relative to MobilePoser's own
        t=0 (its first frame) -- same convention as
        :func:`mobileposer_horizontal_trajectory` and
        :func:`mobileposer_altitude`.
    """
    min_gap_frames = max(1, int(round(min_gap_s * target_hz)))
    out: dict[str, np.ndarray] = {}
    all_events = []
    for ch in range(contact.shape[1]):
        is_down = contact[:, ch] > threshold
        rising = np.where(np.diff(is_down.astype(int)) == 1)[0] + 1
        if len(is_down) and is_down[0]:
            rising = np.concatenate([[0], rising])
        kept = []
        last = -min_gap_frames - 1
        for idx in rising:
            if idx - last >= min_gap_frames:
                kept.append(idx)
                last = idx
        times = np.array(kept, dtype=float) / target_hz
        out[f"channel_{ch}"] = times
        all_events.append(times)
    out["all"] = np.sort(np.concatenate(all_events)) if all_events else np.array([])
    return out


# ---------------------------------------------------------------------------
# Skeleton (pose snapshot) support
# ---------------------------------------------------------------------------

# Standard SMPL/AMASS 24-joint kinematic tree (parent index per joint,
# -1 = root). This is the convention used across SMPL-based tooling and
# should match MobilePoser's `pred_joints` ordering, since MobilePoser is
# trained on AMASS (SMPL-parameterised) data. Only the hierarchy (which
# joint connects to which) is used here for drawing skeleton bones --
# exact joint *names* are for reference only and not load-bearing for the
# plot itself.
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
]
