"""Trajectory engine: turn step events + heading into a 2D path.

Given:
    - a list of step events with times and lengths
    - a heading time-series ψ(t) (radians, CCW from East)
    - an initial position (x0, y0)

We walk forward step by step:

    For each step k at time t_k with length L_k:
        ψ_k = ψ(t_k)                    # heading at the moment of foot strike
        x_k = x_{k-1} + L_k * cos(ψ_k)
        y_k = y_{k-1} + L_k * sin(ψ_k)

The result is a polyline of waypoints, one per step. Between steps we
hold position constant (the user's centre of mass moves smoothly during
a step but the foot-strike-to-foot-strike model treats the whole step
as a discrete jump).

For comparison / fusion with the watch we run this twice and produce two
trajectories. A simple average of the two waypoints can yield a "fused"
trajectory; for indoor PDR this is mostly cosmetic — the bigger fusion
opportunities are using the watch's stronger heading (often less occluded
by a pocket) with the phone's clearer step impacts (closer to the centre
of mass).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .steps import StepEvent


@dataclass
class PDRConfig:
    """Knobs controlling a single PDR run."""
    fs: float = 60.0
    weinberg_k: float = 0.41
    min_peak_height: float = 1.2
    min_peak_distance_s: float = 0.3
    initial_heading: float = 0.0   # radians; only used as fallback
    initial_xy: tuple[float, float] = (0.0, 0.0)


@dataclass
class PDRResult:
    """Outputs of a PDR run."""
    times:    np.ndarray              # (n_steps,) seconds_elapsed at each foot strike
    xy:       np.ndarray              # (n_steps + 1, 2) trajectory in metres, includes start
    headings: np.ndarray              # (n_steps,) heading at each step (rad)
    lengths:  np.ndarray              # (n_steps,) step length per step (m)
    steps:    list[StepEvent] = field(default_factory=list)
    heading_full: Optional[np.ndarray] = None  # (N,) heading time-series used
    t_full:       Optional[np.ndarray] = None  # (N,) seconds_elapsed for heading_full

    @property
    def total_distance(self) -> float:
        return float(self.lengths.sum())

    @property
    def n_steps(self) -> int:
        return len(self.steps)


def compute_trajectory(
    steps: list[StepEvent],
    heading: np.ndarray,
    seconds_elapsed: np.ndarray,
    initial_xy: tuple[float, float] = (0.0, 0.0),
) -> PDRResult:
    """Build a 2D trajectory from step events and a heading time-series.

    Parameters
    ----------
    steps : list of StepEvent
        From ``detect_steps``.
    heading : (N,) array
        Heading in radians at every sample of the resampled stream.
    seconds_elapsed : (N,) array
        Timestamps for each sample of the heading array.
    initial_xy : (x0, y0)
        Starting position in metres. Default (0, 0).

    Returns
    -------
    PDRResult
    """
    n_steps = len(steps)
    xy = np.empty((n_steps + 1, 2), dtype=float)
    xy[0] = initial_xy
    times = np.empty(n_steps)
    headings = np.empty(n_steps)
    lengths = np.empty(n_steps)

    # Fast lookup: heading at any time, by index into seconds_elapsed.
    # The step's `index` field is already the sample index in the resampled
    # accel stream; if heading was computed on the same grid, we can use it
    # directly. If not, fall back to nearest-neighbor interpolation by time.
    same_grid = (len(heading) == len(seconds_elapsed))
    for k, st in enumerate(steps):
        if same_grid and 0 <= st.index < len(heading):
            psi = float(heading[st.index])
        else:
            # Nearest-time lookup — robust when the heading grid differs.
            j = int(np.argmin(np.abs(seconds_elapsed - st.time)))
            psi = float(heading[j])
        times[k] = st.time
        headings[k] = psi
        lengths[k] = st.length
        xy[k + 1, 0] = xy[k, 0] + st.length * np.cos(psi)
        xy[k + 1, 1] = xy[k, 1] + st.length * np.sin(psi)

    return PDRResult(
        times=times,
        xy=xy,
        headings=headings,
        lengths=lengths,
        steps=steps,
        heading_full=heading,
        t_full=seconds_elapsed,
    )


def fuse_trajectories(
    result_a: PDRResult,
    result_b: PDRResult,
    weight_a: float = 0.5,
) -> PDRResult:
    """Naively average two trajectories that have the same number of steps.

    This is the simplest possible fusion: aligned step-by-step weighted
    average. Falls back to whichever has more steps if they disagree.
    A real fused PDR system would run a single EKF with both devices'
    measurements; this is a useful sanity check for "do phone and watch
    agree on where I went?".
    """
    if result_a.n_steps != result_b.n_steps:
        # Different step counts — return whichever recorded more, with a
        # warning printed by the caller. We do *not* try to align by time
        # here because that gets expensive and is best done explicitly.
        return result_a if result_a.n_steps >= result_b.n_steps else result_b

    w = np.clip(weight_a, 0.0, 1.0)
    xy = w * result_a.xy + (1 - w) * result_b.xy
    headings = w * result_a.headings + (1 - w) * result_b.headings
    lengths = w * result_a.lengths + (1 - w) * result_b.lengths
    return PDRResult(
        times=result_a.times,
        xy=xy,
        headings=headings,
        lengths=lengths,
        steps=result_a.steps,  # reuse phone's step events
        heading_full=None,
        t_full=None,
    )
