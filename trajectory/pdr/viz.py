"""Plotting helpers (matplotlib only, no interactive backends).

The map overlay uses ``contextily`` to fetch tiles from OpenStreetMap-
based providers. It needs network access at run-time when first called
on a given area; the tiles get cached locally afterwards. For purely
offline use, drop the ``with_basemap=True`` argument.
"""
from __future__ import annotations

from typing import Optional, Sequence
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .pdr import PDRResult


# ---------------------------------------------------------------------------
# 1. Step-detection diagnostics
# ---------------------------------------------------------------------------

def plot_step_detection(
    seconds_elapsed: np.ndarray,
    accel_xyz: np.ndarray,
    steps,
    title: str = "Step detection",
    annotations: Optional[dict[str, float]] = None,
    ax: Optional[Axes] = None,
) -> Figure:
    """Plot acceleration magnitude with detected step peaks marked."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 3.5))
    else:
        fig = ax.figure
    mag = np.linalg.norm(accel_xyz, axis=1)
    ax.plot(seconds_elapsed, mag, linewidth=0.7, color="#34495e", label="|a|")
    if steps:
        ts = [s.time for s in steps]
        amps = [mag[s.index] if 0 <= s.index < len(mag) else 0 for s in steps]
        ax.scatter(ts, amps, c="#e74c3c", s=20, zorder=5,
                   label=f"{len(steps)} steps")
    if annotations:
        _shade_annotations(ax, annotations, seconds_elapsed[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|a| (m/s²)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _shade_annotations(ax: Axes, annotations: dict[str, float], t_end: float) -> None:
    """Shade activity periods between annotation timestamps."""
    palette = {
        "walking": "#2ecc71", "Walking": "#2ecc71",
        "sitting": "#3498db", "Sitting": "#3498db",
        "standing": "#f1c40f", "Standing": "#f1c40f",
        "upstairs": "#9b59b6", "Upstairs": "#9b59b6",
        "downstairs": "#e67e22", "Downstairs": "#e67e22",
    }
    sorted_anns = sorted(annotations.items(), key=lambda kv: kv[1])
    for i, (text, start) in enumerate(sorted_anns):
        end = sorted_anns[i + 1][1] if i + 1 < len(sorted_anns) else t_end
        ax.axvspan(start, end, alpha=0.12, color=palette.get(text, "#95a5a6"))
        ax.axvline(start, color="red", linestyle="--", alpha=0.4, linewidth=0.8)


# ---------------------------------------------------------------------------
# 2. Heading comparison
# ---------------------------------------------------------------------------

def plot_heading_sources(
    seconds_elapsed: np.ndarray,
    *,
    gyro_only: Optional[np.ndarray] = None,
    compass: Optional[np.ndarray] = None,
    fused: Optional[np.ndarray] = None,
    sensor_fusion: Optional[np.ndarray] = None,
    title: str = "Heading comparison",
    ax: Optional[Axes] = None,
) -> Figure:
    """Overlay multiple heading time-series in degrees, unwrapped for clarity."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 3.5))
    else:
        fig = ax.figure

    sources = [
        ("gyro integrated", gyro_only,        "#3498db"),
        ("compass",         compass,          "#e67e22"),
        ("EKF fused",       fused,            "#27ae60"),
        ("OS / Madgwick",   sensor_fusion,    "#9b59b6"),
    ]
    for label, arr, color in sources:
        if arr is None:
            continue
        ax.plot(seconds_elapsed, np.rad2deg(np.unwrap(arr)),
                linewidth=1.0, alpha=0.85, color=color, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heading (deg, unwrapped)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Trajectory plots
# ---------------------------------------------------------------------------

def plot_trajectory(
    result: PDRResult,
    *,
    label: str = "trajectory",
    color: str = "#2c3e50",
    ax: Optional[Axes] = None,
    show_steps: bool = True,
) -> Figure:
    """Plot a single 2D PDR trajectory with start/end markers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure
    xy = result.xy
    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2, label=label, alpha=0.85)
    if show_steps:
        ax.scatter(xy[1:, 0], xy[1:, 1], color=color, s=12, alpha=0.5, zorder=3)
    ax.scatter(xy[0, 0], xy[0, 1], color="#27ae60", s=80, marker="o", zorder=5,
               edgecolors="white", linewidth=1.5, label="start")
    ax.scatter(xy[-1, 0], xy[-1, 1], color="#e74c3c", s=80, marker="X", zorder=5,
               edgecolors="white", linewidth=1.5, label="end")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_trajectories(
    results: dict[str, PDRResult],
    title: str = "PDR trajectories",
    ax: Optional[Axes] = None,
) -> Figure:
    """Overlay multiple trajectories sharing a start point.

    Parameters
    ----------
    results : dict
        ``{label: PDRResult}``. Each trajectory is drawn in a different colour.
    """
    palette = ["#2c3e50", "#e67e22", "#16a085", "#8e44ad", "#c0392b"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    for i, (label, res) in enumerate(results.items()):
        c = palette[i % len(palette)]
        xy = res.xy
        ax.plot(xy[:, 0], xy[:, 1], color=c, linewidth=2,
                label=f"{label} ({res.n_steps} steps, {res.total_distance:.1f} m)",
                alpha=0.85)
        ax.scatter(xy[-1, 0], xy[-1, 1], color=c, s=60, marker="X",
                   edgecolors="white", linewidth=1.2, zorder=5)
    # Common start
    first = next(iter(results.values()))
    ax.scatter(first.xy[0, 0], first.xy[0, 1], color="#27ae60", s=90,
               marker="o", edgecolors="white", linewidth=1.5,
               zorder=6, label="start")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Stairs (3D-ish: time vs altitude + step count)
# ---------------------------------------------------------------------------

def plot_altitude(
    seconds_elapsed: np.ndarray,
    altitude_m: np.ndarray,
    *,
    annotations: Optional[dict[str, float]] = None,
    title: str = "Barometric altitude",
    ax: Optional[Axes] = None,
) -> Figure:
    """Plot altitude over time with optional activity shading."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 3.5))
    else:
        fig = ax.figure
    ax.plot(seconds_elapsed, altitude_m, linewidth=1.2, color="#8e44ad")
    if annotations:
        _shade_annotations(ax, annotations, seconds_elapsed[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m, relative)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Map overlay
# ---------------------------------------------------------------------------

def _local_xy_to_lonlat(
    xy: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert local East-North metres to lon/lat using a flat-Earth approximation.

    For trajectories under a few kilometres the curvature error is well
    below GPS noise. For city-scale work, switch to ``pyproj``.
    """
    R_earth = 6_378_137.0  # WGS-84 equatorial radius, metres
    lat = origin_lat + np.degrees(xy[:, 1] / R_earth)
    lon = origin_lon + np.degrees(xy[:, 0] / (R_earth * np.cos(np.radians(origin_lat))))
    return lon, lat


def plot_trajectory_on_map(
    result: PDRResult,
    location_df,
    *,
    label: str = "PDR",
    title: str = "PDR trajectory on map",
    with_basemap: bool = True,
    zoom: Optional[int] = None,
    ax: Optional[Axes] = None,
) -> Figure:
    """Anchor a PDR trajectory to a GPS fix and plot it over an OpenStreetMap basemap.

    Parameters
    ----------
    result : PDRResult
        From ``compute_trajectory``. Local East-North metres.
    location_df : pandas.DataFrame
        Sensor Logger's Location.csv. Must have ``latitude``, ``longitude``,
        ``seconds_elapsed`` columns. We use the first fix as the anchor —
        if you want a different anchor (e.g. median of first 5 fixes for
        stability), pre-process the DataFrame before passing it in.
    with_basemap : bool
        If True, fetch OpenStreetMap tiles via ``contextily``. Requires
        network access on first call. Set False for a plain plot.
    zoom : int, optional
        Tile zoom level (10-19). Auto-selected if None.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    GPS indoors is unreliable — the first fix may be off by 10-50 m or
    point at a neighbouring building. The trajectory *shape* is what
    matters; treat the absolute placement on the map as approximate.
    """
    if location_df is None or location_df.empty:
        raise ValueError("Location DataFrame is empty — cannot anchor trajectory.")
    needed = {"latitude", "longitude"}
    if not needed.issubset(location_df.columns):
        raise ValueError(f"Location DataFrame missing columns: {needed}")

    origin_lat = float(location_df["latitude"].iloc[0])
    origin_lon = float(location_df["longitude"].iloc[0])

    # PDR trajectory in lon/lat
    lon, lat = _local_xy_to_lonlat(result.xy, origin_lat, origin_lon)
    # GPS trace as well, for context/comparison
    gps_lon = location_df["longitude"].to_numpy()
    gps_lat = location_df["latitude"].to_numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
    else:
        fig = ax.figure

    if with_basemap:
        # Use Web Mercator for tile alignment.
        try:
            import contextily as cx
            from pyproj import Transformer
            tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            x_pdr, y_pdr = tf.transform(lon, lat)
            x_gps, y_gps = tf.transform(gps_lon, gps_lat)
            ax.plot(x_pdr, y_pdr, color="#c0392b", linewidth=2.5,
                    label=f"{label} ({result.total_distance:.1f} m)", zorder=4)
            ax.scatter(x_pdr[0], y_pdr[0], color="#27ae60", s=80, marker="o",
                       edgecolors="white", linewidth=1.5, zorder=5, label="start")
            ax.scatter(x_pdr[-1], y_pdr[-1], color="#c0392b", s=80, marker="X",
                       edgecolors="white", linewidth=1.5, zorder=5, label="end")
            ax.scatter(x_gps, y_gps, color="#2980b9", s=15, alpha=0.5,
                       zorder=3, label=f"GPS fixes ({len(gps_lon)})")
            # Pad bounds a bit so the basemap has some margin around the path.
            pad = max(40.0, 0.1 * (np.ptp(x_pdr) + np.ptp(y_pdr)))
            xlim = (min(x_pdr.min(), x_gps.min()) - pad,
                    max(x_pdr.max(), x_gps.max()) + pad)
            ylim = (min(y_pdr.min(), y_gps.min()) - pad,
                    max(y_pdr.max(), y_gps.max()) + pad)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_aspect("equal", adjustable="box")
            try:
                cx.add_basemap(ax, crs="EPSG:3857",
                               source=cx.providers.OpenStreetMap.Mapnik,
                               zoom=zoom)
            except Exception as e:
                warnings.warn(f"Could not fetch basemap tiles ({e}); "
                              "falling back to plain plot.")
            ax.set_xlabel("Web Mercator X (m)")
            ax.set_ylabel("Web Mercator Y (m)")
        except ImportError:
            warnings.warn("contextily and/or pyproj not installed; falling back to lon/lat.")
            with_basemap = False

    if not with_basemap:
        ax.plot(lon, lat, color="#c0392b", linewidth=2.5,
                label=f"{label} ({result.total_distance:.1f} m)")
        ax.scatter(gps_lon, gps_lat, color="#2980b9", s=15, alpha=0.6,
                   label=f"GPS fixes ({len(gps_lon)})")
        ax.scatter(lon[0], lat[0], color="#27ae60", s=80, marker="o",
                   edgecolors="white", linewidth=1.5, label="start")
        ax.scatter(lon[-1], lat[-1], color="#c0392b", s=80, marker="X",
                   edgecolors="white", linewidth=1.5, label="end")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_aspect("equal", adjustable="box")

    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
