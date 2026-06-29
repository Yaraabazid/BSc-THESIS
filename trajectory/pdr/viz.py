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
# Global style: larger, readable fonts for all figures (these are sized for
# inclusion in a thesis where figures are often scaled down, so they need to
# stay legible). Applied at import time; individual plots can still override.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size":        14,
    "axes.titlesize":   17,
    "axes.labelsize":   15,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "legend.fontsize":  12,
    "figure.titlesize": 18,
    "axes.titleweight": "bold",
    "lines.linewidth":  1.8,
})


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
    accel_gyro: Optional[np.ndarray] = None,
    sources_dict: Optional[dict[str, np.ndarray]] = None,
    title: str = "Heading comparison",
    ax: Optional[Axes] = None,
) -> Figure:
    """Overlay multiple heading time-series in degrees, unwrapped for clarity.

    If ``sources_dict`` ({label: (N,) array}) is given, it takes precedence
    and every entry is plotted with its own label -- use this to show an
    arbitrary set of methods with explicit, filter-type-aware labels. The
    individual keyword arguments are kept for backward compatibility.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))
    else:
        fig = ax.figure

    palette = ["#3498db", "#c0392b", "#16a085", "#9b59b6", "#27ae60",
               "#e67e22", "#2c3e50"]
    if sources_dict is not None:
        sources = [(label, arr, palette[i % len(palette)])
                   for i, (label, arr) in enumerate(sources_dict.items())]
    else:
        sources = [
            ("gyro integrated (world-frame)", gyro_only,    "#3498db"),
            ("compass",                       compass,      "#e67e22"),
            ("EKF (gyro + magnetometer)",     fused,        "#27ae60"),
            ("orientation quaternion",        sensor_fusion,"#9b59b6"),
            ("accel + gyro only (no mag.)",   accel_gyro,   "#c0392b"),
        ]
    for label, arr, color in sources:
        if arr is None:
            continue
        ax.plot(seconds_elapsed, np.rad2deg(np.unwrap(arr)),
                linewidth=2.0, alpha=0.9, color=color, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heading (deg, unwrapped)")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=True, fontsize=12)
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
    extra_paths: Optional[dict[str, np.ndarray]] = None,
) -> Figure:
    """Overlay multiple trajectories sharing a start point.

    Parameters
    ----------
    results : dict
        ``{label: PDRResult}``. Each trajectory is drawn in a different colour.
    extra_paths : dict, optional
        ``{label: (M, 2) array}`` -- additional paths that are not
        :class:`PDRResult` (e.g. MobilePoser's horizontal translation).
        Drawn as dashed lines, anchored at the same origin as the PDR
        trajectories. Distance is not shown in the legend for these (no
        per-step accounting).
    """
    palette = ["#2c3e50", "#e67e22", "#16a085", "#8e44ad", "#c0392b"]
    extra_palette = ["#d35400", "#34495e", "#1abc9c", "#9b59b6"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 12))
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
    if extra_paths:
        for i, (label, xy) in enumerate(extra_paths.items()):
            c = extra_palette[i % len(extra_palette)]
            dist = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
            ax.plot(xy[:, 0], xy[:, 1], color=c, linewidth=2, linestyle="--",
                    label=f"{label} ({dist:.1f} m)", alpha=0.9)
            ax.scatter(xy[-1, 0], xy[-1, 1], color=c, s=60, marker="X",
                       edgecolors="white", linewidth=1.2, zorder=5)
    # Common start
    if results:
        first = next(iter(results.values())).xy[0]
    else:
        first = next(iter(extra_paths.values()))[0]
    ax.scatter(first[0], first[1], color="#27ae60", s=90,
               marker="o", edgecolors="white", linewidth=1.5,
               zorder=6, label="start")
    ax.set_ylabel("North (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    # Place the legend well below the plot so it never covers the data or the
    # x-axis label. The offset is generous because equal-aspect trajectory
    # axes can be short, which would otherwise bring the legend up too high.
    ncol = 1 if len(results) <= 3 else 2
    ax.set_xlabel("East (m)", labelpad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=ncol, frameon=True, fontsize=11)
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


# ---------------------------------------------------------------------------
# 6. GPS vs PDR side-by-side comparison
# ---------------------------------------------------------------------------

def plot_gps_vs_pdr(
    trajectories: dict[str, "PDRResult"],
    location_df,
    *,
    best_key: Optional[str] = None,
    title: str = "GPS reference vs PDR estimate",
    extra_paths: Optional[dict[str, np.ndarray]] = None,
) -> Figure:
    """Side-by-side GPS trace and PDR trajectories in the same East/North frame.

    The first GPS fix is the shared origin (0, 0).  GPS dots are sized by
    ``horizontalAccuracy`` so fix quality is immediately visible.

    Parameters
    ----------
    trajectories : dict
        ``{label: PDRResult}`` — all variants to overlay on the right panel.
    location_df : DataFrame
        Sensor Logger ``Location.csv`` with ``latitude``, ``longitude``,
        and optionally ``horizontalAccuracy``.
    best_key : str, optional
        Label of the trajectory to highlight in bold.  Defaults to the
        first entry.
    title : str
        Figure suptitle.
    extra_paths : dict, optional
        ``{label: (M, 2) array}`` — additional paths without a
        :class:`PDRResult` (e.g. MobilePoser's horizontal translation),
        drawn as dashed lines on the right panel.

    Returns
    -------
    Figure
    """
    if location_df is None or location_df.empty:
        raise ValueError("No GPS data.")

    lat0 = float(location_df["latitude"].iloc[0])
    lon0 = float(location_df["longitude"].iloc[0])
    R_earth = 6_371_000.0
    gps_n = np.radians(location_df["latitude"].to_numpy()  - lat0) * R_earth
    gps_e = np.radians(location_df["longitude"].to_numpy() - lon0) * R_earth * np.cos(np.radians(lat0))
    acc   = location_df["horizontalAccuracy"].to_numpy() if "horizontalAccuracy" in location_df.columns else None

    if best_key is None:
        best_key = next(iter(trajectories))

    palette = ["#e67e22", "#2980b9", "#7f8c8d", "#16a085", "#c0392b"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # ── Left: GPS ─────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(gps_e, gps_n, color="#2980b9", linewidth=1, linestyle="--", alpha=0.4)
    dot_s = (acc / acc.min() * 30) if acc is not None else 40
    ax.scatter(gps_e, gps_n, c="#2980b9", s=dot_s, alpha=0.85, zorder=4)
    ax.scatter(gps_e[0],  gps_n[0],  color="#27ae60", s=100, zorder=6, label="start")
    ax.scatter(gps_e[-1], gps_n[-1], color="#e74c3c", s=100, marker="X", zorder=6, label="end")
    acc_label = (f"  (accuracy {acc.min():.0f}–{acc.max():.0f} m, {len(location_df)} fixes)"
                 if acc is not None else f"  ({len(location_df)} fixes)")
    ax.set_title("GPS trace" + acc_label)
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_aspect("equal"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── Right: PDR ────────────────────────────────────────────────────────────
    ax2 = axes[1]
    for (label, res), col in zip(trajectories.items(), palette):
        lw    = 2.5 if label == best_key else 1.0
        alpha = 1.0 if label == best_key else 0.4
        ax2.plot(res.xy[:,0], res.xy[:,1], color=col, linewidth=lw,
                 alpha=alpha, label=label.replace("\n", " "))
    extra_palette = ["#d35400", "#34495e", "#1abc9c", "#9b59b6"]
    if extra_paths:
        for i, (label, xy) in enumerate(extra_paths.items()):
            c = extra_palette[i % len(extra_palette)]
            ax2.plot(xy[:, 0], xy[:, 1], color=c, linewidth=1.5,
                     linestyle="--", alpha=0.9, label=label)
    best_xy = trajectories[best_key].xy
    ax2.scatter(0, 0, color="#27ae60", s=100, zorder=6)
    ax2.scatter(best_xy[-1,0], best_xy[-1,1], color="#e74c3c", s=100, marker="X", zorder=6)
    ax2.set_title(f"PDR  (bold = {best_key.replace(chr(10), ' ')})")
    ax2.set_xlabel("East (m)"); ax2.set_ylabel("North (m)")
    ax2.set_aspect("equal")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
               ncol=2, fontsize=10)
    ax2.grid(alpha=0.3)

    # Shared axis limits
    all_x = [gps_e] + [r.xy[:,0] for r in trajectories.values()]
    all_y = [gps_n] + [r.xy[:,1] for r in trajectories.values()]
    if extra_paths:
        all_x += [xy[:, 0] for xy in extra_paths.values()]
        all_y += [xy[:, 1] for xy in extra_paths.values()]
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    span  = max(all_x.max()-all_x.min(), all_y.max()-all_y.min())
    cx, cy = (all_x.max()+all_x.min())/2, (all_y.max()+all_y.min())/2
    half  = span/2 + span*0.15 + 2
    for ax in axes:
        ax.set_xlim(cx-half, cx+half); ax.set_ylim(cy-half, cy+half)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Trajectory + altitude (for stairs recordings)
# ---------------------------------------------------------------------------

def plot_trajectory_and_altitude(
    trajectories: dict[str, "PDRResult"],
    seconds_elapsed: np.ndarray,
    altitude_m: Optional[np.ndarray] = None,
    altitude_watch_m: Optional[np.ndarray] = None,
    *,
    title: str = "Trajectory and altitude",
    extra_paths: Optional[dict[str, np.ndarray]] = None,
    mobileposer_altitude_m: Optional[np.ndarray] = None,
) -> Figure:
    """Side-by-side 2D trajectory and barometric altitude profile.

    Intended for stairs recordings, where the 2D top-down trajectory alone
    does not show the floor change. The left panel overlays every trajectory
    in ``trajectories``; the right panel plots phone (and watch, if given)
    relative altitude against time.

    Parameters
    ----------
    trajectories : dict
        ``{label: PDRResult}`` -- same as :func:`plot_trajectories`.
    seconds_elapsed : (N,) array
        Time grid matching ``altitude_m`` / ``altitude_watch_m`` /
        ``mobileposer_altitude_m``.
    altitude_m : (N,) array, optional
        Phone relative altitude in metres (from ``Barometer.csv``).
    altitude_watch_m : (N,) array, optional
        Watch relative altitude in metres, if available.
    title : str
        Figure suptitle.
    extra_paths : dict, optional
        ``{label: (M, 2) array}`` -- additional horizontal paths (e.g.
        MobilePoser's horizontal translation), drawn as dashed lines on the
        left panel.
    mobileposer_altitude_m : (N,) array, optional
        MobilePoser's height (vertical translation) profile, resampled onto
        ``seconds_elapsed`` and zeroed at the first sample -- plotted
        alongside the barometer traces on the right panel for direct
        comparison.

    Returns
    -------
    Figure
    """
    palette = ["#2c3e50", "#e67e22", "#16a085", "#8e44ad", "#c0392b"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    for i, (label, res) in enumerate(trajectories.items()):
        c = palette[i % len(palette)]
        xy = res.xy
        ax1.plot(xy[:, 0], xy[:, 1], color=c, linewidth=2,
                 label=label, alpha=0.85)
        ax1.scatter(xy[-1, 0], xy[-1, 1], color=c, s=50, marker="X",
                    edgecolors="white", linewidth=1, zorder=5)
    if extra_paths:
        extra_palette = ["#d35400", "#34495e", "#1abc9c", "#9b59b6"]
        for i, (label, xy) in enumerate(extra_paths.items()):
            c = extra_palette[i % len(extra_palette)]
            ax1.plot(xy[:, 0], xy[:, 1], color=c, linewidth=2,
                     linestyle="--", label=label, alpha=0.9)
            ax1.scatter(xy[-1, 0], xy[-1, 1], color=c, s=50, marker="X",
                        edgecolors="white", linewidth=1, zorder=5)
    if trajectories:
        first = next(iter(trajectories.values())).xy[0]
    elif extra_paths:
        first = next(iter(extra_paths.values()))[0]
    else:
        first = None
    if first is not None:
        ax1.scatter(first[0], first[1], color="#27ae60", s=80,
                     marker="o", edgecolors="white", linewidth=1.5,
                     zorder=6, label="start")
    ax1.set_xlabel("East (m)")
    ax1.set_ylabel("North (m)")
    ax1.set_title("2D trajectory (top-down)")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
               ncol=2, fontsize=10)

    if altitude_m is not None:
        ax2.plot(seconds_elapsed, altitude_m, color="#8e44ad",
                 linewidth=1.5, label="phone (barometer)")
    if altitude_watch_m is not None:
        ax2.plot(seconds_elapsed, altitude_watch_m, color="#16a085",
                 linewidth=1.2, alpha=0.8, label="watch (barometer)")
    if mobileposer_altitude_m is not None:
        ax2.plot(seconds_elapsed, mobileposer_altitude_m, color="#d35400",
                 linewidth=1.5, linestyle="--", alpha=0.9, label="MobilePoser (height)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Relative altitude (m)")
    ax2.set_title("Altitude")
    ax2.grid(alpha=0.3)
    if altitude_m is not None or altitude_watch_m is not None or mobileposer_altitude_m is not None:
        ax2.legend(loc="best", fontsize=12)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. Step timing comparison (raster plot)
# ---------------------------------------------------------------------------

def plot_step_timing_comparison(
    step_times: dict[str, np.ndarray],
    title: str = "Step timing comparison",
) -> Figure:
    """Raster-style comparison of step/contact event timestamps across sources.

    Each entry in ``step_times`` is drawn as a row of vertical tick marks at
    its event timestamps, so timing agreement (or disagreement) between
    independent step detectors is visible at a glance -- without depending
    on any of them sharing a coordinate frame or scale.

    Parameters
    ----------
    step_times : dict
        ``{label: (M,) array of seconds}`` -- e.g. watch/phone
        accelerometer-detected step times and MobilePoser foot-contact
        event times (see
        :func:`pdr.mobileposer.mobileposer_foot_contact_events`).
    title : str
        Figure title.

    Returns
    -------
    Figure
    """
    palette = ["#2c3e50", "#e67e22", "#16a085", "#8e44ad", "#c0392b"]
    labels = list(step_times.keys())
    fig, ax = plt.subplots(figsize=(13, 0.85 * len(labels) + 1.6))
    for i, label in enumerate(labels):
        times = step_times[label]
        c = palette[i % len(palette)]
        if len(times):
            ax.eventplot(times, lineoffsets=i, linelengths=0.8,
                         colors=c, linewidths=1.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"{l}  (n={len(step_times[l])})" for l in labels],
                       fontsize=13)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 9. MobilePoser pose snapshots (skeleton stick figures)
# ---------------------------------------------------------------------------

def plot_pose_skeleton_snapshots(
    joints: np.ndarray,
    target_hz: float,
    n_snapshots: int = 6,
    parents: Optional[Sequence[int]] = None,
    title: str = "MobilePoser pose snapshots",
    view: str = "front",
) -> Figure:
    """Grid of stick-figure pose snapshots from MobilePoser's joint output.

    This is a *qualitative* illustration of MobilePoser's pose estimate
    over time -- useful even when the translation/trajectory estimate
    isn't trustworthy, since pose accuracy and translation accuracy are
    largely independent in this model.

    Parameters
    ----------
    joints : (N, 24, 3) array
        MobilePoser's ``pred_joints`` output (``step0_output['joints']``),
        SMPL 24-joint convention, pelvis-rooted.
    target_hz : float
        MobilePoser's frame rate, used to label each snapshot with a
        timestamp.
    n_snapshots : int
        Number of evenly-spaced frames to show.
    parents : sequence of int, optional
        Parent index per joint. Defaults to
        :data:`pdr.mobileposer.SMPL_PARENTS` (the standard SMPL kinematic
        tree) -- pass explicitly if your joints use a different ordering.
    title : str
        Figure suptitle.
    view : str
        ``"front"`` plots (X, Y): left/right vs. up/down.
        ``"side"`` plots (Z, Y): forward/back vs. up/down.

    Returns
    -------
    Figure
    """
    if parents is None:
        from .mobileposer import SMPL_PARENTS
        parents = SMPL_PARENTS

    n = len(joints)
    n_snapshots = min(n_snapshots, n)
    idxs = np.linspace(0, n - 1, n_snapshots, dtype=int)

    ax_pair = (0, 1) if view == "front" else (2, 1)
    ax_labels = ("X (m)", "Y (m)") if view == "front" else ("Z (m)", "Y (m)")

    fig, axes = plt.subplots(1, n_snapshots, figsize=(2.3 * n_snapshots, 3.4),
                              sharey=True)
    if n_snapshots == 1:
        axes = [axes]

    for ax, fi in zip(axes, idxs):
        pts = joints[fi]  # (24, 3)
        for j, p in enumerate(parents):
            if p < 0:
                continue
            x = [pts[j, ax_pair[0]], pts[p, ax_pair[0]]]
            y = [pts[j, ax_pair[1]], pts[p, ax_pair[1]]]
            ax.plot(x, y, color="#2c3e50", linewidth=1.8,
                    marker="o", markersize=2.5,
                    markerfacecolor="#e74c3c", markeredgewidth=0)
        ax.set_title(f"t={fi/target_hz:.1f}s", fontsize=9)
        ax.set_xlabel(ax_labels[0], fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel(ax_labels[1], fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig
