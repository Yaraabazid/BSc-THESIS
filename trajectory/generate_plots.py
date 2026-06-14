"""Generate every comparison plot needed for the paper, for every recording.

Runs the full PDR pipeline (see ``pdr/pipeline.py``) on each recording folder
listed in ``RECORDINGS`` and saves, per recording, into
``trajectory/output/<name>/``:

    step_detection.png/.pdf       -- watch + phone step detection
    heading_comparison.png/.pdf   -- all 4 heading sources overlaid
    trajectory_comparison.png/.pdf-- all 4 heading sources, best step source
    gps_vs_pdr.png/.pdf            -- GPS reference vs PDR (if GPS available)
    trajectory_and_altitude.png/.pdf -- 2D path + barometric altitude profile

It also writes two summary tables across all recordings and methods:

    output/summary.csv
    output/summary.md

Usage
-----
    cd trajectory
    python generate_report_plots.py

Edit ``RECORDINGS`` below to add/remove recordings (each must be a folder
under ``DATA_ROOT`` containing the standard Sensor Logger CSV files).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pdr import run_pipeline, viz


# ── Configuration ────────────────────────────────────────────────────────────
DATA_ROOT  = Path("../data/NU")
OUTPUT_ROOT = Path("output")

RECORDINGS = ["Walking", "Walking-4", "Upstairs", "Downstairs"]

FS = 60.0

# Human-readable labels for each heading source, used in plot legends.
HEADING_LABELS = {
    "quat":       "Orientation quaternion",
    "accel_gyro": "Accel + gyro only (no magnetometer)",
    "gyro":       "Gyro only (world-frame)",
    "ekf":        "EKF (gyro + magnetometer)",
}


def best_step_source(res) -> str:
    """Prefer watch steps (cleaner arm-swing peaks) if available."""
    return "watch" if res.watch_steps else "phone"


def make_step_detection_plot(res, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    annotations = res.rec.annotation_dict()

    p_xyz = res.phone.accel_total[["x", "y", "z"]].to_numpy()
    viz.plot_step_detection(res.t_grid, p_xyz, res.phone_steps,
                             title=f"{res.name} — phone step detection",
                             annotations=annotations, ax=axes[0])

    if res.watch_steps:
        w_xyz = res.watch.accel_total[["x", "y", "z"]].to_numpy()
        viz.plot_step_detection(res.t_grid, w_xyz, res.watch_steps,
                                 title=f"{res.name} — watch step detection",
                                 annotations=annotations, ax=axes[1])
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    _save(fig, out_dir / "step_detection")


def make_heading_comparison_plot(res, out_dir: Path) -> None:
    fig = viz.plot_heading_sources(
        res.t_grid,
        gyro_only=res.headings.get("gyro"),
        fused=res.headings.get("ekf"),
        sensor_fusion=res.headings.get("quat"),
        accel_gyro=res.headings.get("accel_gyro"),
        title=f"{res.name} — heading sources "
              f"(forward axis = {res.fwd_axis_name})",
    )
    _save(fig, out_dir / "heading_comparison")


def make_trajectory_comparison_plot(res, out_dir: Path) -> None:
    src = best_step_source(res)
    subset = {
        HEADING_LABELS[h]: res.trajectories[f"{src} steps + {h}"]
        for h in res.headings
        if f"{src} steps + {h}" in res.trajectories
    }
    fig = viz.plot_trajectories(
        subset,
        title=f"{res.name} — PDR trajectories ({src} steps)",
    )
    _save(fig, out_dir / "trajectory_comparison")


def make_gps_plot(res, out_dir: Path) -> None:
    loc = res.rec.phone.location
    if loc is None or loc.empty:
        return
    src = best_step_source(res)
    subset = {
        HEADING_LABELS[h]: res.trajectories[f"{src} steps + {h}"]
        for h in res.headings
        if f"{src} steps + {h}" in res.trajectories
    }
    best_key = HEADING_LABELS["accel_gyro"]
    fig = viz.plot_gps_vs_pdr(
        subset, loc, best_key=best_key,
        title=f"{res.name} — GPS reference vs PDR ({src} steps)",
    )
    _save(fig, out_dir / "gps_vs_pdr")


def make_altitude_plot(res, out_dir: Path) -> None:
    if res.altitude_phone is None and res.altitude_watch is None:
        return
    src = best_step_source(res)
    subset = {
        HEADING_LABELS[h]: res.trajectories[f"{src} steps + {h}"]
        for h in res.headings
        if f"{src} steps + {h}" in res.trajectories
    }
    fig = viz.plot_trajectory_and_altitude(
        subset, res.t_grid,
        altitude_m=res.altitude_phone,
        altitude_watch_m=res.altitude_watch,
        title=f"{res.name} — trajectory and barometric altitude ({src} steps)",
    )
    _save(fig, out_dir / "trajectory_and_altitude")


def _save(fig, path_no_ext: Path) -> None:
    fig.savefig(path_no_ext.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path_no_ext.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_ROOT.mkdir(exist_ok=True)
    summary_rows = []

    for name in RECORDINGS:
        data_dir = DATA_ROOT / name
        if not data_dir.exists():
            print(f"skip {name}: {data_dir} not found")
            continue

        print(f"=== {name} ===")
        res = run_pipeline(data_dir, fs=FS, verbose=False)
        for w in res.warnings_raised:
            print(f"  note: {w}")

        out_dir = OUTPUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)

        make_step_detection_plot(res, out_dir)
        make_heading_comparison_plot(res, out_dir)
        make_trajectory_comparison_plot(res, out_dir)
        make_gps_plot(res, out_dir)
        make_altitude_plot(res, out_dir)
        print(f"  plots written to {out_dir}/")

        # Summary rows for this recording
        for key, traj in res.trajectories.items():
            step_src, _, h_label = key.partition(" steps + ")
            summary_rows.append({
                "recording":   res.name,
                "step_source": step_src,
                "heading":     HEADING_LABELS.get(h_label, h_label),
                "n_steps":     traj.n_steps,
                "total_distance_m": round(traj.total_distance, 2),
                "closure_error_m":  round(res.closure_errors[key], 2),
                "total_turn_deg":   round(float(np.degrees(res.headings[h_label][-1])), 1),
            })

    if not summary_rows:
        print("No recordings processed.")
        return

    df = pd.DataFrame(summary_rows)
    df.to_csv(OUTPUT_ROOT / "summary.csv", index=False)

    with open(OUTPUT_ROOT / "summary.md", "w") as f:
        for name in df["recording"].unique():
            sub = df[df["recording"] == name].drop(columns="recording")
            f.write(f"### {name}\n\n")
            f.write(sub.to_markdown(index=False))
            f.write("\n\n")

    print(f"\nSummary written to {OUTPUT_ROOT/'summary.csv'} "
          f"and {OUTPUT_ROOT/'summary.md'}")


if __name__ == "__main__":
    main()