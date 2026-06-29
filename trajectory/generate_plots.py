"""Generate every comparison plot needed for the paper, for every recording.

Runs the full PDR pipeline (see ``pdr/pipeline.py``) on each recording folder
listed in ``RECORDINGS`` and saves, per recording, into
``trajectory/output/<name>/``:

    step_detection.png/.pdf       -- watch + phone step detection
    heading_comparison.png/.pdf   -- all 4 heading sources overlaid
    trajectory_comparison.png/.pdf-- all 4 heading sources, best step source
    gps_vs_pdr.png/.pdf            -- GPS reference vs PDR (if GPS available)
    trajectory_and_altitude.png/.pdf -- 2D path + barometric altitude profile

If ``<recording>/processed/step0_output.npz`` exists (produced by
``poser-test-drive/mobileposer_runner.py``), two MobilePoser-based plots are
also written. Note MobilePoser's *translation* is deliberately NOT used as a
trajectory -- with our sparse 2-device combo it doesn't track the real path,
so it would be misleading on the trajectory plots. Instead MobilePoser's more
reliable pose/contact output is used:

    step_timing_comparison.png/.pdf -- watch/phone step times vs. MobilePoser
                                        foot-contact event times (independent
                                        of translation accuracy)
    pose_snapshots.png/.pdf          -- stick-figure pose snapshots over time
                                        (qualitative; uses pred_joints, not
                                        the translation)

For an animated body video (mesh or skeleton) beside the sensor signals, see
``poser-test-drive/render_pose_video.py``.

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
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pdr import run_pipeline, viz, mobileposer


# ── Configuration ────────────────────────────────────────────────────────────
DATA_ROOT  = Path("../data/NU")
OUTPUT_ROOT = Path("output")

RECORDINGS = ["Walking", "Walking-4", "Upstairs", "Downstairs"]

FS = 60.0

# Human-readable labels for each heading source, used in plot legends.
# Each label is explicit about which fusion method / filter it uses.
HEADING_LABELS = {
    "quat":           "Orientation quaternion (OS fusion)",
    "accel_gyro":     "Accel + gyro, complementary filter (no mag.)",
    "accel_gyro_ekf": "Accel + gyro, EKF (no mag.)",
    "gyro":           "Gyro only, integrated (no filter)",
    "ekf_mag":        "Gyro + magnetometer, EKF",
}


def best_step_source(res) -> str:
    """Prefer watch steps (cleaner arm-swing peaks) if available."""
    return "watch" if res.watch_steps else "phone"


def load_mobileposer(data_dir: Path):
    """Load step0_output.npz (if present).

    Returns the raw npz dict, or None if no file is found.

    Note: MobilePoser's *translation* is deliberately NOT used as a
    trajectory here -- with the sparse 2-device combo it doesn't track the
    real path (drifts horizontally, flat vertically). MobilePoser is used
    instead for its pose/contact output (step timing + pose video/snapshots),
    which are far more reliable than its translation.
    """
    npz_path = data_dir / "processed" / "step0_output.npz"
    return mobileposer.load_step0_output(npz_path)


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
    sources_dict = {
        HEADING_LABELS[h]: res.headings[h]
        for h in ["quat", "accel_gyro", "accel_gyro_ekf", "gyro", "ekf_mag"]
        if h in res.headings
    }
    fig = viz.plot_heading_sources(
        res.t_grid,
        sources_dict=sources_dict,
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
        title=f"{res.name} — trajectory and altitude ({src} steps)",
    )
    _save(fig, out_dir / "trajectory_and_altitude")


def make_step_timing_plot(res, out_dir: Path, mp: Optional[dict],
                           t_grid: np.ndarray) -> None:
    """Step timing comparison: watch/phone steps vs. MobilePoser foot contact.

    Doesn't depend on translation accuracy at all -- a useful cross-check
    even when the MobilePoser trajectory itself looks implausible.
    """
    if mp is None or "contact" not in mp:
        return
    t0 = t_grid[0]
    step_times = {}
    if res.watch_steps:
        step_times["Watch (accelerometer)"] = np.array(
            [s.time - t0 for s in res.watch_steps])
    step_times["Phone (accelerometer)"] = np.array(
        [s.time - t0 for s in res.phone_steps])

    target_hz = float(mp["target_hz"])
    events = mobileposer.mobileposer_foot_contact_events(
        mp["contact"], target_hz=target_hz)
    step_times["MobilePoser foot contact (ch. 0)"] = events["channel_0"]
    step_times["MobilePoser foot contact (ch. 1)"] = events["channel_1"]

    fig = viz.plot_step_timing_comparison(
        step_times,
        title=f"{res.name} — step timing: watch/phone vs. MobilePoser foot contact",
    )
    _save(fig, out_dir / "step_timing_comparison")


def make_pose_snapshot_plot(res, out_dir: Path, mp: Optional[dict]) -> None:
    """Qualitative stick-figure pose snapshots from MobilePoser's joints."""
    if mp is None or "joints" not in mp:
        return
    target_hz = float(mp["target_hz"])
    fig = viz.plot_pose_skeleton_snapshots(
        mp["joints"], target_hz, n_snapshots=6,
        title=f"{res.name} — MobilePoser pose snapshots",
    )
    _save(fig, out_dir / "pose_snapshots")


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

        mp = load_mobileposer(data_dir)
        if mp is not None:
            print("  MobilePoser output found (used for pose/step-timing, "
                  "not trajectory):")
            if "tran" in mp:
                print("   ", mobileposer.describe_translation(
                    mp["tran"], float(mp["target_hz"])).replace("\n", "\n    "))
            if "acc" in mp and "ori" in mp:
                print("   ", mobileposer.describe_inputs(
                    mp["acc"], mp["ori"], combo=str(mp.get("combo", "?"))
                ).replace("\n", "\n    "))
        else:
            print("  (no processed/step0_output.npz -- run "
                  "poser-test-drive/mobileposer_runner.py to add it)")

        out_dir = OUTPUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)

        make_step_detection_plot(res, out_dir)
        make_heading_comparison_plot(res, out_dir)
        make_trajectory_comparison_plot(res, out_dir)
        make_gps_plot(res, out_dir)
        make_altitude_plot(res, out_dir)
        make_step_timing_plot(res, out_dir, mp, res.t_grid)
        make_pose_snapshot_plot(res, out_dir, mp)
        print(f"  plots written to {out_dir}/")

        # Summary rows for this recording (PDR methods only -- MobilePoser
        # translation is intentionally excluded as it isn't a usable
        # trajectory with this sparse device combo).
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
            sub = sub.dropna(axis=1, how="all")
            sub = sub.fillna("-")
            f.write(f"### {name}\n\n")
            f.write(sub.to_markdown(index=False))
            f.write("\n\n")

    print(f"\nSummary written to {OUTPUT_ROOT/'summary.csv'} "
          f"and {OUTPUT_ROOT/'summary.md'}")


if __name__ == "__main__":
    main()