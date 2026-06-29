r"""Render MobilePoser pose output as a video + still frames, beside the sensors.

This is the qualitative "what is the body doing" companion to the PDR
trajectory work. MobilePoser's *translation* estimate is unreliable with our
sparse 2-device combo, so we do **not** use it as a trajectory here. Instead
this script visualises MobilePoser's *pose* output (which is far more
trustworthy than translation), animated next to the synchronised IMU signals,
to show that the model produces physically sensible body motion from the
phone+watch sensors.

For each recording it produces, in ``<recording>/processed/``:

    pose_video.mp4        -- animation of the body, in place (no translation),
                             with a synchronised sensor-signal panel and a
                             moving time cursor.
    pose_still_*.png      -- a handful of still frames pulled from the same
                             render, for use as figures in the paper.

Two render modes
----------------
- ``mesh``  (default): full SMPL body mesh, like MobilePoser's own renders.
  Uses MobilePoser's ``articulate.ParametricModel`` (the same code path that
  already works in your environment via ``visualize_sensor.py``) to turn
  ``pred_pose`` into mesh vertices per frame, then draws the mesh with
  matplotlib's 3D ``plot_trisurf``. No extra rendering libraries
  (pyrender / open3d / aitviewer) required -- only matplotlib + imageio,
  which you already have.
- ``skeleton`` (automatic fallback): if the SMPL mesh can't be built for any
  reason (missing body model, dependency issue), the script falls back to a
  3D stick figure from ``pred_joints`` -- no chumpy / SMPL involved at all.
  Still a real "body moving over time" video + stills.

Usage
-----
Edit ``CONFIG`` below (MobilePoser repo path + weights), then::

    python render_pose_video.py

By default it processes every recording under ``DATA_ROOT`` that has a
``processed/step0_output.npz``. Requires ``imageio`` and ``imageio-ffmpeg``
for MP4 output (``pip install imageio imageio-ffmpeg``); if those are missing
it writes an animated GIF instead.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── Python 3.11+ / NumPy 1.24+ compatibility shims for chumpy ────────────────
# Same shims as mobileposer_runner.py -- needed because building the SMPL mesh
# goes through chumpy, which uses APIs removed in modern Python / NumPy.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for _name, _builtin in [("bool", bool), ("int", int), ("float", float),
                         ("complex", complex), ("object", object),
                         ("str", str), ("unicode", str)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)

import matplotlib
matplotlib.use("Agg")  # headless / offscreen rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ── Configuration ────────────────────────────────────────────────────────────
CONFIG = dict(
    mobileposer_root=r"C:\Users\96mal\BSc-THESIS\MobilePoser-main",
    weights_file=r"C:\Users\96mal\BSc-THESIS\MobilePoser-main\weights.pth",
    combo="lw_rp",
)

DATA_ROOT = Path("../data/NU")
RECORDINGS = ["Walking", "Walking-4", "Upstairs", "Downstairs"]

# Render settings
RENDER_MODE = "skeleton"  # "skeleton" (fast, reliable) or "mesh" (full SMPL,
                          # much slower in matplotlib). Skeleton renders 24
                          # joints per frame; mesh renders ~6890 vertices.
FRAME_STRIDE = 1       # render every Nth frame. 1 = every frame = smoothest
                       # video (uses each pose the model produced). Raise to
                       # 2-3 only if rendering is too slow for you.
N_STILLS = 5           # number of still frames to save per recording
DPI = 90               # render resolution
ELEV, AZIM = 10, -70   # 3D view angle (degrees)

# Output video frame rate. None = play back at the source's own rate (its
# target_hz), so the body moves at real-world speed and as smoothly as the
# data allows. Set to a number to force a specific playback rate.
FPS_OUT = None

# Which MobilePoser output files to render per recording. Each becomes its own
# video + stills, named by the file stem, so the 30 Hz (native) and 60 Hz
# (off-spec comparison) outputs sit side by side for comparison.
#   step0_output.npz       -> pose_video.mp4        (30 Hz, native)
#   step0_output_60hz.npz  -> pose_video_60hz.mp4   (60 Hz, comparison)
NPZ_VARIANTS = ["step0_output.npz", "step0_output_60hz.npz"]

# Mesh rendering in matplotlib is slow (thousands of triangles per frame).
# If you only need the still figures for the paper and not the full video,
# set STILLS_ONLY = True for a fast run (renders just N_STILLS frames, no
# video). The video is mainly useful as a supplementary repo artifact.
STILLS_ONLY = False

# Hard cap on rendered video frames, to avoid extremely long runs on long
# recordings. With FRAME_STRIDE applied first, frames beyond this many are
# dropped from the video (stills still span the whole recording). Set to
# None to disable the cap.
MAX_VIDEO_FRAMES = 600

# SMPL 24-joint kinematic tree (for skeleton fallback).
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
]


# ── SMPL mesh construction (primary path) ────────────────────────────────────

def build_mesh_frames(npz: dict, mobileposer_root: str):
    """Return (verts, faces) for the SMPL mesh, or None if unavailable.

    verts : (N, V, 3) float array of mesh vertices per frame (in place,
            translation removed so the body stays centred).
    faces : (F, 3) int array of mesh triangle indices.

    Uses MobilePoser's own ParametricModel -- the same ``articulate`` body
    model that ``visualize_sensor.py`` uses via ``view_motion`` -- so the
    mesh matches theirs. The ``articulate`` library has used a couple of
    slightly different method names across versions, so this tries the known
    variants and returns None (-> skeleton fallback) if none work, rather
    than crashing.
    """
    try:
        import torch
        if mobileposer_root not in sys.path:
            sys.path.insert(0, mobileposer_root)
        from mobileposer.config import paths
        import mobileposer.articulate as art

        pose = torch.from_numpy(npz["pose"]).float().view(-1, 24, 3, 3)
        n = pose.shape[0]
        tran_zero = torch.zeros(n, 3)

        body = art.model.ParametricModel(paths.smpl_file, device="cpu")

        # Locate the triangle faces attribute (name varies by version).
        faces = None
        for attr in ("face", "faces"):
            if hasattr(body, attr):
                faces = np.asarray(getattr(body, attr), dtype=np.int64)
                break

        verts = None
        # Variant 1: forward_kinematics(pose, tran=..., calc_mesh=True)
        #            -> (..., verts) where verts is the last return value.
        try:
            out = body.forward_kinematics(pose, tran=tran_zero, calc_mesh=True)
            cand = out[-1] if isinstance(out, (tuple, list)) else out
            verts = cand.detach().cpu().numpy()
        except Exception:
            verts = None

        # Variant 2: explicit mesh method, if present.
        if verts is None:
            for meth in ("forward_kinematics_mesh", "get_mesh", "mesh"):
                if hasattr(body, meth):
                    try:
                        cand = getattr(body, meth)(pose)
                        cand = cand[-1] if isinstance(cand, (tuple, list)) else cand
                        verts = cand.detach().cpu().numpy()
                        break
                    except Exception:
                        verts = None

        if verts is None or faces is None:
            print("    [mesh] articulate mesh API not found in this version")
            print("    [mesh] falling back to skeleton render")
            return None

        verts = np.asarray(verts).reshape(n, -1, 3)
        return verts, faces
    except Exception as e:
        print(f"    [mesh] could not build SMPL mesh ({type(e).__name__}: {e})")
        print("    [mesh] falling back to skeleton render")
        return None


# ── Frame drawing ─────────────────────────────────────────────────────────────

def _setup_body_axis(ax):
    ax.set_axis_off()
    ax.view_init(elev=ELEV, azim=AZIM)
    try:
        ax.set_box_aspect((1, 1, 1.6))
    except Exception:
        pass


def _draw_mesh(ax, verts_f, faces):
    """Draw one mesh frame (verts_f: (V,3))."""
    # SMPL is Y-up; matplotlib 3D is Z-up. Map (x, y, z)->(x, z, y) so the
    # body stands upright in the plot.
    v = verts_f[:, [0, 2, 1]]
    tris = v[faces]  # (F, 3, 3)
    coll = Poly3DCollection(tris, alpha=0.9, linewidths=0)
    coll.set_facecolor((0.6, 0.7, 0.85))
    coll.set_edgecolor((0.3, 0.4, 0.55, 0.15))
    ax.add_collection3d(coll)
    _set_equal_3d(ax, v)


def _draw_skeleton(ax, joints_f):
    """Draw one skeleton frame (joints_f: (24,3))."""
    j = joints_f[:, [0, 2, 1]]
    for ji, p in enumerate(SMPL_PARENTS):
        if p < 0:
            continue
        ax.plot([j[ji, 0], j[p, 0]], [j[ji, 1], j[p, 1]], [j[ji, 2], j[p, 2]],
                color="#2c3e50", linewidth=2)
    ax.scatter(j[:, 0], j[:, 1], j[:, 2], color="#e74c3c", s=10)
    _set_equal_3d(ax, j)


def _set_equal_3d(ax, pts):
    centre = pts.mean(axis=0)
    r = np.abs(pts - centre).max() * 1.1
    ax.set_xlim(centre[0] - r, centre[0] + r)
    ax.set_ylim(centre[1] - r, centre[1] + r)
    ax.set_zlim(centre[2] - r, centre[2] + r)


def _draw_sensor_panel(ax, t, sig_dict, t_now, title):
    """Static sensor signals with a moving time cursor."""
    ax.clear()
    for label, (sig, color) in sig_dict.items():
        ax.plot(t, sig, color=color, linewidth=0.8, label=label)
    ax.axvline(t_now, color="red", linewidth=1.5)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("accel. magnitude (m/s²)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)


# ── Main per-recording render ─────────────────────────────────────────────────

def render_recording(npz_path: Path, mobileposer_root: str) -> None:
    npz = dict(np.load(npz_path, allow_pickle=True))
    target_hz = float(npz.get("target_hz", 30.0))
    n_total = npz["pose"].shape[0] if "pose" in npz else npz["joints"].shape[0]

    # Output suffix from the npz filename so 30 Hz and 60 Hz renders don't
    # overwrite each other:
    #   step0_output.npz       -> ""        -> pose_video.mp4
    #   step0_output_60hz.npz  -> "_60hz"   -> pose_video_60hz.mp4
    stem = npz_path.stem  # e.g. "step0_output" or "step0_output_60hz"
    suffix = stem.replace("step0_output", "")

    # Build mesh frames only if explicitly requested (mesh is slow in
    # matplotlib); otherwise go straight to the fast skeleton render.
    mesh = None
    if RENDER_MODE == "mesh":
        mesh = build_mesh_frames(npz, mobileposer_root)
        if mesh is None:
            print("    [mode] mesh requested but unavailable -- using skeleton")
    if mesh is not None:
        verts, faces = mesh
        mode = "mesh"
        n_total = verts.shape[0]
    else:
        verts = faces = None
        mode = "skeleton"
        if "joints" not in npz:
            print("    no 'joints' in npz and mesh unavailable -- cannot render")
            return
        joints = npz["joints"].reshape(-1, 24, 3)
        n_total = joints.shape[0]

    # Sensor signals for the side panel: per-active-slot accel magnitude.
    t_full = np.arange(n_total) / target_hz
    sig_dict = {}
    if "acc" in npz:
        acc = npz["acc"]  # (N, 6, 3)
        # Slots populated by lw_rp: 0 = left wrist (watch), 3 = right pocket (phone)
        if acc.shape[1] > 3:
            sig_dict["watch (left wrist)"] = (
                np.linalg.norm(acc[:, 0, :], axis=1), "#16a085")
            sig_dict["phone (right pocket)"] = (
                np.linalg.norm(acc[:, 3, :], axis=1), "#8e44ad")

    frame_idxs = list(range(0, n_total, FRAME_STRIDE))
    if MAX_VIDEO_FRAMES is not None and len(frame_idxs) > MAX_VIDEO_FRAMES:
        # Evenly subsample down to the cap so the video still spans the
        # whole recording, just at a coarser frame rate.
        sel = np.linspace(0, len(frame_idxs) - 1, MAX_VIDEO_FRAMES, dtype=int)
        frame_idxs = [frame_idxs[i] for i in sel]

    out_dir = npz_path.parent
    rec_name = out_dir.parent.name

    # Figure: body (left, 3D) + sensor panel (right, 2D).
    # Size chosen so the pixel dimensions are divisible by 16 (avoids ffmpeg
    # macro-block resize warnings and keeps the video crisp).
    fig = plt.figure(figsize=(10.24, 4.8), dpi=DPI)
    ax_body = fig.add_subplot(1, 2, 1, projection="3d")
    ax_sens = fig.add_subplot(1, 2, 2)

    # Still frames span the entire recording (independent of the video cap).
    still_frames = np.linspace(0, n_total - 1, N_STILLS, dtype=int).tolist()

    def render_at(fi: int):
        ax_body.clear()
        _setup_body_axis(ax_body)
        if mode == "mesh":
            _draw_mesh(ax_body, verts[fi], faces)
        else:
            _draw_skeleton(ax_body, joints[fi])
        ax_body.set_title(f"{rec_name} — MobilePoser pose ({mode})\n"
                          f"t = {fi / target_hz:5.1f} s", fontsize=9)
        if sig_dict:
            _draw_sensor_panel(ax_sens, t_full, sig_dict, fi / target_hz,
                               "Input IMU signals")
        fig.tight_layout()

    # Playback frame rate: if FPS_OUT is None, play at the source's own rate
    # scaled by the stride, so motion runs at real-world speed.
    if FPS_OUT is None:
        out_fps = max(1, int(round(target_hz / FRAME_STRIDE)))
    else:
        out_fps = max(1, int(round(FPS_OUT / FRAME_STRIDE)))

    # --- write video (unless stills-only) ---
    wrote_video = False
    if not STILLS_ONLY:
        video_path = out_dir / f"pose_video{suffix}.mp4"
        wrote_video = _write_animation(
            fig, lambda k: render_at(frame_idxs[k]), len(frame_idxs),
            video_path, out_fps)

    # --- write stills ---
    for s_i, fi in enumerate(still_frames):
        render_at(fi)
        still_path = out_dir / f"pose_still{suffix}_{s_i}.png"
        fig.savefig(still_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    if STILLS_ONLY:
        print(f"    rendered {N_STILLS} stills [{mode}] (stills-only mode)")
    else:
        ext = "mp4" if wrote_video else "gif"
        print(f"    {target_hz:.0f} Hz: rendered {len(frame_idxs)} frames "
              f"[{mode}] @ {out_fps} fps -> pose_video{suffix}.{ext} "
              f"+ {N_STILLS} stills")


def _write_animation(fig, render_frame, n_frames, video_path: Path, fps: int):
    """Write an MP4 via imageio-ffmpeg. Returns True if MP4, False if GIF.

    GIF is a last-resort fallback only -- it is very slow for mesh renders
    and produces huge files, so we try hard to get MP4 working first and
    print the real reason if it fails.
    """
    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio  # type: ignore

    def frames():
        for k in range(n_frames):
            render_frame(k)
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            img = buf.reshape(h, w, 4)[..., :3]
            # ffmpeg/libx264 needs even width & height; crop if odd.
            img = img[: h - (h % 2), : w - (w % 2), :]
            yield np.ascontiguousarray(img)

    # Point imageio at imageio-ffmpeg's bundled binary so the MP4 writer
    # doesn't fail just because ffmpeg isn't on PATH.
    try:
        import imageio_ffmpeg
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE",
                              imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass

    # Confirm ffmpeg is actually available before committing to MP4.
    ffmpeg_ok = False
    try:
        import imageio_ffmpeg  # noqa: F401
        imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_ok = True
    except Exception as e:
        print(f"    [video] imageio-ffmpeg not available ({type(e).__name__}). "
              f"Install it with:  pip install imageio-ffmpeg")

    if ffmpeg_ok:
        try:
            # 'pillow' is not used here; ffmpeg writer needs the explicit
            # format. macro_block_size=1 avoids the resize warning.
            with imageio.get_writer(
                video_path, format="FFMPEG", mode="I", fps=max(1, fps),
                codec="libx264", quality=8, macro_block_size=1,
            ) as w:
                for frame in frames():
                    w.append_data(frame)
            return True
        except Exception as e:
            print(f"    [video] MP4 writing failed ({type(e).__name__}: {e})")

    # Last resort: GIF (slow, large -- only if ffmpeg truly unavailable).
    print("    [video] writing GIF instead (slow; consider installing ffmpeg)")
    gif_path = video_path.with_suffix(".gif")
    with imageio.get_writer(gif_path, mode="I", duration=1.0 / max(1, fps)) as w:
        for frame in frames():
            w.append_data(frame)
    return False


def main() -> None:
    for name in RECORDINGS:
        proc_dir = DATA_ROOT / name / "processed"
        present = [v for v in NPZ_VARIANTS if (proc_dir / v).exists()]
        if not present:
            print(f"skip {name}: no MobilePoser output in {proc_dir} "
                  f"(run mobileposer_runner.py first)")
            continue
        print(f"=== {name} ===")
        for variant in present:
            npz_path = proc_dir / variant
            try:
                render_recording(npz_path, CONFIG["mobileposer_root"])
            except Exception as e:
                import traceback
                print(f"    FAILED ({variant}): {type(e).__name__}: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()