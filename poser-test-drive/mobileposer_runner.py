r"""Run MobilePoser on Sensor Logger recordings (refactored from final-sensor.ipynb).

This is the same pipeline as ``final-sensor.ipynb`` cells 1, 3, 5, 7, 11, 13,
17, 19, 24 -- but as a reusable function, so it can run on every recording
folder (Walking, Walking-4, Upstairs, Downstairs) in one go instead of
re-running the notebook by hand for each one.

For each recording folder it writes ``<recording_dir>/processed/step0_output.npz``
with the same keys as the notebook: ``acc``, ``ori``, ``tran``, ``joints``,
``pose``, ``contact``, ``target_hz``, ``combo``, ``mode``.

``generate_report_plots.py`` (in ``trajectory/``) automatically picks up these
``step0_output.npz`` files and adds MobilePoser's translation to the trajectory
and altitude comparison plots.

Requirements
------------
This needs the MobilePoser repo + pretrained weights, exactly like the
notebook. Edit the paths in ``CONFIG`` below, then run::

    python mobileposer_runner.py

Required asset: SMPL body model file
--------------------------------------
MobilePoser needs an SMPL body model file (``basicmodel_m.pkl``) for its
floor-penetration / vertical-translation correction step. This step only
triggers for recordings with real vertical motion (stairs), so flat-walking
recordings can succeed even if this file is missing -- the failure only
shows up on Upstairs/Downstairs.

If you see an error like::

    FAILED: [Errno 2] No such file or directory: '...\\smpl\\basicmodel_m.pkl'

the asset path is being resolved relative to the directory you launched this
script from, not relative to the MobilePoser repo. The fix is to copy the
file from inside your MobilePoser repo to a ``smpl/`` subfolder next to
wherever you run this script. On Windows, from the folder you run
``mobileposer_runner.py`` in (e.g. ``poser-test-drive``)::

    mkdir smpl
    copy <mobileposer_root>\mobileposer\smpl\basicmodel_m.pkl smpl\

(adjust ``<mobileposer_root>`` to your local MobilePoser repo path -- the
file is typically at ``MobilePoser-main\mobileposer\smpl\basicmodel_m.pkl``).

Notes on time alignment
------------------------
Following the notebook, each sensor stream is zeroed to its own first sample
(``t = (time - time[0]) * 1e-9``). This assumes the phone and watch recordings
were started at (approximately) the same physical instant -- true to within
the second or two it takes to press "record" on both devices. This is
*different* from the ``fix_watch_clock`` correction used in the PDR package
(which corrects a ~17.6-day clock offset using absolute epoch timestamps), but
for MobilePoser's purposes only the *relative* alignment between the two
streams matters, and the per-stream zeroing keeps that error small.

When the resulting ``tran`` array is later compared against the PDR
trajectories (in ``generate_report_plots.py``), the same assumption is made:
both time axes are aligned at their own t=0. Any residual button-press offset
(typically well under 2 s) is not corrected.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

# ── Python 3.11+ compatibility shim for chumpy ──────────────────────────────
# MobilePoser's SMPL body model loading goes through `chumpy`, which still
# calls `inspect.getargspec` -- removed in Python 3.11 (deprecated since
# Python 3.0). chumpy hasn't been updated to use the replacement,
# `inspect.getfullargspec`, which is a superset with the same `.args`,
# `.varargs`, and `.defaults` attributes that chumpy's old code relies on.
# This patches it back in before anything has a chance to import chumpy.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

# ── NumPy 1.24+ compatibility shim for chumpy ───────────────────────────────
# chumpy's __init__.py does
#   `from numpy import bool, int, float, complex, object, unicode, str, nan, inf`
# -- these bare-Python-type aliases on the numpy module were deprecated in
# NumPy 1.20 and removed entirely in NumPy 1.24. They're harmless aliases
# for the builtin types, so it's safe to add them back before chumpy (or
# anything that imports it) gets a chance to import numpy itself.
for _name, _builtin in [("bool", bool), ("int", int), ("float", float),
                          ("complex", complex), ("object", object),
                          ("str", str), ("unicode", str)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)


# ── Configuration ────────────────────────────────────────────────────────────
# Edit these for your local setup.
CONFIG = dict(
    # Path to MobilePoser pretrained weights file.
    weights_file=r"C:\Users\96mal\BSc-THESIS\MobilePoser-main\weights.pth",
    # Path to MobilePoser repo root (so `import mobileposer` works).
    mobileposer_root=r"C:\Users\96mal\BSc-THESIS\MobilePoser-main",
    # 'default' = Android OS fusion (Sensor Logger without Sensor Zoo).
    # 'sensor_zoo' = Madgwick filter (Sensor Logger with Sensor Zoo enabled).
    mode="default",
    # MobilePoser was trained at 30 FPS (mobileposer/config.py -> datasets.fps).
    target_hz=30,
    # 'lw_rp' = left-wrist (watch) + right-pocket (phone). Matches our setup.
    combo="lw_rp",
)

# Recordings to process, relative to DATA_ROOT.
DATA_ROOT = Path("../data/NU")
RECORDINGS = ["Walking", "Walking-4", "Upstairs", "Downstairs"]

# Rates to run MobilePoser at. 30 is the model's NATIVE training rate -- it is
# the one to trust, and the canonical step0_output.npz used by the rest of the
# pipeline. 60 is included only as an off-spec comparison (the watch tops out
# ~73 Hz and the phone ~54 Hz, so 60 is about the highest common rate we can
# actually interpolate to). Running the model above its trained rate is not
# expected to improve pose quality and may change it unpredictably -- the
# point is to *check* that, not to assume 60 is better. The 60 Hz output is
# saved as step0_output_60hz.npz so it never overwrites the canonical file.
TARGET_HZ_LIST = [30, 60]


# ── CSV loading (cells 5 & 7) ─────────────────────────────────────────────────

def _load_csv(folder: Path, filename: str) -> pd.DataFrame:
    path = folder / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    return pd.read_csv(path)


def _extract_time_seconds(df: pd.DataFrame) -> np.ndarray:
    """Relative seconds from the first sample (each stream zeroed independently)."""
    t_ns = df["time"].to_numpy(dtype=np.float64)
    return (t_ns - t_ns[0]) * 1e-9


def _extract_acceleration(df: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (t_seconds, acc_xyz) with acc_xyz shape (N, 3)."""
    t = _extract_time_seconds(df)
    if "x" in df.columns:
        acc = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    else:
        acc = df.iloc[:, 1:4].to_numpy(dtype=np.float32)
    return t, acc


def _extract_quaternion(df: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (t_seconds, quats) with quats shape (N, 4) in [qw, qx, qy, qz]."""
    t = _extract_time_seconds(df)
    if "qw" in df.columns:
        quats = df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=np.float64)
    else:
        quats = df.iloc[:, 1:5].to_numpy(dtype=np.float64)
    return t, quats


# ── Quaternion -> rotation matrix (cell 11) ──────────────────────────────────

def _quat_to_rotmat(quats_wxyz: np.ndarray) -> np.ndarray:
    """(N, 4) [w,x,y,z] -> (N, 3, 3) rotation matrices."""
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
    return R.from_quat(quats_xyzw).as_matrix().astype(np.float32)


# ── Resampling (cell 13) ──────────────────────────────────────────────────────

def _resample_signal(t_orig: np.ndarray, signal: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    D = signal.shape[1]
    out = np.zeros((len(t_new), D), dtype=np.float32)
    for d in range(D):
        f = interp1d(t_orig, signal[:, d], kind="linear",
                      bounds_error=False, fill_value="extrapolate")
        out[:, d] = f(t_new)
    return out


def _resample_rotmat(t_orig: np.ndarray, rotmats: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    n_new = len(t_new)
    flat = rotmats.reshape(-1, 9)
    flat_resampled = _resample_signal(t_orig, flat, t_new)
    return flat_resampled.reshape(n_new, 3, 3)


# ── Main pipeline (cells 17, 19, 24) ─────────────────────────────────────────

def run_mobileposer_on_recording(
    recording_dir: str | Path,
    weights_file: str | Path,
    mobileposer_root: str | Path,
    combo: str = "lw_rp",
    target_hz: int = 30,
    mode: str = "default",
    device: Optional[str] = None,
    save: bool = True,
) -> dict:
    """Run the Sensor Logger -> MobilePoser pipeline on one recording folder.

    Equivalent to running ``final-sensor.ipynb`` end-to-end on
    ``recording_dir``. If ``save`` is True, writes
    ``<recording_dir>/processed/step0_output.npz`` (same format as the
    notebook's Cell 24).

    Parameters
    ----------
    recording_dir : path-like
        Folder with ``TotalAcceleration.csv``, ``Orientation.csv``,
        ``WatchAccelerometer.csv``, ``WatchOrientation.csv``.
    weights_file, mobileposer_root : path-like
        Same as the notebook's Cell 1.
    combo : str
        Device combination. ``'lw_rp'`` = left-wrist watch + right-pocket phone.
    target_hz : int
        Resampling rate (MobilePoser was trained at 30 FPS).
    mode : str
        ``'default'`` or ``'sensor_zoo'`` -- only affects the docstring/printed
        info here, since both extraction functions already select columns by
        name and work for either Sensor Logger mode.
    device : str, optional
        ``'cuda'`` or ``'cpu'``. Defaults to CUDA if available.
    save : bool
        Write ``processed/step0_output.npz``.

    Returns
    -------
    dict
        Same keys as ``step0_output.npz``: ``acc``, ``ori``, ``tran``,
        ``joints``, ``pose``, ``contact``, ``target_hz``, ``combo``, ``mode``.
    """
    recording_dir = Path(recording_dir).resolve()
    mobileposer_root = Path(mobileposer_root).resolve()
    weights_file = Path(weights_file).resolve()
    mobileposer_root_str = str(mobileposer_root)
    if mobileposer_root_str not in sys.path:
        sys.path.insert(0, mobileposer_root_str)

    # Imports that require MOBILEPOSER_ROOT on sys.path.
    from mobileposer.config import amass, model_config
    from mobileposer.utils.model_utils import smooth_avg
    from mobileposer.models import MobilePoserNet

    # --- load CSVs (cell 5) ---------------------------------------------------
    phone_acc_df = _load_csv(recording_dir, "TotalAcceleration.csv")
    phone_ori_df = _load_csv(recording_dir, "Orientation.csv")
    watch_acc_df = _load_csv(recording_dir, "WatchAccelerometer.csv")
    watch_ori_df = _load_csv(recording_dir, "WatchOrientation.csv")

    # --- extract + normalise (cell 7) -----------------------------------------
    t_phone_acc, phone_acc = _extract_acceleration(phone_acc_df, mode)
    t_watch_acc, watch_acc = _extract_acceleration(watch_acc_df, mode)
    t_phone_ori, phone_quat = _extract_quaternion(phone_ori_df, mode)
    t_watch_ori, watch_quat = _extract_quaternion(watch_ori_df, mode)

    # --- quaternion -> rotation matrix (cell 11) ------------------------------
    phone_rotmat = _quat_to_rotmat(phone_quat)
    watch_rotmat = _quat_to_rotmat(watch_quat)

    # --- resample to common grid (cell 13) ------------------------------------
    t_start = max(t_phone_acc[0], t_watch_acc[0], t_phone_ori[0], t_watch_ori[0])
    t_end = min(t_phone_acc[-1], t_watch_acc[-1], t_phone_ori[-1], t_watch_ori[-1])
    duration = t_end - t_start
    n_frames = int(duration * target_hz)
    t_common = np.linspace(t_start, t_end, n_frames)

    phone_acc_r = _resample_signal(t_phone_acc, phone_acc, t_common)
    watch_acc_r = _resample_signal(t_watch_acc, watch_acc, t_common)
    phone_rot_r = _resample_rotmat(t_phone_ori, phone_rotmat, t_common)
    watch_rot_r = _resample_rotmat(t_watch_ori, watch_rotmat, t_common)

    # --- build MobilePoser input tensors (cell 17) ----------------------------
    combo_slots = amass.combos[combo]
    SLOT_LEFT_WRIST = 0
    SLOT_RIGHT_POCKET = 3
    n = n_frames

    acc_6 = np.zeros((n, 6, 3), dtype=np.float32)
    acc_6[:, SLOT_LEFT_WRIST, :] = watch_acc_r
    acc_6[:, SLOT_RIGHT_POCKET, :] = phone_acc_r

    ori_6 = np.tile(np.eye(3, dtype=np.float32), (n, 6, 1, 1))
    ori_6[:, SLOT_LEFT_WRIST, :, :] = watch_rot_r
    ori_6[:, SLOT_RIGHT_POCKET, :, :] = phone_rot_r

    acc_t = torch.from_numpy(acc_6)
    ori_t = torch.from_numpy(ori_6)

    acc_masked = torch.zeros_like(acc_t)
    ori_masked = torch.zeros_like(ori_t)
    acc_masked[:, combo_slots] = acc_t[:, combo_slots]
    ori_masked[:, combo_slots] = ori_t[:, combo_slots]

    acc_5 = acc_masked[:, amass.all_imu_ids] / amass.acc_scale
    ori_5 = ori_masked[:, amass.all_imu_ids]
    acc_5 = smooth_avg(acc_5)

    imu_input = torch.cat([acc_5.flatten(1), ori_5.flatten(1)], dim=1)
    batch_t = imu_input.float().unsqueeze(0)

    if batch_t.shape[-1] != model_config.n_imu:
        raise ValueError(
            f"Input tensor last dim {batch_t.shape[-1]} != "
            f"model_config.n_imu {model_config.n_imu}"
        )

    # --- run model (cell 19) ---------------------------------------------------
    # NOTE: MobilePoser internally loads an SMPL body model asset
    # (.../smpl/basicmodel_m.pkl) for floor-penetration / vertical
    # correction. In testing, this path is resolved relative to the
    # directory the *outer* script was launched from (not fixed up by
    # chdir'ing here), and only for recordings with real vertical motion
    # (stairs) -- so if you see a "No such file or directory" error
    # mentioning basicmodel_m.pkl, copy that file from inside your
    # MobilePoser repo (typically
    # <mobileposer_root>/mobileposer/smpl/basicmodel_m.pkl) to
    # <the folder you run this script from>/smpl/basicmodel_m.pkl.
    # See the module docstring for the exact command.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MobilePoserNet().to(device)
    state = torch.load(weights_file, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        pred_pose, pred_joints, pred_tran, pred_contact = model.forward_offline(
            batch_t.to(device)
        )

    pred_pose = pred_pose.cpu()
    pred_joints = pred_joints.view(-1, 24, 3).cpu()
    pred_tran = pred_tran.cpu()
    pred_contact = pred_contact.cpu()

    save_dict = {
        "acc": acc_6,
        "ori": ori_6,
        "tran": pred_tran.numpy(),
        "joints": pred_joints.numpy(),
        "pose": pred_pose.numpy(),
        "contact": pred_contact.numpy(),
        "target_hz": target_hz,
        "combo": combo,
        "mode": mode,
    }

    if save:
        out_dir = recording_dir / "processed"
        out_dir.mkdir(exist_ok=True)
        # 30 Hz (MobilePoser's native rate) is the canonical file used by the
        # rest of the pipeline; keep its original name for backward
        # compatibility. Any other rate gets a rate-suffixed name so the two
        # don't overwrite each other and can be compared side by side.
        fname = ("step0_output.npz" if int(target_hz) == 30
                 else f"step0_output_{int(target_hz)}hz.npz")
        np.savez(out_dir / fname, **save_dict)
        print(f"  saved {out_dir/fname}  "
              f"({n_frames} frames @ {target_hz} Hz, "
              f"final tran = {pred_tran[-1].numpy()})")

    return save_dict


def main() -> None:
    for name in RECORDINGS:
        rec_dir = DATA_ROOT / name
        if not rec_dir.exists():
            print(f"skip {name}: {rec_dir} not found")
            continue
        print(f"=== {name} ===")
        for hz in TARGET_HZ_LIST:
            tag = "native" if hz == 30 else "off-spec comparison"
            print(f"  -- {hz} Hz ({tag}) --")
            try:
                run_mobileposer_on_recording(
                    rec_dir,
                    weights_file=CONFIG["weights_file"],
                    mobileposer_root=CONFIG["mobileposer_root"],
                    combo=CONFIG["combo"],
                    target_hz=hz,
                    mode=CONFIG["mode"],
                )
            except Exception as e:
                print(f"    FAILED: {e}")


if __name__ == "__main__":
    main()