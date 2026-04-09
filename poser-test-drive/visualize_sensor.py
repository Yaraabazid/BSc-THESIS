"""
Visualize MobilePoser output from sensor data.

Usage
─────
  # 1. Run the notebook first (cells 1-19) to produce step0_output.npz
  #    OR pass --run-pipeline to build everything from raw CSVs here.

  # View the saved .npz from the notebook:
  python visualize_sensor.py --npz ./data/sensor/default/processed/step0_output.npz

  # Or run the full pipeline from raw CSVs and view immediately:
  python visualize_sensor.py --run-pipeline --recording-dir ./data/sensor/default

  # Add --with-tran to include root translation (walking path):
  python visualize_sensor.py --npz ... --with-tran

  # Compare against a DIP test sequence to sanity-check the viewer works:
  python visualize_sensor.py --test-dip --seq-num 5 --with-tran
"""

import os
import sys
import argparse
import numpy as np
import torch

# ── Ensure mobileposer is importable ──────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mobileposer.config import paths, model_config, amass, datasets
from mobileposer.models import MobilePoserNet
from mobileposer.utils.model_utils import smooth_avg
import mobileposer.articulate as art


def load_model(weights_path, device):
    """Load the pretrained MobilePoserNet."""
    model = MobilePoserNet().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def build_imu_from_npz(npz_path, combo='lw_rp', device='cpu'):
    """
    Load step0_output.npz (from the notebook) and rebuild the (1, N, 60)
    input tensor exactly as MobilePoser expects.
    """
    data = np.load(npz_path)
    acc_6 = torch.from_numpy(data['acc'])    # (N, 6, 3)
    ori_6 = torch.from_numpy(data['ori'])    # (N, 6, 3, 3)

    combo_slots = amass.combos[combo]

    acc_masked = torch.zeros_like(acc_6)
    ori_masked = torch.zeros_like(ori_6)
    acc_masked[:, combo_slots] = acc_6[:, combo_slots]
    ori_masked[:, combo_slots] = ori_6[:, combo_slots]

    acc_5 = acc_masked[:, amass.all_imu_ids] / amass.acc_scale
    ori_5 = ori_masked[:, amass.all_imu_ids]
    acc_5 = smooth_avg(acc_5)

    imu = torch.cat([acc_5.flatten(1), ori_5.flatten(1)], dim=1).float()
    return imu.unsqueeze(0).to(device)


def run_pipeline_from_csvs(recording_dir, combo='lw_rp', target_hz=30):
    """
    Replicate the notebook's CSV → tensor pipeline so you can skip
    running Jupyter entirely.
    """
    import pandas as pd
    from scipy.spatial.transform import Rotation as R
    from scipy.interpolate import interp1d

    def load_csv(folder, filename):
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected: {path}")
        return pd.read_csv(path)

    def extract_time(df):
        t_ns = df['time'].values.astype(np.float64)
        return (t_ns - t_ns[0]) * 1e-9

    def extract_acc(df):
        t = extract_time(df)
        acc = df[['x', 'y', 'z']].values.astype(np.float32)
        return t, acc

    def extract_quat(df):
        t = extract_time(df)
        quats = df[['qw', 'qx', 'qy', 'qz']].values.astype(np.float64)
        return t, quats

    def quat_to_rotmat(q_wxyz):
        q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
        return R.from_quat(q_xyzw).as_matrix().astype(np.float32)

    def resample(t_orig, signal, t_new):
        D = signal.shape[1]
        out = np.zeros((len(t_new), D), dtype=np.float32)
        for d in range(D):
            f = interp1d(t_orig, signal[:, d], kind='linear',
                         bounds_error=False, fill_value='extrapolate')
            out[:, d] = f(t_new)
        return out

    # Load CSVs
    print("Loading CSVs...")
    phone_acc_df = load_csv(recording_dir, "TotalAcceleration.csv")
    phone_ori_df = load_csv(recording_dir, "Orientation.csv")
    watch_acc_df = load_csv(recording_dir, "WatchAccelerometer.csv")
    watch_ori_df = load_csv(recording_dir, "WatchOrientation.csv")

    t_pa, phone_acc = extract_acc(phone_acc_df)
    t_wa, watch_acc = extract_acc(watch_acc_df)
    t_po, phone_quat = extract_quat(phone_ori_df)
    t_wo, watch_quat = extract_quat(watch_ori_df)

    phone_rot = quat_to_rotmat(phone_quat)
    watch_rot = quat_to_rotmat(watch_quat)

    # Resample to common grid
    t_start = max(t_pa[0], t_wa[0], t_po[0], t_wo[0])
    t_end = min(t_pa[-1], t_wa[-1], t_po[-1], t_wo[-1])
    n_frames = int((t_end - t_start) * target_hz)
    t_common = np.linspace(t_start, t_end, n_frames)

    phone_acc_rs = resample(t_pa, phone_acc, t_common)
    watch_acc_rs = resample(t_wa, watch_acc, t_common)
    phone_rot_rs = resample(t_po, phone_rot.reshape(-1, 9), t_common).reshape(-1, 3, 3)
    watch_rot_rs = resample(t_wo, watch_rot.reshape(-1, 9), t_common).reshape(-1, 3, 3)

    print(f"Resampled: {n_frames} frames at {target_hz} Hz ({t_end - t_start:.1f}s)")

    # Pack into (N, 6, 3) and (N, 6, 3, 3)
    N = n_frames
    acc_6 = np.zeros((N, 6, 3), dtype=np.float32)
    ori_6 = np.zeros((N, 6, 3, 3), dtype=np.float32)
    for i in range(6):
        ori_6[:, i] = np.eye(3)

    # lw_rp: slot 0 = left_wrist (watch), slot 3 = right_thigh (phone)
    acc_6[:, 0] = watch_acc_rs
    acc_6[:, 3] = phone_acc_rs
    ori_6[:, 0] = watch_rot_rs
    ori_6[:, 3] = phone_rot_rs

    # Build IMU tensor
    acc_t = torch.from_numpy(acc_6)
    ori_t = torch.from_numpy(ori_6)

    combo_slots = amass.combos[combo]
    acc_masked = torch.zeros_like(acc_t)
    ori_masked = torch.zeros_like(ori_t)
    acc_masked[:, combo_slots] = acc_t[:, combo_slots]
    ori_masked[:, combo_slots] = ori_t[:, combo_slots]

    acc_5 = acc_masked[:, amass.all_imu_ids] / amass.acc_scale
    ori_5 = ori_masked[:, amass.all_imu_ids]
    acc_5 = smooth_avg(acc_5)

    imu = torch.cat([acc_5.flatten(1), ori_5.flatten(1)], dim=1).float()
    return imu.unsqueeze(0)


def view_with_smpl(pose, tran, with_tran=True, fps=30):
    """Render the SMPL body mesh using vctoolkit (interactive window)."""
    bodymodel = art.model.ParametricModel(paths.smpl_file, device=pose.device)

    if not with_tran:
        tran = torch.zeros(pose.shape[0], 3, device=pose.device)

    pose = pose.view(-1, 24, 3, 3)
    tran = tran.view(-1, 3)

    print(f"\nLaunching SMPL viewer: {pose.shape[0]} frames at {fps} FPS")
    print("  Controls: mouse to rotate, scroll to zoom, close window to exit")
    bodymodel.view_motion([pose], [tran], fps=fps, distance_between_subjects=0)


def export_unity(pose, tran, output_dir):
    """Save pose.txt + tran.txt for Unity."""
    bodymodel = art.model.ParametricModel(paths.smpl_file, device='cpu')
    bodymodel.save_unity_motion(pose=pose.cpu(), tran=tran.cpu(), output_dir=output_dir)
    print(f"\nUnity files saved to: {output_dir}/")
    print(f"  pose.txt  ({pose.shape[0]} frames)")
    print(f"  tran.txt  ({tran.shape[0]} frames)")


def test_with_dip(seq_num, combo, with_tran, device):
    """
    Run on a DIP test sequence — useful to verify the viewer works
    and to see what 'good' output looks like as a reference.
    """
    from mobileposer.loader import DataLoader as MPDataLoader

    print(f"\n=== Test mode: DIP sequence {seq_num}, combo '{combo}' ===")
    loader = MPDataLoader(dataset='dip', combo=combo, device=device)
    data = loader.load_data(seq_num)

    model = load_model(paths.weights_file, device)
    imu = data['imu'].unsqueeze(0)

    print(f"IMU shape: {tuple(imu.shape)}")
    with torch.no_grad():
        pose_p, joints_p, tran_p, contact_p = model.forward_offline(imu, [imu.shape[1]])

    # Also show ground truth side-by-side
    from mobileposer.viewers import SMPLViewer
    viewer = SMPLViewer(fps=datasets.fps)
    print(f"\nViewing: prediction (red-tinted) vs ground truth (gray)")
    print(f"  {pose_p.shape[0]} frames, with_tran={with_tran}")

    os.environ['GT'] = '1'  # show both prediction and ground truth
    viewer.view(
        pose_p, tran_p,
        data['pose'], data['tran'],
        with_tran=with_tran
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MobilePoser predictions from your sensor data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input source (pick one)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--npz', type=str,
                        help='Path to step0_output.npz from the notebook')
    source.add_argument('--run-pipeline', action='store_true',
                        help='Build tensors from raw CSVs (skip the notebook)')
    source.add_argument('--test-dip', action='store_true',
                        help='Test with a DIP dataset sequence (sanity check)')

    # Options
    parser.add_argument('--recording-dir', type=str, default='./data/sensor/default',
                        help='Sensor Logger CSV folder (for --run-pipeline)')
    parser.add_argument('--weights', type=str, default=str(paths.weights_file),
                        help='Model weights path')
    parser.add_argument('--combo', type=str, default='lw_rp',
                        choices=list(amass.combos.keys()),
                        help='Device combination')
    parser.add_argument('--with-tran', action='store_true',
                        help='Include root translation (walking path)')
    parser.add_argument('--seq-num', type=int, default=1,
                        help='Sequence number (for --test-dip)')
    parser.add_argument('--export-unity', type=str, default=None,
                        help='Also export Unity files to this directory')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu, auto-detected if omitted)')

    args = parser.parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Test mode: DIP dataset ────────────────────────────────────────────
    if args.test_dip:
        test_with_dip(args.seq_num, args.combo, args.with_tran, device)
        return

    # ── Build or load IMU tensor ──────────────────────────────────────────
    if args.npz:
        print(f"Loading from: {args.npz}")
        batch = build_imu_from_npz(args.npz, combo=args.combo, device=device)
    else:
        print(f"Running pipeline from: {args.recording_dir}")
        batch = run_pipeline_from_csvs(
            args.recording_dir, combo=args.combo, target_hz=datasets.fps
        ).to(device)

    print(f"IMU tensor: {tuple(batch.shape)}")

    # ── Run inference ─────────────────────────────────────────────────────
    model = load_model(args.weights, device)
    with torch.no_grad():
        pred_pose, pred_joints, pred_tran, pred_contact = model.forward_offline(batch)

    n_frames = pred_pose.shape[0]
    duration = n_frames / datasets.fps
    tran_np = pred_tran.cpu().numpy()
    total_dist = np.sqrt(np.sum(np.diff(tran_np[:, [0, 2]], axis=0) ** 2, axis=1)).sum()

    print(f"\n--- Results ---")
    print(f"Frames     : {n_frames} ({duration:.1f}s at {datasets.fps} FPS)")
    print(f"Distance   : {total_dist:.2f} m")
    print(f"Avg speed  : {total_dist / duration:.2f} m/s")
    print(f"Height     : {tran_np[:, 1].min():.3f} to {tran_np[:, 1].max():.3f} m")

    # ── Export Unity files if requested ───────────────────────────────────
    if args.export_unity:
        export_unity(pred_pose, pred_tran, args.export_unity)

    # ── Launch SMPL viewer ────────────────────────────────────────────────
    view_with_smpl(pred_pose, pred_tran, with_tran=args.with_tran, fps=datasets.fps)


if __name__ == '__main__':
    main()
