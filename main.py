import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib.animation as animation

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "data\walking-sitting\default"   # <-- change this
OUTPUT_DIR = "data\walking-sitting\output"

PHONE_ACC = os.path.join(DATA_DIR, "TotalAcceleration.csv")
PHONE_ORI = os.path.join(DATA_DIR, "Orientation.csv")

WATCH_ACC = os.path.join(DATA_DIR, "WatchAccelerometer.csv")
WATCH_ORI = os.path.join(DATA_DIR, "WatchOrientation.csv")

STEP_LENGTH = 0.7  # meters (tune later)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# HELPERS
# -------------------------

def load_and_resample(acc_path, ori_path, prefix):
    acc = pd.read_csv(acc_path)
    ori = pd.read_csv(ori_path)

    # --- Standardize timestamp ---
    acc['timestamp'] = pd.to_datetime(acc['time'], unit='ns')
    ori['timestamp'] = pd.to_datetime(ori['time'], unit='ns')

    # --- Remove problematic duplicate columns ---
    for col in ['time', 'seconds_elapsed']:
        if col in acc.columns:
            acc = acc.drop(columns=col)
        if col in ori.columns:
            ori = ori.drop(columns=col)

    acc = acc.set_index('timestamp')
    ori = ori.set_index('timestamp')

    # --- Resample to 60 Hz ---
    acc = acc.resample('16.67ms').mean().interpolate()
    ori = ori.resample('16.67ms').mean().interpolate()

    # --- Prefix columns to avoid collisions ---
    acc = acc.add_prefix(prefix + "_acc_")
    ori = ori.add_prefix(prefix + "_ori_")

    df = pd.concat([acc, ori], axis=1)

    return df


def quaternion_to_yaw(qw, qx, qy, qz):
    siny_cosp = 2 * (qw*qz + qx*qy)
    cosy_cosp = 1 - 2 * (qy*qy + qz*qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def detect_steps(acc_mag):
    peaks, _ = find_peaks(acc_mag, distance=20, prominence=1.0)
    return peaks


def build_trajectory(df, prefix):
    # ⚠️ Adjust names if needed
    ax = df[f"{prefix}_acc_x"]
    ay = df[f"{prefix}_acc_y"]
    az = df[f"{prefix}_acc_z"]

    qw = df[f"{prefix}_ori_qw"]
    qx = df[f"{prefix}_ori_qx"]
    qy = df[f"{prefix}_ori_qy"]
    qz = df[f"{prefix}_ori_qz"]

    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    steps = detect_steps(acc_mag)

    yaw = quaternion_to_yaw(qw, qx, qy, qz)

    x, y = [0], [0]

    for idx in steps:
        dx = STEP_LENGTH * np.cos(yaw.iloc[idx])
        dy = STEP_LENGTH * np.sin(yaw.iloc[idx])

        x.append(x[-1] + dx)
        y.append(y[-1] + dy)

    return np.array(x), np.array(y), acc_mag, steps


def fuse_trajectories(px, py, wx, wy):
    n = min(len(px), len(wx))

    px, py = px[:n], py[:n]
    wx, wy = wx[:n], wy[:n]

    fx = 0.5 * px + 0.5 * wx
    fy = 0.5 * py + 0.5 * wy

    return fx, fy


def save_plot(x, y, title, filename):
    plt.figure()
    plt.plot(x, y)
    plt.scatter(x[0], y[0])
    plt.scatter(x[-1], y[-1])
    plt.title(title)
    plt.axis('equal')
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


def save_comparison(px, py, wx, wy, fx, fy):
    plt.figure()
    plt.plot(px, py, label="Phone")
    plt.plot(wx, wy, label="Watch")
    plt.plot(fx, fy, label="Fused")
    plt.legend()
    plt.title("Trajectory Comparison")
    plt.axis('equal')
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison.png"))
    plt.close()


def animate_trajectories(px, py, wx, wy, fx, fy, filename):
    fig, ax = plt.subplots()

    xmin = min(px.min(), wx.min(), fx.min()) - 1
    xmax = max(px.max(), wx.max(), fx.max()) + 1
    ymin = min(py.min(), wy.min(), fy.min()) - 1
    ymax = max(py.max(), wy.max(), fy.max()) + 1

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    line_p, = ax.plot([], [], label="Phone")
    line_w, = ax.plot([], [], label="Watch")
    line_f, = ax.plot([], [], label="Fused")

    ax.legend()
    ax.set_title("Trajectory Animation")
    ax.grid()

    def update(frame):
        line_p.set_data(px[:frame], py[:frame])
        line_w.set_data(wx[:frame], wy[:frame])
        line_f.set_data(fx[:frame], fy[:frame])
        return line_p, line_w, line_f

    ani = animation.FuncAnimation(
        fig, update, frames=len(fx), interval=100, blit=True
    )

    ani.save(os.path.join(OUTPUT_DIR, filename), writer='pillow')
    plt.close()


def save_step_debug(acc_mag, steps, filename):
    plt.figure()
    plt.plot(acc_mag)
    plt.scatter(steps, acc_mag[steps])
    plt.title("Step Detection")
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


# -------------------------
# MAIN PIPELINE
# -------------------------

print("Loading data...")

phone_df = load_and_resample(PHONE_ACC, PHONE_ORI, "phone")
watch_df = load_and_resample(WATCH_ACC, WATCH_ORI, "watch")

print("Aligning timestamps...")

common_index = phone_df.index.intersection(watch_df.index)
phone_df = phone_df.loc[common_index]
watch_df = watch_df.loc[common_index]

print("Building trajectories...")

px, py, p_acc, p_steps = build_trajectory(phone_df, "phone")
wx, wy, w_acc, w_steps = build_trajectory(watch_df, "watch")

print("Fusing trajectories...")

fx, fy = fuse_trajectories(px, py, wx, wy)

print("Saving plots...")

save_plot(px, py, "Phone Trajectory", "phone_trajectory.png")
save_plot(wx, wy, "Watch Trajectory", "watch_trajectory.png")
save_plot(fx, fy, "Fused Trajectory", "fused_trajectory.png")

save_comparison(px, py, wx, wy, fx, fy)

save_step_debug(p_acc, p_steps, "phone_steps.png")
save_step_debug(w_acc, w_steps, "watch_steps.png")

print("Creating animation...")

animate_trajectories(px, py, wx, wy, fx, fy, "trajectory.gif")

print(f"Done. Results saved in: {OUTPUT_DIR}")