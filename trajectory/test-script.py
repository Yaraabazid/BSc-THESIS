import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib.animation as animation

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "../data/walking-sitting/default"
OUTPUT_DIR = "output/walking-sitting"

STEP_LENGTH = 0.7
DT = 1/60

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# LOAD + RESAMPLE
# -------------------------

def load_and_resample(acc_path, ori_path, prefix):
    acc = pd.read_csv(acc_path)
    ori = pd.read_csv(ori_path)

    acc['timestamp'] = pd.to_datetime(acc['time'], unit='ns')
    ori['timestamp'] = pd.to_datetime(ori['time'], unit='ns')

    for col in ['time', 'seconds_elapsed']:
        if col in acc.columns:
            acc = acc.drop(columns=col)
        if col in ori.columns:
            ori = ori.drop(columns=col)

    acc = acc.set_index('timestamp')
    ori = ori.set_index('timestamp')

    acc = acc.resample('16.67ms').mean().interpolate()
    ori = ori.resample('16.67ms').mean().interpolate()

    acc = acc.add_prefix(prefix + "_acc_")
    ori = ori.add_prefix(prefix + "_ori_")

    return pd.concat([acc, ori], axis=1)

# -------------------------
# CORE
# -------------------------

def quaternion_to_yaw(qw, qx, qy, qz):
    siny = 2 * (qw*qz + qx*qy)
    cosy = 1 - 2 * (qy*qy + qz*qz)
    return np.arctan2(siny, cosy)

def detect_steps(acc_mag):
    peaks, _ = find_peaks(acc_mag, distance=20, prominence=1.0)
    return peaks

def build_trajectory(df, prefix):
    ax = df[f"{prefix}_acc_x"].values
    ay = df[f"{prefix}_acc_y"].values
    az = df[f"{prefix}_acc_z"].values

    qw = df[f"{prefix}_ori_qw"].values
    qx = df[f"{prefix}_ori_qx"].values
    qy = df[f"{prefix}_ori_qy"].values
    qz = df[f"{prefix}_ori_qz"].values

    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    steps = detect_steps(acc_mag)

    yaw = quaternion_to_yaw(qw, qx, qy, qz)

    x, y = [0.0], [0.0]

    for i in steps:
        x.append(x[-1] + STEP_LENGTH * np.cos(yaw[i]))
        y.append(y[-1] + STEP_LENGTH * np.sin(yaw[i]))

    return np.array(x), np.array(y), acc_mag, steps

# -------------------------
# ALIGNMENT
# -------------------------

def align_trajectory(ref_x, ref_y, x, y):
    n = min(len(ref_x), len(x))
    ref = np.vstack([ref_x[:n], ref_y[:n]]).T
    pts = np.vstack([x[:n], y[:n]]).T

    ref_mean = ref.mean(axis=0)
    pts_mean = pts.mean(axis=0)

    ref_c = ref - ref_mean
    pts_c = pts - pts_mean

    U, _, Vt = np.linalg.svd(pts_c.T @ ref_c)
    R = U @ Vt

    scale = np.linalg.norm(ref_c) / np.linalg.norm(pts_c)

    aligned = (pts_c @ R) * scale + ref_mean
    return aligned[:,0], aligned[:,1]

# -------------------------
# FUSION
# -------------------------

def fuse(px, py, wx, wy):
    n = min(len(px), len(wx))
    return 0.5*px[:n] + 0.5*wx[:n], 0.5*py[:n] + 0.5*wy[:n]

# -------------------------
# GPS
# -------------------------

def latlon_to_xy(lat, lon, lat0, lon0):
    R = 6378137
    x = (lon - lon0) * np.cos(np.radians(lat0)) * R
    y = (lat - lat0) * R
    return x, y

def xy_to_latlon(x, y, lat0, lon0):
    R = 6378137
    lat = y / R + lat0
    lon = x / (R * np.cos(np.radians(lat0))) + lon0
    return lat, lon

def load_gps(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['time'], unit='ns')
    df = df.set_index('timestamp')

    lat0 = df['latitude'].iloc[0]
    lon0 = df['longitude'].iloc[0]

    x, y = latlon_to_xy(df['latitude'], df['longitude'], lat0, lon0)
    df['x'], df['y'] = x, y

    return df, lat0, lon0

# -------------------------
# KALMAN
# -------------------------

class KF:
    def __init__(self):
        self.x = np.zeros((4,1))
        self.F = np.array([[1,0,DT,0],[0,1,0,DT],[0,0,1,0],[0,0,0,1]])
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.P = np.eye(4)
        self.Q = np.eye(4)*0.01
        self.R = np.eye(2)*5

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = z.reshape(2,1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def pos(self):
        return self.x[0,0], self.x[1,0]

def run_kf(px, py, timestamps, gps_df):
    kf = KF()
    traj_x, traj_y = [], []

    gps_t = gps_df.index
    gx, gy = gps_df['x'].values, gps_df['y'].values
    j = 0

    for i,t in enumerate(timestamps[:len(px)]):

        if i>0:
            kf.x[2] = px[i]-px[i-1]
            kf.x[3] = py[i]-py[i-1]

        kf.predict()

        if j < len(gps_t):
            if abs((t - gps_t[j]).total_seconds()) < 0.2:
                kf.update(np.array([gx[j], gy[j]]))
                j+=1

        x,y = kf.pos()
        traj_x.append(x)
        traj_y.append(y)

    return np.array(traj_x), np.array(traj_y)

# -------------------------
# PLOTTING
# -------------------------

def plot_compare(px, py, wx, wy, fx, fy):
    plt.figure()
    plt.plot(px, py, label="Phone")
    plt.plot(wx, wy, label="Watch")
    plt.plot(fx, fy, label="Fused")
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.title("Trajectory Comparison")
    plt.savefig(os.path.join(OUTPUT_DIR,"comparison.png"))
    plt.close()

def plot_steps(acc, steps, name):
    plt.figure()
    plt.plot(acc)
    plt.scatter(steps, acc[steps])
    plt.title(name)
    plt.savefig(os.path.join(OUTPUT_DIR,name))
    plt.close()

def plot_gps_map(gps_df):
    plt.figure()
    plt.plot(gps_df['longitude'], gps_df['latitude'])
    plt.xlabel("Lon")
    plt.ylabel("Lat")
    plt.title("GPS Trajectory")
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR,"gps_map.png"))
    plt.close()

def plot_kalman_vs_gps(kx, ky, gps_df, lat0, lon0):
    lat, lon = xy_to_latlon(kx, ky, lat0, lon0)

    plt.figure()
    plt.plot(gps_df['longitude'], gps_df['latitude'], label="GPS")
    plt.plot(lon, lat, label="Kalman")
    plt.legend()
    plt.title("Kalman vs GPS")
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR,"kalman_vs_gps.png"))
    plt.close()

def animate_all(px, py, fx, fy, kx, ky):
    fig, ax = plt.subplots()

    def update(i):
        ax.clear()
        ax.plot(px[:i], py[:i], label="PDR")
        ax.plot(fx[:i], fy[:i], label="Fused")
        ax.plot(kx[:i], ky[:i], label="Kalman")
        ax.legend()
        ax.grid()

    ani = animation.FuncAnimation(fig, update, frames=len(kx))
    ani.save(os.path.join(OUTPUT_DIR,"animation.gif"), writer="pillow")

# -------------------------
# MAIN
# -------------------------

phone_df = load_and_resample(
    os.path.join(DATA_DIR,"TotalAcceleration.csv"),
    os.path.join(DATA_DIR,"Orientation.csv"),
    "phone"
)

watch_df = load_and_resample(
    os.path.join(DATA_DIR,"WatchAccelerometer.csv"),
    os.path.join(DATA_DIR,"WatchOrientation.csv"),
    "watch"
)

common = phone_df.index.intersection(watch_df.index)
phone_df = phone_df.loc[common]
watch_df = watch_df.loc[common]

px, py, p_acc, p_steps = build_trajectory(phone_df,"phone")
wx, wy, w_acc, w_steps = build_trajectory(watch_df,"watch")

wx, wy = align_trajectory(px, py, wx, wy)
fx, fy = fuse(px, py, wx, wy)

gps_df, lat0, lon0 = load_gps(os.path.join(DATA_DIR,"WatchLocation.csv"))

kx, ky = run_kf(fx, fy, phone_df.index, gps_df)

# plots
plot_compare(px, py, wx, wy, fx, fy)
plot_steps(p_acc, p_steps, "phone_steps.png")
plot_steps(w_acc, w_steps, "watch_steps.png")
plot_gps_map(gps_df)
plot_kalman_vs_gps(kx, ky, gps_df, lat0, lon0)

animate_all(px, py, fx, fy, kx, ky)

print("Done. Check output folder.")