"""Overlay the estimated Walking trajectory on the NU 5th-floor plan.

Produces figures/walking_floorplan_overlay.pdf.

The trajectory is placed on the plan with a SIMILARITY transform only --
rotation, uniform scale, and translation. A similarity transform cannot shear
or stretch, so the trajectory's shape is preserved exactly as estimated (it is
the same shape shown in the trajectory-comparison figure); only its position,
size, and orientation on the page are chosen to line it up with the corridor
loop around the inner courtyard. The placement parameters below were set by
visual alignment.

(An affine fit on four corners was tried first but rejected: affine allows
shear, which distorted the path to hit the corner points and made it look less
like the true estimate. Similarity is the honest choice here.)

Usage: edit FLOORPLAN / PLACEMENT and run from the trajectory/ directory.
"""
import sys
sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pdr import run_pipeline

FLOORPLAN = "../data/NU/NU-5th.png"
RECORDING = "../data/NU/Walking"
METHOD_KEY = "watch steps + accel_gyro_ekf"
OUT = "output/walking_floorplan_overlay.pdf"

# Similarity placement (set by visual alignment to the courtyard corridor):
SCALE = 15.00     # pixels per metre
ROT_DEG = 105.0    # clockwise rotation to match the building's orientation
TX, TY = 430, 495 # translation of the path centroid, in pixels


def main():
    res = run_pipeline(Path(RECORDING), fs=60.0, verbose=False)
    xy = res.trajectories[METHOD_KEY].xy.copy()
    xy[:, 1] = -xy[:, 1]                 # image y is downward
    p0 = xy - xy.mean(0)                 # centre the path

    th = np.radians(ROT_DEG)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    p = SCALE * (R @ p0.T).T
    p[:, 0] += TX
    p[:, 1] += TY

    arr = np.array(Image.open(FLOORPLAN).convert("RGB"))
    fig, ax = plt.subplots(figsize=(8.5, 9.5))
    ax.imshow(arr, alpha=0.45)
    ax.plot(p[:, 0], p[:, 1], color="#c0392b", lw=3.2, solid_capstyle="round",
            label="Estimated path (accel.+gyro EKF)")
    ax.scatter(*p[0], s=180, color="#27ae60", zorder=6,
               edgecolors="white", linewidth=2, label="Start / end")
    ax.annotate("", xy=p[8], xytext=p[0],
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=3))
    ax.set_title("Walking loop overlaid on the NU 5th-floor plan")
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()