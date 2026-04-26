# Dead Reckoning Toolkit

A reusable Python module for pedestrian dead reckoning (PDR) from Sensor
Logger smartphone and smartwatch recordings, plus a Jupyter notebook that
walks through walking and stairs analyses end-to-end.

## What's here

```
dead_reckoning.ipynb       Notebook: load → step detect → EKF → trajectory → map
pdr/                        Reusable Python package
├── __init__.py             Public API
├── io.py                   Load Sensor Logger CSVs into a Recording object
├── preprocess.py           Resample sensors, align phone + watch
├── steps.py                Step detection + Weinberg step-length estimation
├── heading.py              Gyro integration, magnetometer compass, 1-D Kalman EKF
├── pdr.py                  2-D trajectory engine (step + heading → x, y)
├── altitude.py             Barometric pressure → altitude (for stairs)
└── viz.py                  matplotlib + contextily visualisations
```

## Install dependencies

```bash
pip install numpy pandas scipy matplotlib contextily pyproj
```

`contextily` is only needed for the map overlay; everything else is plain
SciPy stack. The map fetch needs network on first call — tiles are
cached locally afterwards.

## Quick start

```python
from pdr import (load_recording, align_phone_watch, detect_steps,
                 HeadingEKF, magnetometer_heading, compute_trajectory, viz)

rec = load_recording('data/walking-sitting/default')
phone, watch = align_phone_watch(rec.phone, rec.watch, fs=60.0)

# Steps from total acceleration magnitude
accel = phone.accel_total[['x', 'y', 'z']].to_numpy()
t     = phone.accel_total['seconds_elapsed'].to_numpy()
steps = detect_steps(accel, fs=60.0, seconds_elapsed=t,
                     min_peak_height=1.2, weinberg_k=0.41,
                     use_total_accel=True)

# Heading from gyro (predict) + compass (update)
gz   = phone.gyro['z'].to_numpy()
mag  = phone.magnet[['x','y','z']].to_numpy()
grav = phone.gravity[['x','y','z']].to_numpy()
comp = magnetometer_heading(mag, gravity_xyz=grav)
heading = HeadingEKF(Q=1e-4, R=0.25).run(gz, comp, fs=60.0)

# Trajectory and plot
result = compute_trajectory(steps, heading, t)
viz.plot_trajectory(result, label='phone PDR')
```

## Calibration tips

- **Step length**: walk a known distance (e.g. a 20 m corridor, count
  steps), then adjust `weinberg_k` so `result.total_distance` matches
  reality. Generic K = 0.41 is a starting point; it varies by user
  height, pace, and how the phone is held.
- **Step threshold**: `min_peak_height` should be raised for hand-held
  phones (impacts are damped) and lowered for chest-mounted setups.
- **EKF tuning**: increase `R` to trust the compass less (less
  magnetic-disturbance reaction); increase `Q` to trust the gyro less.

## Verifying the pipeline

The included `test_synthetic.py` (in the development repo) generates a
4-leg square walk, runs the full pipeline, and asserts the trajectory
closes. Sample output:

```
Steps detected: 40  (expected: 40)
Final heading error — gyro only: 3.7°,  EKF: 0.9°
Closure error: ground truth 0.01 m, EKF 0.12 m, gyro only 0.21 m
```

The EKF reduces both heading error and trajectory closure compared to
gyro-only, which is the expected sign of a working magnetometer update.
