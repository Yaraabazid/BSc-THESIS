# Pure-Walking PDR

Pedestrian dead reckoning from a phone + smartwatch recorded with Sensor Logger,
tested on a rectangular indoor walk in the NU building at VU Amsterdam.

## Setup

- **Phone:** Samsung Galaxy S21, right pocket, port up, screen toward leg
- **Watch:** WearOS, left wrist
- **Route:** one clockwise rectangular loop, NU building

## What we tried

### OS yaw from `Orientation.csv` → didn't work
The `yaw` Euler angle assumes the phone is held upright. In a pocket the phone is
rotated ~90°, so Android maps the walking rotation onto roll/pitch instead — yaw
stays constant. Result: straight line going east, 133 m closure error.

### Gravity projection + per-step `atan2` in phone frame → didn't work
Project out gravity using `Gravity.csv`, take `atan2` of the horizontal acceleration
at each step peak. Two bugs: (1) with port-up orientation, gravity is along y so
`a_horiz_y ≈ 0` after projection — `atan2(~0, ax)` is always ~0. (2) Even with the
right axes, single-step `atan2` picks up body sway on every step, accumulating into
a full 360° drift. Result: smooth circle, ~20 m closure error but wrong shape.

### Quaternion → rotation matrix → forward axis projection → worked
Convert `Orientation.csv` quaternions to rotation matrices, project the phone's
forward axis into the world frame, take `atan2(north, east)`. Works regardless of
pocket orientation. To find the correct forward axis (not obvious for this pocket
orientation) we try all 6 candidates (±X/Y/Z) and pick the one whose total heading
change is closest to ±360° for the closed loop. Applied a 0.5 Hz low-pass filter to
remove step-frequency oscillations (~1.5 Hz) while keeping turns intact.

**Best result:** watch steps + phone quaternion heading → rough quadrilateral,
~21 m closure error.

## Known limitations

- **Magnetometer drift (~155° over 110 s):** the Android orientation filter uses the
  magnetometer for yaw. Indoor metal distorts the field, causing slow heading drift
  that can't be corrected in post-processing.
- **Step length uncalibrated (Weinberg K = 0.41):** PDR path is ~2× too small vs GPS.
  K needs calibrating against a measured corridor.
- **Indoor GPS accuracy (10–34 m):** useful as a shape reference, not ground truth.

## To do

- [ ] Calibrate Weinberg K against a measured corridor
- [ ] Implement gyro-integration + bias estimation (EKF) to replace magnetometer-dependent heading
- [ ] Record annotated routes with known waypoints for quantitative accuracy evaluation
- [ ] Activity recognition (walking / stairs / standing) using step signal + barometer
- [ ] Compare classical PDR vs MobilePoser on the same routes

