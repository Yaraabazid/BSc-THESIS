# Pure-Walking PDR

Pedestrian dead reckoning from a phone + smartwatch recorded with Sensor Logger,
tested on a rectangular indoor walk in the NU building at VU Amsterdam.

## Setup

- **Phone:** Samsung Galaxy S21, right pocket, port up, screen toward leg
- **Watch:** WearOS, left wrist
- **Route:** one clockwise rectangular loop, NU building

## What we tried

### Watch clock desync → `ValueError: no time overlap`
Sensor Logger derives `seconds_elapsed` from each device's own clock. The watch was
~17.6 days behind the phone, so `align_phone_watch` saw no overlap. Fixed by shifting
all watch DataFrames by the nanosecond offset between the two devices' first `time`
values. Added as `fix_watch_clock()` in `preprocess.py`.

### OS yaw from `Orientation.csv` → straight line east
The `yaw` Euler angle assumes the phone is held upright. In a pocket the phone is
rotated ~90° so Android maps walking rotation onto roll/pitch — yaw stays constant.
133 m closure error.

### Gravity projection + per-step `atan2` in phone frame → circle
Project out gravity, take `atan2` of horizontal acceleration at each step peak.
Two bugs: (1) with port-up orientation, gravity is along y so `a_horiz_y ≈ 0` —
`atan2(~0, ax)` is always ~0. (2) Single-step `atan2` picks up body sway on every
step, accumulating into a full 360° drift. Smooth circle, ~20 m closure error.

### Quaternion → rotation matrix → forward axis projection → worked
Convert orientation quaternions to rotation matrices, project the phone's forward axis
into the world frame, `atan2(north, east)`. Works regardless of pocket orientation.
Auto-select the forward axis by trying all ±X/Y/Z candidates and picking whichever
gives closest to ±360° total turn for the closed loop. Applied a 0.5 Hz low-pass
filter to remove step-frequency oscillations (~1.5 Hz) while preserving turns.  
**Result:** watch steps + phone quaternion heading → rough quadrilateral, ~21 m closure error.

### Raw `gyro_z` for EKF and gyro-only → wrong direction/magnitude
Feeding raw `phone.gyro['z']` (phone-frame z-axis rate) to the EKF and gyro
integrator gave the wrong sign and magnitude — the phone z-axis points sideways
out of the pocket, not toward world vertical. Fixed by rotating the full gyro vector
into the world frame using the orientation quaternion and taking the vertical component
(`world_yaw_rate()` in `heading.py`). After this fix:
- **Gyro-only: −380°** (correct direction, only ~20° drift over 110 s — good baseline)
- **EKF: still broken** — the magnetometer in the NU building is so distorted by
  steel structure that every compass update pulls the heading toward a false direction,
  overriding the correct gyro input. The `HeadingEKF` already has a `compass_mask`
  parameter for this; not yet implemented.

## Package (`pdr/`)

Changes made to the package to support arbitrary phone orientation:

| File | What was added |
|---|---|
| `preprocess.py` | `fix_watch_clock(watch, phone_t0_ns)` |
| `heading.py` | `quat_to_R()`, `heading_from_quaternion()`, `select_forward_axis()`, `world_yaw_rate()` |
| `viz.py` | `plot_gps_vs_pdr()` |

## Results summary

| Variant | Steps | Heading | Closure error |
|---|---|---|---|
| watch + quaternion | watch | quat + 0.5 Hz LP | ~21 m |
| phone + quaternion | phone | quat + 0.5 Hz LP | ~29 m |
| phone + gyro only | phone | world yaw rate integrated | ~12 m |
| phone + EKF | phone | world yaw rate + compass | ~105 m (compass corrupted) |

Gyro-only currently has the best closure error (~12 m) because the magnetometer
is too distorted to help. Quaternion has better shape (recognisably rectangular)
because it benefits from the OS orientation filter which smooths over short-term
gyro noise.

## Accel+gyro-only heading (no magnetometer at all)

Per supervisor feedback: the EKF above still uses the magnetometer. Added a second
filter that uses *only* the accelerometer and gyroscope — a complementary filter
where the gyro integrates attitude and the accelerometer (low-passed total
acceleration, gravity direction) corrects roll/pitch only. Since this correction is
orthogonal to yaw, heading is still purely gyro-driven, but the rotation matrix used
to project it into the world frame no longer depends on Android's (magnetometer-
influenced) fused quaternion at all.

**Result: this is the best method so far.** Closure error on the Walking route drops
to ~0.6 m (watch steps) / ~1.0 m (phone steps) — far better than gyro-only (~10–13 m)
or quaternion (~28–37 m). Implemented as `heading_from_accel_gyro()` in `heading.py`.

## Generalised pipeline + report script

The whole pipeline (clock fix, resampling, step detection, all 4 heading sources,
trajectories, altitude) is now wrapped in `pdr.run_pipeline(data_dir)`, which works
on any recording folder — tested on `Walking`, `Walking-4`, `Upstairs`, and
`Downstairs`.

`generate_report_plots.py` runs this on all four recordings and writes, per
recording, to `output/<name>/`: step detection, heading comparison, trajectory
comparison, GPS-vs-PDR, and a combined trajectory+altitude plot (for stairs). A
summary table (`output/summary.csv` / `.md`) collects closure error, total turn, and
distance for every method × recording.

**New finding from the stairs recordings:** the phone barometer (in-pocket) is very
noisy — large spikes from pocket pressure changes during steps — while the watch
barometer (wrist) gives a clean, monotonic altitude profile (~+10.5 m for Upstairs,
~−15 m for Downstairs). The watch barometer looks like the better altitude source.

**Walking-4** (a second, independent rectangular-walk recording) confirms the
accel+gyro result is not a fluke: closure error ~1.6 m (watch) / ~2.7 m (phone),
consistently best across both walking recordings.

## Known limitations

- **Magnetometer unusable indoors** in this building — both the EKF and the OS
  orientation quaternion drift because of it.
- **Step length uncalibrated** (Weinberg K = 0.41) — path is ~2× too small vs GPS.
- **Indoor GPS accuracy 10–34 m** — shape reference only, not ground truth.
- **Stairs heading is hard to validate** — the ±360° single-loop check used for
  Walking doesn't apply to switchback staircases, so forward-axis selection and
  total-turn sanity checks are less reliable for Upstairs/Downstairs.

## To do

- [ ] Calibrate Weinberg K against a measured corridor
- [ ] Use MobilePoser on the same recordings (Walking, Upstairs, Downstairs) and
  compare against this baseline — was future work, but the data is ready now
- [ ] Investigate map matching (https://en.wikipedia.org/wiki/Map_matching) to
  snap the estimated path onto the building floorplan once available
- [ ] Record annotated routes with known waypoints for quantitative accuracy evaluation
- [ ] Activity recognition (walking / stairs / standing) using step signal + barometer