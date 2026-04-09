# Sensor Logger to MobilePoser Pipeline

## Goal
Take raw CSVs from **Sensor Logger** (phone + watch) and feed them into the pre-trained **MobilePoser** model.

**Device Combination:** `lw_rp` = left-wrist (watch) + right-pocket (phone). This matches the pre-trained model's default mapping.

---

## Processing Pipeline
The notebook performs the following steps:

1. **Ingest:** Loads raw phone/watch accelerometer and orientation CSVs.
2. **Normalize:** Handles column naming differences between Default (Android) and Sensor Zoo modes.
3. **Convert:** Transforms quaternions into 3×3 rotation matrices.
4. **Synchronize:** Resamples both devices to a common 30 Hz timeline.
5. **Format:** Packages data into a (1, N, 60) tensor (5 sensors x [3 acc + 9 ori]).
6. **Inference:** Runs forward_offline() to calculate pose and translation.
7. **Visualize/Export:** Animates the skeleton and prepares Unity files.

---

## CLI Usage
After running the notebook through Cell 11, use these terminal commands for visualization:

### 1. Basic Visualization
**Pose only (in-place):**
`python visualize_sensor.py --npz ./data/sensor/default/processed/step0_output.npz`

**Pose with walking trajectory:**
`python visualize_sensor.py --npz ./data/sensor/default/processed/step0_output.npz --with-tran`

### 2. Advanced Options
**Run pipeline directly from CSVs (skips notebook):**
`python visualize_sensor.py --run-pipeline --recording-dir ./data/sensor/default --with-tran`

**Export to Unity:**
`python visualize_sensor.py --npz ... --with-tran --export-unity ./unity_output`

**Sanity Check (DIP test sequence):**
`python visualize_sensor.py --test-dip --seq-num 5 --with-tran`

---

## Troubleshooting & Verification

### Quality Check List
* **SMPL Viewer:** Best for verifying natural limb movement.
* **Unity Export:** Best for driving 3D avatars.
* **Matplotlib Plots:** Look for the following indicators:
    * Trajectory: Does the path look like your actual walk?
    * Speed: Normal walking is ~1.2-1.5 m/s.
    * Height: Should stay near 0 (floor level), not drift upward.
    * Foot Contact: Should alternate left/right during walking.

### Key Technical Specs
* **FPS:** 30 (Matches mobileposer/config.py).
* **Acc Scaling:** Acceleration is divided by 30.
* **Slot Mapping:** [0=left_wrist, 1=right_wrist, 2=left_thigh, 3=right_thigh, 4=head].
* **Combo lw_rp:** Uses slots [0, 3].
* **Note:** forward_offline() automatically handles velocity integration and floor-penetration removal.