# Step 0: Observations & Findings

## Recording Setup
- **App:** Sensor Logger v1.55.0 on Samsung SM-S911B (Android 36)
- **Watch:** WearOS smartwatch, synced via Phone & Watch mode
- **Activity:** ~25s walking, ~10s sitting, annotated manually
- **Two versions:** Default (Android OS fusion) and Sensor Zoo (Madgwick orientation, Adaptive Threshold steps, Tilt-Compensated compass)
- **Requested sampling rate:** 100Hz
- **Standardisation:** Enabled

---

## Key Findings

### 1. Actual Sampling Rates ≠ Requested

| Sensor | Requested | Actual (Phone) | Actual (Watch) |
|--------|-----------|----------------|----------------|
| Accelerometer/Gyroscope | 100 Hz | ~54 Hz | ~73 Hz |
| Orientation | 100 Hz | ~52 Hz | ~71 Hz |
| Magnetometer/Compass | 100 Hz | ~100 Hz | ~70 Hz |
| Barometer    | Max | 25 Hz | ~10 Hz |

**Implication:** The phone's IMU hardware caps at ~54Hz regardless of the requested rate. This is close to MobilePoser's required 60Hz, so we can interpolate up slightly rather than downsample. Watch samples faster (~73Hz) so phone is the bottleneck. All sensors need to be resampled to a common rate for synchronization.

### 2. Walking vs. Sitting is Clearly Distinguishable

In both recording versions:
- **Accelerometer:** Walking shows rhythmic spikes (step impacts), sitting is nearly flat
- **Gyroscope:** Walking shows periodic rotation patterns (body sway), sitting is quiet
- **Barometer:** Flat for same-floor walking (as expected — no altitude change in this test)

**Implication:** Activity recognition from raw sensor data should be achievable even with simple classifiers. The signal difference is visually obvious.

### 3. Default vs. Sensor Zoo: Different Noise Profiles

| Sensor | Default (Android OS) | Sensor Zoo (Madgwick) |
|--------|---------------------|----------------------|
| Accelerometer | Smoother (OS applies filtering) | Noisier (raw-er signal) |
| Gyroscope | Smoother | Noisier |
| Orientation | Noisier | Smoother (Madgwick fusion) |

**Explanation:** Android's built-in fusion applies proprietary smoothing to accelerometer and gyroscope outputs, making them appear cleaner but potentially hiding real dynamics. Madgwick doesn't filter the raw IMU signals — it only fuses accel+gyro+magnetometer for orientation estimation, where it produces a cleaner result.

**Implication for dead reckoning:** We could mix approaches — use default (smoother) accel/gyro for step detection, but Madgwick (smoother) orientation for heading. Or test both end-to-end and compare.

### 4. Column Order Differs Between Versions

| File | Default columns | Sensor Zoo columns |
|------|----------------|--------------------|
| Accelerometer | z, y, x | x, y, z |
| Orientation quaternion | qz, qy, qx, qw | qw, qx, qy, qz |

**Implication:** Preprocessing must handle column ordering explicitly by name, not position.

### 5. Built-in Activity Detection is Insufficient

Sensor Logger's Activity.csv only detected "walking" and "unknown" — it never detected "sitting." The detection is too coarse and infrequent (only 4-5 entries for a 35s recording) to be useful for our activity recognition task.

**Implication:** Manual annotations via presets are essential for ground truth. The built-in activity detection cannot replace them.

### 6. Wi-Fi Scans All Nearby Access Points ✓

- 56 unique BSSIDs detected across 2 scan intervals
- 23 APs per scan on average
- Includes SSID, BSSID, signal level (RSSI), and frequency
- Scans are infrequent (~2 per 40s recording)

**Implication:** Wi-Fi RSSI fingerprinting is possible with this setup for future work. However, scan frequency is low (~1 per 20s), so it can only provide occasional absolute position corrections, not continuous tracking.

### 7. MobilePoser Compatibility

| Requirement | Available? | Notes |
|-------------|-----------|-------|
| Phone acceleration (3-axis) | ✓ | TotalAcceleration.csv (includes gravity) |
| Phone orientation (quaternion) | ✓ | Orientation.csv — needs conversion to rotation matrix |
| Watch acceleration (3-axis) | ✓ | WatchAccelerometer.csv |
| Watch orientation (quaternion) | ✓ | WatchOrientation.csv — needs conversion to rotation matrix |
| 60 Hz sampling | ~ | Actual ~54Hz phone, ~73Hz watch — resample to 60Hz |
| Synchronized timestamps | ✓ | Same epoch time base |

**Conversion needed:**
1. Quaternion → 3×3 rotation matrix
2. Resample all sensors to common 60Hz timeline
3. Align phone and watch data to same timestamps
4. Verify coordinate system conventions match MobilePoser's expectations (check source code)

### 8. Pedometer Counts Differ

For similar-length recordings, Default and Sensor Zoo step counters gave different totals. This needs further investigation with a counted ground truth (actually count your steps next time).

---

## Data Summary

| Category | Phone Sensors | Watch Sensors |
|----------|--------------|---------------|
| Motion | Accelerometer, Gyroscope, Gravity, TotalAcceleration (+ uncalibrated variants) | Accelerometer, Gyroscope, Gravity, TotalAcceleration |
| Orientation | Orientation (quaternion + Euler), Magnetometer, Compass | Orientation (quaternion + Euler), Magnetometer |
| Environment | Barometer (25Hz), GPS/Location | Barometer, Location |
| Connectivity | Wi-Fi (all APs, ~1 scan/20s), Network | — |
| Activity | Pedometer, Activity (coarse), Annotations (manual) | — |

---