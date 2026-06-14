"""Pedestrian Dead Reckoning toolkit for Sensor Logger recordings.

Modules:
    io          : Load Sensor Logger CSV files into a Recording object.
    preprocess  : Resample sensors onto a common time base; align phone+watch.
    steps       : Step detection and step-length estimation (Weinberg).
    heading     : Heading estimation and EKF fusion (gyro + magnetometer).
    pdr         : Trajectory engine that walks 2D position forward step by step.
    altitude    : Barometric altitude for stairs.
    viz         : Static plotting helpers (matplotlib + contextily).
"""
from .io import Recording, load_recording
from .preprocess import resample_to, align_phone_watch, fix_watch_clock
from .steps import detect_steps, weinberg_step_length
from .heading import (HeadingEKF, integrate_gyro_heading, magnetometer_heading,
                       quat_to_R, heading_from_quaternion, select_forward_axis,
                       world_yaw_rate,
                       complementary_filter_attitude, heading_from_accel_gyro)
from .pdr import compute_trajectory, PDRConfig, PDRResult
from .altitude import pressure_to_altitude
from .pipeline import run_pipeline, PipelineResult
from . import viz

__all__ = [
    "Recording", "load_recording",
    "resample_to", "align_phone_watch", "fix_watch_clock",
    "detect_steps", "weinberg_step_length",
    "HeadingEKF", "integrate_gyro_heading", "magnetometer_heading",
    "quat_to_R", "heading_from_quaternion", "select_forward_axis",
    "world_yaw_rate",
    "complementary_filter_attitude", "heading_from_accel_gyro",
    "compute_trajectory", "PDRConfig", "PDRResult",
    "pressure_to_altitude",
    "run_pipeline", "PipelineResult",
    "viz",
]