"""Shared policy joint ordering and generated-standing posture for G1."""

import numpy as np

VLM_STANDING_LEAN_DEG = 5.0

POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


def generated_stand_joint_delta(lean_rad: float) -> np.ndarray:
    if not np.isfinite(lean_rad) or abs(lean_rad) > 0.52:
        raise ValueError("Standing waist lean must be finite and within +/-0.52 rad")
    delta = np.zeros(len(POLICY_JOINT_NAMES), dtype=np.float32)
    delta[POLICY_JOINT_NAMES.index("waist_pitch_joint")] = lean_rad
    return delta
