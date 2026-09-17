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


def align_stand_to_current_feet(model, current_qpos, standing_qpos) -> np.ndarray:
    """Translate a stand in XY to match the live ankle midpoint, using FK only.

    Standing height, heading and all joint angles remain unchanged. This does
    not pin each foot separately or compensate for different stance widths.
    """
    import mujoco

    current = np.asarray(current_qpos, dtype=np.float64)
    standing = np.asarray(standing_qpos, dtype=np.float64).copy()
    if any(q.shape != (model.nq,) or not np.all(np.isfinite(q)) for q in (current, standing)):
        raise ValueError("Foot anchoring requires finite full-model qpos vectors")
    feet = [model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    data = mujoco.MjData(model)
    data.qpos[:] = current
    mujoco.mj_forward(model, data)
    current_midpoint = data.xpos[feet, :2].mean(axis=0).copy()
    data.qpos[:] = standing
    mujoco.mj_forward(model, data)
    standing[:2] += current_midpoint - data.xpos[feet, :2].mean(axis=0)
    return standing
