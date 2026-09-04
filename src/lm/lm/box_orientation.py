"""Pure quaternion helpers for optional target-box orientation correction."""

from __future__ import annotations

import numpy as np

from lm.box_config import parse_orientation_offset_rpy_deg


def normalize_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Normalize a finite WXYZ quaternion."""
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Cannot normalize a zero or non-finite quaternion")
    return quat / norm


def multiply_quat_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Return Hamilton product ``q1 * q2`` for WXYZ quaternions."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_wxyz_from_rpy_deg(rpy_deg: np.ndarray) -> np.ndarray:
    """Return WXYZ for ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` in degrees."""
    roll, pitch, yaw = np.deg2rad(
        parse_orientation_offset_rpy_deg(rpy_deg)
    )
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return normalize_quat_wxyz(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )
    )


def apply_target_box_orientation_offset(
    target_quat_wxyz: np.ndarray,
    offset_rpy_deg: np.ndarray,
) -> np.ndarray:
    """Post-rotate a target orientation about the retargeted box's local axes.

    The convention is intrinsic XYZ roll/pitch/yaw in degrees:
    ``q_corrected = q_retargeted * q_offset``.
    """
    target_quat = normalize_quat_wxyz(target_quat_wxyz)
    offset_quat = quat_wxyz_from_rpy_deg(offset_rpy_deg)
    return normalize_quat_wxyz(multiply_quat_wxyz(target_quat, offset_quat))
