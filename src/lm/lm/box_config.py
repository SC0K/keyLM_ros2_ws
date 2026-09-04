"""Canonical box geometry for keyframe retargeting and visualization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_VALID_AXIS_LABELS = frozenset({"x", "-x", "y", "-y", "z", "-z"})


def parse_box_size_xyz(value) -> np.ndarray:
    """Parse and validate three positive XYZ box dimensions."""
    if isinstance(value, str):
        text = value.replace("[", " ").replace("]", " ").replace(",", " ")
        parts = text.split()
        arr = np.asarray([float(x) for x in parts], dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)

    if arr.size != 3:
        raise ValueError(f"box_size_xyz must contain 3 values, got {arr.tolist()}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"box_size_xyz must contain positive finite values, got {arr.tolist()}")
    return arr


def parse_orientation_offset_rpy_deg(value) -> np.ndarray:
    """Parse a finite roll/pitch/yaw correction expressed in degrees."""
    if isinstance(value, str):
        text = value.replace("[", " ").replace("]", " ").replace(",", " ")
        parts = text.split()
        arr = np.asarray([float(x) for x in parts], dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)

    if arr.size != 3:
        raise ValueError(
            "target_box_orientation_offset_rpy_deg must contain 3 values, "
            f"got {arr.tolist()}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "target_box_orientation_offset_rpy_deg must contain finite values, "
            f"got {arr.tolist()}"
        )
    return arr


@dataclass(frozen=True)
class BoxGeometry:
    """Static box dimensions and semantic frame conventions."""

    size_xyz: tuple[float, float, float]
    forward_axis: str
    up_axis: str

    def __post_init__(self) -> None:
        """Normalize and validate an immutable geometry profile."""
        size = parse_box_size_xyz(self.size_xyz)
        forward_axis = str(self.forward_axis).strip().lower()
        up_axis = str(self.up_axis).strip().lower()
        if forward_axis not in _VALID_AXIS_LABELS:
            raise ValueError(f"Unsupported box forward axis: {self.forward_axis}")
        if up_axis not in _VALID_AXIS_LABELS:
            raise ValueError(f"Unsupported box up axis: {self.up_axis}")
        if forward_axis.lstrip("-") == up_axis.lstrip("-"):
            raise ValueError("Box forward and up axes must use different dimensions")
        object.__setattr__(self, "size_xyz", tuple(float(value) for value in size))
        object.__setattr__(self, "forward_axis", forward_axis)
        object.__setattr__(self, "up_axis", up_axis)


# Edit these profiles to change retargeting geometry everywhere. Source poses
# remain in the keyframe NPZ files; current/target world poses arrive at runtime.
# 0.345, 0.250, 0.285
SOURCE_BOX_GEOMETRY = BoxGeometry(
    size_xyz=(0.350, 0.350, 0.350),
    forward_axis="y",
    up_axis="-z",
)
REAL_TARGET_BOX_GEOMETRY = BoxGeometry(
    size_xyz=(0.350, 0.350, 0.350),
    forward_axis="x",
    up_axis="z",
)
SIM_TARGET_BOX_GEOMETRY = BoxGeometry(
    size_xyz=(0.30, 0.30, 0.30),
    forward_axis="x",
    up_axis="z",
)
DEFAULT_TARGET_BOX_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)
# Optional post-retarget correction.  Keep zero so deployment and sim2sim use
# authored/observed physical-box orientations unless a launch explicitly opts in.
DEFAULT_TARGET_BOX_ORIENTATION_OFFSET_RPY_DEG = (0.0, 0.0, 0.0)

# Backward-compatible alias for code outside this workspace. New code should
# select REAL_TARGET_BOX_GEOMETRY or SIM_TARGET_BOX_GEOMETRY explicitly.
DEFAULT_BOX_SIZE_XYZ = SIM_TARGET_BOX_GEOMETRY.size_xyz


def format_box_size_xyz(value) -> str:
    """Return a stable ROS launch-parameter representation of an XYZ size."""
    size = parse_box_size_xyz(value)
    return " ".join(f"{component:.15g}" for component in size)


def format_orientation_offset_rpy_deg(value) -> str:
    """Return a stable ROS representation of an RPY correction in degrees."""
    rpy_deg = parse_orientation_offset_rpy_deg(value)
    return " ".join(f"{component:.15g}" for component in rpy_deg)
