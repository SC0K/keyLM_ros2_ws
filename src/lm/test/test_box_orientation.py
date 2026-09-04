"""Tests for target-local post-retarget orientation correction."""

import numpy as np
import pytest

from lm.box_orientation import (
    apply_target_box_orientation_offset,
    quat_wxyz_from_rpy_deg,
)


def _rotmat_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_zero_offset_is_an_exact_orientation_no_op() -> None:
    target = quat_wxyz_from_rpy_deg(np.array([21.0, -17.0, 43.0]))
    corrected = apply_target_box_orientation_offset(target, np.zeros(3))
    np.testing.assert_allclose(corrected, target, rtol=0.0, atol=1e-15)


def test_offset_is_postmultiplied_in_target_local_frame() -> None:
    target = quat_wxyz_from_rpy_deg(np.array([28.0, -13.0, 41.0]))
    offset = np.array([-7.0, 11.0, 19.0])
    corrected = apply_target_box_orientation_offset(target, offset)

    expected_rotation = (
        _rotmat_from_quat_wxyz(target)
        @ _rotmat_from_quat_wxyz(quat_wxyz_from_rpy_deg(offset))
    )
    np.testing.assert_allclose(
        _rotmat_from_quat_wxyz(corrected),
        expected_rotation,
        rtol=0.0,
        atol=1e-12,
    )


def test_vlm_latches_the_corrected_target_once() -> None:
    pytest.importorskip("rclpy")
    pytest.importorskip("lm_interfaces.srv")
    from lm.vml import VLMClientNode

    class _Logger:
        def info(self, _message: str) -> None:
            pass

    node = VLMClientNode.__new__(VLMClientNode)
    node._task_target_box_center = None
    node._task_target_box_quat_wxyz = None
    node._has_actual_box_pose = True
    node._has_robot_root_pose = True
    node._has_monitor = False
    node._default_target_box_quat_wxyz = quat_wxyz_from_rpy_deg(
        np.array([5.0, 7.0, 11.0])
    )
    node._target_box_orientation_offset_rpy_deg = np.array([1.0, 2.0, 13.0])
    node._update_box_forward_axis_from_robot_once = lambda: True
    node._default_task_target_box_center = lambda: np.array([1.0, 2.0, 0.15])
    node.get_logger = lambda: _Logger()
    node.publish_status = lambda *_args, **_kwargs: None
    node.box_forward_axis = "x"

    assert node.initialize_task_target_once()
    expected = apply_target_box_orientation_offset(
        node._default_target_box_quat_wxyz,
        node._target_box_orientation_offset_rpy_deg,
    )
    np.testing.assert_allclose(node._task_target_box_quat_wxyz, expected)

    latched = node._task_target_box_quat_wxyz.copy()
    node._target_box_orientation_offset_rpy_deg[:] = 90.0
    assert node.initialize_task_target_once()
    np.testing.assert_array_equal(node._task_target_box_quat_wxyz, latched)
