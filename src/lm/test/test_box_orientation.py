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


def test_vlm_latches_the_nominal_physical_target_once() -> None:
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
    node._update_box_forward_axis_from_robot_once = lambda: True
    node._default_task_target_box_center = lambda: np.array([1.0, 2.0, 0.15])
    node.get_logger = lambda: _Logger()
    node.publish_status = lambda *_args, **_kwargs: None
    node.box_forward_axis = "x"

    assert node.initialize_task_target_once()
    np.testing.assert_allclose(
        node._task_target_box_quat_wxyz,
        node._default_target_box_quat_wxyz,
    )

    latched = node._task_target_box_quat_wxyz.copy()
    node._default_target_box_quat_wxyz = quat_wxyz_from_rpy_deg(
        np.array([90.0, 90.0, 90.0])
    )
    assert node.initialize_task_target_once()
    np.testing.assert_array_equal(node._task_target_box_quat_wxyz, latched)


@pytest.mark.parametrize("preinitialized_sim_target", [False, True])
def test_bucket_task_keeps_observed_heading_instead_of_default_or_nearest_face(preinitialized_sim_target):
    from unittest.mock import Mock
    from lm.vml import VLMClientNode

    node = VLMClientNode.__new__(VLMClientNode)
    node._selected_object_type = None if preinitialized_sim_target else "bucket"
    node._task_target_box_center = np.array([2., 1., .15]) if preinitialized_sim_target else None
    node._task_target_box_quat_wxyz = np.array([1., 0., 0., 0.]) if preinitialized_sim_target else None
    node._has_actual_box_pose = node._has_robot_root_pose = True
    start = np.array([1., 1., .02])
    observed = quat_wxyz_from_rpy_deg(np.array([0., 0., 73.]))
    node._fixed_start_box_pose = lambda: (start.copy(), observed.copy())
    node._default_task_target_box_center = lambda: np.array([2., 1., .02])
    node._default_target_box_quat_wxyz = np.array([1., 0., 0., 0.])
    node._box_forward_axis_initialized_from_robot = True
    node.box_forward_axis = "-y"
    node._stand_before_pick_root_pose = Mock(side_effect=AssertionError("No bucket face-to-centre alignment"))
    node.get_logger = Mock(return_value=Mock())
    node.publish_status = Mock()

    assert node.initialize_task_target_once("bucket")
    np.testing.assert_allclose(node._task_target_box_quat_wxyz, observed)
    np.testing.assert_allclose(node._task_target_box_center, [2., 1., .02])
    assert node.box_forward_axis == "x"
    # Later observations must not make a fixed placement target drift.
    latched = node._task_target_box_quat_wxyz.copy()
    observed[:] = [1., 0., 0., 0.]
    assert node.initialize_task_target_once("bucket")
    np.testing.assert_array_equal(node._task_target_box_quat_wxyz, latched)


@pytest.mark.parametrize("source_height", [0.73, 0.9336])
def test_vlm_retargeter_applies_offset_only_after_robot_ik(source_height) -> None:
    """The VLM path sends physical orientation to IK and offsets only its goal."""
    pytest.importorskip("rclpy")
    pytest.importorskip("lm_interfaces.srv")
    from lm.keyframe_retargeter_node import KeyframeRetargeterNode

    node = KeyframeRetargeterNode.__new__(KeyframeRetargeterNode)
    node._target_box_center = np.array([1.2, -0.4, 0.15])
    node._target_box_quat_wxyz = quat_wxyz_from_rpy_deg(
        np.array([4.0, -6.0, 21.0])
    )
    node._target_box_orientation_offset_rpy_deg = np.array([7.0, 2.0, -13.0])

    ik_call: dict[str, np.ndarray] = {}

    def capture_ik(
        _payload: dict[str, np.ndarray],
        center: np.ndarray,
        quat_wxyz: np.ndarray,
    ) -> None:
        ik_call["center"] = np.asarray(center).copy()
        ik_call["quat_wxyz"] = np.asarray(quat_wxyz).copy()

    node._apply_box_ik = capture_ik
    payload = {
        "object_position_xyz": np.array([0.3, 0.0, source_height], dtype=np.float32),
        "object_quat_wxyz": np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        ),
        "dof_positions": np.arange(29, dtype=np.float32),
        "body_positions": np.arange(12, dtype=np.float32).reshape(4, 3),
        "body_rotations": np.arange(16, dtype=np.float32).reshape(4, 4),
    }
    robot_fields_before = {
        key: payload[key].copy()
        for key in ("dof_positions", "body_positions", "body_rotations")
    }

    assert (
        node._retarget_for_box_task("stand_before_place", payload)
        == "ik_to_box_above_place_target"
    )
    np.testing.assert_allclose(
        ik_call["quat_wxyz"], node._target_box_quat_wxyz, atol=0.0
    )
    np.testing.assert_allclose(ik_call["center"], [1.2, -0.4, source_height])
    np.testing.assert_allclose(payload["object_position_xyz"][2], source_height)
    expected_policy_quat = apply_target_box_orientation_offset(
        node._target_box_quat_wxyz,
        node._target_box_orientation_offset_rpy_deg,
    )
    np.testing.assert_allclose(
        _rotmat_from_quat_wxyz(payload["object_quat_wxyz"]),
        _rotmat_from_quat_wxyz(expected_policy_quat),
        atol=1e-7,
    )
    for key, expected in robot_fields_before.items():
        np.testing.assert_array_equal(payload[key], expected)
