"""Pickup approach mode is fixed locally, independent of VLM output or range."""

from types import SimpleNamespace

import numpy as np
import pytest

from lm.keyframe_modes import MANIPULATION_KEYFRAMES
from lm.vml import APPROACH_XY_OFFSET_M, VLMClientNode, _quat_wxyz_to_rotmat


def planner_at_distance(distance):
    node = VLMClientNode.__new__(VLMClientNode)
    node._has_robot_root_pose = node._has_monitor = True
    node._has_actual_box_pose = distance is not None
    node._current_robot_center = np.zeros(3)
    node._current_box_center = np.array([distance or 0., 0., 10.])
    node._pick_max_horizontal_distance_m = .45
    node._bucket_pick_max_horizontal_distance_m = .60
    node._task_target_box_center = node._starting_box_center = None
    node._context_target_box_center = lambda: None
    return node


@pytest.mark.parametrize("kind,threshold", [("box", .45), ("bucket", .60)])
@pytest.mark.parametrize("distance", [.45, .4501, .55, .60, .6001, None])
def test_object_specific_reach_threshold(kind, threshold, distance):
    node = planner_at_distance(distance)
    node._selected_object_type = kind
    context = node._distance_context()
    assert context["pick_max_horizontal_distance_m"] == threshold
    assert context["pick_within_horizontal_reach"] == (distance is not None and distance <= threshold)


@pytest.mark.parametrize("phase", ["approach", "stand_before_pick"])
def test_bucket_reach_success_and_switch_back_to_box(phase):
    node = planner_at_distance(.55)
    node._tracking_metric = lambda _: 1.0
    node._object_error_to_last_target = lambda: (1.0, None)
    node._mean_body_success_threshold_m = node._root_position_success_threshold_m = .3
    node._root_orientation_success_threshold_rad = .8
    node._object_position_success_threshold_m = .45
    for kind, expected in [("bucket", True), ("box", False)]:
        node._selected_object_type = kind
        node._last_action_name = f"{phase}_{kind}"
        assert node.evaluate_last_action_success() is expected
    node._selected_object_type = "bucket"
    node._bucket_pick_max_horizontal_distance_m = .50
    assert not node._distance_context()["pick_within_horizontal_reach"]


@pytest.mark.parametrize("distance", [2., .4501, .45, .4, None])
@pytest.mark.parametrize("vlm_flag", [True, False])
def test_approach_is_always_locomotion_regardless_of_range_or_vlm_flag(distance, vlm_flag):
    node = planner_at_distance(distance)
    response = SimpleNamespace(next_keyframe="approach", object_in_manipulation=vlm_flag)
    assert node._effective_object_to_manipulate(response) is False


@pytest.mark.parametrize("name", sorted(MANIPULATION_KEYFRAMES))
def test_remaining_actions_stay_object_aware(name):
    assert name != "approach"
    node = planner_at_distance(3.)
    assert node._effective_object_to_manipulate(SimpleNamespace(next_keyframe=name, object_in_manipulation=False))


@pytest.mark.parametrize("name,expected", [("approach", True), ("stand_before_pick", False)])
def test_only_approach_omits_object_tracking_from_success(name, expected):
    node = planner_at_distance(2.)
    node._last_action_name = name
    node._tracking_metric = lambda _: .01
    node._object_error_to_last_target = lambda: (None, None)
    node._mean_body_success_threshold_m = node._root_position_success_threshold_m = .3
    node._root_orientation_success_threshold_rad = .8
    node._object_position_success_threshold_m = .45
    assert node.evaluate_last_action_success() is expected
    assert ("object_position_error_m" in node._action_success_checks) is (name != "approach")


def test_approach_reuses_stand_library_and_is_an_allowed_vlm_action():
    from lm.keyframe_modes import source_keyframe_name
    from lm.vlm_service import KeyframeDecision, SYSTEM_PROMPT

    assert source_keyframe_name("approach") == "stand_before_pick"
    assert KeyframeDecision(next_keyframe="approach_box", object_in_manipulation=False, task_completion=False).next_keyframe == "approach_box"
    assert "choose approach" in SYSTEM_PROMPT


@pytest.mark.parametrize("xy", [[2., 1.], [-2., 1.], [0., 0.]])
@pytest.mark.parametrize("yaw", [0., .7, -1.4])
@pytest.mark.parametrize("axis,local_forward", [("x", [1., 0., 0.]), ("-x", [-1., 0., 0.]),
                                               ("y", [0., 1., 0.]), ("-y", [0., -1., 0.])])
def test_approach_root_stops_030m_along_observed_object_axis(xy, yaw, axis, local_forward):
    node = planner_at_distance(2.)
    node._current_box_center = np.array([*xy, .15])
    node._current_robot_quat_wxyz = np.array([1., 0., 0., 0.])
    node._current_box_quat_wxyz = np.array([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)])
    node.box_forward_axis = axis
    center, quat = node._approach_root_pose()
    assert APPROACH_XY_OFFSET_M == .30
    np.testing.assert_allclose(np.linalg.norm(center[:2] - xy), .30)
    np.testing.assert_allclose(np.asarray(xy) - center[:2], .30 * _quat_wxyz_to_rotmat(quat)[:2, 0], atol=1e-8)
    forward = (_quat_wxyz_to_rotmat(node._current_box_quat_wxyz) @ local_forward)[:2]
    np.testing.assert_allclose(center[:2], np.asarray(xy) - .30 * forward, atol=1e-8)
    assert center[2] == node._current_box_center[2]
    assert np.all(np.isfinite(quat))
    np.testing.assert_allclose(np.linalg.norm(quat), 1.)


@pytest.mark.parametrize("kind", ["box", "bucket"])
@pytest.mark.parametrize("robot_xy,expected", [([-2., 0.], "x"), ([2., 0.], "-x"),
                                               ([0., -2.], "y"), ([0., 2.], "-y")])
@pytest.mark.parametrize("yaw", [0., .7])
def test_both_objects_infer_and_latch_nearest_pickup_axis(kind, robot_xy, expected, yaw):
    from unittest.mock import Mock
    node = planner_at_distance(1.)
    node._selected_object_type = kind
    node._box_forward_axis_initialized_from_robot = False
    node._current_box_center = np.array([.4, .2, .15])
    node._current_box_quat_wxyz = np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])
    rotation = _quat_wxyz_to_rotmat(node._current_box_quat_wxyz)
    node._current_robot_center = node._current_box_center + rotation @ [*robot_xy, .65]
    node._box_size_xyz = np.array([.35, .35, .35])
    node._stand_before_pick_distance_m = .4
    node._min_stand_root_height_m = .78
    node._default_target_root_center = np.array([0., 0., .8])
    node.get_logger = Mock(return_value=Mock())
    assert node._update_box_forward_axis_from_robot_once()
    assert node.box_forward_axis == expected
    node._current_robot_center[:2] *= -1
    assert node._update_box_forward_axis_from_robot_once()
    assert node.box_forward_axis == expected
