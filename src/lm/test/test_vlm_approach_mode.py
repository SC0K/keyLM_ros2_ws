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
    node._task_target_box_center = node._starting_box_center = None
    node._context_target_box_center = lambda: None
    return node


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
def test_approach_root_stops_030m_from_current_box_xy(xy):
    node = planner_at_distance(2.)
    node._current_box_center = np.array([*xy, .15])
    node._current_robot_quat_wxyz = np.array([1., 0., 0., 0.])
    center, quat = node._approach_root_pose()
    assert APPROACH_XY_OFFSET_M == .30
    np.testing.assert_allclose(np.linalg.norm(center[:2] - xy), .30)
    np.testing.assert_allclose(np.asarray(xy) - center[:2], .30 * _quat_wxyz_to_rotmat(quat)[:2, 0])
    if np.linalg.norm(xy) > 0:
        np.testing.assert_allclose(center[:2], np.asarray(xy) - .30 * np.asarray(xy) / np.linalg.norm(xy))
    assert center[2] == node._current_box_center[2]
    assert np.all(np.isfinite(quat))
    np.testing.assert_allclose(np.linalg.norm(quat), 1.)
