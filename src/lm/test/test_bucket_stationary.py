"""Bucket swing tolerance must not relax box or robot stationary checks."""

from types import SimpleNamespace

import pytest

from lm.vml import VLMClientNode


@pytest.mark.parametrize("kind", ["bucket", "box", None])
@pytest.mark.parametrize("linear,angular", [(.15, .30), (.30, .60), (.301, .1), (.1, .601)])
@pytest.mark.parametrize("robot_speed", [0., .11])
def test_stationary_limits_follow_selected_object(kind, linear, angular, robot_speed):
    node = SimpleNamespace(
        _selected_object_type=kind, _has_monitor=True, _has_actual_box_pose=True,
        _robot_linear_speed=robot_speed, _robot_angular_speed=0.,
        _robot_linear_stationary_threshold_mps=.1,
        _robot_angular_stationary_threshold_radps=.15,
        _object_linear_speed=linear, _object_angular_speed=angular,
        _object_linear_stationary_threshold_mps=.15,
        _object_angular_stationary_threshold_radps=.30,
        _bucket_linear_stationary_threshold_mps=.30,
        _bucket_angular_stationary_threshold_radps=.60,
    )
    robot = robot_speed <= .1
    obj = linear <= (.30 if kind == "bucket" else .15) and angular <= (.60 if kind == "bucket" else .30)
    assert VLMClientNode._stationary_flags(node) == (robot, obj, robot and obj)

    node._has_actual_box_pose = False
    assert VLMClientNode._stationary_flags(node) == (robot, False, False)
