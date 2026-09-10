"""Object identity is selected by the VLM, never by whichever pose arrives last."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped
from pydantic import ValidationError
from std_msgs.msg import Empty
from std_srvs.srv import Trigger

from lm.keyframe_modes import MANIPULATION_PHASES, PLANNER_KEYFRAMES
from lm.tracked_objects import TrackedObjects
from lm.vlm_service import KeyframeDecision
from lm.vml import VLMClientNode
from lm.keyframe_retargeter_node import KeyframeRetargeterNode
from test_supervised_goals import controller, message


def pose(x):
    msg = PoseStamped()
    msg.header.frame_id = "world"
    msg.pose.position.x = x
    msg.pose.orientation.w = 1.
    return msg


@pytest.mark.parametrize("name", ["approach", *sorted(MANIPULATION_PHASES)])
def test_unsuffixed_actions_rejected_by_schema_and_retarget_service(name):
    with pytest.raises(ValidationError):
        KeyframeDecision(next_keyframe=name, object_in_manipulation=True, task_completion=False)
    node = KeyframeRetargeterNode.__new__(KeyframeRetargeterNode)
    result = node._on_retarget_keyframe_request(SimpleNamespace(keyframe_name=name), SimpleNamespace())
    assert not result.success
    assert set(KeyframeDecision.model_json_schema()["properties"]["next_keyframe"]["enum"]) == set(PLANNER_KEYFRAMES)


def test_tracking_is_separate_and_missing_or_invalid_never_falls_back():
    poses = TrackedObjects()
    assert poses.update("box", pose(1.))
    assert poses.get("bucket") is None
    assert poses.update("bucket", pose(4.))
    assert poses.get("box").pose.position.x == 1.
    bad = pose(float("nan"))
    assert not poses.update("bucket", bad)
    cached, _ = poses.poses["bucket"]
    poses.poses["bucket"] = (cached, 0.)
    assert poses.get("bucket") is None
    assert poses.get("box") is not None


@pytest.mark.parametrize("kind", ["box", "bucket"])
def test_normal_keyframe_routes_matching_mocap_without_extra_query(kind):
    node = VLMClientNode.__new__(VLMClientNode)
    node._selected_object_type = None
    node._tracked_objects = TrackedObjects()
    node._tracked_objects.update("box", pose(1.))
    node._tracked_objects.update("bucket", pose(4.))
    node.publish_status = Mock()
    node.send_request = Mock()
    node._on_actual_box_pose = Mock()
    assert node._route_keyframe_object(kind)
    assert node._selected_object_type == kind
    assert node._on_actual_box_pose.call_args.args[0].pose.position.x == (1. if kind == "box" else 4.)
    node.send_request.assert_not_called()


def test_task_reset_does_not_query_vlm(monkeypatch):
    node = VLMClientNode.__new__(VLMClientNode)
    node.send_request = Mock()
    node._reset_retarget_task_client = Mock()
    node._retarget_timeout_sec = 1.
    monkeypatch.setattr("lm.vml.rclpy.spin_until_future_complete", Mock())
    node.reset_retarget_task()
    node.send_request.assert_not_called()
    node._reset_retarget_task_client.call_async.assert_called_once()


def test_first_normal_action_is_retained_while_waiting_for_tracking(monkeypatch):
    node = VLMClientNode.__new__(VLMClientNode)
    node._selected_object_type = None
    node._tracked_objects = TrackedObjects()
    node._tracked_objects.update("box", pose(1.))
    node.publish_status = Mock()
    node.send_request = Mock()
    node._on_actual_box_pose = Mock()
    node.robot_and_object_stationary = Mock(side_effect=[False, True])
    response = SimpleNamespace(next_keyframe="approach_bucket")
    monkeypatch.setattr("lm.vml.rclpy.ok", lambda: True)
    def spin(*args, **kwargs):
        node._on_tracked_object("bucket", pose(4.))
    monkeypatch.setattr("lm.vml.rclpy.spin_once", spin)
    assert node.wait_for_first_goal_tracking(response)
    assert node._selected_object_type == "bucket"
    assert node.robot_and_object_stationary.call_count == 2
    assert response.next_keyframe == "approach_bucket"
    node.send_request.assert_not_called()
    assert node._on_actual_box_pose.call_args.args[0].pose.position.x == 4.


def test_first_request_needs_no_object_tracking_or_selection_query(monkeypatch):
    node = VLMClientNode.__new__(VLMClientNode)
    node.mocap_object_selection = True
    node._selected_object_type = None
    node._stationary_since = None
    node._stationary_hold_sec = .5
    node._stationary_flags = Mock(return_value=(True, False, False))
    monkeypatch.setattr("lm.vml.time.monotonic", lambda: 10.)
    assert not node.ready_for_next_request()
    monkeypatch.setattr("lm.vml.time.monotonic", lambda: 10.6)
    assert node.ready_for_next_request()
    context = json.loads(node.build_planner_context())
    assert context["selected_object_type"] is None
    assert "selection_only" not in context
    assert "available_objects" not in context
    assert context["distance_context"]["pick_within_horizontal_reach"] is None
    node._stationary_flags.return_value = (False, False, False)
    assert not node.ready_for_next_request()


def test_missing_chosen_object_does_not_route_available_other_object():
    node = VLMClientNode.__new__(VLMClientNode)
    node.mocap_object_selection = True
    node._selected_object_type = None
    node._tracked_objects = TrackedObjects()
    node._tracked_objects.update("box", pose(1.))
    node.publish_status = Mock()
    node.request_retargeted_keyframe = Mock()
    assert not node.publish_planner_outputs(SimpleNamespace(next_keyframe="approach_bucket"))
    assert node._selected_object_type == "bucket"
    assert not node._has_actual_box_pose
    node.request_retargeted_keyframe.assert_not_called()
    assert not node._route_keyframe_object("box")
    node._tracked_objects.update("bucket", pose(4.))
    node._on_actual_box_pose = Mock()
    assert node._route_keyframe_object("bucket")
    assert node._on_actual_box_pose.call_args.args[0].pose.position.x == 4.


def setup_tracked_controller(node):
    node.mocap_object_selection = True
    node._tracked_objects = TrackedObjects()
    node._active_object_type = "box"
    node.current_object_pos_w = np.zeros(3)
    node.current_object_quat_w = np.array([1., 0., 0., 0.])
    node._active_object_pose_publisher = Mock()
    node._on_tracked_object("box", pose(1.))
    node._on_tracked_object("bucket", pose(4.))


def test_supervised_object_switch_occurs_on_approval_not_preview(controller):
    setup_tracked_controller(controller)
    controller._on_keyframe_preview(message(kind="bucket"))
    assert controller._active_object_type == "box"
    assert controller.current_object_pos_w[0] == 1.
    controller._on_manual_goal_advance(Empty())
    assert controller._active_object_type == "bucket"
    assert controller.current_object_pos_w[0] == 4.
    controller._on_tracked_object("box", pose(2.))
    assert controller.current_object_pos_w[0] == 4.
    controller._on_tracked_object("bucket", pose(5.))
    assert controller.current_object_pos_w[0] == 5.
    assert controller._active_object_pose_publisher.publish.call_args.args[0].pose.position.x == 5.


def test_missing_selected_tracking_blocks_approval_without_switching(controller):
    setup_tracked_controller(controller)
    controller._on_keyframe_preview(message(kind="bucket"))
    msg, _ = controller._tracked_objects.poses["bucket"]
    controller._tracked_objects.poses["bucket"] = (msg, 0.)
    controller._on_manual_goal_advance(Empty())
    assert controller._active_object_type == "box" and not controller.have_goal
    controller._goal_approved_publisher.publish.assert_not_called()
    assert controller._pending_keyframe is not None


def test_retargeter_reset_clears_all_old_object_pose_latches():
    node = KeyframeRetargeterNode.__new__(KeyframeRetargeterNode)
    node._fixed_start_box_center = np.ones(3)
    result = node._on_reset_task(Trigger.Request(), Trigger.Response())
    assert result.success
    for field in ("_fixed_start_box_center", "_fixed_start_box_quat_wxyz", "_fixed_target_box_center",
                  "_fixed_target_box_quat_wxyz", "_fixed_box_hold_forward_axis"):
        assert getattr(node, field) is None


def test_colliding_pose_topics_rejected():
    from lm.tracked_objects import validate_object_topics
    with pytest.raises(ValueError, match="distinct"):
        validate_object_topics("/same", "/same", "/active")
    with pytest.raises(ValueError, match="distinct"):
        validate_object_topics("/box", "/bucket", "/box")


def test_bridge_drops_untracked_data_instead_of_refreshing_stale_pose():
    from builtin_interfaces.msg import Time
    from optitrack_msgs.msg import RigidbodyData
    from crl_g1_goalcontroller_python.rigidbody_to_pose_stamped import RigidbodyToPoseStamped
    node = RigidbodyToPoseStamped.__new__(RigidbodyToPoseStamped)
    node.require_tracking_valid = True
    node.publisher = Mock()
    node.frame_id = "world"
    node.get_clock = Mock()
    node.get_clock.return_value.now.return_value.to_msg.return_value = Time()
    msg = RigidbodyData(x=1., qw=1., params=0)
    node._on_rigidbody(msg)
    node.publisher.publish.assert_not_called()
    msg.params = 1
    node._on_rigidbody(msg)
    node.publisher.publish.assert_called_once()
    assert node.publisher.publish.call_args.args[0].pose.position.x == 1.
