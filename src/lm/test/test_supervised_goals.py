"""Approval gate tests: no ROS nodes, network inference, or hardware required."""

from io import BytesIO
from unittest.mock import Mock
from types import SimpleNamespace

import numpy as np
import pytest
from std_msgs.msg import Empty, MultiArrayDimension, String, UInt8MultiArray
from std_srvs.srv import SetBool

from crl_g1_goalcontroller_python import g1_keyframe_controller as control
from lm import vml
from lm.supervised_goal import PREVIEW_LEASE_SEC, preview_id
from lm.vlm_planner_app import VLMPlannerApp, planner_ros_arguments, PLANNER_EXTRA_DEFAULTS


def message(token="preview-1", manipulation=True):
    stream = BytesIO()
    np.savez(stream, goal_root_pos=[1., 2., .8], goal_root_mat_2x3=[1., 0., 0., 0., 1., 0.],
             goal_joint_pos_delta=np.zeros(29), goal_body_pos_b=np.zeros((14, 3)),
             goal_body_mat_b_3x2=np.zeros((14, 6)), masked_goal_object_pos=[1.4, 2., .15],
             masked_goal_object_quat=[1., 0., 0., 0.], object_to_manipulate=[manipulation])
    msg = UInt8MultiArray(data=list(stream.getvalue()))
    msg.layout.dim = [MultiArrayDimension(label=token)]
    return msg


@pytest.fixture
def controller():
    node = control.G1KeyframeController.__new__(control.G1KeyframeController)
    node.supervised_mode = True
    node._pending_keyframe = node._last_approved_preview_id = None
    node.default_angles = np.zeros(29)
    node.default_object_to_manipulate = True
    node.object_to_manipulate_override = None
    node.object_to_manipulate = False
    node.local_keyframes = node.prepend_default_goal_frame = node.append_default_goal_frame = False
    node.manual_goal_advance = False
    node.goal_sequence = np.zeros((1, 171))
    node.fixed_inserted_default_goals = {}
    node.have_goal = False
    node.current_fsm_state = control.GOAL_FSM_STATE
    node.goal_step_counter = 37
    node.current_goal_index = 0
    node.loop_goal_frames = False
    node.goal_hold_steps = 100
    node.get_logger = Mock(return_value=Mock())
    node._reset_policy_history = Mock()
    node._publish_keyframe_visualization = Mock()
    node._goal_approved_publisher = Mock()
    return node


def test_preview_does_not_change_policy_state_then_approves_exact_goal(controller):
    old_goal = controller.goal_sequence.copy()
    controller._on_keyframe_preview(message())
    pending = controller._pending_keyframe["goals"].copy()
    np.testing.assert_array_equal(controller.goal_sequence, old_goal)
    assert not controller.have_goal and not controller.object_to_manipulate
    assert controller.goal_step_counter == 37
    controller._reset_policy_history.assert_not_called()
    np.testing.assert_array_equal(controller._publish_keyframe_visualization.call_args.args[0], pending[0])
    controller._on_manual_goal_advance(Empty())
    np.testing.assert_array_equal(controller.goal_sequence, pending)
    assert controller.have_goal and controller.object_to_manipulate
    assert controller.goal_step_counter == 0
    assert controller._pending_keyframe is None
    controller._reset_policy_history.assert_called_once()
    assert controller._goal_approved_publisher.publish.call_args.args[0].data == "preview-1"
    controller._on_manual_goal_advance(Empty())
    controller._on_keyframe_preview(message())  # lost ACK retry
    controller._reset_policy_history.assert_called_once()
    assert controller._pending_keyframe is None


def test_locomotion_override_is_prepared_once_not_resampled_on_approval(controller):
    def walking(goals):
        goals = goals.copy()
        goals[:, 9:38] = .123
        goals[:, -7:] = 0
        return goals
    controller._replace_walking_goal_pose_with_default_noise = Mock(side_effect=walking)
    controller.object_to_manipulate = True  # keep old manipulation goal until approved
    controller._on_keyframe_preview(message(manipulation=False))
    preview = controller._pending_keyframe["goals"].copy()
    controller._on_keyframe_preview(message(manipulation=False))
    assert controller.object_to_manipulate
    controller._on_manual_goal_advance(Empty())
    assert not controller.object_to_manipulate
    np.testing.assert_array_equal(controller.goal_sequence, preview)
    assert not np.any(controller.goal_sequence[:, -7:])
    controller._replace_walking_goal_pose_with_default_noise.assert_called_once()


def test_expiry_cancel_and_wrong_fsm_do_not_activate(controller):
    controller._on_keyframe_preview(message())
    controller.current_fsm_state = -1
    controller._on_manual_goal_advance(Empty())
    assert controller._pending_keyframe is not None and not controller.have_goal
    controller.current_fsm_state = control.GOAL_FSM_STATE
    controller._pending_keyframe["updated"] -= PREVIEW_LEASE_SEC + 1
    controller._on_manual_goal_advance(Empty())
    assert controller._pending_keyframe is None and not controller.have_goal
    controller._on_keyframe_preview(message("new"))
    controller._on_cancel_preview(String(data="old"))
    assert controller._pending_keyframe is not None
    controller._on_cancel_preview(String(data="new"))
    controller._on_manual_goal_advance(Empty())
    assert not controller.have_goal
    controller._reset_policy_history.assert_not_called()


def test_direct_goals_cannot_bypass_supervision_and_automatic_mode_unchanged(controller):
    controller._on_keyframe(message())
    assert not controller.have_goal
    controller.supervised_mode = False
    controller._on_keyframe_preview(message())
    assert not controller.have_goal
    controller._on_keyframe(message())
    assert controller.have_goal and controller.object_to_manipulate


def test_invalid_preview_does_not_modify_active_goal(controller):
    controller._on_keyframe_preview(UInt8MultiArray(data=[1, 2, 3]))
    assert controller._pending_keyframe is None
    controller.local_keyframes = True
    controller._on_keyframe_preview(message())
    assert controller._pending_keyframe is None and not controller.have_goal


def test_monitor_uses_pending_pose_without_changing_active_goal(controller):
    controller._on_keyframe_preview(message())
    pending = controller._pending_keyframe["goals"][0].copy()
    controller._decode_compact_goal_world = Mock(return_value=([], [], np.zeros(3), np.zeros(4)))
    controller.get_clock = Mock()
    from builtin_interfaces.msg import Time
    controller.get_clock.return_value.now.return_value.to_msg.return_value = Time()
    controller.keyframe_visualization_publishers = [Mock()]
    control.G1KeyframeController._publish_keyframe_visualization(controller, np.zeros(171))
    np.testing.assert_array_equal(controller._decode_compact_goal_world.call_args.args[0], pending)
    assert not np.any(controller.goal_sequence)


def test_planner_waits_for_matching_ack_not_an_old_one(monkeypatch):
    node = vml.VLMClientNode.__new__(vml.VLMClientNode)
    node.publish_status = Mock()
    node._preview_pub = Mock()
    node._cancel_preview_pub = Mock()
    node._last_action_sent_time = 123.
    calls = []
    monkeypatch.setattr(vml.rclpy, "ok", lambda: True)
    def spin(*args, **kwargs):
        calls.append(1)
        assert node._last_action_sent_time == 123.
        token = preview_id(node._preview_pub.publish.call_args.args[0])
        node._on_goal_approved(String(data="stale" if len(calls) == 1 else token))
    monkeypatch.setattr(vml.rclpy, "spin_once", spin)
    assert node._wait_for_goal_approval(message(), "stand_before_pick")
    assert len(calls) == 2
    assert node.publish_status.call_args.args[0] == "awaiting_approval"
    assert node._cancel_preview_pub.publish.call_args.args[0].data == node._approved_preview_id


@pytest.mark.parametrize("have_active_goal", [False, True])
def test_policy_timer_never_consumes_pending_goal(controller, have_active_goal):
    controller.have_goal = have_active_goal
    controller.have_monitor = True
    controller.require_root_pose = False
    controller.have_object_pose = False
    controller._logged_default_goal_fallback = False
    controller.inserted_default_goal_indices = set()
    controller._default_goal_fallback = Mock(return_value=np.zeros(171))
    controller._build_observation = Mock(return_value=np.zeros(1))
    controller._publish_tracking_errors = Mock()
    controller._publish_control = Mock()
    controller.obs_history = np.zeros((2, 1))
    controller.input_name, controller.output_name = "obs", "act"
    controller.session = Mock()
    controller.session.run.return_value = [np.zeros(29)]
    controller.action_scale = 1.
    controller._on_keyframe_preview(message())
    controller._on_timer()
    np.testing.assert_array_equal(controller._build_observation.call_args.args[0], np.zeros(171))
    assert not controller.object_to_manipulate
    assert controller._pending_keyframe is not None
    controller._publish_control.assert_called_once()


def test_cannot_approve_a_preview_that_failed_to_display(controller):
    controller._publish_keyframe_visualization.side_effect = ValueError("invalid display pose")
    controller._on_keyframe_preview(message())
    controller._on_manual_goal_advance(Empty())
    assert controller._pending_keyframe is None and not controller.have_goal


@pytest.mark.parametrize("enabled", [False, True])
def test_mode_service_discards_preview_without_executing_it(controller, enabled):
    controller._on_keyframe_preview(message())
    controller.set_parameters = Mock(return_value=[SimpleNamespace(successful=True)])
    result = controller._on_set_supervised_mode(SetBool.Request(data=enabled), SetBool.Response())
    assert result.success and controller.supervised_mode == enabled
    assert controller._pending_keyframe is None and not controller.have_goal
    controller._reset_policy_history.assert_not_called()
    assert controller.set_parameters.call_args.args[0][0].value == enabled


@pytest.fixture
def gui():
    app = VLMPlannerApp.__new__(VLMPlannerApp)
    app.proc = None
    app._mode_sync = None
    app.task_text = Mock()
    app.task_text.get.return_value = "Pick up the box"
    app.supervised_selection = Mock()
    app.supervised_selection.get.return_value = True
    app.supervised_toggle = Mock()
    app.node = Mock()
    app.node.supervised_mode_client.service_is_ready.return_value = True
    app.node.supervised_mode_client.call_async.return_value.done.return_value = False
    app.bottom_status = Mock()
    app._append_status = Mock()
    app._launch_planner = Mock()
    app.root = Mock()
    return app


@pytest.mark.parametrize("enabled", [False, True])
def test_gui_waits_for_controller_confirmation_before_starting(gui, enabled):
    gui.supervised_selection.get.return_value = enabled
    gui.start_planner()
    gui._launch_planner.assert_not_called()
    assert gui.node.supervised_mode_client.call_async.call_args.args[0].data == enabled
    gui.supervised_toggle.configure.assert_called_with(state="disabled")
    future = gui.node.supervised_mode_client.call_async.return_value
    future.done.return_value = True
    future.result.return_value = SetBool.Response(success=True)
    gui._sync_supervised_mode()
    gui._launch_planner.assert_called_once_with("Pick up the box", enabled)
    assert gui._mode_sync is None


@pytest.mark.parametrize("failure", ["timeout", "rejected", "cancelled"])
def test_gui_does_not_start_after_mode_sync_failure_or_cancel(gui, failure):
    gui.start_planner()
    future = gui.node.supervised_mode_client.call_async.return_value
    if failure == "timeout":
        task, enabled, _, pending = gui._mode_sync
        gui._mode_sync = (task, enabled, 0., pending)
    elif failure == "rejected":
        future.done.return_value = True
        future.result.return_value = SetBool.Response(success=False, message="Rejected")
    else:
        gui.stop_planner()
        future.done.return_value = True
        future.result.return_value = SetBool.Response(success=True)
    gui._sync_supervised_mode()
    gui._launch_planner.assert_not_called()
    assert gui._mode_sync is None
    gui.supervised_toggle.configure.assert_called_with(state="normal")


def test_gui_cannot_change_controller_mode_while_planner_runs(gui):
    gui.proc = Mock()
    gui.proc.poll.return_value = None
    gui.start_planner()
    gui.node.supervised_mode_client.call_async.assert_not_called()
    gui._update_process_status()
    gui.supervised_toggle.configure.assert_called_with(state="disabled")


@pytest.mark.parametrize("enabled", [False, True])
def test_checkbox_overrides_launch_default_in_planner_arguments(enabled):
    params = dict(PLANNER_EXTRA_DEFAULTS, supervised_mode=not enabled)
    node = SimpleNamespace(get_parameter=lambda key: SimpleNamespace(value=params.get(key, "")))
    args = planner_ros_arguments(node, supervised_mode=enabled)
    assert f"supervised_mode:={str(enabled).lower()}" in args
    assert sum(arg.startswith("supervised_mode:=") for arg in args) == 1
