"""Resolve combined launch wiring without starting ROS, GUIs, SSH or hardware."""

import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.utilities import evaluate_parameters, normalize_parameters

from lm.vlm_planner_app import VLMPlannerApp, build_arg_parser


@pytest.fixture
def launch_module(monkeypatch, tmp_path):
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "ros_logs"))
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "launch/vlm_experiment_launch.py"))


def context_for(module, **overrides):
    context = LaunchContext()
    context.launch_configurations.update(overrides)
    for action in module["generate_launch_description"]().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
        elif isinstance(action, IncludeLaunchDescription):
            # Apply include overrides and declarations only. Never visit nodes.
            for name, value in action.launch_arguments:
                context.launch_configurations[name] = perform_substitutions(context, normalize_to_list_of_substitutions(value))
            description = action.launch_description_source.get_launch_description(context)
            for declaration in description.entities:
                if isinstance(declaration, DeclareLaunchArgument):
                    declaration.execute(context)
    return context


@pytest.mark.parametrize("mode,backend,monitor,robot_launch", [
    ("sim", "mujoco", "/g1_sim/monitor", "g1_keyframe_sim.py"),
    ("real", "usb", "/g1_hardware/monitor", "g1_keyframe.py"),
])
def test_mode_wiring_without_duplicate_planner(launch_module, monkeypatch, mode, backend, monitor, robot_launch):
    context = context_for(launch_module, mode=mode, start_client="true")
    assert context.launch_configurations["camera_backend"] == backend
    assert context.launch_configurations["monitor_topic"] == monitor
    assert context.launch_configurations["start_client"] == "false"
    app_fn = launch_module["_planner_app"]
    monkeypatch.setitem(app_fn.__globals__, "Node", lambda **kwargs: kwargs)
    app = app_fn(context)[0]
    assert ("--real" in app["arguments"]) == (mode == "real")
    params = evaluate_parameters(context, normalize_parameters(app["parameters"]))[0]
    assert params["monitor_topic"] == monitor
    assert params["robot_root_pose_topic"] == ""
    include = launch_module["_robot_launch"](context)[0]
    include.launch_description_source.get_launch_description(context)
    assert include.launch_description_source.location.endswith(robot_launch)


def test_custom_topics_server_and_external_tunnel(launch_module, monkeypatch):
    context = context_for(
        launch_module, mode="real", server="case", manage_tunnel="false", start_robot="false",
        local_port="11435", actual_box_pose_topic="/custom/box", retargeted_keyframe_topic="/custom/goals",
        retarget_keyframe_service="/custom/retarget", box_size_xyz="0.3 0.4 0.5",
        box_hold_forward_axis="y", task_text="Place the box.", stand_before_pick_distance_m="0.42",
    )
    app_fn = launch_module["_planner_app"]
    monkeypatch.setitem(app_fn.__globals__, "Node", lambda **kwargs: kwargs)
    app = app_fn(context)[0]
    assert "--no-tunnel" in app["arguments"]
    assert app["arguments"][:4] == ["--server", "case", "--local-port", "11435"]
    assert "default_box_forward_axis:='y'" in app["arguments"]
    assert "Place the box." in app["arguments"]
    assert context.launch_configurations["ollama_host"] == "http://localhost:11435"
    params = evaluate_parameters(context, normalize_parameters(app["parameters"]))[0]
    assert params["actual_box_pose_topic"] == "/custom/box"
    assert params["retargeted_keyframe_topic"] == "/custom/goals"
    assert params["retarget_keyframe_service"] == "/custom/retarget"
    assert tuple(params["box_size_xyz"]) == (.3, .4, .5)
    assert params["stand_before_pick_distance_m"] == .42
    assert launch_module["_robot_launch"](context) == []


def test_gui_task_and_shutdown_are_explicit_and_idempotent():
    assert build_arg_parser().parse_args(["--task", "Pick up the box."]).task == "Pick up the box."
    app = VLMPlannerApp.__new__(VLMPlannerApp)
    app.proc, app.tunnel_proc, app.root = Mock(), Mock(), Mock()
    app.proc.poll.return_value = app.tunnel_proc.poll.return_value = None
    app.shutdown()
    app.shutdown()
    for proc in (app.proc, app.tunnel_proc):
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=2.0)
    app.root.destroy.assert_called_once()


@pytest.mark.parametrize("mode", ["sim", "real"])
def test_supervision_forwarded_to_robot_and_gui(launch_module, monkeypatch, mode):
    context = context_for(launch_module, mode=mode, supervised_mode="true")
    app_fn = launch_module["_planner_app"]
    monkeypatch.setitem(app_fn.__globals__, "Node", lambda **kwargs: kwargs)
    app = app_fn(context)[0]
    params = evaluate_parameters(context, normalize_parameters(app["parameters"]))[0]
    assert params["supervised_mode"] is True
    include = launch_module["_robot_launch"](context)[0]
    forwarded = {name: perform_substitutions(context, normalize_to_list_of_substitutions(value))
                 for name, value in include.launch_arguments}
    assert forwarded["supervised_mode"] == "true"
    from lm.vlm_planner_app import PLANNER_EXTRA_DEFAULTS
    assert "supervised_mode" in PLANNER_EXTRA_DEFAULTS
