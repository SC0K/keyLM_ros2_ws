"""One entry point for robot/monitor, VLM services, and the planner GUI."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from lm.box_config import parse_box_size_xyz
from lm.vlm_connection import DEFAULT_LOCAL_PORT, DEFAULT_SERVER, SERVER_PROFILES


def _mode_default(sim_value, real_value):
    return PythonExpression([
        repr(real_value), " if '", LaunchConfiguration("mode"), "' == 'real' else ", repr(sim_value),
    ])


def _robot_launch(context):
    if not IfCondition(LaunchConfiguration("start_robot")).evaluate(context):
        return []
    real = LaunchConfiguration("mode").perform(context) == "real"
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("crl_g1_goalcontroller"), "launch",
            "g1_keyframe.py" if real else "g1_keyframe_sim.py",
        )),
        launch_arguments={
            "current_object_pose_topic": LaunchConfiguration("actual_box_pose_topic"),
            "retargeted_keyframe_topic": LaunchConfiguration("retargeted_keyframe_topic"),
            "supervised_mode": LaunchConfiguration("supervised_mode"),
        }.items(),
    )]


def _planner_app(context):
    value = lambda name: LaunchConfiguration(name).perform(context)
    arguments = ["--server", value("server"), "--local-port", value("local_port"),
                 "--service", value("service_name"), "--task", value("task_text")]
    if value("mode") == "real":
        arguments.append("--real")
    if not IfCondition(LaunchConfiguration("manage_tunnel")).evaluate(context):
        arguments.append("--no-tunnel")
    parameters = {name: ParameterValue(value(name), value_type=str) for name in (
        "monitor_topic", "actual_box_pose_topic", "robot_root_pose_topic", "tracking_error_topic",
        "retarget_keyframe_service", "retargeted_keyframe_topic", "retargeted_info_topic",
    )}
    parameters["box_size_xyz"] = parse_box_size_xyz(value("box_size_xyz")).tolist()
    parameters["stand_before_pick_distance_m"] = float(value("stand_before_pick_distance_m"))
    parameters["supervised_mode"] = IfCondition(LaunchConfiguration("supervised_mode")).evaluate(context)
    # Send the axis as a quoted CLI literal (ROS Humble interprets bare YAML y
    # as boolean even when the launch parameter is explicitly typed as str).
    arguments += ["--ros-args", "-p", "default_box_forward_axis:=" + repr(value("box_hold_forward_axis"))]
    return [Node(package="lm", executable="vlm_planner_app", output="screen",
                 arguments=arguments, parameters=[parameters])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="sim", choices=["sim", "real"],
                              description="Real starts hardware/OptiTrack; neither mode starts a task automatically."),
        DeclareLaunchArgument("start_robot", default_value="true",
                              description="Start robot/controller and monitor. False if already running."),
        DeclareLaunchArgument("supervised_mode", default_value="false",
                              description="Preview VLM goals; approve with N in the monitor or R1+A."),
        DeclareLaunchArgument("server", default_value=DEFAULT_SERVER, choices=list(SERVER_PROFILES)),
        DeclareLaunchArgument("manage_tunnel", default_value="true",
                              description="Let the GUI manage SSH. False for an existing external tunnel."),
        DeclareLaunchArgument("local_port", default_value=str(DEFAULT_LOCAL_PORT)),
        DeclareLaunchArgument("ollama_host", default_value=["http://localhost:", LaunchConfiguration("local_port")]),
        DeclareLaunchArgument("camera_backend", default_value=_mode_default("mujoco", "usb")),
        DeclareLaunchArgument("monitor_topic", default_value=_mode_default("/g1_sim/monitor", "/g1_hardware/monitor")),
        DeclareLaunchArgument("camera_frame_id", default_value=_mode_default("vlm_camera", "usb_camera_optical_frame")),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory("lm"), "launch", "vlm_launch.py",
            )),
            # GUI Start/Stop exclusively owns the planner subprocess. Never
            # launch a second vlm_client alongside it, even with a CLI override.
            launch_arguments={"start_client": "false"}.items(),
        ),
        OpaqueFunction(function=_robot_launch),
        OpaqueFunction(function=_planner_app),
    ])
