"""One entry point for robot/monitor, VLM services, and the planner GUI."""

import os
import uuid

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


def _configure_sim_scene(context):
    if LaunchConfiguration("mode").perform(context) != "sim":
        return []
    object_type = LaunchConfiguration("scene_object").perform(context)
    scene = LaunchConfiguration("sim_scene_xml").perform(context).strip()
    scene = scene or f"g1_description/scene_crl_with_{object_type}.xml"
    # The simulator, monitor and image renderer must load the same model.
    context.launch_configurations["sim_scene_xml"] = scene
    context.launch_configurations["camera_robot_xml"] = os.path.join(
        get_package_share_directory("crl_humanoid_commons"), "data", "robots", scene,
    )
    context.launch_configurations["camera_object_joint_name"] = f"{object_type}_freejoint"
    return []


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
            "reset_keyframe_on_goal_transition": "true",
            "scene_object": LaunchConfiguration("scene_object"),
            "mocap_object_selection": LaunchConfiguration("mocap_object_selection"),
            "tracked_box_pose_topic": LaunchConfiguration("tracked_box_pose_topic"),
            "tracked_bucket_pose_topic": LaunchConfiguration("tracked_bucket_pose_topic"),
            "tracked_object_timeout_sec": LaunchConfiguration("tracked_object_timeout_sec"),
            "sim_scene_xml": LaunchConfiguration("sim_scene_xml"),
            "initial_root_pos": LaunchConfiguration("initial_root_pos"),
            "monitor_camera_name": LaunchConfiguration("monitor_camera_name"),
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
        "tracked_box_pose_topic", "tracked_bucket_pose_topic",
    )}
    parameters["box_size_xyz"] = parse_box_size_xyz(value("box_size_xyz")).tolist()
    parameters["vlm_request_image_topic"] = ParameterValue(value("request_image_topic"), value_type=str)
    parameters["stand_before_pick_distance_m"] = float(value("stand_before_pick_distance_m"))
    parameters["bucket_pick_max_horizontal_distance_m"] = float(value("bucket_pick_max_horizontal_distance_m"))
    parameters["supervised_mode"] = IfCondition(LaunchConfiguration("supervised_mode")).evaluate(context)
    parameters["mocap_object_selection"] = IfCondition(LaunchConfiguration("mocap_object_selection")).evaluate(context)
    parameters["tracked_object_timeout_sec"] = float(value("tracked_object_timeout_sec"))
    # Send the axis as a quoted CLI literal (ROS Humble interprets bare YAML y
    # as boolean even when the launch parameter is explicitly typed as str).
    arguments += ["--ros-args", "-p", "default_box_forward_axis:=" + repr(value("box_hold_forward_axis"))]
    return [Node(package="lm", executable="vlm_planner_app", output="screen",
                 arguments=arguments, parameters=[parameters])]


def generate_launch_description():
    experiment_prefix = f"/vlm/experiment_{uuid.uuid4().hex}"
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="sim", choices=["sim", "real"],
                              description="Real starts hardware/OptiTrack; neither mode starts a task automatically."),
        DeclareLaunchArgument("start_robot", default_value="true",
                              description="Start robot/controller and monitor. False if already running."),
        DeclareLaunchArgument("supervised_mode", default_value="false",
                              description="Preview VLM goals; approve with N in the monitor or R1+A."),
        DeclareLaunchArgument("scene_object", default_value="box", choices=["box", "bucket"],
                              description="Physical/visualized object for this experiment, not the VLM's library decision."),
        DeclareLaunchArgument("mocap_object_selection", default_value="false"),
        DeclareLaunchArgument("server", default_value=DEFAULT_SERVER, choices=list(SERVER_PROFILES)),
        DeclareLaunchArgument("manage_tunnel", default_value="true",
                              description="Let the GUI manage SSH. False for an existing external tunnel."),
        DeclareLaunchArgument("local_port", default_value=str(DEFAULT_LOCAL_PORT)),
        DeclareLaunchArgument("sim_scene_xml", default_value="", description="Optional simulation scene override."),
        DeclareLaunchArgument("initial_root_pos", default_value="", description="Optional simulation root XYZ override."),
        DeclareLaunchArgument("monitor_camera_name", default_value="", description="Named fixed MuJoCo monitor camera."),
        DeclareLaunchArgument("ollama_host", default_value=["http://localhost:", LaunchConfiguration("local_port")]),
        DeclareLaunchArgument("camera_backend", default_value=_mode_default("mujoco", "usb")),
        DeclareLaunchArgument("monitor_topic", default_value=_mode_default("/g1_sim/monitor", "/g1_hardware/monitor")),
        DeclareLaunchArgument("camera_frame_id", default_value=_mode_default("vlm_camera", "usb_camera_optical_frame")),
        # A previous launch's renderer must not feed this experiment's VLM.
        # Both the camera and server inherit this launch's private image topic.
        DeclareLaunchArgument("image_topic", default_value=experiment_prefix + "/image_raw",
                              description="Camera input shared by this experiment's renderer and VLM server."),
        DeclareLaunchArgument("service_name", default_value=experiment_prefix + "/query"),
        DeclareLaunchArgument("request_image_topic", default_value=experiment_prefix + "/request_image"),
        DeclareLaunchArgument("render_image_service", default_value=_mode_default(experiment_prefix + "/render_image", "")),
        OpaqueFunction(function=_configure_sim_scene),
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
