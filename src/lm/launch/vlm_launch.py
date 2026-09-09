import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from lm.box_config import (
    DEFAULT_TARGET_BOX_ORIENTATION_OFFSET_RPY_DEG,
    REAL_TARGET_BOX_GEOMETRY,
    SIM_TARGET_BOX_GEOMETRY,
    SOURCE_BOX_GEOMETRY,
    format_box_size_xyz,
    format_orientation_offset_rpy_deg,
)
from lm.vlm_connection import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST
from lm.generated_stand import VLM_STANDING_LEAN_DEG


def generate_launch_description() -> LaunchDescription:
    image_topic = LaunchConfiguration("image_topic")
    rate_hz = LaunchConfiguration("rate_hz")
    camera_backend = LaunchConfiguration("camera_backend")
    real_image_topic = LaunchConfiguration("real_image_topic")
    camera_width = LaunchConfiguration("camera_width")
    camera_height = LaunchConfiguration("camera_height")
    camera_frame_id = LaunchConfiguration("camera_frame_id")
    camera_robot_xml = LaunchConfiguration("camera_robot_xml")
    retargeter_robot_xml = LaunchConfiguration("retargeter_robot_xml")
    camera_name = LaunchConfiguration("camera_name")
    camera_lookat = LaunchConfiguration("camera_lookat")
    camera_distance = LaunchConfiguration("camera_distance")
    camera_azimuth = LaunchConfiguration("camera_azimuth")
    camera_elevation = LaunchConfiguration("camera_elevation")
    service_name = LaunchConfiguration("service_name")
    start_client = LaunchConfiguration("start_client")
    client_delay_sec = LaunchConfiguration("client_delay_sec")
    task_text = LaunchConfiguration("task_text")
    start_visualizer = LaunchConfiguration("start_visualizer")
    start_camera = LaunchConfiguration("start_camera")
    actual_box_pose_topic = LaunchConfiguration("actual_box_pose_topic")
    robot_root_pose_topic = LaunchConfiguration("robot_root_pose_topic")
    monitor_topic = LaunchConfiguration("monitor_topic")
    box_size_xyz = LaunchConfiguration("box_size_xyz")
    source_box_size_xyz = LaunchConfiguration("source_box_size_xyz")
    source_box_forward_axis = LaunchConfiguration("source_box_forward_axis")
    source_box_up_axis = LaunchConfiguration("source_box_up_axis")
    box_hold_forward_axis = LaunchConfiguration("box_hold_forward_axis")
    box_hold_up_axis = LaunchConfiguration("box_hold_up_axis")
    target_box_orientation_offset_rpy_deg = LaunchConfiguration(
        "target_box_orientation_offset_rpy_deg"
    )
    ik_max_residual_m = LaunchConfiguration("ik_max_residual_m")
    tracking_error_topic = LaunchConfiguration("tracking_error_topic")
    retarget_keyframe_service = LaunchConfiguration("retarget_keyframe_service")
    retargeted_keyframe_topic = LaunchConfiguration("retargeted_keyframe_topic")
    retargeted_info_topic = LaunchConfiguration("retargeted_info_topic")

    default_retargeter_robot_xml = os.path.join(
        get_package_share_directory("crl_humanoid_commons"),
        "data",
        "robots",
        "g1_description",
        "g1_29dof_crl.xml",
    )
    source_box_size_default = format_box_size_xyz(SOURCE_BOX_GEOMETRY.size_xyz)
    real_box_size_default = format_box_size_xyz(REAL_TARGET_BOX_GEOMETRY.size_xyz)
    sim_box_size_default = format_box_size_xyz(SIM_TARGET_BOX_GEOMETRY.size_xyz)

    return LaunchDescription(
        [
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("rate_hz", default_value="2.0"),
            DeclareLaunchArgument("camera_backend", default_value="mujoco"),
            DeclareLaunchArgument("camera_device", default_value="/dev/video0",
                                  description="USB/V4L2 capture device; preferably a /dev/v4l/by-id path."),
            DeclareLaunchArgument("capture_fps", default_value="30.0"),
            DeclareLaunchArgument("real_image_topic", default_value="/real_camera/image_raw"),
            DeclareLaunchArgument("camera_width", default_value="640"),
            DeclareLaunchArgument("camera_height", default_value="480"),
            DeclareLaunchArgument("camera_frame_id", default_value="vlm_camera"),
            DeclareLaunchArgument("camera_robot_xml", default_value=""),
            DeclareLaunchArgument("retargeter_robot_xml", default_value=default_retargeter_robot_xml),
            DeclareLaunchArgument("retarget_object_type", default_value="box", choices=["box", "bucket"],
                                  description="Grasp retargeting: scaled two-hand box or rigid right-hand bucket. Does not select a keyframe library."),
            DeclareLaunchArgument("camera_name", default_value=""),
            DeclareLaunchArgument("camera_lookat", default_value="0.7 0.0 0.55"),
            DeclareLaunchArgument("camera_distance", default_value="2.4"),
            DeclareLaunchArgument("camera_azimuth", default_value="-135.0"),
            DeclareLaunchArgument("camera_elevation", default_value="-18.0"),
            DeclareLaunchArgument("service_name", default_value="/vlm/query"),
            DeclareLaunchArgument("ollama_host", default_value=DEFAULT_OLLAMA_HOST,
                                  description="Local Ollama endpoint exposed by either SSH tunnel."),
            DeclareLaunchArgument("model_name", default_value=DEFAULT_MODEL,
                                  description="Ollama model installed on the selected remote server."),
            DeclareLaunchArgument("start_client", default_value="false"),
            DeclareLaunchArgument("client_delay_sec", default_value="2.0"),
            # Real robot topics: actual_box_pose_topic="/red_box/pose", robot_root_pose_topic="/g1_torso/pose".
            DeclareLaunchArgument("actual_box_pose_topic", default_value="/actual_box_pose"),
            DeclareLaunchArgument("robot_root_pose_topic", default_value=""),
            DeclareLaunchArgument("monitor_topic", default_value="/g1_sim/monitor"),
            DeclareLaunchArgument("tracking_error_topic", default_value="/tracking_errors"),
            DeclareLaunchArgument("stand_before_pick_distance_m", default_value="0.4",
                                  description="Stand-before-pick root distance from box center in XY (meters)."),
            DeclareLaunchArgument(
                "box_size_xyz",
                default_value=PythonExpression(
                    [
                        repr(sim_box_size_default),
                        " if '",
                        camera_backend,
                        "' == 'mujoco' else ",
                        repr(real_box_size_default),
                    ]
                ),
                description=(
                    "Target box dimensions in XYZ. Defaults to the MuJoCo box "
                    "for simulation and the real target profile otherwise."
                ),
            ),
            DeclareLaunchArgument("source_box_size_xyz", default_value=source_box_size_default),
            DeclareLaunchArgument(
                "source_box_forward_axis",
                default_value=SOURCE_BOX_GEOMETRY.forward_axis,
            ),
            DeclareLaunchArgument(
                "source_box_up_axis",
                default_value=SOURCE_BOX_GEOMETRY.up_axis,
            ),
            DeclareLaunchArgument(
                "box_hold_forward_axis",
                default_value=PythonExpression(
                    [
                        repr(SIM_TARGET_BOX_GEOMETRY.forward_axis),
                        " if '",
                        camera_backend,
                        "' == 'mujoco' else ",
                        repr(REAL_TARGET_BOX_GEOMETRY.forward_axis),
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "box_hold_up_axis",
                default_value=PythonExpression(
                    [
                        repr(SIM_TARGET_BOX_GEOMETRY.up_axis),
                        " if '",
                        camera_backend,
                        "' == 'mujoco' else ",
                        repr(REAL_TARGET_BOX_GEOMETRY.up_axis),
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "target_box_orientation_offset_rpy_deg",
                default_value=format_orientation_offset_rpy_deg(
                    DEFAULT_TARGET_BOX_ORIENTATION_OFFSET_RPY_DEG
                ),
                description=(
                    "Optional local-frame XYZ roll/pitch/yaw correction applied only "
                    "to the object goal after robot IK; it does not alter root, hand, "
                    "or joint keyframes."
                ),
            ),
            DeclareLaunchArgument("ik_max_residual_m", default_value="0.01"),
            DeclareLaunchArgument("retarget_ik_enabled", default_value="false",
                                  description="False: rigid XY/yaw only, preserve authored joint poses and Z. True: grasp IK."),
            DeclareLaunchArgument("standing_config_file", default_value="",
                                  description="Default: controller g1_keyframe_tracking_obj.yaml; shared default joint angles."),
            DeclareLaunchArgument("standing_waist_pitch_deg", default_value=str(VLM_STANDING_LEAN_DEG),
                                  description="VLM generated standing forward lean, independent of test-sequence lean."),
            DeclareLaunchArgument("retarget_keyframe_service", default_value="/retargeter/generate_keyframe"),
            DeclareLaunchArgument("retargeted_keyframe_topic", default_value="/retargeter/output_keyframe"),
            DeclareLaunchArgument("retargeted_info_topic", default_value="/retargeter/output_info"),
            DeclareLaunchArgument("supervised_mode", default_value="false",
                                  description="Preview each VLM goal and wait for N or R1+A approval."),
            DeclareLaunchArgument(
                "task_text",
                default_value="Pick up the box on the ground and place it 1m at the front.",
            ),
            DeclareLaunchArgument("start_visualizer", default_value="true"),
            DeclareLaunchArgument("start_camera", default_value="true"),
            Node(
                package="lm",
                executable="scene_camera",
                name="scene_camera",
                output="screen",
                condition=IfCondition(start_camera),
                parameters=[
                    {
                        "backend": camera_backend,
                        "camera_device": ParameterValue(LaunchConfiguration("camera_device"), value_type=str),
                        "capture_fps": ParameterValue(LaunchConfiguration("capture_fps"), value_type=float),
                        "topic": image_topic,
                        "rate_hz": rate_hz,
                        "real_image_topic": real_image_topic,
                        "width": camera_width,
                        "height": camera_height,
                        "frame_id": camera_frame_id,
                        "robot_xml": camera_robot_xml,
                        "monitor_topic": monitor_topic,
                        "object_pose_topic": actual_box_pose_topic,
                        "camera_name": camera_name,
                        "camera_lookat": camera_lookat,
                        "camera_distance": camera_distance,
                        "camera_azimuth": camera_azimuth,
                        "camera_elevation": camera_elevation,
                    }
                ],
            ),
            Node(
                package="lm",
                executable="vlm_server",
                name="vlm_server",
                output="screen",
                parameters=[
                    {
                        "service_name": service_name,
                        "ollama_host": ParameterValue(LaunchConfiguration("ollama_host"), value_type=str),
                        "model_name": ParameterValue(LaunchConfiguration("model_name"), value_type=str),
                        "image_topic": image_topic,
                    }
                ],
            ),
            Node(
                package="lm",
                executable="keyframe_retargeter",
                name="keyframe_retargeter",
                output="screen",
                # Humble's parameter-file writer emits even ParameterValue("y",
                # value_type=str) as bare YAML y; rcl then reads it as True.
                # Quoted CLI literals preserve strings through both parsers.
                arguments=[
                    "--ros-args",
                    "-p", ["source_box_forward_axis:='", source_box_forward_axis, "'"],
                    "-p", ["source_box_up_axis:='", source_box_up_axis, "'"],
                    "-p", ["box_hold_forward_axis:='", box_hold_forward_axis, "'"],
                    "-p", ["box_hold_up_axis:='", box_hold_up_axis, "'"],
                ],
                parameters=[
                    {
                        "retarget_keyframe_service": retarget_keyframe_service,
                        "robot_xml": retargeter_robot_xml,
                        "retarget_object_type": LaunchConfiguration("retarget_object_type"),
                        "retarget_ik_enabled": ParameterValue(LaunchConfiguration("retarget_ik_enabled"), value_type=bool),
                        "standing_config_file": ParameterValue(LaunchConfiguration("standing_config_file"), value_type=str),
                        "standing_waist_pitch_deg": ParameterValue(LaunchConfiguration("standing_waist_pitch_deg"), value_type=float),
                        "box_size_xyz": box_size_xyz,
                        "source_box_size_xyz": source_box_size_xyz,
                        "target_box_orientation_offset_rpy_deg": (
                            target_box_orientation_offset_rpy_deg
                        ),
                        "ik_max_residual_m": ParameterValue(
                            ik_max_residual_m,
                            value_type=float,
                        ),
                    }
                ],
            ),
            # Node(
            #     package="lm",
            #     executable="mujoco_visualizer",
            #     name="mujoco_visualizer",
            #     output="screen",
            #     condition=IfCondition(start_visualizer),
            # ),
            TimerAction(
                period=client_delay_sec,
                actions=[
                    Node(
                        package="lm",
                        executable="vlm_client",
                        name="vlm_client",
                        output="screen",
                        condition=IfCondition(start_client),
                        arguments=[
                            "--task",
                            task_text,
                            "--service",
                            service_name,
                        ],
                        parameters=[
                            {
                                "actual_box_pose_topic": actual_box_pose_topic,
                                "robot_root_pose_topic": robot_root_pose_topic,
                                "monitor_topic": monitor_topic,
                                "tracking_error_topic": tracking_error_topic,
                                "stand_before_pick_distance_m": ParameterValue(
                                    LaunchConfiguration("stand_before_pick_distance_m"), value_type=float),
                                "box_size_xyz": box_size_xyz,
                                "default_box_forward_axis": box_hold_forward_axis,
                                "retarget_keyframe_service": retarget_keyframe_service,
                                "retargeted_keyframe_topic": retargeted_keyframe_topic,
                                "retargeted_info_topic": retargeted_info_topic,
                                "supervised_mode": ParameterValue(LaunchConfiguration("supervised_mode"), value_type=bool),
                            }
                        ],
                    )
                ],
            ),
        ]
    )
