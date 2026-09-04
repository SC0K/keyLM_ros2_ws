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
    stand_after_pick_height_m = LaunchConfiguration("stand_after_pick_height_m")
    stand_before_place_height_m = LaunchConfiguration("stand_before_place_height_m")
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
            DeclareLaunchArgument("real_image_topic", default_value="/real_camera/image_raw"),
            DeclareLaunchArgument("camera_width", default_value="640"),
            DeclareLaunchArgument("camera_height", default_value="480"),
            DeclareLaunchArgument("camera_frame_id", default_value="vlm_camera"),
            DeclareLaunchArgument("camera_robot_xml", default_value=""),
            DeclareLaunchArgument("retargeter_robot_xml", default_value=default_retargeter_robot_xml),
            DeclareLaunchArgument("camera_name", default_value=""),
            DeclareLaunchArgument("camera_lookat", default_value="0.7 0.0 0.55"),
            DeclareLaunchArgument("camera_distance", default_value="2.4"),
            DeclareLaunchArgument("camera_azimuth", default_value="-135.0"),
            DeclareLaunchArgument("camera_elevation", default_value="-18.0"),
            DeclareLaunchArgument("service_name", default_value="/vlm/query"),
            DeclareLaunchArgument("start_client", default_value="false"),
            DeclareLaunchArgument("client_delay_sec", default_value="2.0"),
            # Real robot topics: actual_box_pose_topic="/red_box/pose", robot_root_pose_topic="/g1_torso/pose".
            DeclareLaunchArgument("actual_box_pose_topic", default_value="/actual_box_pose"),
            DeclareLaunchArgument("robot_root_pose_topic", default_value=""),
            DeclareLaunchArgument("monitor_topic", default_value="/g1_sim/monitor"),
            DeclareLaunchArgument("tracking_error_topic", default_value="/tracking_errors"),
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
                    "Optional local-frame XYZ roll/pitch/yaw correction in degrees, "
                    "post-multiplied onto the retargeted target-box orientation."
                ),
            ),
            DeclareLaunchArgument("stand_after_pick_height_m", default_value="0.8"),
            DeclareLaunchArgument("stand_before_place_height_m", default_value="0.8"),
            DeclareLaunchArgument("ik_max_residual_m", default_value="0.01"),
            DeclareLaunchArgument("retarget_keyframe_service", default_value="/retargeter/generate_keyframe"),
            DeclareLaunchArgument("retargeted_keyframe_topic", default_value="/retargeter/output_keyframe"),
            DeclareLaunchArgument("retargeted_info_topic", default_value="/retargeter/output_info"),
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
                        "image_topic": image_topic,
                    }
                ],
            ),
            Node(
                package="lm",
                executable="keyframe_retargeter",
                name="keyframe_retargeter",
                output="screen",
                parameters=[
                    {
                        "retarget_keyframe_service": retarget_keyframe_service,
                        "robot_xml": retargeter_robot_xml,
                        "box_size_xyz": box_size_xyz,
                        "source_box_size_xyz": source_box_size_xyz,
                        "source_box_forward_axis": source_box_forward_axis,
                        "source_box_up_axis": source_box_up_axis,
                        "box_hold_forward_axis": box_hold_forward_axis,
                        "box_hold_up_axis": box_hold_up_axis,
                        "stand_after_pick_height_m": stand_after_pick_height_m,
                        "stand_before_place_height_m": stand_before_place_height_m,
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
                                "box_size_xyz": box_size_xyz,
                                "default_box_forward_axis": box_hold_forward_axis,
                                "target_box_orientation_offset_rpy_deg": (
                                    target_box_orientation_offset_rpy_deg
                                ),
                                "stand_after_pick_height_m": stand_after_pick_height_m,
                                "stand_before_place_height_m": stand_before_place_height_m,
                                "retarget_keyframe_service": retarget_keyframe_service,
                                "retargeted_keyframe_topic": retargeted_keyframe_topic,
                                "retargeted_info_topic": retargeted_info_topic,
                            }
                        ],
                    )
                ],
            ),
        ]
    )
