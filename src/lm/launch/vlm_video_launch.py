"""Fixed-camera, white-grid box or bucket experiment with optional MP4 capture."""

from datetime import datetime
import os
from uuid import uuid4

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    scene = ["g1_description/scene_crl_with_", LaunchConfiguration("scene_object"), "_video.xml"]
    scene_path = PathJoinSubstitution([
        get_package_share_directory("crl_humanoid_commons"), "data", "robots", LaunchConfiguration("sim_scene_xml")])
    video_path = os.path.expanduser(os.path.join(
        "~/Videos/vlm", f"vlm_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}.mp4"))
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="sim", choices=["sim"]),
        DeclareLaunchArgument("scene_object", default_value="box", choices=["box", "bucket"]),
        DeclareLaunchArgument("camera_backend", default_value="mujoco", choices=["mujoco"]),
        DeclareLaunchArgument("initial_root_pos", default_value="[-2.0, 0.0, 0.8]"),
        DeclareLaunchArgument("sim_scene_xml", default_value=scene),
        DeclareLaunchArgument("camera_robot_xml", default_value=scene_path),
        DeclareLaunchArgument("camera_object_joint_name", default_value=[LaunchConfiguration("scene_object"), "_freejoint"]),
        DeclareLaunchArgument("camera_name", default_value="experiment_video"),
        DeclareLaunchArgument("monitor_camera_name", default_value="experiment_video"),
        DeclareLaunchArgument("task_text", default_value=["Pick up the ", LaunchConfiguration("scene_object"),
                                                        " and place it 1 metre forward along world +X."]),
        DeclareLaunchArgument("record_video", default_value="true"),
        DeclareLaunchArgument("video_path", default_value=video_path,
                              description="MP4 output; existing files are never overwritten. Ctrl+C finalizes the file."),
        DeclareLaunchArgument("video_fps", default_value="30.0"),
        DeclareLaunchArgument("video_width", default_value="1920"),
        DeclareLaunchArgument("video_height", default_value="1080"),
        DeclareLaunchArgument("video_show_robot_goal", default_value="true"),
        DeclareLaunchArgument("video_keyframe_target_topic", default_value="/g1_sim/keyframe_target_poses"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("lm"), "launch", "vlm_experiment_launch.py")),
            launch_arguments={"mode": "sim", "scene_object": LaunchConfiguration("scene_object"),
                              "camera_backend": "mujoco"}.items()),
        # Separate renderer: high-rate recording does not raise the VLM image
        # rate or change policy timing. Target points mirror the monitor feed.
        Node(package="lm", executable="scene_camera", name="experiment_video", output="screen",
             condition=IfCondition(LaunchConfiguration("record_video")),
             parameters=[{
                 "backend": "mujoco",
                 "robot_xml": LaunchConfiguration("camera_robot_xml"),
                 "camera_name": LaunchConfiguration("camera_name"),
                 "monitor_topic": LaunchConfiguration("monitor_topic"),
                 "object_pose_topic": LaunchConfiguration("actual_box_pose_topic"),
                 "object_joint_name": LaunchConfiguration("camera_object_joint_name"),
                 "topic": "/experiment_video/image_raw",
                 "publish_images": False,
                 "hide_sites": True,
                 "show_robot_goal": ParameterValue(LaunchConfiguration("video_show_robot_goal"), value_type=bool),
                 "keyframe_target_topic": LaunchConfiguration("video_keyframe_target_topic"),
                 "recording_path": ParameterValue(LaunchConfiguration("video_path"), value_type=str),
                 "rate_hz": ParameterValue(LaunchConfiguration("video_fps"), value_type=float),
                 "width": ParameterValue(LaunchConfiguration("video_width"), value_type=int),
                 "height": ParameterValue(LaunchConfiguration("video_height"), value_type=int),
             }]),
    ])
