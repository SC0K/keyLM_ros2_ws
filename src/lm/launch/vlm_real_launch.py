"""USB-camera VLM services for real G1; does not start hardware or a task."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare real defaults before including the shared stack. All shared launch
    # arguments (device, image topic, server/model, retargeter, task) remain usable.
    return LaunchDescription([
        DeclareLaunchArgument("camera_backend", default_value="usb"),
        DeclareLaunchArgument("monitor_topic", default_value="/g1_hardware/monitor"),
        DeclareLaunchArgument("camera_frame_id", default_value="usb_camera_optical_frame"),
        DeclareLaunchArgument("start_client", default_value="false",
                              description="Do not issue goals automatically; start the planner explicitly."),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory("lm"), "launch", "vlm_launch.py",
            )),
            launch_arguments={key: LaunchConfiguration(key) for key in (
                "camera_backend", "monitor_topic", "camera_frame_id", "start_client",
            )}.items(),
        ),
    ])
