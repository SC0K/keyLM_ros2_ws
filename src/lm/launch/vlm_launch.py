from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    image_topic = LaunchConfiguration("image_topic")
    image_path = LaunchConfiguration("image_path")
    rate_hz = LaunchConfiguration("rate_hz")
    service_name = LaunchConfiguration("service_name")
    start_client = LaunchConfiguration("start_client")
    client_delay_sec = LaunchConfiguration("client_delay_sec")
    task_text = LaunchConfiguration("task_text")
    start_visualizer = LaunchConfiguration("start_visualizer")

    return LaunchDescription(
        [
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("image_path", default_value="/home/sitongchen/pics/standupwithbox.png"),
            DeclareLaunchArgument("rate_hz", default_value="2.0"),
            DeclareLaunchArgument("service_name", default_value="/vlm/query"),
            DeclareLaunchArgument("start_client", default_value="true"),
            DeclareLaunchArgument("client_delay_sec", default_value="2.0"),
            DeclareLaunchArgument(
                "task_text",
                default_value="Pick up the box on the ground and place it on the table.",
            ),
            DeclareLaunchArgument("start_visualizer", default_value="true"),
            Node(
                package="lm",
                executable="dummy_camera",
                name="dummy_camera",
                output="screen",
                parameters=[
                    {
                        "topic": image_topic,
                        "image_path": image_path,
                        "rate_hz": rate_hz,
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
                executable="keyframe_retargetor",
                name="keyframe_retargetor",
                output="screen",
            ),
            Node(
                package="lm",
                executable="mujoco_visualizer",
                name="mujoco_visualizer",
                output="screen",
                condition=IfCondition(start_visualizer),
            ),
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
                    )
                ],
            ),
        ]
    )
