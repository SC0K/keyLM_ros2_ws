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
    actual_box_pose_topic = LaunchConfiguration("actual_box_pose_topic")
    robot_root_pose_topic = LaunchConfiguration("robot_root_pose_topic")
    monitor_topic = LaunchConfiguration("monitor_topic")
    retarget_keyframe_service = LaunchConfiguration("retarget_keyframe_service")
    retargeted_keyframe_topic = LaunchConfiguration("retargeted_keyframe_topic")
    retargeted_info_topic = LaunchConfiguration("retargeted_info_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("image_path", default_value="/home/sitongchen/pics/standupwithbox.png"),
            DeclareLaunchArgument("rate_hz", default_value="2.0"),
            DeclareLaunchArgument("service_name", default_value="/vlm/query"),
            DeclareLaunchArgument("start_client", default_value="true"),
            DeclareLaunchArgument("client_delay_sec", default_value="2.0"),
            # Real robot topics: actual_box_pose_topic="/red_box/pose", robot_root_pose_topic="/g1_torso/pose".
            DeclareLaunchArgument("actual_box_pose_topic", default_value="/actual_box_pose"),
            DeclareLaunchArgument("robot_root_pose_topic", default_value=""),
            DeclareLaunchArgument("monitor_topic", default_value="/g1_sim/monitor"),
            DeclareLaunchArgument("retarget_keyframe_service", default_value="/retargeter/generate_keyframe"),
            DeclareLaunchArgument("retargeted_keyframe_topic", default_value="/retargeter/output_keyframe"),
            DeclareLaunchArgument("retargeted_info_topic", default_value="/retargeter/output_info"),
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
                executable="keyframe_retargeter",
                name="keyframe_retargeter",
                output="screen",
                parameters=[
                    {
                        "retarget_keyframe_service": retarget_keyframe_service,
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
