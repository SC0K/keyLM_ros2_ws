from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Bool, String

from lm_interfaces.srv import VLMQuery


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class VLMClientNode(Node):
    def __init__(self, service_name: str) -> None:
        super().__init__("vlm_client_node")
        self._client = self.create_client(VLMQuery, service_name)

        self.declare_parameter("selected_keyframe_topic", "/vlm/selected_keyframe")
        self.declare_parameter("object_to_manipulate_topic", "/vlm/object_to_manipulate")
        self.declare_parameter("current_box_pose_topic", "/vlm/current_box_pose")
        self.declare_parameter("target_box_pose_topic", "/vlm/target_box_pose")
        self.declare_parameter("target_root_pose_topic", "/vlm/target_root_pose")
        self.declare_parameter("current_box_center", [0.7, 0.0, 0.175])
        self.declare_parameter("current_box_quat_wxyz", [1.0, 0.0, 0.0, 0.0])           #TODO: update to actual box pose if available
        self.declare_parameter("default_place_distance_m", 1.6)
        self.declare_parameter("default_target_root_center", [1.6, 0.0, 0.7])
        self.declare_parameter("default_target_root_quat_wxyz", [1.0, 0.0, 0.0, 0.0])

        self._current_box_center = np.asarray(self.get_parameter("current_box_center").value, dtype=np.float64)
        self._current_box_quat_wxyz = np.asarray(self.get_parameter("current_box_quat_wxyz").value, dtype=np.float64)
        self._default_place_distance_m = float(self.get_parameter("default_place_distance_m").value)
        self._default_target_root_center = np.asarray(
            self.get_parameter("default_target_root_center").value, dtype=np.float64
        )
        self._default_target_root_quat_wxyz = np.asarray(
            self.get_parameter("default_target_root_quat_wxyz").value, dtype=np.float64
        )

        selected_keyframe_topic = self.get_parameter("selected_keyframe_topic").value
        object_to_manipulate_topic = self.get_parameter("object_to_manipulate_topic").value
        current_box_pose_topic = self.get_parameter("current_box_pose_topic").value
        target_box_pose_topic = self.get_parameter("target_box_pose_topic").value
        target_root_pose_topic = self.get_parameter("target_root_pose_topic").value
        self._selected_keyframe_pub = self.create_publisher(String, selected_keyframe_topic, 10)
        self._object_to_manipulate_pub = self.create_publisher(Bool, object_to_manipulate_topic, 10)
        self._current_box_pose_pub = self.create_publisher(PoseStamped, current_box_pose_topic, 10)
        self._target_box_pose_pub = self.create_publisher(PoseStamped, target_box_pose_topic, 10)
        self._target_root_pose_pub = self.create_publisher(PoseStamped, target_root_pose_topic, 10)

    def send_request(
        self,
        task_text: str,
        planner_context: str,
        timeout_sec: float,
    ) -> VLMQuery.Response | None:
        if not self._client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error("VLM service not available")
            return None

        request = VLMQuery.Request()
        request.task_text = task_text
        request.planner_context = planner_context

        future = self._client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done() or future.result() is None:
            self.get_logger().error("Service call timed out or failed")
            return None
        return future.result()

    @staticmethod
    def _pose_stamped_from(center: np.ndarray, quat_wxyz: np.ndarray, stamp) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "robot_root"
        msg.pose.position.x = float(center[0])
        msg.pose.position.y = float(center[1])
        msg.pose.position.z = float(center[2])
        msg.pose.orientation.w = float(quat_wxyz[0])
        msg.pose.orientation.x = float(quat_wxyz[1])
        msg.pose.orientation.y = float(quat_wxyz[2])
        msg.pose.orientation.z = float(quat_wxyz[3])
        return msg

    def publish_planner_outputs(self, response: VLMQuery.Response) -> None:
        keyframe_msg = String()
        keyframe_msg.data = response.next_keyframe
        self._selected_keyframe_pub.publish(keyframe_msg)

        object_flag_msg = Bool()
        object_flag_msg.data = bool(response.object_in_manipulation)
        self._object_to_manipulate_pub.publish(object_flag_msg)

        current_box_pose_msg = self._pose_stamped_from(
            center=self._current_box_center,
            quat_wxyz=self._current_box_quat_wxyz,
            stamp=response.image_stamp,
        )
        self._current_box_pose_pub.publish(current_box_pose_msg)

        box_rot = _quat_wxyz_to_rotmat(self._current_box_quat_wxyz)
        front_dir = box_rot[:, 0] # x_front
        target_box_center = self._current_box_center + self._default_place_distance_m * front_dir
        target_box_pose_msg = self._pose_stamped_from(
            center=target_box_center,
            quat_wxyz=self._current_box_quat_wxyz,
            stamp=response.image_stamp,     
        )       # TODO: use actual box pose from vision instead of VLM timestamp 
        self._target_box_pose_pub.publish(target_box_pose_msg)

        target_root_pose_msg = self._pose_stamped_from(
            center=self._default_target_root_center,
            quat_wxyz=self._default_target_root_quat_wxyz,
            stamp=response.image_stamp,
        )
        self._target_root_pose_pub.publish(target_root_pose_msg)

        self.get_logger().info(f"Published retarget context for keyframe: {response.next_keyframe}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLM ROS2 service client")
    parser.add_argument("--task", type=str, default=None, help="Task instruction text")
    parser.add_argument("--service", default="/vlm/query", help="Service name")
    parser.add_argument("--timeout", type=float, default=120.0, help="Wait timeout in seconds")
    return parser


def main(args: list[str] | None = None) -> None:
    if args is None:
        args = sys.argv[1:]
    ros_filtered_args = rclpy.utilities.remove_ros_args(args)

    parser = build_arg_parser()
    parsed = parser.parse_args(args=ros_filtered_args)

    rclpy.init(args=None)
    node = VLMClientNode(parsed.service)
    try:
        task = parsed.task.strip() if parsed.task else "Pick up the box on the ground and place it on the table."
        context = """
        Current phase: stand_after_pick
        Last action: stand_after_pick
        Last action finished: true
        Robot root distance to object: 0.5m
        """.strip()
        response = node.send_request(
            task_text=task,
            planner_context=context,
            timeout_sec=parsed.timeout,
        )

        if response is None:
            raise RuntimeError("No response from service")
        if not response.success:
            raise RuntimeError(response.error_message)

        node.publish_planner_outputs(response)
        rclpy.spin_once(node, timeout_sec=0.05)

        output = {
            "next_keyframe": response.next_keyframe,
            "object_in_manipulation": response.object_in_manipulation,
            "task_completion": response.task_completion,
            "latency_sec": response.latency_sec,
            "image_stamp": {
                "sec": int(response.image_stamp.sec),
                "nanosec": int(response.image_stamp.nanosec),
            },
            "raw_json": response.raw_json,
        }
        print(json.dumps(output, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
