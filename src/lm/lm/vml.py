from __future__ import annotations

import argparse
import json
import math
import sys
import time

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


def _yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float64)


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _pose_to_arrays(msg: PoseStamped) -> tuple[np.ndarray, np.ndarray]:
    pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64)
    quat_wxyz = np.array(
        [
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ],
        dtype=np.float64,
    )
    return pos, quat_wxyz


def _infer_axis_label_from_world_dir(box_quat_wxyz: np.ndarray, world_dir: np.ndarray) -> str:
    rot = _quat_wxyz_to_rotmat(box_quat_wxyz)
    d = np.asarray(world_dir, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-9:
        d = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        d = d / n
    candidates = {
        "x": rot[:, 0],
        "-x": -rot[:, 0],
        "y": rot[:, 1],
        "-y": -rot[:, 1],
        "z": rot[:, 2],
        "-z": -rot[:, 2],
    }
    return max(candidates.keys(), key=lambda k: float(np.dot(candidates[k], d)))


class VLMClientNode(Node):
    def __init__(self, service_name: str) -> None:
        super().__init__("vlm_client_node")
        self._client = self.create_client(VLMQuery, service_name)

        self.declare_parameter("selected_keyframe_topic", "/vlm/selected_keyframe")
        self.declare_parameter("object_to_manipulate_topic", "/vlm/object_to_manipulate")
        self.declare_parameter("actual_box_pose_topic", "/actual_box_pose")
        self.declare_parameter("current_box_pose_topic", "/vlm/current_box_pose")
        self.declare_parameter("target_box_pose_topic", "/vlm/target_box_pose")
        self.declare_parameter("target_root_pose_topic", "/vlm/target_root_pose")
        self.declare_parameter("box_forward_axis_topic", "/vlm/box_forward_axis")
        self.declare_parameter("current_box_center", [0.7, 0.5, 0.15])
        self.declare_parameter("current_box_quat_wxyz", [1.0, 0.0, 0.0, 0.0])
        self.declare_parameter("actual_box_pose_timeout_sec", 2.0)
        self.declare_parameter("box_size_xyz", [0.3, 0.3, 0.3])
        self.declare_parameter("default_place_distance_m", 1.6)
        self.declare_parameter("stand_before_pick_offset_m", 0.2)
        self.declare_parameter("min_stand_root_height_m", 0.78)
        self.declare_parameter("default_target_root_center", [2.3, 0.5, 0.78])  # TODO: find the correct target root pose for root mode (navifation)
        self.declare_parameter("default_target_root_quat_wxyz", [ 0.707, 0.0, 0.0,  0.707])
        self.declare_parameter("default_target_box_quat_wxyz", [0.707, 0.0, 0.0,  0.707])
        self.declare_parameter("default_box_forward_axis", "x")     # TODO: compute the forward axis at pickup.

        self._current_box_center = np.asarray(self.get_parameter("current_box_center").value, dtype=np.float64)
        self._current_box_quat_wxyz = _normalize_quat_wxyz(
            np.asarray(self.get_parameter("current_box_quat_wxyz").value, dtype=np.float64)
        )
        self._has_actual_box_pose = False
        self._current_box_pose_stamp = None
        self._current_box_frame_id = "world"
        self._box_size_xyz = np.asarray(self.get_parameter("box_size_xyz").value, dtype=np.float64)
        self._default_place_distance_m = float(self.get_parameter("default_place_distance_m").value)
        self._stand_before_pick_offset_m = float(self.get_parameter("stand_before_pick_offset_m").value)
        self._min_stand_root_height_m = float(self.get_parameter("min_stand_root_height_m").value)
        self._default_target_root_center = np.asarray(
            self.get_parameter("default_target_root_center").value, dtype=np.float64
        )
        self._default_target_root_quat_wxyz = np.asarray(
            self.get_parameter("default_target_root_quat_wxyz").value, dtype=np.float64
        )
        self._default_target_box_quat_wxyz = np.asarray(
            self.get_parameter("default_target_box_quat_wxyz").value, dtype=np.float64
        )
        self.box_forward_axis = str(self.get_parameter("default_box_forward_axis").value).strip()

        selected_keyframe_topic = self.get_parameter("selected_keyframe_topic").value
        object_to_manipulate_topic = self.get_parameter("object_to_manipulate_topic").value
        actual_box_pose_topic = self.get_parameter("actual_box_pose_topic").value
        current_box_pose_topic = self.get_parameter("current_box_pose_topic").value
        target_box_pose_topic = self.get_parameter("target_box_pose_topic").value
        target_root_pose_topic = self.get_parameter("target_root_pose_topic").value
        box_forward_axis_topic = self.get_parameter("box_forward_axis_topic").value
        self._actual_box_pose_sub = self.create_subscription(
            PoseStamped,
            actual_box_pose_topic,
            self._on_actual_box_pose,
            10,
        )
        self._selected_keyframe_pub = self.create_publisher(String, selected_keyframe_topic, 10)
        self._object_to_manipulate_pub = self.create_publisher(Bool, object_to_manipulate_topic, 10)
        self._current_box_pose_pub = self.create_publisher(PoseStamped, current_box_pose_topic, 10)
        self._target_box_pose_pub = self.create_publisher(PoseStamped, target_box_pose_topic, 10)
        self._target_root_pose_pub = self.create_publisher(PoseStamped, target_root_pose_topic, 10)
        self._box_forward_axis_pub = self.create_publisher(String, box_forward_axis_topic, 10)
        self.get_logger().info(
            f"VLM client will load actual box pose from {actual_box_pose_topic} and republish it on {current_box_pose_topic}"
        )

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
    def _pose_stamped_from(center: np.ndarray, quat_wxyz: np.ndarray, stamp, frame_id: str) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(center[0])
        msg.pose.position.y = float(center[1])
        msg.pose.position.z = float(center[2])
        msg.pose.orientation.w = float(quat_wxyz[0])
        msg.pose.orientation.x = float(quat_wxyz[1])
        msg.pose.orientation.y = float(quat_wxyz[2])
        msg.pose.orientation.z = float(quat_wxyz[3])
        return msg

    def _on_actual_box_pose(self, msg: PoseStamped) -> None:
        self._current_box_center, self._current_box_quat_wxyz = _pose_to_arrays(msg)
        self._current_box_quat_wxyz = _normalize_quat_wxyz(self._current_box_quat_wxyz)
        self._current_box_pose_stamp = msg.header.stamp
        self._current_box_frame_id = msg.header.frame_id or "world"
        if not self._has_actual_box_pose:
            self.get_logger().info(
                "Loaded actual box pose from simulator: center=%s quat=%s frame=%s"
                % (
                    np.array2string(self._current_box_center, precision=3),
                    np.array2string(self._current_box_quat_wxyz, precision=3),
                    self._current_box_frame_id,
                )
            )
        self._has_actual_box_pose = True

    def wait_for_actual_box_pose(self, timeout_sec: float) -> bool:
        if self._has_actual_box_pose:
            return True
        if timeout_sec <= 0.0:
            return False

        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not self._has_actual_box_pose:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            rclpy.spin_once(self, timeout_sec=min(0.1, remaining))

        if not self._has_actual_box_pose:
            self.get_logger().warn(
                "No actual box pose received before timeout; using configured current_box_center/current_box_quat_wxyz fallback."
            )
            return False
        return True

    def publish_planner_outputs(self, response: VLMQuery.Response) -> None:
        pose_stamp = self._current_box_pose_stamp if self._current_box_pose_stamp is not None else response.image_stamp
        pose_frame_id = self._current_box_frame_id or "world"

        keyframe_msg = String()
        keyframe_msg.data = response.next_keyframe
        self._selected_keyframe_pub.publish(keyframe_msg)

        object_flag_msg = Bool()
        object_flag_msg.data = bool(response.object_in_manipulation)
        self._object_to_manipulate_pub.publish(object_flag_msg)

        current_box_pose_msg = self._pose_stamped_from(
            center=self._current_box_center,
            quat_wxyz=self._current_box_quat_wxyz,
            stamp=pose_stamp,
            frame_id=pose_frame_id,
        )
        self._current_box_pose_pub.publish(current_box_pose_msg)

        box_rot = _quat_wxyz_to_rotmat(self._current_box_quat_wxyz)
        front_dir = box_rot[:, 0] # x_front
        target_box_center = self._current_box_center + self._default_place_distance_m * front_dir
        target_box_center[2] = self._box_size_xyz[2] / 2.0 # TODO: compute actural target box pose.
        target_box_quat = self._default_target_box_quat_wxyz.copy()
        try:
            raw = json.loads(response.raw_json) if response.raw_json else {}
            if isinstance(raw, dict):
                q = raw.get("target_box_quat_wxyz")
                if isinstance(q, (list, tuple)) and len(q) == 4:
                    q_arr = np.asarray(q, dtype=np.float64)
                    if np.linalg.norm(q_arr) > 1e-9:
                        target_box_quat = q_arr / np.linalg.norm(q_arr)
        except Exception:
            pass
        target_box_pose_msg = self._pose_stamped_from(
            center=target_box_center,
            quat_wxyz=target_box_quat,
            stamp=pose_stamp,
            frame_id=pose_frame_id,
        )
        self._target_box_pose_pub.publish(target_box_pose_msg)

        target_root_center = self._default_target_root_center.copy()
        target_root_quat = self._default_target_root_quat_wxyz.copy()
        if response.next_keyframe == "stand_before_pick":
            # Compute stand-before-pick root target in VLM client.
            rot = _quat_wxyz_to_rotmat(self._current_box_quat_wxyz)
            hx, hy = 0.5 * float(self._box_size_xyz[0]), 0.5 * float(self._box_size_xyz[1])
            edge_centers_local = np.array([[hx, 0, 0], [-hx, 0, 0], [0, hy, 0], [0, -hy, 0]], dtype=np.float64)
            edge_centers_world = edge_centers_local @ rot.T + self._current_box_center[None, :]
            dists = np.linalg.norm(edge_centers_world[:, :2], axis=1)
            nearest_idx = int(np.argmin(dists))
            edge_center = edge_centers_world[nearest_idx]
            outward = edge_center[:2] - self._current_box_center[:2]
            n = np.linalg.norm(outward)
            if n < 1e-9:
                outward = np.array([1.0, 0.0], dtype=np.float64)
            else:
                outward = outward / n
            root_xy = edge_center[:2] + self._stand_before_pick_offset_m * outward
            facing_dir = -outward
            root_z = max(float(self._default_target_root_center[2]), self._min_stand_root_height_m)
            target_root_center = np.array([root_xy[0], root_xy[1], root_z], dtype=np.float64)
            target_root_quat = _yaw_to_quat_wxyz(float(math.atan2(facing_dir[1], facing_dir[0])))
            self.box_forward_axis = _infer_axis_label_from_world_dir(
                self._current_box_quat_wxyz, np.array([facing_dir[0], facing_dir[1], 0.0], dtype=np.float64)
            )

        target_root_pose_msg = self._pose_stamped_from(
            center=target_root_center,
            quat_wxyz=target_root_quat,
            stamp=pose_stamp,
            frame_id=pose_frame_id,
        )
        self._target_root_pose_pub.publish(target_root_pose_msg)

        # Default/manual target forward axis; can be overridden by VLM raw_json if present.
        try:
            raw = json.loads(response.raw_json) if response.raw_json else {}
            if isinstance(raw, dict):
                parsed = raw.get("box_forward_axis")
                if isinstance(parsed, str) and parsed.strip():
                    self.box_forward_axis = parsed.strip()
                else:
                    parsed_hold = raw.get("box_hold_forward_axis")
                    if isinstance(parsed_hold, str) and parsed_hold.strip():
                        self.box_forward_axis = parsed_hold.strip()
        except Exception:
            pass
        forward_axis_msg = String()
        forward_axis_msg.data = self.box_forward_axis
        self._box_forward_axis_pub.publish(forward_axis_msg)

        self.get_logger().info(
            "Published retarget context for keyframe: %s, current_box_source=%s, target_box_quat=%s, box_forward_axis=%s"
            % (
                response.next_keyframe,
                "actual_box_pose" if self._has_actual_box_pose else "configured_fallback",
                target_box_quat.tolist(),
                self.box_forward_axis,
            )
        )


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

        node.wait_for_actual_box_pose(float(node.get_parameter("actual_box_pose_timeout_sec").value))
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
