from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import rclpy
from crl_humanoid_msgs.msg import Monitor
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray

from lm.box_config import DEFAULT_BOX_SIZE_XYZ, parse_box_size_xyz
from lm_interfaces.srv import RetargetKeyframe, VLMQuery


_AXIS_TO_LOCAL_VEC = {
    "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "-y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "-z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
}

_OBJECT_REQUIRED_KEYFRAMES = frozenset({"stand_before_place"})

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


def _quat_angle_error(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = _normalize_quat_wxyz(q1)
    q2 = _normalize_quat_wxyz(q2)
    dot = float(np.clip(abs(float(np.dot(q1, q2))), 0.0, 1.0))
    return float(2.0 * math.acos(dot))


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


def _normalize_axis_label(text: str) -> str:
    key = str(text).strip().lower()
    if key not in _AXIS_TO_LOCAL_VEC:
        raise ValueError(f"Unsupported box forward axis '{text}'. Use one of: {list(_AXIS_TO_LOCAL_VEC.keys())}")
    return key


def _normalize_vec(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    out = np.asarray(v, dtype=np.float64).copy()
    n = float(np.linalg.norm(out))
    if n >= 1e-9:
        return out / n
    fb = np.asarray(fallback, dtype=np.float64).copy()
    n_fb = float(np.linalg.norm(fb))
    if n_fb < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return fb / n_fb


def _box_axis_world(box_quat_wxyz: np.ndarray, axis_label: str) -> np.ndarray:
    rot = _quat_wxyz_to_rotmat(box_quat_wxyz)
    return _normalize_vec(
        rot @ _AXIS_TO_LOCAL_VEC[_normalize_axis_label(axis_label)],
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )


def _infer_axis_label_from_world_dir(box_quat_wxyz: np.ndarray, world_dir: np.ndarray) -> str:
    rot = _quat_wxyz_to_rotmat(box_quat_wxyz)
    d = np.asarray(world_dir, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-9:
        d = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        d = d / n
    candidates = {label: rot @ vec for label, vec in _AXIS_TO_LOCAL_VEC.items()}
    return max(candidates.keys(), key=lambda k: float(np.dot(candidates[k], d)))


def _optional_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


class VLMClientNode(Node):
    def __init__(self, service_name: str) -> None:
        super().__init__("vlm_client_node")
        self._client = self.create_client(VLMQuery, service_name)

        # Real robot topics: actual_box_pose_topic="/red_box/pose", robot_root_pose_topic="/g1_torso/pose".
        self.declare_parameter("actual_box_pose_topic", "/actual_box_pose")
        self.declare_parameter("robot_root_pose_topic", "")
        self.declare_parameter("monitor_topic", "/g1_sim/monitor")
        self.declare_parameter("tracking_error_topic", "/tracking_errors")
        self.declare_parameter("retarget_keyframe_service", "/retargeter/generate_keyframe")
        self.declare_parameter("retargeted_keyframe_topic", "/retargeter/output_keyframe")
        self.declare_parameter("retargeted_info_topic", "/retargeter/output_info")
        self.declare_parameter("planner_status_topic", "/vlm_planner/status")
        self.declare_parameter("planner_decision_topic", "/vlm_planner/decision")
        self.declare_parameter("retarget_timeout_sec", 10.0)
        self.declare_parameter("current_box_quat_wxyz", [1.0, 0.0, 0.0, 0.0])
        self.declare_parameter("actual_box_pose_timeout_sec", 2.0)
        self.declare_parameter("monitor_timeout_sec", 2.0)
        self.declare_parameter(
            "box_size_xyz",
            list(DEFAULT_BOX_SIZE_XYZ),
            descriptor=ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter("default_place_distance_m", 1.0)
        self.declare_parameter("stand_before_pick_offset_m", 0.2)
        self.declare_parameter("stand_after_pick_height_m", 1.0)
        self.declare_parameter("stand_before_place_height_m", 1.0)
        self.declare_parameter("min_stand_root_height_m", 0.78)
        self.declare_parameter("default_target_root_center", [0.0, 0.0, 0.78])  # TODO: find the correct target root pose for root mode (navifation)
        self.declare_parameter("default_target_root_quat_wxyz", [ 1.0, 0.0, 0.0,  0.0])
        self.declare_parameter("default_target_box_quat_wxyz", [1.0, 0.0, 0.0,  0.0])
        self.declare_parameter("default_box_forward_axis", "x")     # TODO: compute the forward axis at pickup.
        self.declare_parameter("stationary_hold_sec", 1.0)
        self.declare_parameter("min_action_duration_sec", 1.0)
        self.declare_parameter("robot_linear_stationary_threshold_mps", 0.05)
        self.declare_parameter("robot_angular_stationary_threshold_radps", 0.12)
        self.declare_parameter("object_linear_stationary_threshold_mps", 0.05)
        self.declare_parameter("object_angular_stationary_threshold_radps", 0.20)
        self.declare_parameter("mean_body_success_threshold_m", 0.20)
        self.declare_parameter("root_position_success_threshold_m", 0.25)
        self.declare_parameter("root_orientation_success_threshold_rad", 0.4)
        self.declare_parameter("object_position_success_threshold_m", 0.25)
        self.declare_parameter("object_orientation_success_threshold_rad", 1.00)
        self.declare_parameter("task_object_position_threshold_m", 0.25)
        self.declare_parameter("task_object_orientation_threshold_rad", 5.00)

        self._current_box_center = np.zeros(3, dtype=np.float64)
        self._current_box_quat_wxyz = _normalize_quat_wxyz(
            np.asarray(self.get_parameter("current_box_quat_wxyz").value, dtype=np.float64)
        )
        self._has_actual_box_pose = False
        self._current_box_pose_stamp = None
        self._current_box_frame_id = "world"
        self._current_robot_center = np.zeros(3, dtype=np.float64)
        self._current_robot_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._has_robot_root_pose = False
        self._has_monitor = False
        self._box_size_xyz = parse_box_size_xyz(self.get_parameter("box_size_xyz").value)
        self._tracking_errors: dict | None = None
        self._robot_linear_speed = float("inf")
        self._robot_angular_speed = float("inf")
        self._object_linear_speed = float("inf")
        self._object_angular_speed = float("inf")
        self._last_box_pose_sample: tuple[np.ndarray, np.ndarray, float] | None = None
        self._stationary_since: float | None = None
        self._last_action_name: str | None = None
        self._last_action_sent_time: float | None = None
        self._last_action_success: bool | None = None
        self._last_retargeted_info: str | None = None
        self._last_target_box_center: np.ndarray | None = None
        self._last_target_box_quat_wxyz: np.ndarray | None = None
        self._last_target_root_center: np.ndarray | None = None
        self._last_target_root_quat_wxyz: np.ndarray | None = None
        self._task_target_box_center: np.ndarray | None = None
        self._task_target_box_quat_wxyz: np.ndarray | None = None
        self._last_object_to_manipulate = True
        self._default_place_distance_m = float(self.get_parameter("default_place_distance_m").value)
        self._stand_before_pick_offset_m = float(self.get_parameter("stand_before_pick_offset_m").value)
        self._stand_after_pick_height_m = float(self.get_parameter("stand_after_pick_height_m").value)
        self._stand_before_place_height_m = float(self.get_parameter("stand_before_place_height_m").value)
        self._min_stand_root_height_m = float(self.get_parameter("min_stand_root_height_m").value)
        self._stationary_hold_sec = float(self.get_parameter("stationary_hold_sec").value)
        self._min_action_duration_sec = float(self.get_parameter("min_action_duration_sec").value)
        self._robot_linear_stationary_threshold_mps = float(self.get_parameter("robot_linear_stationary_threshold_mps").value)
        self._robot_angular_stationary_threshold_radps = float(self.get_parameter("robot_angular_stationary_threshold_radps").value)
        self._object_linear_stationary_threshold_mps = float(self.get_parameter("object_linear_stationary_threshold_mps").value)
        self._object_angular_stationary_threshold_radps = float(self.get_parameter("object_angular_stationary_threshold_radps").value)
        self._mean_body_success_threshold_m = float(self.get_parameter("mean_body_success_threshold_m").value)
        self._root_position_success_threshold_m = float(self.get_parameter("root_position_success_threshold_m").value)
        self._root_orientation_success_threshold_rad = float(self.get_parameter("root_orientation_success_threshold_rad").value)
        self._object_position_success_threshold_m = float(self.get_parameter("object_position_success_threshold_m").value)
        self._object_orientation_success_threshold_rad = float(self.get_parameter("object_orientation_success_threshold_rad").value)
        self._task_object_position_threshold_m = float(self.get_parameter("task_object_position_threshold_m").value)
        self._task_object_orientation_threshold_rad = float(self.get_parameter("task_object_orientation_threshold_rad").value)
        self._default_target_root_center = np.asarray(
            self.get_parameter("default_target_root_center").value, dtype=np.float64
        )
        self._default_target_root_quat_wxyz = np.asarray(
            self.get_parameter("default_target_root_quat_wxyz").value, dtype=np.float64
        )
        self._default_target_box_quat_wxyz = np.asarray(
            self.get_parameter("default_target_box_quat_wxyz").value, dtype=np.float64
        )
        self.box_forward_axis = _normalize_axis_label(self.get_parameter("default_box_forward_axis").value)
        self._box_forward_axis_initialized_from_robot = False

        actual_box_pose_topic = self.get_parameter("actual_box_pose_topic").value
        robot_root_pose_topic = str(self.get_parameter("robot_root_pose_topic").value).strip()
        monitor_topic = self.get_parameter("monitor_topic").value
        tracking_error_topic = str(self.get_parameter("tracking_error_topic").value)
        retarget_keyframe_service = str(self.get_parameter("retarget_keyframe_service").value)
        retargeted_keyframe_topic = str(self.get_parameter("retargeted_keyframe_topic").value)
        retargeted_info_topic = str(self.get_parameter("retargeted_info_topic").value)
        planner_status_topic = str(self.get_parameter("planner_status_topic").value)
        planner_decision_topic = str(self.get_parameter("planner_decision_topic").value)
        self._retarget_timeout_sec = float(self.get_parameter("retarget_timeout_sec").value)
        self._actual_box_pose_sub = self.create_subscription(
            PoseStamped,
            actual_box_pose_topic,
            self._on_actual_box_pose,
            10,
        )
        self._robot_root_pose_sub = None
        if robot_root_pose_topic:
            self._robot_root_pose_sub = self.create_subscription(
                PoseStamped,
                robot_root_pose_topic,
                self._on_robot_root_pose,
                10,
            )
        self._monitor_sub = self.create_subscription(
            Monitor,
            monitor_topic,
            self._on_monitor,
            10,
        )
        self._tracking_error_sub = self.create_subscription(
            String,
            tracking_error_topic,
            self._on_tracking_errors,
            10,
        )
        self._retarget_client = self.create_client(RetargetKeyframe, retarget_keyframe_service)
        self._retargeted_keyframe_pub = self.create_publisher(UInt8MultiArray, retargeted_keyframe_topic, 10)
        self._retargeted_info_pub = self.create_publisher(String, retargeted_info_topic, 10)
        self._planner_status_pub = self.create_publisher(String, planner_status_topic, 10)
        self._planner_decision_pub = self.create_publisher(String, planner_decision_topic, 10)
        self.get_logger().info(
            f"VLM client will load actual box pose from {actual_box_pose_topic}, "
            f"robot root pose from monitor {monitor_topic}"
            f"{' with optional external topic ' + robot_root_pose_topic if robot_root_pose_topic else ''}, "
            f"tracking errors from {tracking_error_topic}, "
            f"call retargeter service {retarget_keyframe_service}, "
            f"publish retargeted keyframes on {retargeted_keyframe_topic}, "
            f"and publish planner status on {planner_status_topic}"
        )

    def publish_status(self, state: str, message: str = "", **extra) -> None:
        payload = {
            "stamp_monotonic": time.monotonic(),
            "state": state,
            "message": message,
        }
        payload.update(extra)
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._planner_status_pub.publish(msg)

    def publish_decision(self, step_index: int, response: VLMQuery.Response, published: bool) -> None:
        object_to_manipulate = self._effective_object_to_manipulate(response)
        payload = {
            "stamp_monotonic": time.monotonic(),
            "step_index": int(step_index),
            "next_keyframe": response.next_keyframe,
            "object_in_manipulation": object_to_manipulate,
            "object_to_manipulate": object_to_manipulate,
            "task_completion": bool(response.task_completion),
            "measured_task_completion": self.measured_task_completion(),
            "published": bool(published),
            "latency_sec": float(response.latency_sec),
            "raw_json": response.raw_json,
        }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._planner_decision_pub.publish(msg)

    @staticmethod
    def _effective_object_to_manipulate(response: VLMQuery.Response) -> bool:
        return bool(response.object_in_manipulation) or response.next_keyframe in _OBJECT_REQUIRED_KEYFRAMES

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

    def request_retargeted_keyframe(
        self,
        keyframe_name: str,
        object_to_manipulate: bool,
        current_box_pose: PoseStamped,
        target_box_pose: PoseStamped,
        target_root_pose: PoseStamped,
        box_forward_axis: str,
    ) -> RetargetKeyframe.Response | None:
        if not self._retarget_client.wait_for_service(timeout_sec=self._retarget_timeout_sec):
            self.get_logger().error("Retargeter service not available")
            self.publish_status("retargeter_unavailable", "Retargeter service not available")
            return None

        self.publish_status("calling_retargeter", f"Retargeting keyframe {keyframe_name}", keyframe=keyframe_name)
        request = RetargetKeyframe.Request()
        request.keyframe_name = keyframe_name
        request.object_to_manipulate = bool(object_to_manipulate)
        request.current_box_pose = current_box_pose
        request.target_box_pose = target_box_pose
        request.target_root_pose = target_root_pose
        request.box_forward_axis = box_forward_axis

        future = self._retarget_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._retarget_timeout_sec)
        if not future.done() or future.result() is None:
            self.get_logger().error("Retargeter service call timed out or failed")
            self.publish_status("retargeter_timeout", "Retargeter service call timed out or failed", keyframe=keyframe_name)
            return None
        return future.result()

    def _on_actual_box_pose(self, msg: PoseStamped) -> None:
        now = time.monotonic()
        center, quat_wxyz = _pose_to_arrays(msg)
        quat_wxyz = _normalize_quat_wxyz(quat_wxyz)
        if self._last_box_pose_sample is not None:
            prev_center, prev_quat, prev_time = self._last_box_pose_sample
            dt = max(now - prev_time, 1e-6)
            self._object_linear_speed = float(np.linalg.norm(center - prev_center) / dt)
            self._object_angular_speed = float(_quat_angle_error(prev_quat, quat_wxyz) / dt)
        else:
            self._object_linear_speed = 0.0
            self._object_angular_speed = 0.0
        self._last_box_pose_sample = (center.copy(), quat_wxyz.copy(), now)
        self._current_box_center = center
        self._current_box_quat_wxyz = quat_wxyz
        self._current_box_pose_stamp = msg.header.stamp
        self._current_box_frame_id = msg.header.frame_id or "world"
        if not self._has_actual_box_pose:
            self.get_logger().info(
                "Loaded actual box pose: center=%s quat=%s frame=%s"
                % (
                    np.array2string(self._current_box_center, precision=3),
                    np.array2string(self._current_box_quat_wxyz, precision=3),
                    self._current_box_frame_id,
                )
            )
        self._has_actual_box_pose = True

    def _on_robot_root_pose(self, msg: PoseStamped) -> None:
        center, quat_wxyz = _pose_to_arrays(msg)
        self._current_robot_center = np.asarray(center, dtype=np.float64)
        self._current_robot_quat_wxyz = _normalize_quat_wxyz(quat_wxyz)
        self._has_robot_root_pose = True

    def _on_monitor(self, msg: Monitor) -> None:
        self._current_robot_center = np.array(
            [
                msg.state.base_pose.pose.position.x,
                msg.state.base_pose.pose.position.y,
                msg.state.base_pose.pose.position.z,
            ],
            dtype=np.float64,
        )
        state_quat = np.array(
            [
                msg.state.base_pose.pose.orientation.w,
                msg.state.base_pose.pose.orientation.x,
                msg.state.base_pose.pose.orientation.y,
                msg.state.base_pose.pose.orientation.z,
            ],
            dtype=np.float64,
        )
        imu_quat = np.array(
            [
                msg.sensor.imu.orientation.w,
                msg.sensor.imu.orientation.x,
                msg.sensor.imu.orientation.y,
                msg.sensor.imu.orientation.z,
            ],
            dtype=np.float64,
        )
        self._current_robot_quat_wxyz = _normalize_quat_wxyz(
            state_quat if np.linalg.norm(state_quat) > 1e-3 else imu_quat
        )
        root_linear = np.array(
            [
                msg.state.base_twist.twist.linear.x,
                msg.state.base_twist.twist.linear.y,
                msg.state.base_twist.twist.linear.z,
            ],
            dtype=np.float64,
        )
        root_angular = np.array(
            [
                msg.state.base_twist.twist.angular.x,
                msg.state.base_twist.twist.angular.y,
                msg.state.base_twist.twist.angular.z,
            ],
            dtype=np.float64,
        )
        self._robot_linear_speed = float(np.linalg.norm(root_linear))
        self._robot_angular_speed = float(np.linalg.norm(root_angular))
        self._has_monitor = True

    def _on_tracking_errors(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"Ignoring malformed tracking error JSON: {msg.data[:120]}")
            return
        if isinstance(data, dict):
            self._tracking_errors = data

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
                "No actual box pose received before timeout; waiting for actual box pose before planning."
            )
            return False
        return True

    def wait_for_robot_pose(self, timeout_sec: float) -> bool:
        if self._has_robot_root_pose or self._has_monitor:
            return True
        if timeout_sec <= 0.0:
            return False

        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not (self._has_robot_root_pose or self._has_monitor):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            rclpy.spin_once(self, timeout_sec=min(0.1, remaining))

        if not (self._has_robot_root_pose or self._has_monitor):
            self.get_logger().warn(
                "No robot root pose or monitor message received before timeout; using identity robot pose for box forward axis."
            )
            return False
        return True

    def _update_box_forward_axis_from_robot(self) -> None:
        previous_axis = self.box_forward_axis
        robot_forward_world = _quat_wxyz_to_rotmat(self._current_robot_quat_wxyz)[:, 0]
        robot_forward_world[2] = 0.0
        self.box_forward_axis = _infer_axis_label_from_world_dir(
            self._current_box_quat_wxyz,
            robot_forward_world,
        )
        if not self._box_forward_axis_initialized_from_robot:
            self.get_logger().info(
                "Initialized box_forward_axis from robot orientation: %s"
                % self.box_forward_axis
            )
        elif self.box_forward_axis != previous_axis:
            self.get_logger().info(
                "Updated box_forward_axis from robot/object orientation: %s"
                % self.box_forward_axis
            )
        self._box_forward_axis_initialized_from_robot = True

    def _stationary_flags(self) -> tuple[bool, bool, bool]:
        robot_stationary = (
            self._has_monitor
            and self._robot_linear_speed <= self._robot_linear_stationary_threshold_mps
            and self._robot_angular_speed <= self._robot_angular_stationary_threshold_radps
        )
        object_stationary = (
            self._has_actual_box_pose
            and self._object_linear_speed <= self._object_linear_stationary_threshold_mps
            and self._object_angular_speed <= self._object_angular_stationary_threshold_radps
        )
        return robot_stationary, object_stationary, robot_stationary and object_stationary

    def robot_and_object_stationary(self) -> bool:
        now = time.monotonic()
        _, _, stationary = self._stationary_flags()
        if not stationary:
            self._stationary_since = None
            return False
        if self._stationary_since is None:
            self._stationary_since = now
        return (now - self._stationary_since) >= self._stationary_hold_sec

    def _tracking_error_flags(self) -> tuple[bool, bool, bool, bool, dict]:
        metrics = {
            "mean_body_position_error_m": None,
            "root_position_error_m": None,
            "root_orientation_error_rad": None,
        }
        if self._tracking_errors is not None:
            for name in metrics:
                metrics[name] = _optional_float(self._tracking_errors.get(name))

        body_ok = (
            metrics["mean_body_position_error_m"] is not None
            and metrics["mean_body_position_error_m"] <= self._mean_body_success_threshold_m
        )
        root_position_ok = (
            metrics["root_position_error_m"] is not None
            and metrics["root_position_error_m"] <= self._root_position_success_threshold_m
        )
        root_orientation_ok = (
            metrics["root_orientation_error_rad"] is not None
            and metrics["root_orientation_error_rad"] <= self._root_orientation_success_threshold_rad
        )
        tracking_ready = bool(body_ok and root_position_ok and root_orientation_ok)
        return body_ok, root_position_ok, root_orientation_ok, tracking_ready, metrics

    def ready_for_next_request(self) -> bool:
        if not self.robot_and_object_stationary():
            return False
        if self._last_action_name is None or self._last_action_sent_time is None:
            return True
        if (time.monotonic() - self._last_action_sent_time) < self._min_action_duration_sec:
            return False
        return True

    def _object_error_to_last_target(self) -> tuple[float | None, float | None]:
        if (
            self._last_target_box_center is None
            or self._last_target_box_quat_wxyz is None
            or not self._has_actual_box_pose
        ):
            return None, None
        pos_error = float(np.linalg.norm(self._current_box_center - self._last_target_box_center))
        quat_error = _quat_angle_error(self._current_box_quat_wxyz, self._last_target_box_quat_wxyz)
        return pos_error, quat_error

    def _object_error_to_task_target(self) -> tuple[float | None, float | None]:
        if (
            self._task_target_box_center is None
            or self._task_target_box_quat_wxyz is None
            or not self._has_actual_box_pose
        ):
            return None, None
        pos_error = float(np.linalg.norm(self._current_box_center - self._task_target_box_center))
        quat_error = _quat_angle_error(self._current_box_quat_wxyz, self._task_target_box_quat_wxyz)
        return pos_error, quat_error

    def _expected_object_target_for_action(
        self,
        action: str,
        place_target_center: np.ndarray,
        place_target_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if action in ("stand_after_pick",):
            target = self._current_box_center.copy()
            target[2] = self._stand_after_pick_height_m
            return target, self._current_box_quat_wxyz.copy()
        if action in ("stand_before_place",):
            target = np.asarray(place_target_center, dtype=np.float64).copy()
            target[2] = self._stand_before_place_height_m
            return target, np.asarray(place_target_quat, dtype=np.float64).copy()
        if action in ("crouch_to_place", "stand_after_place"):
            target = np.asarray(place_target_center, dtype=np.float64).copy()
            target[2] = 0.5 * float(self._box_size_xyz[2])
            return target, np.asarray(place_target_quat, dtype=np.float64).copy()
        return self._current_box_center.copy(), self._current_box_quat_wxyz.copy()

    def _tracking_metric(self, name: str) -> float | None:
        if self._tracking_errors is None:
            return None
        return _optional_float(self._tracking_errors.get(name))

    def _default_task_target_box_center(self) -> np.ndarray:
        front_dir = _box_axis_world(self._current_box_quat_wxyz, self.box_forward_axis)
        front_dir[2] = 0.0
        front_dir = _normalize_vec(front_dir, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        target = self._current_box_center + self._default_place_distance_m * front_dir
        target[2] = self._box_size_xyz[2] / 2.0
        return target

    def initialize_task_target_once(self) -> bool:
        if self._task_target_box_center is not None:
            return True
        if not self._has_actual_box_pose:
            return False

        self._update_box_forward_axis_from_robot()
        self._task_target_box_center = self._default_task_target_box_center()
        self._task_target_box_quat_wxyz = self._default_target_box_quat_wxyz.copy()
        self.get_logger().info(
            "Initialized fixed task target box position: %s"
            % np.array2string(self._task_target_box_center, precision=3)
        )
        self.publish_status(
            "task_target_initialized",
            "Initialized fixed task target box position",
            target_box_position_xyz=self._task_target_box_center.tolist(),
            box_forward_axis=self.box_forward_axis,
        )
        return True

    def _context_target_box_center(self) -> np.ndarray | None:
        if self._task_target_box_center is not None:
            return self._task_target_box_center.copy()
        if self._has_actual_box_pose:
            return self._default_task_target_box_center()
        return None

    def _distance_context(self) -> dict:
        have_robot = self._has_robot_root_pose or self._has_monitor
        robot_to_object = None
        robot_to_object_xy = None
        if have_robot and self._has_actual_box_pose:
            robot_to_object_vec = self._current_box_center - self._current_robot_center
            robot_to_object = float(np.linalg.norm(robot_to_object_vec))
            robot_to_object_xy = float(np.linalg.norm(robot_to_object_vec[:2]))

        target_box_center = self._context_target_box_center()
        if self._task_target_box_center is not None:
            target_box_source = "active_task_target"
        elif target_box_center is not None:
            target_box_source = "default_preview"
        else:
            target_box_source = "unavailable"
        object_to_target = None
        object_to_target_xy = None
        if target_box_center is not None and self._has_actual_box_pose:
            object_to_target_vec = target_box_center - self._current_box_center
            object_to_target = float(np.linalg.norm(object_to_target_vec))
            object_to_target_xy = float(np.linalg.norm(object_to_target_vec[:2]))

        return {
            "robot_to_object_distance_m": robot_to_object,
            "robot_to_object_xy_distance_m": robot_to_object_xy,
            "object_to_target_distance_m": object_to_target,
            "object_to_target_xy_distance_m": object_to_target_xy,
            "target_box_position_xyz": None if target_box_center is None else target_box_center.tolist(),
            "target_box_source": target_box_source,
        }

    def evaluate_last_action_success(self) -> bool | None:
        if self._last_action_name is None:
            return None

        mean_body_error = self._tracking_metric("mean_body_position_error_m")
        root_position_error = self._tracking_metric("root_position_error_m")
        root_orientation_error = self._tracking_metric("root_orientation_error_rad")
        object_position_error, _object_orientation_error = self._object_error_to_last_target()

        checks = [
            mean_body_error is not None and mean_body_error <= self._mean_body_success_threshold_m,
            root_position_error is not None and root_position_error <= self._root_position_success_threshold_m,
            root_orientation_error is not None and root_orientation_error <= self._root_orientation_success_threshold_rad,
            object_position_error is not None and object_position_error <= self._object_position_success_threshold_m,
        ]
        self._last_action_success = bool(all(checks))
        return self._last_action_success

    def measured_task_completion(self) -> bool:
        object_position_error, _object_orientation_error = self._object_error_to_task_target()
        if object_position_error is None:
            return False
        return object_position_error <= self._task_object_position_threshold_m

    def build_planner_context(self) -> str:
        robot_stationary, object_stationary, raw_stationary = self._stationary_flags()
        body_tracking_ready, root_position_ready, root_orientation_ready, tracking_ready, tracking_metrics = (
            self._tracking_error_flags()
        )
        finished = self.ready_for_next_request()
        success = self.evaluate_last_action_success() if finished else self._last_action_success
        object_position_error, object_orientation_error = self._object_error_to_last_target()
        task_object_position_error, task_object_orientation_error = self._object_error_to_task_target()
        context = {
            "previous_action": self._last_action_name or "none",
            "previous_action_finished": bool(finished),
            "previous_action_success": None if self._last_action_name is None else success,
            "measured_task_completion": self.measured_task_completion(),
            "stationary": {
                "robot_stationary": bool(robot_stationary),
                "object_stationary": bool(object_stationary),
                "robot_and_object_stationary": bool(raw_stationary),
                "hold_required_sec": self._stationary_hold_sec,
                "robot_linear_speed_mps": self._robot_linear_speed,
                "robot_angular_speed_radps": self._robot_angular_speed,
                "object_linear_speed_mps": self._object_linear_speed,
                "object_angular_speed_radps": self._object_angular_speed,
            },
            "tracking_ready": {
                "body_tracking_ready": bool(body_tracking_ready),
                "root_position_ready": bool(root_position_ready),
                "root_orientation_ready": bool(root_orientation_ready),
                "tracking_ready": bool(tracking_ready),
                "mean_body_position_error_m": tracking_metrics["mean_body_position_error_m"],
                "root_position_error_m": tracking_metrics["root_position_error_m"],
                "root_orientation_error_rad": tracking_metrics["root_orientation_error_rad"],
                "mean_body_threshold_m": self._mean_body_success_threshold_m,
                "root_position_threshold_m": self._root_position_success_threshold_m,
                "root_orientation_threshold_rad": self._root_orientation_success_threshold_rad,
            },
            "current_state": {
                "robot_root_position_xyz": self._current_robot_center.tolist(),
                "robot_root_quat_wxyz": self._current_robot_quat_wxyz.tolist(),
                "box_position_xyz": self._current_box_center.tolist(),
                "box_quat_wxyz": self._current_box_quat_wxyz.tolist(),
                "box_forward_axis": self.box_forward_axis,
            },
            "previous_targets": {
                "last_action_object_target_position_xyz": None
                if self._last_target_box_center is None
                else self._last_target_box_center.tolist(),
                "last_action_object_target_quat_wxyz": None
                if self._last_target_box_quat_wxyz is None
                else self._last_target_box_quat_wxyz.tolist(),
                "target_root_position_xyz": None
                if self._last_target_root_center is None
                else self._last_target_root_center.tolist(),
                "target_root_quat_wxyz": None
                if self._last_target_root_quat_wxyz is None
                else self._last_target_root_quat_wxyz.tolist(),
                "last_action_object_position_error_m": object_position_error,
                "last_action_object_orientation_error_rad": object_orientation_error,
                "task_target_box_position_xyz": None
                if self._task_target_box_center is None
                else self._task_target_box_center.tolist(),
                "task_target_box_quat_wxyz": None
                if self._task_target_box_quat_wxyz is None
                else self._task_target_box_quat_wxyz.tolist(),
                "task_object_position_error_m": task_object_position_error,
                "task_object_orientation_error_rad": task_object_orientation_error,
            },
            "tracking_errors": self._tracking_errors or {},
            "distance_context": self._distance_context(),
            "success_thresholds": {
                "mean_body_position_error_m": self._mean_body_success_threshold_m,
                "root_position_error_m": self._root_position_success_threshold_m,
                "root_orientation_error_rad": self._root_orientation_success_threshold_rad,
                "object_position_error_m": self._object_position_success_threshold_m,
                "object_orientation_error_rad": None,
                "object_orientation_ignored": True,
            },
            "task_completion_thresholds": {
                "object_position_error_m": self._task_object_position_threshold_m,
                "object_orientation_error_rad": None,
                "object_orientation_ignored": True,
            },
            "request_policy": (
                "This request is made only when robot_and_object_stationary has held long enough. "
                "For the first request previous_action is none. For later requests previous_action is the keyframe selected by the previous VLM response. "
                "If previous_action_finished is true and previous_action_success is false, the previous keyframe stopped with tracking or object error above threshold. "
                "Object success and task completion use box position only; object orientation errors are diagnostic and ignored. "
                "Use the image to check whether the robot is actually holding the box with two hands during object-aware carry/place phases, or whether the box has slipped, dropped, or is not controlled. "
                "On failure, do not advance to the next semantic phase; retry the previous keyframe when safe, or choose a safe standing/setup keyframe before retrying. "
                "For failed pick actions such as crouch_to_pick or stand_after_pick, recover with stand_before_pick first, then retry crouch_to_pick. "
                "For failed place actions such as stand_before_place or crouch_to_place, retry the failed place keyframe if still safe, or recover with stand_before_place before retrying crouch_to_place. "
                "For failed final standby, retry stand_after_place. "
                "Set task_completion true only when measured_task_completion is true and the selected next keyframe leaves the robot in the final required task state. "
                "The VLM response field object_in_manipulation is the same effective flag as object_to_manipulate: true means both retargeting and policy should consider the object. "
                "Set it true for object-aware pick/place frames such as crouch_to_pick, stand_after_pick, stand_before_place, and crouch_to_place. It can be false for pure standing/root/standby frames such as stand_before_pick and final stand_after_place."
            ),
        }
        return json.dumps(context, indent=2)

    def publish_planner_outputs(self, response: VLMQuery.Response) -> bool:
        if not self._has_actual_box_pose:
            self.get_logger().error("Cannot publish planner outputs without an actual box pose.")
            self.publish_status("missing_actual_box_pose", "Cannot publish planner outputs without an actual box pose")
            return False

        pose_stamp = self._current_box_pose_stamp if self._current_box_pose_stamp is not None else response.image_stamp
        pose_frame_id = self._current_box_frame_id or "world"
        # The service field name is kept for compatibility; this is the single
        # object-aware retargeting and policy mask.
        object_to_manipulate = self._effective_object_to_manipulate(response)
        if object_to_manipulate and not bool(response.object_in_manipulation):
            self.get_logger().info(
                "Forcing object_to_manipulate=true for %s because this keyframe requires object-aware retargeting."
                % response.next_keyframe
            )
            response.object_in_manipulation = True
        if object_to_manipulate:
            self._update_box_forward_axis_from_robot()

        current_box_pose_msg = self._pose_stamped_from(
            center=self._current_box_center,
            quat_wxyz=self._current_box_quat_wxyz,
            stamp=pose_stamp,
            frame_id=pose_frame_id,
        )

        if not self.initialize_task_target_once():
            self.get_logger().error("Cannot publish planner outputs without a fixed task target box position.")
            self.publish_status(
                "missing_task_target",
                "Cannot publish planner outputs without a fixed task target box position",
            )
            return False

        target_box_center = self._task_target_box_center.copy()
        target_box_quat = (
            self._task_target_box_quat_wxyz.copy()
            if self._task_target_box_quat_wxyz is not None
            else self._default_target_box_quat_wxyz.copy()
        )
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

        target_root_center = self._default_target_root_center.copy()
        target_root_quat = self._default_target_root_quat_wxyz.copy()
        if response.next_keyframe == "stand_before_pick":
            # Compute stand-before-pick root target in VLM client.
            rot = _quat_wxyz_to_rotmat(self._current_box_quat_wxyz)
            hx, hy = 0.5 * float(self._box_size_xyz[0]), 0.5 * float(self._box_size_xyz[1])
            edge_centers_local = np.array([[hx, 0, 0], [-hx, 0, 0], [0, hy, 0], [0, -hy, 0]], dtype=np.float64)
            edge_centers_world = edge_centers_local @ rot.T + self._current_box_center[None, :]
            robot_xy = self._current_robot_center[:2] if (self._has_robot_root_pose or self._has_monitor) else np.zeros(2)
            dists = np.linalg.norm(edge_centers_world[:, :2] - robot_xy[None, :], axis=1)
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
            root_z_candidates = [float(self._default_target_root_center[2]), self._min_stand_root_height_m]
            if (self._has_robot_root_pose or self._has_monitor) and self._current_robot_center[2] > 0.0:
                root_z_candidates.append(float(self._current_robot_center[2]))
            root_z = max(root_z_candidates)
            target_root_center = np.array([root_xy[0], root_xy[1], root_z], dtype=np.float64)
            target_root_quat = _yaw_to_quat_wxyz(float(math.atan2(facing_dir[1], facing_dir[0])))
        elif response.next_keyframe == "stand_after_place":
            if self._has_robot_root_pose or self._has_monitor:
                target_root_center = self._current_robot_center.copy()
                target_root_quat = self._current_robot_quat_wxyz.copy()
            else:
                self.get_logger().warn(
                    "No current robot root pose available for stand_after_place; using default target root pose."
                )

        action_object_target_center, action_object_target_quat = self._expected_object_target_for_action(
            response.next_keyframe,
            target_box_center,
            target_box_quat,
        )
        target_root_pose_msg = self._pose_stamped_from(
            center=target_root_center,
            quat_wxyz=target_root_quat,
            stamp=pose_stamp,
            frame_id=pose_frame_id,
        )
        retarget_response = self.request_retargeted_keyframe(
            keyframe_name=response.next_keyframe,
            object_to_manipulate=object_to_manipulate,
            current_box_pose=current_box_pose_msg,
            target_box_pose=target_box_pose_msg,
            target_root_pose=target_root_pose_msg,
            box_forward_axis=self.box_forward_axis,
        )
        if retarget_response is None:
            return False
        if not retarget_response.success:
            self.get_logger().error(f"Retargeter failed: {retarget_response.error_message}")
            self.publish_status(
                "retargeter_failed",
                retarget_response.error_message,
                keyframe=response.next_keyframe,
            )
            return False

        keyframe_msg = UInt8MultiArray()
        keyframe_msg.data = list(retarget_response.retargeted_keyframe)
        self._retargeted_keyframe_pub.publish(keyframe_msg)

        if retarget_response.retargeted_info:
            info_msg = String()
            info_msg.data = retarget_response.retargeted_info
            self._retargeted_info_pub.publish(info_msg)

        self._last_action_name = response.next_keyframe
        self._last_action_sent_time = time.monotonic()
        self._last_action_success = None
        self._last_retargeted_info = retarget_response.retargeted_info or None
        self._last_target_box_center = action_object_target_center.copy()
        self._last_target_box_quat_wxyz = action_object_target_quat.copy()
        self._last_target_root_center = target_root_center.copy()
        self._last_target_root_quat_wxyz = target_root_quat.copy()
        self._task_target_box_quat_wxyz = target_box_quat.copy()
        self._last_object_to_manipulate = object_to_manipulate
        self._stationary_since = None

        self.get_logger().info(
            "Published VLM-generated retargeted keyframe: %s, current_box_source=actual_box_pose, target_box_quat=%s, box_forward_axis=%s"
            % (
                response.next_keyframe,
                target_box_quat.tolist(),
                self.box_forward_axis,
            )
        )
        self.publish_status(
            "keyframe_published",
            f"Published retargeted keyframe {response.next_keyframe}",
            keyframe=response.next_keyframe,
            object_in_manipulation=object_to_manipulate,
            object_to_manipulate=object_to_manipulate,
        )
        return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLM ROS2 service client")
    parser.add_argument("--task", type=str, default=None, help="Task instruction text")
    parser.add_argument("--service", default="/vlm/query", help="Service name")
    parser.add_argument("--timeout", type=float, default=120.0, help="Wait timeout in seconds")
    parser.add_argument("--poll-period", type=float, default=0.1, help="Seconds between readiness checks")
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
        node.publish_status("connected", "VLM planner client started")
        task = parsed.task.strip() if parsed.task else "Pick up the box on the ground and place it on the table."
        node.wait_for_actual_box_pose(float(node.get_parameter("actual_box_pose_timeout_sec").value))
        node.wait_for_robot_pose(float(node.get_parameter("monitor_timeout_sec").value))
        node.initialize_task_target_once()

        step_index = 0
        last_waiting_status_time = 0.0
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=max(0.0, parsed.poll_period))
            if not node.ready_for_next_request():
                now = time.monotonic()
                if now - last_waiting_status_time > 1.0:
                    robot_stationary, object_stationary, raw_stationary = node._stationary_flags()
                    body_ready, root_pos_ready, root_ori_ready, tracking_ready, tracking_metrics = (
                        node._tracking_error_flags()
                    )
                    node.publish_status(
                        "waiting_ready_for_request",
                        "Waiting for robot/object stationary hold and minimum action duration",
                        previous_action=node._last_action_name or "none",
                        robot_stationary=robot_stationary,
                        object_stationary=object_stationary,
                        robot_and_object_stationary=raw_stationary,
                        body_tracking_ready=body_ready,
                        root_position_ready=root_pos_ready,
                        root_orientation_ready=root_ori_ready,
                        tracking_ready=tracking_ready,
                        tracking_errors=tracking_metrics,
                    )
                    last_waiting_status_time = now
                continue

            context = node.build_planner_context()
            node.get_logger().info(
                "Sending VLM request step=%d previous_action=%s"
                % (step_index, node._last_action_name or "none")
            )
            node.publish_status(
                "sending_vlm_request",
                "Sending request to VLM service",
                step_index=step_index,
                previous_action=node._last_action_name or "none",
            )
            response = node.send_request(
                task_text=task,
                planner_context=context,
                timeout_sec=parsed.timeout,
            )

            if response is None:
                node.publish_status("vlm_request_failed", "No response from VLM service", step_index=step_index)
                raise RuntimeError("No response from service")
            if not response.success:
                node.publish_status("vlm_request_failed", response.error_message, step_index=step_index)
                raise RuntimeError(response.error_message)

            node.publish_status(
                "answer_received",
                f"VLM answer received: {response.next_keyframe}",
                step_index=step_index,
                next_keyframe=response.next_keyframe,
                latency_sec=float(response.latency_sec),
                task_completion=bool(response.task_completion),
            )
            node.wait_for_actual_box_pose(float(node.get_parameter("actual_box_pose_timeout_sec").value))
            node.wait_for_robot_pose(float(node.get_parameter("monitor_timeout_sec").value))
            published = node.publish_planner_outputs(response)
            node.publish_decision(step_index, response, published)
            rclpy.spin_once(node, timeout_sec=0.05)

            object_to_manipulate = node._effective_object_to_manipulate(response)
            output = {
                "step_index": step_index,
                "next_keyframe": response.next_keyframe,
                "object_in_manipulation": object_to_manipulate,
                "object_to_manipulate": object_to_manipulate,
                "task_completion": response.task_completion,
                "measured_task_completion": node.measured_task_completion(),
                "published": published,
                "latency_sec": response.latency_sec,
                "image_stamp": {
                    "sec": int(response.image_stamp.sec),
                    "nanosec": int(response.image_stamp.nanosec),
                },
                "raw_json": response.raw_json,
            }
            print(json.dumps(output, indent=2), flush=True)

            step_index += 1
            if response.task_completion:
                node.get_logger().info("VLM marked task complete; stopping planner client loop.")
                node.publish_status("task_complete", "VLM marked task complete")
                break
    finally:
        node.publish_status("stopped", "VLM planner client stopped")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
