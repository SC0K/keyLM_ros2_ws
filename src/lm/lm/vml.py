from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from io import BytesIO

import numpy as np
import rclpy
from crl_humanoid_msgs.msg import Monitor
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from std_msgs.msg import MultiArrayDimension, String, UInt8MultiArray
from std_srvs.srv import Trigger
from lm.supervised_goal import APPROVED_SUFFIX, CANCEL_SUFFIX, PREVIEW_SUFFIX, PREVIEW_REFRESH_SEC

from lm.box_config import (
    DEFAULT_TARGET_BOX_QUAT_WXYZ,
    REAL_TARGET_BOX_GEOMETRY,
    parse_box_size_xyz,
)
from lm.keyframe_modes import MANIPULATION_KEYFRAMES, keyframe_phase, keyframe_object_type
from lm_interfaces.srv import RetargetKeyframe, VLMQuery
from lm.tracked_objects import TRACKED_OBJECT_DEFAULTS, TrackedObjects, validate_object_topics


_AXIS_TO_LOCAL_VEC = {
    "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "-y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "-z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
}

_GLOBAL_X_WORLD = np.array([1.0, 0.0, 0.0], dtype=np.float64)
APPROACH_XY_OFFSET_M = 0.30
_PICK_POSE_KEYFRAMES = frozenset({"stand_before_pick", "crouch_to_pick", "stand_after_pick"})


def published_goal_targets(data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read success targets from the exact payload sent to the policy, not guesses."""
    with np.load(BytesIO(bytes(data)), allow_pickle=True) as payload:
        names = [str(name) for name in payload["body_names"]]
        pelvis = names.index("pelvis")
        root = np.asarray(payload["body_positions"], dtype=np.float64).reshape(-1, len(names), 3)[0, pelvis].copy()
        root_quat = np.asarray(payload["body_rotations"], dtype=np.float64).reshape(-1, len(names), 4)[0, pelvis].copy()
        obj = np.asarray(payload["object_position_xyz"], dtype=np.float64).reshape(-1, 3)[0].copy()
        obj_quat = np.asarray(payload["object_quat_wxyz"], dtype=np.float64).reshape(-1, 4)[0].copy()
    if not all(np.all(np.isfinite(v)) for v in (root, root_quat, obj, obj_quat)):
        raise ValueError("Retargeted goal contains non-finite poses")
    return obj, obj_quat, root, root_quat


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
        for name, default in TRACKED_OBJECT_DEFAULTS.items():
            self.declare_parameter(name, default)
        self.mocap_object_selection = bool(self.get_parameter("mocap_object_selection").value)
        self._tracked_objects = TrackedObjects(self.get_parameter("tracked_object_timeout_sec").value)
        self.declare_parameter("robot_root_pose_topic", "")
        self.declare_parameter("monitor_topic", "/g1_sim/monitor")
        self.declare_parameter("tracking_error_topic", "/tracking_errors")
        self.declare_parameter("retarget_keyframe_service", "/retargeter/generate_keyframe")
        self.declare_parameter("retargeted_keyframe_topic", "/retargeter/output_keyframe")
        self.declare_parameter("supervised_mode", False)
        self.declare_parameter("retargeted_info_topic", "/retargeter/output_info")
        self.declare_parameter("planner_status_topic", "/vlm_planner/status")
        self.declare_parameter("planner_decision_topic", "/vlm_planner/decision")
        self.declare_parameter("retarget_timeout_sec", 10.0)
        self.declare_parameter("current_box_quat_wxyz", list(DEFAULT_TARGET_BOX_QUAT_WXYZ))
        self.declare_parameter("actual_box_pose_timeout_sec", 2.0)
        self.declare_parameter("monitor_timeout_sec", 2.0)
        self.declare_parameter(
            "box_size_xyz",
            list(REAL_TARGET_BOX_GEOMETRY.size_xyz),
            descriptor=ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter("default_place_distance_m", 1.0)
        # XY root-to-box-center distance, not clearance from the box surface.
        self.declare_parameter("stand_before_pick_distance_m", 0.4)
        self.declare_parameter("pick_max_horizontal_distance_m", 0.45)
        self.declare_parameter("min_stand_root_height_m", 0.78)
        self.declare_parameter("default_target_root_center", [0.0, 0.0, 0.78])  # TODO: find the correct target root pose for root mode (navifation)
        self.declare_parameter("default_target_root_quat_wxyz", [ 1.0, 0.0, 0.0,  0.0])
        self.declare_parameter(
            "default_target_box_quat_wxyz",
            list(DEFAULT_TARGET_BOX_QUAT_WXYZ),
        )
        self.declare_parameter(
            "default_box_forward_axis",
            REAL_TARGET_BOX_GEOMETRY.forward_axis,
        )
        self.declare_parameter("stationary_hold_sec", 0.5)
        self.declare_parameter("min_action_duration_sec", 1.0)
        self.declare_parameter("robot_linear_stationary_threshold_mps", 0.1)
        self.declare_parameter("robot_angular_stationary_threshold_radps", 0.15)
        self.declare_parameter("object_linear_stationary_threshold_mps", 0.15)
        self.declare_parameter("object_angular_stationary_threshold_radps", 0.30)
        self.declare_parameter("mean_body_success_threshold_m", 0.30)
        self.declare_parameter("root_position_success_threshold_m", 0.3)
        self.declare_parameter("root_orientation_success_threshold_rad", 0.8)
        self.declare_parameter("object_position_success_threshold_m", 0.45)
        self.declare_parameter("object_orientation_success_threshold_rad", 1.00)
        self.declare_parameter("task_object_position_threshold_m", 0.45)
        self.declare_parameter("task_object_orientation_threshold_rad", 5.00)

        self._current_box_center = np.zeros(3, dtype=np.float64)
        self._current_box_quat_wxyz = _normalize_quat_wxyz(
            np.asarray(self.get_parameter("current_box_quat_wxyz").value, dtype=np.float64)
        )
        self._starting_box_center: np.ndarray | None = None
        self._starting_box_quat_wxyz: np.ndarray | None = None
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
        self._selected_object_type: str | None = None
        self._last_action_sent_time: float | None = None
        self._last_action_success: bool | None = None
        self._action_success_checks: dict = {}
        self._last_retargeted_info: str | None = None
        self._last_target_box_center: np.ndarray | None = None
        self._last_target_box_quat_wxyz: np.ndarray | None = None
        self._last_target_root_center: np.ndarray | None = None
        self._last_target_root_quat_wxyz: np.ndarray | None = None
        self._task_target_box_center: np.ndarray | None = None
        self._task_target_box_quat_wxyz: np.ndarray | None = None
        self._task_target_initialized_time: float | None = None
        self._last_object_to_manipulate = True
        self._default_place_distance_m = float(self.get_parameter("default_place_distance_m").value)
        self._stand_before_pick_distance_m = float(self.get_parameter("stand_before_pick_distance_m").value)
        if not math.isfinite(self._stand_before_pick_distance_m) or self._stand_before_pick_distance_m <= 0.0:
            raise ValueError("stand_before_pick_distance_m must be finite and positive")
        self._pick_max_horizontal_distance_m = float(
            self.get_parameter("pick_max_horizontal_distance_m").value
        )
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
        if self.mocap_object_selection:
            validate_object_topics(*(self.resolve_topic_name(str(self.get_parameter(name).value)) for name in
                                     ("tracked_box_pose_topic", "tracked_bucket_pose_topic", "actual_box_pose_topic")))
            for kind in ("box", "bucket"):
                self.create_subscription(PoseStamped, str(self.get_parameter(f"tracked_{kind}_pose_topic").value),
                                         lambda msg, kind=kind: self._on_tracked_object(kind, msg), 10)
        else:
            self._actual_box_pose_sub = self.create_subscription(PoseStamped, actual_box_pose_topic, self._on_actual_box_pose, 10)
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
        self._reset_retarget_task_client = self.create_client(Trigger, retarget_keyframe_service + "/reset_task")
        self._retargeted_keyframe_pub = self.create_publisher(UInt8MultiArray, retargeted_keyframe_topic, 10)
        self.supervised_mode = bool(self.get_parameter("supervised_mode").value)
        self._approved_preview_id = None
        self._preview_pub = self.create_publisher(UInt8MultiArray, retargeted_keyframe_topic + PREVIEW_SUFFIX, 10)
        self._cancel_preview_pub = self.create_publisher(String, retargeted_keyframe_topic + CANCEL_SUFFIX, 10)
        self.create_subscription(String, retargeted_keyframe_topic + APPROVED_SUFFIX, self._on_goal_approved, 10)
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
            "action_success_checks": getattr(self, "_action_success_checks", {}),
        }
        payload.update(extra)
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._planner_status_pub.publish(msg)

    def publish_decision(self, response: VLMQuery.Response, published: bool) -> None:
        object_to_manipulate = self._effective_object_to_manipulate(response)
        payload = {
            "stamp_monotonic": time.monotonic(),
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
        if keyframe_phase(response.next_keyframe) == "approach":
            return False
        return (
            bool(response.object_in_manipulation)
            or response.next_keyframe in MANIPULATION_KEYFRAMES
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
            self._starting_box_center = center.copy()
            self._starting_box_quat_wxyz = quat_wxyz.copy()
            self.get_logger().info(
                "Loaded actual box pose: center=%s quat=%s frame=%s"
                % (
                    np.array2string(self._current_box_center, precision=3),
                    np.array2string(self._current_box_quat_wxyz, precision=3),
                    self._current_box_frame_id,
                )
            )
        self._has_actual_box_pose = True

    def _on_tracked_object(self, kind, msg):
        if self._tracked_objects.update(kind, msg) and kind == self._selected_object_type:
            self._on_actual_box_pose(msg)

    def reset_retarget_task(self):
        """Clear the previous task's pose latches without querying the VLM."""
        if not self._reset_retarget_task_client.wait_for_service(timeout_sec=self._retarget_timeout_sec):
            raise RuntimeError("Retargeter reset service unavailable; cannot safely initialize this object's task")
        future = self._reset_retarget_task_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._retarget_timeout_sec)
        if not future.done() or future.result() is None or not future.result().success:
            raise RuntimeError("Could not reset retargeter task pose latches")

    def _route_keyframe_object(self, object_type):
        """Use the normal decision's suffix, never tracking availability, as identity."""
        selected = self._selected_object_type
        if selected is not None and selected != object_type:
            self.publish_status("object_type_mismatch", "Rejected object-library switch during an active task.")
            return False
        if selected is None:
            self._selected_object_type = object_type
            self._has_actual_box_pose = False
            self._last_box_pose_sample = None
            self._stationary_since = None
        msg = self._tracked_objects.get(object_type)
        if msg is None:
            self.publish_status("missing_object_tracking", f"No fresh mocap pose for {object_type}; goal not sent")
            return False
        if not self._has_actual_box_pose:
            self._on_actual_box_pose(msg)
        return True

    def wait_for_first_goal_tracking(self, response):
        """Keep the first normal decision while its chosen object becomes ready."""
        from lm.keyframe_modes import PLANNER_KEYFRAMES
        if response.next_keyframe not in PLANNER_KEYFRAMES:
            return False
        self._route_keyframe_object(keyframe_object_type(response.next_keyframe))
        self.publish_status(
            "waiting_goal_tracking",
            "Waiting for the chosen object's fresh pose and robot/object stationary hold",
            keyframe=response.next_keyframe,
        )
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.robot_and_object_stationary():
                return True
        return False

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
                "No robot root pose or monitor message received before timeout; "
                "cannot determine the desired pickup approach."
            )
            return False
        return True

    def _stand_before_pick_root_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Face the nearest box side at the configured XY distance from its center."""
        if not self._has_actual_box_pose:
            raise RuntimeError("Cannot compute a pickup approach without an actual box pose")
        if not (self._has_robot_root_pose or self._has_monitor):
            raise RuntimeError("Cannot compute a pickup approach without a robot root pose")

        start_box_center, start_box_quat = self._fixed_start_box_pose()
        rot = _quat_wxyz_to_rotmat(start_box_quat)
        hx = 0.5 * float(self._box_size_xyz[0])
        hy = 0.5 * float(self._box_size_xyz[1])
        edge_centers_local = np.array(
            [[hx, 0.0, 0.0], [-hx, 0.0, 0.0], [0.0, hy, 0.0], [0.0, -hy, 0.0]],
            dtype=np.float64,
        )
        edge_centers_world = edge_centers_local @ rot.T + start_box_center[None, :]
        robot_xy = self._current_robot_center[:2]
        nearest_idx = int(
            np.argmin(np.linalg.norm(edge_centers_world[:, :2] - robot_xy[None, :], axis=1))
        )
        edge_center = edge_centers_world[nearest_idx]
        outward = edge_center[:2] - start_box_center[:2]
        outward_norm = float(np.linalg.norm(outward))
        if outward_norm < 1e-9:
            outward = np.array([1.0, 0.0], dtype=np.float64)
        else:
            outward = outward / outward_norm

        root_xy = start_box_center[:2] + self._stand_before_pick_distance_m * outward
        root_z_candidates = [
            float(self._default_target_root_center[2]),
            self._min_stand_root_height_m,
        ]
        if self._current_robot_center[2] > 0.0:
            root_z_candidates.append(float(self._current_robot_center[2]))
        root_center = np.array(
            [root_xy[0], root_xy[1], max(root_z_candidates)],
            dtype=np.float64,
        )
        facing_dir = -outward
        root_quat = _yaw_to_quat_wxyz(float(math.atan2(facing_dir[1], facing_dir[0])))
        return root_center, root_quat

    def _approach_root_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Stop 0.30 m from the current object centre, on the robot-facing side."""
        center = self._current_box_center.copy()
        direction = center[:2] - self._current_robot_center[:2]
        quat = self._current_robot_quat_wxyz.copy()
        if np.linalg.norm(direction) > 1e-9:
            quat = _yaw_to_quat_wxyz(float(math.atan2(direction[1], direction[0])))
        else:
            # Coincident XY: use current heading to choose a stable approach side.
            forward = _quat_wxyz_to_rotmat(quat)[:2, 0]
            quat = _yaw_to_quat_wxyz(float(math.atan2(forward[1], forward[0])))
        center[:2] -= APPROACH_XY_OFFSET_M * _quat_wxyz_to_rotmat(quat)[:2, 0]
        # The retargeter supplies root Z from stand_before_pick.npz.
        return center, quat

    def _update_box_forward_axis_from_robot_once(self) -> bool:
        """Latch the physical box axis aligned with the desired pickup approach."""
        if self._box_forward_axis_initialized_from_robot:
            return True
        if not self._has_actual_box_pose:
            self.get_logger().warn(
                "Cannot latch box_forward_axis before receiving the actual box pose."
            )
            return False
        if not (self._has_robot_root_pose or self._has_monitor):
            self.get_logger().warn(
                "Cannot latch box_forward_axis before receiving a valid robot root pose."
            )
            return False

        pickup_root_center, pickup_root_quat = self._stand_before_pick_root_pose()
        pickup_forward_world = _quat_wxyz_to_rotmat(pickup_root_quat)[:, 0]
        _, start_box_quat = self._fixed_start_box_pose()
        self.box_forward_axis = _infer_axis_label_from_world_dir(
            start_box_quat,
            pickup_forward_world,
        )
        self._box_forward_axis_initialized_from_robot = True
        self.get_logger().info(
            "Latched box_forward_axis from desired nearest-edge pickup approach: "
            "axis=%s pickup_root=%s"
            % (
                self.box_forward_axis,
                np.array2string(pickup_root_center, precision=3),
            )
        )
        return True

    def _stationary_flags(self) -> tuple[bool, bool, bool]:
        robot_stationary = (
            self._has_monitor
            and self._robot_linear_speed <= self._robot_linear_stationary_threshold_mps
            and self._robot_angular_speed <= self._robot_angular_stationary_threshold_radps
        )
        object_stationary = (
            self._has_actual_box_pose
            and (not getattr(self, "mocap_object_selection", False)
                 or self._tracked_objects.get(self._selected_object_type) is not None)
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
        if getattr(self, "mocap_object_selection", False) and self._selected_object_type is None:
            # No object has been chosen yet. Wait for the robot only; waiting
            # for a selected object's pose here would deadlock the first query.
            robot_stationary, _, _ = self._stationary_flags()
            if not robot_stationary:
                self._stationary_since = None
                return False
            now = time.monotonic()
            if self._stationary_since is None:
                self._stationary_since = now
            return now - self._stationary_since >= self._stationary_hold_sec
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

    def _fixed_start_box_pose(self) -> tuple[np.ndarray, np.ndarray]:
        center = (
            self._starting_box_center
            if self._starting_box_center is not None
            else self._current_box_center
        )
        quat = (
            self._starting_box_quat_wxyz
            if self._starting_box_quat_wxyz is not None
            else self._current_box_quat_wxyz
        )
        return center.copy(), quat.copy()

    def _tracking_metric(self, name: str) -> float | None:
        if self._tracking_errors is None:
            return None
        return _optional_float(self._tracking_errors.get(name))

    def _default_task_target_box_center(self) -> np.ndarray:
        source_center = (
            self._starting_box_center
            if self._starting_box_center is not None
            else self._current_box_center
        )
        target = source_center + self._default_place_distance_m * _GLOBAL_X_WORLD
        if getattr(self, "_selected_object_type", None) != "bucket":
            target[2] = self._box_size_xyz[2] / 2.0
        return target

    def initialize_task_target_once(self) -> bool:
        if self._task_target_box_center is not None:
            return True
        if not self._has_actual_box_pose or not (self._has_robot_root_pose or self._has_monitor):
            return False

        if not self._update_box_forward_axis_from_robot_once():
            return False
        self._task_target_box_center = self._default_task_target_box_center()
        nominal_target_quat = _normalize_quat_wxyz(
            self._default_target_box_quat_wxyz.copy()
        )
        # This is the physical placement orientation used by geometric
        # retargeting and task-success checks.  Any policy-only correction is
        # applied by the retargeter after IK when it writes the object goal.
        self._task_target_box_quat_wxyz = nominal_target_quat
        self._task_target_initialized_time = time.monotonic()
        self.get_logger().info(
            "Initialized fixed physical task target box pose: position=%s quat=%s"
            % (
                np.array2string(self._task_target_box_center, precision=3),
                np.array2string(self._task_target_box_quat_wxyz, precision=3),
            )
        )
        self.publish_status(
            "task_target_initialized",
            "Initialized fixed task target box pose",
            target_box_position_xyz=self._task_target_box_center.tolist(),
            target_box_quat_wxyz=self._task_target_box_quat_wxyz.tolist(),
            target_direction_world_xyz=_GLOBAL_X_WORLD.tolist(),
            target_source="starting_box_pose_plus_global_x",
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
        pick_within_horizontal_reach = (
            robot_to_object_xy is not None
            and robot_to_object_xy <= self._pick_max_horizontal_distance_m
        )

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
            "pick_max_horizontal_distance_m": self._pick_max_horizontal_distance_m,
            "pick_within_horizontal_reach": bool(pick_within_horizontal_reach),
            "object_to_target_distance_m": object_to_target,
            "object_to_target_xy_distance_m": object_to_target_xy,
            "target_box_position_xyz": None if target_box_center is None else target_box_center.tolist(),
            "target_box_source": target_box_source,
            "target_direction_world_xyz": _GLOBAL_X_WORLD.tolist(),
            "starting_box_position_xyz": None
            if self._starting_box_center is None
            else self._starting_box_center.tolist(),
        }

    def evaluate_last_action_success(self) -> bool | None:
        if self._last_action_name is None:
            return None

        mean_body_error = self._tracking_metric("mean_body_position_error_m")
        root_position_error = self._tracking_metric("root_position_error_m")
        root_orientation_error = self._tracking_metric("root_orientation_error_rad")
        object_position_error, _object_orientation_error = self._object_error_to_last_target()

        metrics = {
            "mean_body_position_error_m": (mean_body_error, self._mean_body_success_threshold_m),
            "root_position_error_m": (root_position_error, self._root_position_success_threshold_m),
            "root_orientation_error_rad": (root_orientation_error, self._root_orientation_success_threshold_rad),
            "object_position_error_m": (object_position_error, self._object_position_success_threshold_m),
        }
        if keyframe_phase(self._last_action_name or "") == "approach":
            metrics.pop("object_position_error_m")
        self._action_success_checks = {
            name: {"value": value, "threshold": threshold, "passed": value is not None and value <= threshold}
            for name, (value, threshold) in metrics.items()
        }
        generic_success = all(check["passed"] for check in self._action_success_checks.values())
        distance_context = self._distance_context()
        stand_before_pick_reach_success = bool(
            keyframe_phase(self._last_action_name or "") in ("approach", "stand_before_pick")
            and distance_context["pick_within_horizontal_reach"]
        )
        self._last_action_success = bool(generic_success or stand_before_pick_reach_success)
        return self._last_action_success

    def measured_task_completion(self) -> bool:
        object_position_error, _object_orientation_error = self._object_error_to_task_target()
        if object_position_error is None:
            return False
        return object_position_error <= self._task_object_position_threshold_m

    def build_planner_context(self) -> str:
        if getattr(self, "mocap_object_selection", False) and self._selected_object_type is None:
            # This is a normal action request, not an object-selection query.
            # Object identity comes only from task/image, not mocap labels/poses.
            return json.dumps({
                "selected_object_type": None,
                "previous_action": "none",
                "previous_action_finished": False,
                "measured_task_completion": False,
                "distance_context": {"pick_within_horizontal_reach": None},
                "request_policy": (
                    "The robot is stationary. Choose the next keyframe from the task text and image. "
                    "Object distance is not yet available. Before pickup, choose approach_box or "
                    "approach_bucket when reach is unknown. Never approach while holding an object. "
                    "Keep task_completion false. The chosen action will be retargeted and sent "
                    "through the normal execution/approval path."
                ),
            })
        robot_stationary, object_stationary, raw_stationary = self._stationary_flags()
        body_tracking_ready, root_position_ready, root_orientation_ready, tracking_ready, tracking_metrics = (
            self._tracking_error_flags()
        )
        finished = self.ready_for_next_request()
        success = self.evaluate_last_action_success() if finished else self._last_action_success
        context = {
            "selected_object_type": getattr(self, "_selected_object_type", None),
            "previous_action_phase": keyframe_phase(self._last_action_name or "none"),
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
            "tracking_errors": self._tracking_errors or {},
            "action_success_checks": self._action_success_checks,
            "distance_context": self._distance_context(),
            "success_thresholds": {
                "mean_body_position_error_m": self._mean_body_success_threshold_m,
                "root_position_error_m": self._root_position_success_threshold_m,
                "root_orientation_error_rad": self._root_orientation_success_threshold_rad,
                "object_position_error_m": self._object_position_success_threshold_m,
                "object_orientation_error_rad": None,
                "object_orientation_ignored": True,
                "stand_before_pick_robot_to_box_xy_m": self._pick_max_horizontal_distance_m,
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
                f"The approach and stand_before_pick actions are also successful when the robot root is within {self._pick_max_horizontal_distance_m:g} m in the XY plane of the current box center. "
                "Choose the object library from the task text and image: _box for two-hand box motions, _bucket for right-hand bucket/handle motions. All phase names in these notes require that suffix. "
                "Use the image to check the appropriate grasp (two hands for a box, right hand for a bucket) during carry/place phases, or whether the object has slipped, dropped, or is not controlled. "
                f"Select crouch_to_pick only when distance_context.pick_within_horizontal_reach is true, meaning robot_to_object_xy_distance_m is at most {self._pick_max_horizontal_distance_m:g} m. "
                f"Before pickup, if that distance is greater than {self._pick_max_horizontal_distance_m:g} m or unavailable, select approach (locomotion). Once within reach, select stand_before_pick to prepare the grasp. Never use approach while holding the box. "
                "On failure, do not advance to the next semantic phase; retry the previous keyframe when safe, or choose a safe standing/setup keyframe before retrying. "
                "For failed pick actions, use approach if out of reach and not holding the box; otherwise recover with stand_before_pick before retrying crouch_to_pick. "
                "For failed place actions such as stand_before_place or crouch_to_place, retry the failed place keyframe if still safe, or recover with stand_before_place before retrying crouch_to_place. "
                "For failed final standby, retry stand_after_place. "
                "Required placement order: stand_before_place -> crouch_to_place -> stand_after_place. "
                "stand_before_place holds the object above the destination; it does not place or release it. "
                "Never select stand_after_place directly after stand_before_place, even if measured_task_completion is true. "
                "On successful stand_before_place select crouch_to_place; on failure retry/recover without skipping placement. "
                "measured_task_completion is only a position-tolerance check, not evidence of placement or release. "
                "Keep task_completion false when first selecting stand_after_place after crouch_to_place. "
                "Set task_completion true only after previous_action is stand_after_place, previous_action_finished and previous_action_success are true, "
                "measured_task_completion is true, and the image confirms final standby with the object supported at the destination and no longer held. "
                "Do not predict completion of the newly selected action: task_completion true stops the planner immediately. "
                "The VLM response field object_in_manipulation is the same effective flag as object_to_manipulate: true means both retargeting and policy should consider the object. "
                "The node fixes this flag: false for approach, true for all six pick/place keyframes. The VLM's returned flag is ignored."
            ),
        }
        return json.dumps(context, indent=2)

    def _on_goal_approved(self, msg: String) -> None:
        self._approved_preview_id = msg.data

    def _wait_for_goal_approval(self, message: UInt8MultiArray, name: str) -> bool:
        token = uuid.uuid4().hex
        message.layout.dim = [MultiArrayDimension(label=token, size=len(message.data), stride=len(message.data))]
        self._approved_preview_id = None
        self.publish_status("awaiting_approval", f"Preview: {name}. Press N in the monitor or R1+A to execute.",
                            keyframe=name, preview_id=token)
        try:
            next_refresh = 0.0
            while rclpy.ok():
                now = time.monotonic()
                if now >= next_refresh:
                    # Refresh a short lease; a stopped/crashed planner leaves no
                    # indefinitely approvable goal. Duplicate IDs never reapply.
                    self._preview_pub.publish(message)
                    next_refresh = now + PREVIEW_REFRESH_SEC
                rclpy.spin_once(self, timeout_sec=0.1)
                if self._approved_preview_id == token:
                    return True
            return False
        finally:
            if rclpy.ok():
                self._cancel_preview_pub.publish(String(data=token))

    def publish_planner_outputs(self, response: VLMQuery.Response) -> bool:
        from lm.keyframe_modes import PLANNER_KEYFRAMES
        if response.next_keyframe not in PLANNER_KEYFRAMES:
            self.publish_status("invalid_keyframe", "Rejected unsuffixed or unknown VLM action")
            return False
        phase = keyframe_phase(response.next_keyframe)
        object_type = keyframe_object_type(response.next_keyframe)
        selected = getattr(self, "_selected_object_type", None)
        if selected is not None and selected != object_type:
            self.publish_status("object_type_mismatch", "Rejected object-library switch during an active task.")
            return False
        if getattr(self, "mocap_object_selection", False) and not self._route_keyframe_object(object_type):
            return False
        if object_type == "bucket" and selected is None and self._task_target_box_center is not None:
            # Bucket poses use the mesh base, not a box centre at half height.
            self._task_target_box_center[2] = self._fixed_start_box_pose()[0][2]
        if not self._has_actual_box_pose:
            self.get_logger().error("Cannot publish planner outputs without an actual box pose.")
            self.publish_status("missing_actual_box_pose", "Cannot publish planner outputs without an actual box pose")
            return False

        pose_stamp = self._current_box_pose_stamp if self._current_box_pose_stamp is not None else response.image_stamp
        pose_frame_id = self._current_box_frame_id or "world"
        # The service field name is kept for compatibility; this is the single
        # object-aware retargeting and policy mask.
        object_to_manipulate = self._effective_object_to_manipulate(response)
        if object_to_manipulate != bool(response.object_in_manipulation):
            self.get_logger().info(
                "Setting object_to_manipulate=%s for %s from the fixed keyframe-mode mapping."
                % (object_to_manipulate, response.next_keyframe)
            )
        response.object_in_manipulation = object_to_manipulate
        if object_to_manipulate:
            self._update_box_forward_axis_from_robot_once()

        start_box_center, start_box_quat = self._fixed_start_box_pose()
        retarget_current_box_source = (
            "fixed_start_box_pose"
            if phase in _PICK_POSE_KEYFRAMES
            else "current_box_pose"
        )
        retarget_current_box_center = (
            start_box_center
            if phase in _PICK_POSE_KEYFRAMES
            else self._current_box_center
        )
        retarget_current_box_quat = (
            start_box_quat
            if phase in _PICK_POSE_KEYFRAMES
            else self._current_box_quat_wxyz
        )
        current_box_pose_msg = self._pose_stamped_from(
            center=retarget_current_box_center,
            quat_wxyz=retarget_current_box_quat,
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
        target_box_pose_msg = self._pose_stamped_from(
            center=target_box_center,
            quat_wxyz=target_box_quat,
            stamp=pose_stamp,
            frame_id=pose_frame_id,
        )

        target_root_center = self._default_target_root_center.copy()
        target_root_quat = self._default_target_root_quat_wxyz.copy()
        # Bucket stand-before-pick ignores this root hint: the retargeter places
        # the library's left-offset stance relative to the observed bucket.
        if phase == "approach":
            target_root_center, target_root_quat = self._approach_root_pose()
        elif phase == "stand_before_pick" and object_type == "box":
            target_root_center, target_root_quat = self._stand_before_pick_root_pose()
        elif phase == "stand_after_place":
            if self._has_robot_root_pose or self._has_monitor:
                target_root_center = self._current_robot_center.copy()
                target_root_quat = self._current_robot_quat_wxyz.copy()
            else:
                self.get_logger().warn(
                    "No current robot root pose available for stand_after_place; using default target root pose."
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

        try:
            (action_object_target_center, action_object_target_quat,
             target_root_center, target_root_quat) = published_goal_targets(retarget_response.retargeted_keyframe)
        except (ValueError, KeyError, OSError) as exc:
            self.get_logger().error(f"Invalid retargeted goal: {exc}")
            self.publish_status("retargeter_failed", f"Invalid retargeted goal: {exc}")
            return False

        keyframe_msg = UInt8MultiArray()
        keyframe_msg.data = list(retarget_response.retargeted_keyframe)
        if self.supervised_mode:
            self.publish_decision(response, published=False)
            if not self._wait_for_goal_approval(keyframe_msg, response.next_keyframe):
                return False
        else:
            self._retargeted_keyframe_pub.publish(keyframe_msg)

        if retarget_response.retargeted_info:
            info_msg = String()
            info_msg.data = retarget_response.retargeted_info
            self._retargeted_info_pub.publish(info_msg)

        self._last_action_name = response.next_keyframe
        self._selected_object_type = object_type
        self._last_action_sent_time = time.monotonic()
        self._last_action_success = None
        self._action_success_checks = {}
        self._last_retargeted_info = retarget_response.retargeted_info or None
        self._last_target_box_center = action_object_target_center.copy()
        self._last_target_box_quat_wxyz = action_object_target_quat.copy()
        self._last_target_root_center = target_root_center.copy()
        self._last_target_root_quat_wxyz = target_root_quat.copy()
        self._last_object_to_manipulate = object_to_manipulate
        self._stationary_since = None

        self.get_logger().info(
            "Published VLM-generated retargeted keyframe: %s, current_box_source=%s, target_box_quat=%s, box_forward_axis=%s"
            % (
                response.next_keyframe,
                retarget_current_box_source,
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
            current_box_source=retarget_current_box_source,
            current_box_position_xyz=retarget_current_box_center.tolist(),
            target_box_position_xyz=target_box_center.tolist(),
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
        task = parsed.task.strip() if parsed.task else "Pick up the box on the ground and place it 1m at the front."
        if node.mocap_object_selection:
            node.reset_retarget_task()
        if not node.mocap_object_selection and not node.wait_for_actual_box_pose(float(node.get_parameter("actual_box_pose_timeout_sec").value)):
            node.publish_status(
                "missing_start_box_pose",
                "Cannot start VLM planner until the starting box pose has been received.",
            )
            raise RuntimeError("Cannot initialize fixed task target without starting box pose")
        if not node.wait_for_robot_pose(float(node.get_parameter("monitor_timeout_sec").value)):
            node.publish_status(
                "missing_start_robot_pose",
                "Cannot determine the desired pickup approach without a robot root pose.",
            )
            raise RuntimeError("Cannot initialize pickup approach without robot root pose")
        if not node.mocap_object_selection and not node.initialize_task_target_once():
            node.publish_status(
                "missing_task_target",
                "Cannot start VLM planner until the fixed target box pose is initialized.",
            )
            raise RuntimeError("Cannot initialize fixed task target box pose")

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
                "Sending VLM request previous_action=%s"
                % (node._last_action_name or "none")
            )
            node.publish_status(
                "sending_vlm_request",
                "Sending request to VLM service",
                previous_action=node._last_action_name or "none",
            )
            response = node.send_request(
                task_text=task,
                planner_context=context,
                timeout_sec=parsed.timeout,
            )

            if response is None:
                node.publish_status("vlm_request_failed", "No response from VLM service")
                raise RuntimeError("No response from service")
            if not response.success:
                node.publish_status("vlm_request_failed", response.error_message)
                raise RuntimeError(response.error_message)

            node.publish_status(
                "answer_received",
                f"VLM answer received: {response.next_keyframe}",
                next_keyframe=response.next_keyframe,
                latency_sec=float(response.latency_sec),
                task_completion=bool(response.task_completion),
            )
            if node.mocap_object_selection and node._selected_object_type is None:
                # Keep this normal action, rather than sending another VLM
                # request after binding the chosen object's measured pose.
                if not node.wait_for_first_goal_tracking(response):
                    break
            if not node.mocap_object_selection:
                node.wait_for_actual_box_pose(float(node.get_parameter("actual_box_pose_timeout_sec").value))
            node.wait_for_robot_pose(float(node.get_parameter("monitor_timeout_sec").value))
            published = node.publish_planner_outputs(response)
            node.publish_decision(response, published)
            rclpy.spin_once(node, timeout_sec=0.05)

            object_to_manipulate = node._effective_object_to_manipulate(response)
            output = {
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
