from __future__ import annotations

import json
import math
from io import BytesIO
from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node

from lm.box_config import (
    DEFAULT_TARGET_BOX_QUAT_WXYZ,
    REAL_TARGET_BOX_GEOMETRY,
    SOURCE_BOX_GEOMETRY,
    parse_box_size_xyz,
)
from lm_interfaces.srv import RetargetKeyframe

from lm.keyframe_box_retarget import (
    BoxFrame,
    _default_robot_xml,
    _get_body_pos,
    _pick_existing_default_ee,
    _pick_existing_default_feet,
    retarget_qpos_for_box_grasp,
)

_AXIS_TO_LOCAL_VEC = {
    "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "-y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "-z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
}

_OBJECT_REQUIRED_KEYFRAMES = frozenset({"stand_before_place"})
_PICK_POSE_KEYFRAMES = frozenset({"stand_before_pick", "crouch_to_pick", "stand_after_pick"})

def _quat_wxyz_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_wxyz_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_wxyz_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


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


def _yaw_from_quat_wxyz(q: np.ndarray) -> float:
    rot = _quat_wxyz_to_rotmat(q)
    return float(math.atan2(rot[1, 0], rot[0, 0]))


def _yaw_quat_from_forward_xy(forward_world: np.ndarray, fallback_quat_wxyz: np.ndarray) -> np.ndarray:
    v = np.asarray(forward_world, dtype=np.float64).copy()
    v[2] = 0.0
    n = float(np.linalg.norm(v[:2]))
    if n < 1e-9:
        yaw = _yaw_from_quat_wxyz(fallback_quat_wxyz)
    else:
        yaw = float(math.atan2(v[1], v[0]))
    return _yaw_to_quat_wxyz(yaw)


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
    if n < 1e-9:
        fb = np.asarray(fallback, dtype=np.float64).copy()
        n_fb = float(np.linalg.norm(fb))
        if n_fb < 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return fb / n_fb
    return out / n


def _forward_world_from_axis(quat_wxyz: np.ndarray, axis_label: str) -> np.ndarray:
    rot = _quat_wxyz_to_rotmat(quat_wxyz)
    return _normalize_vec(rot @ _AXIS_TO_LOCAL_VEC[axis_label], np.array([1.0, 0.0, 0.0], dtype=np.float64))


def _quat_wxyz_from_rotmat(rot: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=np.float64).reshape(9))
    return _quat_wxyz_normalize(quat)


def _axis_dimension_index(axis_label: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis_label.lstrip("-")]


def _infer_box_axis_from_world_direction(
    box_quat_wxyz: np.ndarray,
    world_direction: np.ndarray,
    excluded_axis: str | None = None,
) -> str:
    excluded_index = None if excluded_axis is None else _axis_dimension_index(excluded_axis)
    direction = _normalize_vec(world_direction, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    rot = _quat_wxyz_to_rotmat(box_quat_wxyz)
    candidates = {
        label: _normalize_vec(rot @ local_axis, direction)
        for label, local_axis in _AXIS_TO_LOCAL_VEC.items()
        if excluded_index is None or _axis_dimension_index(label) != excluded_index
    }
    return max(candidates, key=lambda label: float(np.dot(candidates[label], direction)))


def _matched_box_quat(
    box_quat_wxyz: np.ndarray,
    forward_axis: str,
    up_axis: str,
) -> np.ndarray:
    """Build a right-handed forward/side/up frame from two physical box axes."""
    if _axis_dimension_index(forward_axis) == _axis_dimension_index(up_axis):
        raise ValueError("Box forward and up axes must use different physical dimensions")
    rot = _quat_wxyz_to_rotmat(box_quat_wxyz)
    forward = _normalize_vec(rot @ _AXIS_TO_LOCAL_VEC[forward_axis], np.array([1.0, 0.0, 0.0]))
    up_raw = rot @ _AXIS_TO_LOCAL_VEC[up_axis]
    up = _normalize_vec(up_raw - float(np.dot(up_raw, forward)) * forward, np.array([0.0, 0.0, 1.0]))
    side = _normalize_vec(np.cross(up, forward), np.array([0.0, 1.0, 0.0]))
    up = _normalize_vec(np.cross(forward, side), up)
    return _quat_wxyz_from_rotmat(np.column_stack([forward, side, up]))


def _box_size_in_matched_frame(
    size_xyz: np.ndarray,
    forward_axis: str,
    up_axis: str,
) -> np.ndarray:
    """Express physical dimensions in the matched forward/side/up frame."""
    size = np.asarray(size_xyz, dtype=np.float64).reshape(3)
    forward_index = _axis_dimension_index(forward_axis)
    up_index = _axis_dimension_index(up_axis)
    if forward_index == up_index:
        raise ValueError("Box forward and up axes must use different physical dimensions")
    side_index = next(index for index in range(3) if index not in (forward_index, up_index))
    return size[[forward_index, side_index, up_index]].copy()


def map_orientation_by_frame_transform(
    src_quat_wxyz: np.ndarray,
    dst_quat_wxyz: np.ndarray,
    quat_wxyz: np.ndarray,
) -> np.ndarray:
    src_q = _quat_wxyz_normalize(src_quat_wxyz)
    dst_q = _quat_wxyz_normalize(dst_quat_wxyz)
    q = _quat_wxyz_normalize(quat_wxyz)
    delta_q = _quat_wxyz_multiply(dst_q, _quat_wxyz_conj(src_q))
    return _quat_wxyz_normalize(_quat_wxyz_multiply(delta_q, q))


class KeyframeRetargeterNode(Node):
    def __init__(self) -> None:
        super().__init__("keyframe_retargeter_node")

        self.declare_parameter("retarget_keyframe_service", "/retargeter/generate_keyframe")
        self.declare_parameter("library_dir", "")
        self.declare_parameter(
            "box_size_xyz",
            list(REAL_TARGET_BOX_GEOMETRY.size_xyz),
            descriptor=ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter(
            "source_box_size_xyz",
            list(SOURCE_BOX_GEOMETRY.size_xyz),
            descriptor=ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter(
            "source_box_forward_axis",
            SOURCE_BOX_GEOMETRY.forward_axis,
        )
        self.declare_parameter(
            "box_hold_forward_axis",
            REAL_TARGET_BOX_GEOMETRY.forward_axis,
        )
        self.declare_parameter("source_box_up_axis", SOURCE_BOX_GEOMETRY.up_axis)
        self.declare_parameter("box_hold_up_axis", REAL_TARGET_BOX_GEOMETRY.up_axis)
        self.declare_parameter("stand_before_pick_offset_m", 0.2)
        self.declare_parameter("stand_after_pick_height_m", 0.9)
        self.declare_parameter("stand_before_place_height_m", 0.9)
        # Retained as accepted legacy parameters while the shared solver uses
        # explicit fixed-foot constraints below.
        self.declare_parameter("foot_motion_penalty_weight", 0.1)
        self.declare_parameter("ee_root_penalty_weight", 2.0)
        self.declare_parameter("ik_max_residual_m", 0.01)
        self.declare_parameter("ik_foot_constraint_weight", 6.0)
        self.declare_parameter("ik_max_foot_residual_m", 0.001)
        self.declare_parameter("robot", "g1")
        self.declare_parameter("robot_xml", "")

        retarget_keyframe_service = str(self.get_parameter("retarget_keyframe_service").value)
        self._box_size_xyz = parse_box_size_xyz(self.get_parameter("box_size_xyz").value)
        self._source_box_size_xyz = parse_box_size_xyz(
            self.get_parameter("source_box_size_xyz").value
        )
        self._source_box_forward_axis = _normalize_axis_label(
            str(self.get_parameter("source_box_forward_axis").value)
        )
        self._box_hold_forward_axis = _normalize_axis_label(
            str(self.get_parameter("box_hold_forward_axis").value)
        )
        self._source_box_up_axis = _normalize_axis_label(
            str(self.get_parameter("source_box_up_axis").value)
        )
        self._box_hold_up_axis = _normalize_axis_label(
            str(self.get_parameter("box_hold_up_axis").value)
        )
        self._stand_before_pick_offset_m = float(self.get_parameter("stand_before_pick_offset_m").value)
        self._stand_after_pick_height_m = float(self.get_parameter("stand_after_pick_height_m").value)
        self._stand_before_place_height_m = float(self.get_parameter("stand_before_place_height_m").value)
        self._foot_motion_penalty_weight = float(self.get_parameter("foot_motion_penalty_weight").value)
        self._ee_root_penalty_weight = float(self.get_parameter("ee_root_penalty_weight").value)
        self._ik_max_residual_m = float(self.get_parameter("ik_max_residual_m").value)
        self._ik_foot_constraint_weight = float(self.get_parameter("ik_foot_constraint_weight").value)
        self._ik_max_foot_residual_m = float(self.get_parameter("ik_max_foot_residual_m").value)
        if self._ik_max_residual_m <= 0.0 or self._ik_max_foot_residual_m <= 0.0:
            raise ValueError("IK residual limits must be positive")
        if self._ik_foot_constraint_weight <= 0.0:
            raise ValueError("ik_foot_constraint_weight must be positive")
        robot_xml = str(self.get_parameter("robot_xml").value).strip()
        if robot_xml:
            self._ik_model = mujoco.MjModel.from_xml_path(robot_xml)
        else:
            robot_name = str(self.get_parameter("robot").value)
            self._ik_model = mujoco.MjModel.from_xml_path(str(_default_robot_xml(robot_name)))
        self._ik_data = mujoco.MjData(self._ik_model)
        ee_names = _pick_existing_default_ee(self._ik_model)
        self._ik_ee_body_ids = [
            mujoco.mj_name2id(self._ik_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ee_names
        ]
        foot_names = _pick_existing_default_feet(self._ik_model)
        if foot_names is None:
            raise ValueError("Could not infer two foot bodies required for grounded box IK")
        self._ik_foot_body_ids = [
            mujoco.mj_name2id(self._ik_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in foot_names
        ]

        configured_library_dir = str(self.get_parameter("library_dir").value).strip()
        self._library_dir = self._resolve_library_dir(configured_library_dir)
        # The semantic source face belongs to the reference motion, not to an
        # individual carried/placed posture. Infer it once from the canonical
        # pickup frame and retain it for the entire task.
        self._source_box_forward_axis = self._infer_library_source_forward_axis(
            self._source_box_forward_axis
        )

        self._object_to_manipulate = True
        self._current_box_center = np.array([10.0, 10.0, self._box_size_xyz[2] * 0.5], dtype=np.float64)
        self._current_box_quat_wxyz = np.asarray(
            DEFAULT_TARGET_BOX_QUAT_WXYZ,
            dtype=np.float64,
        ).copy()
        self._target_box_center = np.array([11.0, 10.0, self._box_size_xyz[2] * 0.5], dtype=np.float64)
        self._target_box_quat_wxyz = np.asarray(
            DEFAULT_TARGET_BOX_QUAT_WXYZ,
            dtype=np.float64,
        ).copy()
        self._target_root_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._target_root_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._has_current_box_pose = False
        self._fixed_start_box_center: np.ndarray | None = None
        self._fixed_start_box_quat_wxyz: np.ndarray | None = None
        self._fixed_target_box_center: np.ndarray | None = None
        self._fixed_target_box_quat_wxyz: np.ndarray | None = None
        self._fixed_box_hold_forward_axis: str | None = None
        self._latest_retargeted_keyframe_data: list[int] | None = None
        self._latest_retargeted_info_data: str | None = None
        self._latest_retargeted_keyframe_name: str | None = None
        self._latest_retargeted_object_to_manipulate = True

        self._service = self.create_service(
            RetargetKeyframe,
            retarget_keyframe_service,
            self._on_retarget_keyframe_request,
        )

        self.get_logger().info(
            "Retargeter service ready. service=%s. keyframes=%s. "
            "source_box_size=%s. target_box_size=%s. "
            "source_axes=%s/%s. target_axes=%s/%s"
            % (
                retarget_keyframe_service,
                self._library_dir,
                np.array2string(self._source_box_size_xyz, precision=3),
                np.array2string(self._box_size_xyz, precision=3),
                self._source_box_forward_axis,
                self._source_box_up_axis,
                self._box_hold_forward_axis,
                self._box_hold_up_axis,
            )
        )

    def _resolve_library_dir(self, configured_path: str) -> Path:
        if configured_path:
            path = Path(configured_path).expanduser()
            if path.exists():
                return path
            raise FileNotFoundError(f"Configured library_dir does not exist: {path}")

        candidates = [Path(__file__).resolve().parents[1] / "keyframes"]
        try:
            from ament_index_python.packages import get_package_share_directory

            candidates.append(Path(get_package_share_directory("lm")) / "keyframes")
        except Exception:
            pass

        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not locate keyframe library in: {', '.join(str(c) for c in candidates)}")

    def _process_keyframe(self, keyframe_name: str, object_to_manipulate: bool | None = None) -> tuple[bytes, str]:
        if object_to_manipulate is not None:
            self._object_to_manipulate = bool(object_to_manipulate)
        if keyframe_name in _OBJECT_REQUIRED_KEYFRAMES:
            self._object_to_manipulate = True
        payload = self._load_payload(keyframe_name)
        if keyframe_name == "stand_after_place":
            self._object_to_manipulate = False
            mode = self._retarget_stand_after_place(payload)
        elif self._object_to_manipulate:
            mode = self._retarget_for_box_task(keyframe_name, payload)
        else:
            mode = self._retarget_root_only(payload)
        payload["object_to_manipulate"] = np.asarray([self._object_to_manipulate], dtype=np.bool_)
        payload_bytes = self._serialize_payload(payload)

        self._latest_retargeted_keyframe_data = list(payload_bytes)
        self._latest_retargeted_keyframe_name = keyframe_name
        self._latest_retargeted_object_to_manipulate = self._object_to_manipulate
        info_data = json.dumps(
            {
                "input_keyframe": keyframe_name,
                "serialized_npz_bytes": len(payload_bytes),
                "mode": mode,
                "object_to_manipulate": bool(self._object_to_manipulate),
                "current_box_position_xyz": self._current_box_center.tolist(),
                "target_box_position_xyz": self._target_box_center.tolist(),
                "fixed_start_box_position_xyz": None
                if self._fixed_start_box_center is None
                else self._fixed_start_box_center.tolist(),
                "fixed_target_box_position_xyz": None
                if self._fixed_target_box_center is None
                else self._fixed_target_box_center.tolist(),
            }
        )
        self._latest_retargeted_info_data = info_data
        self.get_logger().info(
            f"Retargeted {keyframe_name} "
            f"(service response, {mode}, object_to_manipulate={self._object_to_manipulate})"
        )
        return payload_bytes, info_data

    def _on_retarget_keyframe_request(
        self,
        request: RetargetKeyframe.Request,
        response: RetargetKeyframe.Response,
    ) -> RetargetKeyframe.Response:
        keyframe_name = request.keyframe_name.strip()
        if not keyframe_name:
            response.success = False
            response.error_message = "keyframe_name is empty"
            return response

        try:
            self._object_to_manipulate = bool(request.object_to_manipulate)
            request_current_box_center, request_current_box_quat_wxyz = _pose_to_arrays(request.current_box_pose)
            request_target_box_center, request_target_box_quat_wxyz = _pose_to_arrays(request.target_box_pose)
            self._target_root_center, self._target_root_quat_wxyz = _pose_to_arrays(request.target_root_pose)

            if self._fixed_start_box_center is None or self._fixed_start_box_quat_wxyz is None:
                self._fixed_start_box_center = request_current_box_center.copy()
                self._fixed_start_box_quat_wxyz = request_current_box_quat_wxyz.copy()
                self.get_logger().info(
                    "Latched fixed start box pose: position=%s quat=%s"
                    % (
                        np.array2string(self._fixed_start_box_center, precision=3),
                        np.array2string(self._fixed_start_box_quat_wxyz, precision=3),
                    )
                )
            if self._fixed_target_box_center is None or self._fixed_target_box_quat_wxyz is None:
                self._fixed_target_box_center = request_target_box_center.copy()
                self._fixed_target_box_quat_wxyz = request_target_box_quat_wxyz.copy()
                self.get_logger().info(
                    "Latched fixed target box pose: position=%s quat=%s"
                    % (
                        np.array2string(self._fixed_target_box_center, precision=3),
                        np.array2string(self._fixed_target_box_quat_wxyz, precision=3),
                    )
                )

            if keyframe_name in _PICK_POSE_KEYFRAMES:
                self._current_box_center = self._fixed_start_box_center.copy()
                self._current_box_quat_wxyz = self._fixed_start_box_quat_wxyz.copy()
            else:
                self._current_box_center = request_current_box_center
                self._current_box_quat_wxyz = request_current_box_quat_wxyz
            self._target_box_center = self._fixed_target_box_center.copy()
            self._target_box_quat_wxyz = self._fixed_target_box_quat_wxyz.copy()
            self._has_current_box_pose = True
            requested_forward_axis = _normalize_axis_label(request.box_forward_axis)
            if self._fixed_box_hold_forward_axis is None:
                self._fixed_box_hold_forward_axis = requested_forward_axis
                self.get_logger().info(
                    "Latched robot-relative box forward axis for this task: %s"
                    % self._fixed_box_hold_forward_axis
                )
            elif requested_forward_axis != self._fixed_box_hold_forward_axis:
                self.get_logger().warn(
                    "Ignoring box forward-axis change %s -> %s to preserve the "
                    "post-pick robot/box orientation"
                    % (self._fixed_box_hold_forward_axis, requested_forward_axis)
                )
            self._box_hold_forward_axis = self._fixed_box_hold_forward_axis
            payload_bytes, info_data = self._process_keyframe(keyframe_name, self._object_to_manipulate)
            response.success = True
            response.error_message = ""
            response.retargeted_keyframe = list(payload_bytes)
            response.retargeted_info = info_data
        except Exception as exc:
            response.success = False
            response.error_message = str(exc)
            response.retargeted_keyframe = []
            response.retargeted_info = ""
            self.get_logger().error(f"Failed retargeting service request for '{keyframe_name}': {exc}")
        return response

    @staticmethod
    def _extract_body_arrays(payload: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        body_positions = np.asarray(payload["body_positions"], dtype=np.float64).copy()
        body_rotations = np.asarray(payload["body_rotations"], dtype=np.float64).copy()
        if body_positions.ndim == 3:
            body_positions = body_positions[0]
        if body_rotations.ndim == 3:
            body_rotations = body_rotations[0]
        return body_positions, body_rotations

    @staticmethod
    def _write_body_arrays(payload: dict[str, np.ndarray], body_positions: np.ndarray, body_rotations: np.ndarray) -> None:
        if payload["body_positions"].ndim == 3:
            payload["body_positions"][0] = body_positions.astype(payload["body_positions"].dtype)
        else:
            payload["body_positions"] = body_positions.astype(payload["body_positions"].dtype)
        if payload["body_rotations"].ndim == 3:
            payload["body_rotations"][0] = body_rotations.astype(payload["body_rotations"].dtype)
        else:
            payload["body_rotations"] = body_rotations.astype(payload["body_rotations"].dtype)

    def _apply_root_pose(self, payload: dict[str, np.ndarray], root_center_new: np.ndarray, root_quat_new: np.ndarray) -> None:
        body_names = [str(x) for x in payload["body_names"]]
        pelvis_idx = body_names.index("pelvis") if "pelvis" in body_names else 0
        body_positions, body_rotations = self._extract_body_arrays(payload)

        root_center_old = body_positions[pelvis_idx].copy()
        root_quat_old = body_rotations[pelvis_idx].copy()
        yaw_old = _yaw_from_quat_wxyz(root_quat_old)
        yaw_new = _yaw_from_quat_wxyz(root_quat_new)
        delta_yaw = yaw_new - yaw_old
        delta_q = _yaw_to_quat_wxyz(delta_yaw)
        rot_delta = _quat_wxyz_to_rotmat(delta_q)

        body_positions = (body_positions - root_center_old[None, :]) @ rot_delta.T + root_center_new[None, :]
        body_rotations = np.vstack([_quat_wxyz_multiply(delta_q, q) for q in body_rotations])
        self._write_body_arrays(payload, body_positions, body_rotations)

    def _build_qpos_from_payload(self, payload: dict[str, np.ndarray]) -> np.ndarray:
        q = np.zeros(self._ik_model.nq, dtype=np.float64)
        body_names = [str(x) for x in payload["body_names"]]
        pelvis_idx = body_names.index("pelvis") if "pelvis" in body_names else 0
        body_positions, body_rotations = self._extract_body_arrays(payload)
        q[0:3] = body_positions[pelvis_idx]
        quat = body_rotations[pelvis_idx].astype(np.float64)
        q[3:7] = quat / max(np.linalg.norm(quat), 1e-12)
        if "dof_positions" in payload:
            dof = np.asarray(payload["dof_positions"], dtype=np.float64)
            if dof.ndim == 2:
                dof = dof[0]
            n = min(dof.shape[0], self._ik_model.nq - 7)
            q[7 : 7 + n] = dof[:n]
        return q

    def _write_ik_result_to_payload(self, payload: dict[str, np.ndarray], q_new: np.ndarray) -> None:
        self._ik_data.qpos[:] = q_new
        mujoco.mj_forward(self._ik_model, self._ik_data)

        if "dof_positions" in payload:
            dof = np.asarray(payload["dof_positions"])
            n = dof.shape[0] if dof.ndim == 1 else dof.shape[-1]
            new_dof = q_new[7 : 7 + n].astype(payload["dof_positions"].dtype)
            payload["dof_positions"] = new_dof

        if "body_positions" in payload and "body_names" in payload:
            body_names = [str(x) for x in payload["body_names"]]
            body_positions, body_rotations = self._extract_body_arrays(payload)
            for i, name in enumerate(body_names):
                bid = mujoco.mj_name2id(self._ik_model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid < 0:
                    continue
                body_positions[i] = self._ik_data.xpos[bid]
                body_rotations[i] = self._ik_data.xquat[bid]
            self._write_body_arrays(payload, body_positions, body_rotations)

    def _infer_source_forward_axis(self, root_quat_wxyz: np.ndarray, src_box_quat_wxyz: np.ndarray) -> str:
        root_rot = _quat_wxyz_to_rotmat(root_quat_wxyz)
        root_forward = root_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        root_forward[2] = 0.0
        root_forward = _normalize_vec(root_forward, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        box_rot = _quat_wxyz_to_rotmat(src_box_quat_wxyz)
        source_up_dimension = _axis_dimension_index(self._source_box_up_axis)
        axis_world = {
            label: _normalize_vec(box_rot @ vec, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            for label, vec in _AXIS_TO_LOCAL_VEC.items()
            if _axis_dimension_index(label) != source_up_dimension
        }
        return max(axis_world.keys(), key=lambda k: float(np.dot(axis_world[k], root_forward)))

    def _source_object_quat(
        self,
        payload: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Return the authored source orientation without training-time offsets."""
        stored_src_quat = np.asarray(payload["object_quat_wxyz"], dtype=np.float64).copy()
        if np.linalg.norm(stored_src_quat) < 1e-12:
            stored_src_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        stored_frame_value = np.asarray(
            payload.get("object_quat_frame", np.asarray("unspecified"))
        ).reshape(-1)[0]
        if isinstance(stored_frame_value, bytes):
            stored_frame = stored_frame_value.decode("utf-8")
        else:
            stored_frame = str(stored_frame_value)
        src_quat = _quat_wxyz_normalize(stored_src_quat)
        return src_quat, stored_src_quat, stored_frame

    def _infer_library_source_forward_axis(self, fallback_axis: str) -> str:
        pickup_path = self._library_dir / "crouch_to_pick.npz"
        if not pickup_path.exists():
            self.get_logger().warning(
                "No crouch_to_pick.npz in the keyframe library; using configured "
                f"source_box_forward_axis={fallback_axis}."
            )
            return fallback_axis
        payload = self._load_payload("crouch_to_pick")
        if "object_quat_wxyz" not in payload:
            self.get_logger().warning(
                "crouch_to_pick has no object quaternion; using configured "
                f"source_box_forward_axis={fallback_axis}."
            )
            return fallback_axis
        qpos = self._build_qpos_from_payload(payload)
        source_quat, _, _ = self._source_object_quat(payload)
        inferred_axis = self._infer_source_forward_axis(qpos[3:7], source_quat)
        self.get_logger().info(
            "Latched reference pickup axis from crouch_to_pick: %s"
            % inferred_axis
        )
        return inferred_axis

    def _apply_box_ik(self, payload: dict[str, np.ndarray], dst_center: np.ndarray, dst_quat: np.ndarray) -> None:
        q0 = self._build_qpos_from_payload(payload)
        if "object_position_xyz" in payload and "object_quat_wxyz" in payload:
            src_center = np.asarray(payload["object_position_xyz"], dtype=np.float64).copy()
            src_quat, stored_src_quat, stored_frame = self._source_object_quat(payload)
            self.get_logger().info(
                "Retargeting with object. "
                f"src_center={src_center}, stored_src_quat={stored_src_quat}, "
                f"stored_frame={stored_frame}, source_quat={src_quat}, "
                f"dest_center={dst_center}, dst_quat={dst_quat}"
            )
        else:
            self._ik_data.qpos[:] = q0
            mujoco.mj_forward(self._ik_model, self._ik_data)
            ee_world = np.vstack(
                [_get_body_pos(self._ik_data, body_id) for body_id in self._ik_ee_body_ids]
            )
            src_center = ee_world.mean(axis=0)
            src_quat = np.asarray(dst_quat, dtype=np.float64).copy()
            self.get_logger().info(f"Retargeting without object. Using ee to estimate: src_center={src_center}, src_quat={src_quat}")

        src_box = BoxFrame(
            center=src_center,
            size=self._source_box_size_xyz.copy(),
            quat_wxyz=src_quat,
        )
        preferred_dst_quat = np.asarray(dst_quat, dtype=np.float64).copy()
        dst_box_used = BoxFrame(
            center=np.asarray(dst_center, dtype=np.float64),
            size=self._box_size_xyz.copy(),
            quat_wxyz=preferred_dst_quat,
        )
        result = retarget_qpos_for_box_grasp(
            self._ik_model,
            self._ik_data,
            qpos=q0,
            source_box=src_box,
            target_box=dst_box_used,
            hand_body_ids=self._ik_ee_body_ids,
            foot_body_ids=self._ik_foot_body_ids,
            source_forward_axis=self._source_box_forward_axis,
            source_up_axis=self._source_box_up_axis,
            target_forward_axis=self._box_hold_forward_axis,
            target_up_axis=self._box_hold_up_axis,
            fixed_foot_weight=self._ik_foot_constraint_weight,
            max_foot_residual_m=self._ik_max_foot_residual_m,
        )
        self.get_logger().info(
            "Grounded matched box frames | src forward=%s up=%s size=%s | "
            "tgt forward=%s up=%s size=%s | yaw_delta_deg=%.2f | "
            "root_corner=%s | foot_residual=%.6f m"
            % (
                result.source_forward_axis,
                result.source_up_axis,
                np.array2string(result.source_matched_size, precision=3),
                result.target_forward_axis,
                result.target_up_axis,
                np.array2string(result.target_matched_size, precision=3),
                math.degrees(result.yaw_delta_rad),
                np.array2string(result.root_corner_code, precision=0),
                result.foot_residual_m,
            )
        )
        if not np.isfinite(result.hand_residual_m) or result.hand_residual_m > self._ik_max_residual_m:
            raise RuntimeError(
                f"Retargeting IK residual {result.hand_residual_m:.4f} m exceeds "
                f"limit {self._ik_max_residual_m:.4f} m"
            )
        self.get_logger().info(
            f"Retargeting IK residual={result.hand_residual_m:.6f} m"
        )
        self._write_ik_result_to_payload(payload, result.qpos)

    def _nearest_edge_root_pose(self, box_center: np.ndarray, box_quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rot = _quat_wxyz_to_rotmat(box_quat_wxyz)
        hx, hy = float(self._box_size_xyz[0] * 0.5), float(self._box_size_xyz[1] * 0.5)
        edge_centers_local = np.array([[hx, 0, 0], [-hx, 0, 0], [0, hy, 0], [0, -hy, 0]], dtype=np.float64)
        edge_centers_world = edge_centers_local @ rot.T + box_center[None, :]
        dists = np.linalg.norm(edge_centers_world[:, :2], axis=1)
        nearest_idx = int(np.argmin(dists))
        edge_center = edge_centers_world[nearest_idx]

        outward = edge_center[:2] - box_center[:2]
        norm = np.linalg.norm(outward)
        if norm < 1e-9:
            outward = np.array([1.0, 0.0], dtype=np.float64)
        else:
            outward = outward / norm

        root_xy = edge_center[:2] + self._stand_before_pick_offset_m * outward
        facing_dir = -outward
        yaw = float(math.atan2(facing_dir[1], facing_dir[0]))
        root_quat = _yaw_to_quat_wxyz(yaw)
        return np.array([root_xy[0], root_xy[1], 0.0], dtype=np.float64), root_quat

    def _load_payload(self, keyframe_name: str) -> dict[str, np.ndarray]:
        path = self._library_dir / f"{keyframe_name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Keyframe not found: {path}")
        with np.load(path, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}

    @staticmethod
    def _write_physical_object_pose(
        payload: dict[str, np.ndarray],
        center: np.ndarray,
        quat_wxyz: np.ndarray,
    ) -> None:
        """Write a canonical physical pose without retaining legacy markers."""
        payload["object_position_xyz"] = np.asarray(center).astype(
            payload["object_position_xyz"].dtype
        )
        payload["object_quat_wxyz"] = np.asarray(quat_wxyz).astype(
            payload["object_quat_wxyz"].dtype
        )
        payload["object_quat_frame"] = np.asarray("physical_box")
        payload.pop("object_mesh_offset_removed_rpy_deg", None)

    @staticmethod
    def _serialize_payload(payload: dict[str, np.ndarray]) -> bytes:
        buf = BytesIO()
        np.savez(buf, **payload)
        return buf.getvalue()

    def _retarget_for_box_task(self, keyframe_name: str, payload: dict[str, np.ndarray]) -> str:
        if keyframe_name == "crouch_to_pick":
            self._apply_box_ik(payload, self._current_box_center, self._current_box_quat_wxyz)
            self._write_physical_object_pose(
                payload,
                self._current_box_center,
                self._current_box_quat_wxyz,
            )
            return "ik_to_current_box_pick"

        if keyframe_name == "stand_after_pick":
            lifted_box_center = self._current_box_center.copy()
            lifted_box_center[2] = self._stand_after_pick_height_m
            self._apply_box_ik(payload, lifted_box_center, self._current_box_quat_wxyz)
            self._write_physical_object_pose(
                payload,
                lifted_box_center,
                self._current_box_quat_wxyz,
            )
            return "ik_to_lifted_box_stand_after_pick"

        if keyframe_name == "stand_before_pick":
            self._apply_root_pose(payload, self._target_root_center, self._target_root_quat_wxyz)

            if self._has_current_box_pose:
                self._write_physical_object_pose(
                    payload,
                    self._current_box_center,
                    self._current_box_quat_wxyz,
                )

            return "stand_before_pick_root_from_vlm"

        if keyframe_name == "stand_before_place":
            above_target = self._target_box_center.copy()
            above_target[2] = self._stand_before_place_height_m
            self._apply_box_ik(payload, above_target, self._target_box_quat_wxyz)
            self._write_physical_object_pose(
                payload,
                above_target,
                self._target_box_quat_wxyz,
            )
            return "ik_to_box_above_place_target"

        if keyframe_name == "crouch_to_place":
            place_target = self._target_box_center.copy()
            place_target[2] = 0.5 * self._box_size_xyz[2]
            self._apply_box_ik(payload, place_target, self._target_box_quat_wxyz)
            self._write_physical_object_pose(
                payload,
                place_target,
                self._target_box_quat_wxyz,
            )
            return "ik_to_box_on_ground_at_place_target"

        if keyframe_name == "stand_after_place":
            return self._retarget_stand_after_place(payload)

        return "unsupported_keyframe_no_change"

    def _zero_object_targets(self, payload: dict[str, np.ndarray]) -> None:
        for key in (
            "object_pose",
            "object_position_xyz",
            "object_quat_wxyz",
            "masked_goal_object_pos",
            "masked_goal_object_quat",
        ):
            if key in payload:
                payload[key] = np.zeros_like(payload[key])
        payload.pop("object_mesh_offset_removed_rpy_deg", None)

    def _retarget_stand_after_place(self, payload: dict[str, np.ndarray]) -> str:
        self._apply_root_pose(payload, self._target_root_center, self._target_root_quat_wxyz)
        self._zero_object_targets(payload)
        return "stand_after_place_current_root"

    def _retarget_root_only(self, payload: dict[str, np.ndarray]) -> str:
        self._apply_root_pose(payload, self._target_root_center, self._target_root_quat_wxyz)
        self._zero_object_targets(payload)
        return "root_only_retarget"

def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = KeyframeRetargeterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
