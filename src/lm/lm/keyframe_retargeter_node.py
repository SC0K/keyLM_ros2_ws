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

from lm.box_config import DEFAULT_BOX_SIZE_XYZ, parse_box_size_xyz
from lm_interfaces.srv import RetargetKeyframe

from lm.keyframe_box_retarget import (
    BoxFrame,
    _default_robot_xml,
    _get_body_pos,
    _pick_existing_default_ee,
    _pick_existing_default_feet,
    infer_scaled_targets,
    infer_scaled_targets_with_corner_surface_alignment,
    solve_multi_ee_ik,
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


def map_points_by_frame_transform(
    src_pos: np.ndarray,
    src_quat_wxyz: np.ndarray,
    dst_pos: np.ndarray,
    dst_quat_wxyz: np.ndarray,
    pts_world: np.ndarray,
) -> np.ndarray:
    '''Map points by applying the relative transform from source frame to destination frame.'''
    src_rot = _quat_wxyz_to_rotmat(_quat_wxyz_normalize(src_quat_wxyz))
    dst_rot = _quat_wxyz_to_rotmat(_quat_wxyz_normalize(dst_quat_wxyz))
    local = (np.asarray(pts_world, dtype=np.float64) - np.asarray(src_pos, dtype=np.float64)) @ src_rot
    return local @ dst_rot.T + np.asarray(dst_pos, dtype=np.float64)


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
            list(DEFAULT_BOX_SIZE_XYZ),
            descriptor=ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter("box_hold_forward_axis", "x")
        self.declare_parameter("stand_before_pick_offset_m", 0.2)
        self.declare_parameter("stand_after_pick_height_m", 0.9)
        self.declare_parameter("stand_before_place_height_m", 0.9)
        self.declare_parameter("foot_motion_penalty_weight", 0.1)
        self.declare_parameter("ee_root_penalty_weight", 2.0)
        self.declare_parameter("robot", "g1")
        self.declare_parameter("robot_xml", "")

        retarget_keyframe_service = str(self.get_parameter("retarget_keyframe_service").value)
        self._box_size_xyz = parse_box_size_xyz(self.get_parameter("box_size_xyz").value)
        self._box_hold_forward_axis = _normalize_axis_label(
            str(self.get_parameter("box_hold_forward_axis").value)
        )
        self._stand_before_pick_offset_m = float(self.get_parameter("stand_before_pick_offset_m").value)
        self._stand_after_pick_height_m = float(self.get_parameter("stand_after_pick_height_m").value)
        self._stand_before_place_height_m = float(self.get_parameter("stand_before_place_height_m").value)
        self._foot_motion_penalty_weight = float(self.get_parameter("foot_motion_penalty_weight").value)
        self._ee_root_penalty_weight = float(self.get_parameter("ee_root_penalty_weight").value)
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
        foot_names = _pick_existing_default_feet(self._ik_model) or []
        self._ik_foot_body_ids = [
            mujoco.mj_name2id(self._ik_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in foot_names
            if mujoco.mj_name2id(self._ik_model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
        ]

        configured_library_dir = str(self.get_parameter("library_dir").value).strip()
        self._library_dir = self._resolve_library_dir(configured_library_dir)

        self._object_to_manipulate = True
        self._current_box_center = np.array([10.0, 10.0, self._box_size_xyz[2] * 0.5], dtype=np.float64)
        self._current_box_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._target_box_center = np.array([11.0, 10.0, self._box_size_xyz[2] * 0.5], dtype=np.float64)
        self._target_box_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._target_root_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._target_root_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._has_current_box_pose = False
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
            "Retargeter service ready. service=%s. keyframes=%s. target_forward_axis=%s"
            % (
                retarget_keyframe_service,
                self._library_dir,
                self._box_hold_forward_axis,
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
            self._current_box_center, self._current_box_quat_wxyz = _pose_to_arrays(request.current_box_pose)
            self._target_box_center, self._target_box_quat_wxyz = _pose_to_arrays(request.target_box_pose)
            self._target_root_center, self._target_root_quat_wxyz = _pose_to_arrays(request.target_root_pose)
            self._has_current_box_pose = True
            self._box_hold_forward_axis = _normalize_axis_label(request.box_forward_axis)
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
        axis_world = {
            label: _normalize_vec(box_rot @ vec, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            for label, vec in _AXIS_TO_LOCAL_VEC.items()
        }
        return max(axis_world.keys(), key=lambda k: float(np.dot(axis_world[k], root_forward)))

    def _apply_box_ik(self, payload: dict[str, np.ndarray], dst_center: np.ndarray, dst_quat: np.ndarray) -> None:
        q0 = self._build_qpos_from_payload(payload)
        self._ik_data.qpos[:] = q0
        mujoco.mj_forward(self._ik_model, self._ik_data)

        ee_world = np.vstack([_get_body_pos(self._ik_data, bid) for bid in self._ik_ee_body_ids])
        if "object_position_xyz" in payload and "object_quat_wxyz" in payload:
            src_center = np.asarray(payload["object_position_xyz"], dtype=np.float64).copy()
            src_quat = np.asarray(payload["object_quat_wxyz"], dtype=np.float64).copy()
            self.get_logger().info(f"Retargeting with object. src_center={src_center}, src_quat={src_quat}, dest_center={dst_center}, dst_quat={dst_quat}")
            if np.linalg.norm(src_quat) < 1e-12:
                src_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            src_center = ee_world.mean(axis=0)
            src_quat = np.asarray(dst_quat, dtype=np.float64).copy()
            self.get_logger().info(f"Retargeting without object. Using ee to estimate: src_center={src_center}, src_quat={src_quat}")

        src_box = BoxFrame(center=src_center, size=self._box_size_xyz.copy(), quat_wxyz=src_quat)
        preferred_dst_quat = np.asarray(dst_quat, dtype=np.float64).copy()
        dst_box_used = BoxFrame(
            center=np.asarray(dst_center, dtype=np.float64),
            size=self._box_size_xyz.copy(),
            quat_wxyz=preferred_dst_quat,
        )
        # Stage 1: keep source orientation for EE target inference.
        dst_box_infer = BoxFrame(
            center=src_box.center.copy(),
            size=dst_box_used.size.copy(),
            quat_wxyz=src_box.quat_wxyz.copy(),
        )
        targets_infer = infer_scaled_targets_with_corner_surface_alignment(
            src_box=src_box,
            dst_box=dst_box_infer,
            ee_world=ee_world,
            robot_root=q0[0:3].copy(),
        )

        src_forward_axis = self._infer_source_forward_axis(q0[3:7], src_box.quat_wxyz)
        src_forward_world = _forward_world_from_axis(src_box.quat_wxyz, src_forward_axis)
        dst_forward_world = _forward_world_from_axis(dst_box_used.quat_wxyz, self._box_hold_forward_axis)
        # Keep only horizontal (yaw) rotation for source->target transform.
        src_stage2_quat = _yaw_quat_from_forward_xy(src_forward_world, src_box.quat_wxyz)
        dst_stage2_quat = _yaw_quat_from_forward_xy(dst_forward_world, dst_box_used.quat_wxyz)
        # Stage 2: rigidly rotate/translate inferred targets to final target position with yaw-only rotation.
        targets = map_points_by_frame_transform(
            src_pos=dst_box_infer.center,
            src_quat_wxyz=src_stage2_quat,
            dst_pos=dst_box_used.center,
            dst_quat_wxyz=dst_stage2_quat,
            pts_world=targets_infer,
        )
        self.get_logger().info(
            "Forward dirs | src forward(%s)=%s | tgt forward(%s)=%s"
            % (
                src_forward_axis,
                np.array2string(src_forward_world, precision=3),
                self._box_hold_forward_axis,
                np.array2string(dst_forward_world, precision=3),
            )
        )
        # Move base together with the source->target box frame transform.
        q_init = q0.copy()
        base_before = q0[0:3].copy()
        base_mapped = map_points_by_frame_transform(
            src_pos=src_box.center,
            src_quat_wxyz=src_stage2_quat,
            dst_pos=dst_box_used.center,
            dst_quat_wxyz=dst_stage2_quat,
            pts_world=base_before[None, :],
        )[0]
        q_init[0:2] = base_mapped[0:2]
        q_init[3:7] = map_orientation_by_frame_transform(
            src_quat_wxyz=src_stage2_quat,
            dst_quat_wxyz=dst_stage2_quat,
            quat_wxyz=q0[3:7],
        )

        relative_body_ids: list[int] = []
        relative_offsets: list[np.ndarray] = []
        relative_masks: list[np.ndarray] = []
        relative_weights: list[float] = []
        if self._ik_foot_body_ids:
            self._ik_data.qpos[:] = q_init
            mujoco.mj_forward(self._ik_model, self._ik_data)
            foot_positions = np.vstack([_get_body_pos(self._ik_data, bid) for bid in self._ik_foot_body_ids])
            relative_body_ids.extend(self._ik_foot_body_ids)
            relative_offsets.append(foot_positions - q_init[0:3][None, :])
            relative_masks.append(np.ones((len(self._ik_foot_body_ids), 3), dtype=np.float64))
            relative_weights.extend([self._foot_motion_penalty_weight] * len(self._ik_foot_body_ids))

        if self._ee_root_penalty_weight > 0.0:
            relative_body_ids.extend(self._ik_ee_body_ids)
            relative_offsets.append(targets - q_init[0:3][None, :])
            relative_masks.append(np.ones((len(self._ik_ee_body_ids), 3), dtype=np.float64))
            relative_weights.extend([self._ee_root_penalty_weight] * len(self._ik_ee_body_ids))

        relative_body_offsets = np.vstack(relative_offsets) if relative_offsets else None
        relative_body_mask = np.vstack(relative_masks) if relative_masks else None
        relative_body_weight = np.asarray(relative_weights, dtype=np.float64) if relative_weights else 1.0

        q_new, _ = solve_multi_ee_ik(
            self._ik_model,
            self._ik_data,
            q_init=q_init,
            body_ids=self._ik_ee_body_ids,
            target_positions=targets,
            relative_body_ids=relative_body_ids,
            relative_body_offsets=relative_body_offsets,
            relative_body_mask=relative_body_mask,
            relative_body_weight=relative_body_weight,
        )
        self._write_ik_result_to_payload(payload, q_new)

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
    def _serialize_payload(payload: dict[str, np.ndarray]) -> bytes:
        buf = BytesIO()
        np.savez(buf, **payload)
        return buf.getvalue()

    def _retarget_for_box_task(self, keyframe_name: str, payload: dict[str, np.ndarray]) -> str:
        if keyframe_name == "crouch_to_pick":
            self._apply_box_ik(payload, self._current_box_center, self._current_box_quat_wxyz)
            payload["object_position_xyz"] = self._current_box_center.astype(payload["object_position_xyz"].dtype)
            payload["object_quat_wxyz"] = self._current_box_quat_wxyz.astype(payload["object_quat_wxyz"].dtype)
            return "ik_to_current_box_pick"

        if keyframe_name == "stand_after_pick":
            lifted_box_center = self._current_box_center.copy()
            lifted_box_center[2] = self._stand_after_pick_height_m
            self._apply_box_ik(payload, lifted_box_center, self._current_box_quat_wxyz)
            payload["object_position_xyz"] = lifted_box_center.astype(payload["object_position_xyz"].dtype)
            payload["object_quat_wxyz"] = self._current_box_quat_wxyz.astype(payload["object_quat_wxyz"].dtype)
            return "ik_to_lifted_box_stand_after_pick"

        if keyframe_name == "stand_before_pick":
            self._apply_root_pose(payload, self._target_root_center, self._target_root_quat_wxyz)

            if self._has_current_box_pose:
                payload["object_position_xyz"] = self._current_box_center.astype(payload["object_position_xyz"].dtype)
                payload["object_quat_wxyz"] = self._current_box_quat_wxyz.astype(payload["object_quat_wxyz"].dtype)

            return "stand_before_pick_root_from_vlm"

        if keyframe_name == "stand_before_place":
            above_target = self._target_box_center.copy()
            above_target[2] = self._stand_before_place_height_m
            self._apply_box_ik(payload, above_target, self._target_box_quat_wxyz)
            payload["object_position_xyz"] = above_target.astype(payload["object_position_xyz"].dtype)
            payload["object_quat_wxyz"] = self._target_box_quat_wxyz.astype(payload["object_quat_wxyz"].dtype)
            return "ik_to_box_above_place_target"

        if keyframe_name == "crouch_to_place":
            place_target = self._target_box_center.copy()
            place_target[2] = 0.5 * self._box_size_xyz[2]
            self._apply_box_ik(payload, place_target, self._target_box_quat_wxyz)
            payload["object_position_xyz"] = place_target.astype(payload["object_position_xyz"].dtype)
            payload["object_quat_wxyz"] = self._target_box_quat_wxyz.astype(payload["object_quat_wxyz"].dtype)
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
