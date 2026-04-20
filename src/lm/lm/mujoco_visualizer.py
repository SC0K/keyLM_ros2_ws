from __future__ import annotations

from io import BytesIO
from pathlib import Path
import threading

import mujoco  # type: ignore[import-not-found]
import mujoco.viewer  # type: ignore[import-not-found]
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_to_arrays(msg: PoseStamped) -> tuple[np.ndarray, np.ndarray]:
    pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64)
    quat = np.array(
        [
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ],
        dtype=np.float64,
    )
    return pos, quat


class MujocoVisualizerNode(Node):
    def __init__(self) -> None:
        super().__init__("mujoco_visualizer")

        self.declare_parameter("robot", "g1")
        self.declare_parameter("robot_xml", "")
        self.declare_parameter("current_box_pose_topic", "/vlm/current_box_pose")
        self.declare_parameter("target_box_pose_topic", "/vlm/target_box_pose")
        self.declare_parameter("retargeted_keyframe_topic", "/retargetor/output_keyframe")
        self.declare_parameter("box_size_xyz", [0.35, 0.35, 0.35])

        self._lock = threading.Lock()
        self._box_size_xyz = np.asarray(self.get_parameter("box_size_xyz").value, dtype=np.float64)
        self._current_box_center: np.ndarray | None = None
        self._current_box_quat: np.ndarray | None = None
        self._target_box_center: np.ndarray | None = None
        self._target_box_quat: np.ndarray | None = None
        self._latest_qpos: np.ndarray | None = None
        self._have_new_qpos = False
        self._overlap_warned = False

        robot_xml = str(self.get_parameter("robot_xml").value).strip()
        if robot_xml:
            xml_path = Path(robot_xml).expanduser()
        else:
            robot = str(self.get_parameter("robot").value)
            xml_path = self._default_robot_xml(robot)

        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)

        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("current_box_pose_topic").value),
            self._on_current_box_pose,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("target_box_pose_topic").value),
            self._on_target_box_pose,
            10,
        )
        self.create_subscription(
            UInt8MultiArray,
            str(self.get_parameter("retargeted_keyframe_topic").value),
            self._on_retargeted_keyframe,
            10,
        )
        self.get_logger().info(f"MuJoCo visualizer using model: {xml_path}")

    def _default_robot_xml(self, robot: str) -> Path:
        if robot.lower() != "g1":
            raise ValueError(f"Unsupported robot '{robot}'. Only g1 is supported.")
        return self._resolve_model_path("models/g1/g1_29dof.xml")

    def _resolve_model_path(self, rel: str) -> Path:
        candidates = [Path(__file__).resolve().parents[1] / rel]
        try:
            from ament_index_python.packages import get_package_share_directory

            candidates.append(Path(get_package_share_directory("lm")) / rel)
        except Exception:
            pass
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(f"Could not find model '{rel}' in: {', '.join(str(x) for x in candidates)}")

    def _on_current_box_pose(self, msg: PoseStamped) -> None:
        pos, quat = _pose_to_arrays(msg)
        with self._lock:
            self._current_box_center = pos
            self._current_box_quat = quat

    def _on_target_box_pose(self, msg: PoseStamped) -> None:
        pos, quat = _pose_to_arrays(msg)
        with self._lock:
            self._target_box_center = pos
            self._target_box_quat = quat

    def _on_retargeted_keyframe(self, msg: UInt8MultiArray) -> None:
        try:
            payload = self._decode_payload(msg.data)
            qpos = self._qpos_from_payload(payload)
            with self._lock:
                self._latest_qpos = qpos
                self._have_new_qpos = True
        except Exception as exc:
            self.get_logger().error(f"Failed decoding retargeted keyframe: {exc}")

    @staticmethod
    def _decode_payload(data_list: list[int]) -> dict[str, np.ndarray]:
        blob = bytes(data_list)
        with np.load(BytesIO(blob), allow_pickle=True) as npz:
            return {k: npz[k] for k in npz.files}

    def _qpos_from_payload(self, payload: dict[str, np.ndarray]) -> np.ndarray:
        qpos = self._data.qpos.copy()

        body_names = [str(x) for x in payload.get("body_names", [])]
        if "pelvis" in body_names:
            pelvis_idx = body_names.index("pelvis")
        else:
            pelvis_idx = 0

        if "body_positions" in payload:
            bp = np.asarray(payload["body_positions"], dtype=np.float64)
            if bp.ndim == 3:
                bp = bp[0]
            qpos[0:3] = bp[pelvis_idx]

        if "body_rotations" in payload:
            br = np.asarray(payload["body_rotations"], dtype=np.float64)
            if br.ndim == 3:
                br = br[0]
            q = br[pelvis_idx]
            qpos[3:7] = q / max(np.linalg.norm(q), 1e-12)

        if "dof_positions" in payload:
            dof = np.asarray(payload["dof_positions"], dtype=np.float64)
            if dof.ndim == 2:
                dof = dof[0]
            n = min(dof.shape[0], self._model.nq - 7)
            qpos[7 : 7 + n] = dof[:n]

        return qpos

    def _add_box_geom(self, viewer, center: np.ndarray, quat: np.ndarray, rgba: np.ndarray) -> None:
        scn = viewer.user_scn
        if scn.ngeom >= scn.maxgeom:
            return
        geom = scn.geoms[scn.ngeom]
        size = (0.5 * self._box_size_xyz).astype(np.float64)
        mat = _quat_wxyz_to_rotmat(quat).reshape(-1)
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=center.astype(np.float64),
            mat=mat,
            rgba=rgba.astype(np.float32),
        )
        scn.ngeom += 1

    def _add_axes(self, viewer, center: np.ndarray, quat: np.ndarray, axis_len: float = 0.2, radius: float = 0.006) -> None:
        scn = viewer.user_scn
        rot = _quat_wxyz_to_rotmat(quat)
        axes = [
            (rot[:, 0], np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float32)),  # X red
            (rot[:, 1], np.array([0.0, 1.0, 0.0, 0.9], dtype=np.float32)),  # Y green
            (rot[:, 2], np.array([0.0, 0.4, 1.0, 0.9], dtype=np.float32)),  # Z blue-ish
        ]
        for direction, rgba in axes:
            if scn.ngeom >= scn.maxgeom:
                return
            p0 = center.astype(np.float64)
            p1 = (center + axis_len * direction).astype(np.float64)
            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.array([radius, radius, radius], dtype=np.float64),
                pos=np.zeros(3, dtype=np.float64),
                mat=np.eye(3, dtype=np.float64).reshape(-1),
                rgba=rgba,
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                radius,
                p0,
                p1,
            )
            scn.ngeom += 1

    def _render_once(self, viewer) -> None:
        with self._lock:
            if self._have_new_qpos and self._latest_qpos is not None:
                self._data.qpos[:] = self._latest_qpos
                self._have_new_qpos = False
            cur_c = None if self._current_box_center is None else self._current_box_center.copy()
            cur_q = None if self._current_box_quat is None else self._current_box_quat.copy()
            tgt_c = None if self._target_box_center is None else self._target_box_center.copy()
            tgt_q = None if self._target_box_quat is None else self._target_box_quat.copy()

        mujoco.mj_forward(self._model, self._data)

        viewer.user_scn.ngeom = 0
        if cur_c is not None and cur_q is not None:
            self._add_box_geom(viewer, cur_c, cur_q, np.array([0.95, 0.65, 0.15, 0.35], dtype=np.float32))
            self._add_axes(viewer, cur_c, cur_q)
        if tgt_c is not None and tgt_q is not None:
            self._add_box_geom(viewer, tgt_c, tgt_q, np.array([0.15, 0.65, 0.95, 0.35], dtype=np.float32))
            self._add_axes(viewer, tgt_c, tgt_q)
        if cur_c is not None and tgt_c is not None:
            if np.linalg.norm(cur_c - tgt_c) < 1e-4:
                if not self._overlap_warned:
                    self.get_logger().warn(
                        "Current and target box poses are nearly identical; boxes overlap in visualization."
                    )
                    self._overlap_warned = True
            else:
                self._overlap_warned = False
        viewer.sync()

    def run(self) -> None:
        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            self.get_logger().info("MuJoCo viewer started.")
            while rclpy.ok() and viewer.is_running():
                rclpy.spin_once(self, timeout_sec=0.01)
                self._render_once(viewer)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MujocoVisualizerNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
