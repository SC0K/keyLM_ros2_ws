from __future__ import annotations

from pathlib import Path
import threading
import time

import mujoco  # type: ignore[import-not-found]
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from crl_humanoid_msgs.msg import Monitor
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import Image


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _parse_vec3(value, fallback: tuple[float, float, float]) -> np.ndarray:
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        if len(parts) == 3:
            return np.asarray([float(x) for x in parts], dtype=np.float64)
        return np.asarray(fallback, dtype=np.float64)
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 3:
            return arr
    except Exception:
        pass
    return np.asarray(fallback, dtype=np.float64)


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
    return pos, _normalize_quat_wxyz(quat)


class SceneCameraNode(Node):
    def __init__(self) -> None:
        super().__init__("scene_camera_node")

        self.declare_parameter("backend", "mujoco")
        self.declare_parameter("topic", "/camera/image_raw")
        self.declare_parameter("real_image_topic", "/real_camera/image_raw")
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("frame_id", "vlm_camera")
        self.declare_parameter("robot_xml", "")
        self.declare_parameter("monitor_topic", "/g1_sim/monitor")
        self.declare_parameter("object_pose_topic", "/actual_box_pose")
        self.declare_parameter("object_joint_name", "box_freejoint")
        self.declare_parameter("camera_name", "")
        self.declare_parameter(
            "camera_lookat",
            [0.7, 0.0, 0.55],
            descriptor=ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter("camera_distance", 2.4)
        self.declare_parameter("camera_azimuth", -135.0)
        self.declare_parameter("camera_elevation", -18.0)

        self._backend = str(self.get_parameter("backend").value).strip().lower()
        self._topic = str(self.get_parameter("topic").value)
        self._real_image_topic = str(self.get_parameter("real_image_topic").value)
        self._rate_hz = float(self.get_parameter("rate_hz").value)
        self._width = int(self.get_parameter("width").value)
        self._height = int(self.get_parameter("height").value)
        self._frame_id = str(self.get_parameter("frame_id").value)

        if self._rate_hz <= 0.0:
            raise ValueError("rate_hz must be > 0")
        if self._width <= 0 or self._height <= 0:
            raise ValueError("width and height must be > 0")

        self._pub = self.create_publisher(Image, self._topic, 10)

        if self._backend == "mujoco":
            self._init_mujoco_backend()
        elif self._backend in ("real", "passthrough"):
            self._init_passthrough_backend()
        else:
            raise ValueError("backend must be one of: mujoco, real, passthrough")

    def _init_passthrough_backend(self) -> None:
        if self._real_image_topic == self._topic:
            self.get_logger().warn(
                "real_image_topic is the same as output topic; scene_camera will not republish to avoid a loop."
            )
            return
        self.create_subscription(Image, self._real_image_topic, self._on_real_image, 10)
        self.get_logger().info(
            f"Scene camera passthrough from {self._real_image_topic} to {self._topic}"
        )

    def _on_real_image(self, msg: Image) -> None:
        if self._frame_id:
            msg.header.frame_id = self._frame_id
        self._pub.publish(msg)

    def _init_mujoco_backend(self) -> None:
        self._lock = threading.Lock()
        self._xml_path = self._resolve_robot_xml(str(self.get_parameter("robot_xml").value).strip())
        self._model = mujoco.MjModel.from_xml_path(str(self._xml_path))
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=self._height, width=self._width)
        self._camera_name = str(self.get_parameter("camera_name").value).strip()
        self._camera = self._make_free_camera()
        self._published_first_image = False
        self._last_render_error_log_time = 0.0

        self._object_joint_qpos_adr = self._joint_qpos_adr(
            str(self.get_parameter("object_joint_name").value)
        )

        self.create_subscription(
            Monitor,
            str(self.get_parameter("monitor_topic").value),
            self._on_monitor,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("object_pose_topic").value),
            self._on_object_pose,
            10,
        )
        self._timer = self.create_timer(1.0 / self._rate_hz, self._publish_mujoco_image)
        self.get_logger().info(
            f"Scene camera rendering {self._xml_path} to {self._topic} at {self._rate_hz:.2f} Hz"
        )
        self._publish_mujoco_image()

    def _resolve_robot_xml(self, configured_path: str) -> Path:
        if configured_path:
            path = Path(configured_path).expanduser()
            if path.exists():
                return path
            raise FileNotFoundError(f"Configured robot_xml does not exist: {path}")

        candidates = []
        try:
            candidates.append(
                Path(get_package_share_directory("crl_humanoid_commons"))
                / "data"
                / "robots"
                / "g1_description"
                / "scene_crl_with_box.xml"
            )
        except Exception:
            pass
        candidates.append(
            Path("/home/sitongchen/keyLM_ros2_ws/src/crl-humanoid-ros")
            / "crl_humanoid_commons"
            / "data"
            / "robots"
            / "g1_description"
            / "scene_crl_with_box.xml"
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not resolve default scene XML from: {candidates}")

    def _make_free_camera(self):
        if self._camera_name:
            cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, self._camera_name)
            if cam_id < 0:
                raise ValueError(f"MuJoCo camera '{self._camera_name}' not found in {self._xml_path}")
            return self._camera_name

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        lookat = _parse_vec3(
            self.get_parameter("camera_lookat").value,
            (0.7, 0.0, 0.55),
        )
        camera.lookat[:] = lookat
        camera.distance = float(self.get_parameter("camera_distance").value)
        camera.azimuth = float(self.get_parameter("camera_azimuth").value)
        camera.elevation = float(self.get_parameter("camera_elevation").value)
        return camera

    def _joint_qpos_adr(self, joint_name: str) -> int:
        if not joint_name:
            return -1
        joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            self.get_logger().warn(f"Joint '{joint_name}' not found in MuJoCo model.")
            return -1
        return int(self._model.jnt_qposadr[joint_id])

    def _on_monitor(self, msg: Monitor) -> None:
        root_pos = np.array(
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
        root_quat = _normalize_quat_wxyz(
            state_quat if float(np.linalg.norm(state_quat)) > 1e-3 else imu_quat
        )

        with self._lock:
            self._data.qpos[0:3] = root_pos
            self._data.qpos[3:7] = root_quat
            for name, pos in zip(msg.sensor.joint.name, msg.sensor.joint.position):
                joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
                if joint_id < 0:
                    continue
                qpos_adr = int(self._model.jnt_qposadr[joint_id])
                if 0 <= qpos_adr < self._model.nq:
                    self._data.qpos[qpos_adr] = float(pos)

    def _on_object_pose(self, msg: PoseStamped) -> None:
        if self._object_joint_qpos_adr < 0:
            return
        pos, quat = _pose_to_arrays(msg)
        adr = self._object_joint_qpos_adr
        with self._lock:
            if adr + 6 < self._model.nq:
                self._data.qpos[adr : adr + 3] = pos
                self._data.qpos[adr + 3 : adr + 7] = quat

    def _publish_mujoco_image(self) -> None:
        try:
            with self._lock:
                mujoco.mj_forward(self._model, self._data)
                self._renderer.update_scene(self._data, camera=self._camera)
                bgr = np.ascontiguousarray(self._renderer.render()[:, :, ::-1])
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_render_error_log_time > 2.0:
                self.get_logger().error(f"Scene camera render failed: {exc}")
                self._last_render_error_log_time = now
            return

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.height = self._height
        msg.width = self._width
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = self._width * 3
        msg.data = bgr.astype(np.uint8).tobytes()
        self._pub.publish(msg)
        if not self._published_first_image:
            self._published_first_image = True
            self.get_logger().info(
                f"Scene camera published first image on {self._topic} ({self._width}x{self._height})"
            )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SceneCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
