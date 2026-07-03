from __future__ import annotations

import argparse
import json
import math
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import rclpy
from crl_humanoid_msgs.msg import Monitor
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import String

from lm.box_config import DEFAULT_BOX_SIZE_XYZ

GOAL_BODY_LINKS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (7, 11),
    (11, 12),
    (12, 13),
)

try:
    from PIL import Image as PILImage
    from PIL import ImageTk
except ImportError:
    PILImage = None
    ImageTk = None


def _quat_wxyz_to_yaw(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return 0.0
    w, x, y, z = q / n
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _quat_angle_error(q1: np.ndarray, q2: np.ndarray) -> float | None:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    n1 = float(np.linalg.norm(q1))
    n2 = float(np.linalg.norm(q2))
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    dot = float(np.clip(abs(float(np.dot(q1 / n1, q2 / n2))), 0.0, 1.0))
    return float(2.0 * math.acos(dot))


def _pose_stamped_to_arrays(msg: PoseStamped) -> tuple[np.ndarray, np.ndarray]:
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


def _stamp_to_text(msg: ImageMsg) -> str:
    return f"{int(msg.header.stamp.sec)}.{int(msg.header.stamp.nanosec):09d}"


def _image_msg_to_rgb_array(msg: ImageMsg) -> np.ndarray:
    encoding = msg.encoding.lower()
    if encoding in ("rgb8", "bgr8"):
        channels = 3
    elif encoding in ("rgba8", "bgra8"):
        channels = 4
    elif encoding == "mono8":
        channels = 1
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    expected_row_bytes = width * channels
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    rows = raw.reshape(height, step)[:, :expected_row_bytes]
    image = rows.reshape(height, width, channels)

    if encoding == "rgb8":
        return image.copy()
    if encoding == "bgr8":
        return image[:, :, ::-1].copy()
    if encoding == "rgba8":
        return image[:, :, :3].copy()
    if encoding == "bgra8":
        return image[:, :, 2::-1].copy()
    return np.repeat(image, 3, axis=2).copy()


class PlannerAppNode(Node):
    def __init__(self) -> None:
        super().__init__("vlm_planner_app_node")

        self.declare_parameter("monitor_topic", "/g1_sim/monitor")
        self.declare_parameter("actual_box_pose_topic", "/actual_box_pose")
        self.declare_parameter("keyframe_visualization_topic", "/keyframe_target_poses")
        self.declare_parameter("tracking_error_topic", "/tracking_errors")
        self.declare_parameter("retargeted_info_topic", "/retargeter/output_info")
        self.declare_parameter("planner_status_topic", "/vlm_planner/status")
        self.declare_parameter("planner_decision_topic", "/vlm_planner/decision")
        self.declare_parameter("vlm_request_image_topic", "/vlm/request_image")

        self._lock = threading.Lock()
        self.robot_pos: np.ndarray | None = None
        self.robot_quat: np.ndarray | None = None
        self.box_pos: np.ndarray | None = None
        self.box_quat: np.ndarray | None = None
        self.keyframe_points: list[np.ndarray] = []
        self.keyframe_object: np.ndarray | None = None
        self.keyframe_object_quat: np.ndarray | None = None
        self.tracking_errors: dict = {}
        self.retargeted_info: dict = {}
        self.planner_status: dict = {}
        self.last_decision: dict = {}
        self.request_image_rgb: np.ndarray | None = None
        self.request_image_stamp = ""

        self.create_subscription(
            Monitor,
            str(self.get_parameter("monitor_topic").value),
            self._on_monitor,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("actual_box_pose_topic").value),
            self._on_actual_box_pose,
            10,
        )
        self.create_subscription(
            PoseArray,
            str(self.get_parameter("keyframe_visualization_topic").value),
            self._on_keyframe_visualization,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("tracking_error_topic").value),
            self._on_tracking_errors,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("retargeted_info_topic").value),
            self._on_retargeted_info,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("planner_status_topic").value),
            self._on_planner_status,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("planner_decision_topic").value),
            self._on_planner_decision,
            10,
        )
        self.create_subscription(
            ImageMsg,
            str(self.get_parameter("vlm_request_image_topic").value),
            self._on_vlm_request_image,
            10,
        )

    def _on_monitor(self, msg: Monitor) -> None:
        pos = np.array(
            [
                msg.state.base_pose.pose.position.x,
                msg.state.base_pose.pose.position.y,
                msg.state.base_pose.pose.position.z,
            ],
            dtype=np.float64,
        )
        quat = np.array(
            [
                msg.state.base_pose.pose.orientation.w,
                msg.state.base_pose.pose.orientation.x,
                msg.state.base_pose.pose.orientation.y,
                msg.state.base_pose.pose.orientation.z,
            ],
            dtype=np.float64,
        )
        with self._lock:
            self.robot_pos = pos
            self.robot_quat = quat

    def _on_actual_box_pose(self, msg: PoseStamped) -> None:
        pos, quat = _pose_stamped_to_arrays(msg)
        with self._lock:
            self.box_pos = pos
            self.box_quat = quat

    def _on_keyframe_visualization(self, msg: PoseArray) -> None:
        points: list[np.ndarray] = []
        object_quat = None
        for pose in msg.poses:
            points.append(np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64))
        if msg.poses:
            last_pose = msg.poses[-1]
            object_quat = np.array(
                [
                    last_pose.orientation.w,
                    last_pose.orientation.x,
                    last_pose.orientation.y,
                    last_pose.orientation.z,
                ],
                dtype=np.float64,
            )
        has_valid_object = object_quat is not None and float(np.linalg.norm(object_quat)) > 1e-9
        with self._lock:
            if has_valid_object:
                self.keyframe_points = points[:-1] if len(points) > 1 else points
                self.keyframe_object = points[-1] if points else None
                self.keyframe_object_quat = object_quat
            else:
                self.keyframe_points = points[:-1] if len(points) > 1 else points
                self.keyframe_object = None
                self.keyframe_object_quat = None

    def _on_tracking_errors(self, msg: String) -> None:
        data = _json_or_text(msg.data)
        with self._lock:
            self.tracking_errors = data if isinstance(data, dict) else {"raw": msg.data}

    def _on_retargeted_info(self, msg: String) -> None:
        data = _json_or_text(msg.data)
        with self._lock:
            self.retargeted_info = data if isinstance(data, dict) else {"raw": msg.data}

    def _on_planner_status(self, msg: String) -> None:
        data = _json_or_text(msg.data)
        with self._lock:
            self.planner_status = data if isinstance(data, dict) else {"raw": msg.data}

    def _on_planner_decision(self, msg: String) -> None:
        data = _json_or_text(msg.data)
        with self._lock:
            self.last_decision = data if isinstance(data, dict) else {"raw": msg.data}

    def _on_vlm_request_image(self, msg: ImageMsg) -> None:
        try:
            image_rgb = _image_msg_to_rgb_array(msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode VLM request image: {exc}")
            return
        with self._lock:
            self.request_image_rgb = image_rgb
            self.request_image_stamp = _stamp_to_text(msg)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "robot_pos": None if self.robot_pos is None else self.robot_pos.copy(),
                "robot_quat": None if self.robot_quat is None else self.robot_quat.copy(),
                "box_pos": None if self.box_pos is None else self.box_pos.copy(),
                "box_quat": None if self.box_quat is None else self.box_quat.copy(),
                "keyframe_points": [p.copy() for p in self.keyframe_points],
                "keyframe_object": None if self.keyframe_object is None else self.keyframe_object.copy(),
                "keyframe_object_quat": None
                if self.keyframe_object_quat is None
                else self.keyframe_object_quat.copy(),
                "tracking_errors": dict(self.tracking_errors),
                "retargeted_info": dict(self.retargeted_info),
                "planner_status": dict(self.planner_status),
                "last_decision": dict(self.last_decision),
                "request_image_rgb": None if self.request_image_rgb is None else self.request_image_rgb.copy(),
                "request_image_stamp": self.request_image_stamp,
            }


def _json_or_text(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class VLMPlannerApp:
    def __init__(self, node: PlannerAppNode, args: argparse.Namespace) -> None:
        self.node = node
        self.args = args
        self.root = tk.Tk()
        self.root.title("VLM Planner")
        self.root.geometry("1600x950")
        self.root.minsize(1200, 760)

        self.proc: subprocess.Popen | None = None
        self.tunnel_proc: subprocess.Popen | None = None
        self.event_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.bottom_status = tk.StringVar(value="Starting")
        self.tunnel_status = tk.StringVar(value="Tunnel: starting")
        self.planner_status = tk.StringVar(value="Planner: idle")
        self.request_photo = None
        self._request_photo_cache_key = None

        self._build_ui()
        if not args.no_tunnel:
            self._start_tunnel()
        else:
            self.tunnel_status.set("Tunnel: disabled")

        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)
        self.root.after(100, self._refresh)

    def _build_ui(self) -> None:
        self.root.configure(bg="#edf0f2")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)

        center = ttk.Frame(self.root, padding=(0, 10, 0, 10))
        center.grid(row=0, column=1, sticky="nsew")
        center.rowconfigure(0, weight=4)
        center.rowconfigure(2, weight=1)
        center.columnconfigure(0, weight=1)

        right = ttk.Frame(self.root, padding=10)
        right.grid(row=0, column=2, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)
        right.rowconfigure(6, weight=2)

        bottom = ttk.Frame(self.root, padding=(10, 4))
        bottom.grid(row=1, column=0, columnspan=3, sticky="ew")
        bottom.columnconfigure(1, weight=1)

        ttk.Label(left, text="VLM Status", font=("TkDefaultFont", 13, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(left, textvariable=self.tunnel_status).grid(row=1, column=0, sticky="w", pady=(8, 4))
        ttk.Label(left, textvariable=self.planner_status).grid(row=2, column=0, sticky="w", pady=(0, 8))

        self.status_text = tk.Text(left, width=40, height=16, wrap="word", state="disabled")
        self.status_text.grid(row=3, column=0, sticky="nsew")
        left.rowconfigure(3, weight=1)

        ttk.Label(left, text="Last Decision", font=("TkDefaultFont", 11, "bold")).grid(row=4, column=0, sticky="w", pady=(12, 4))
        self.decision_text = tk.Text(left, width=40, height=10, wrap="word", state="disabled")
        self.decision_text.grid(row=5, column=0, sticky="ew")

        ttk.Label(left, text="Tracking Errors", font=("TkDefaultFont", 11, "bold")).grid(row=6, column=0, sticky="w", pady=(12, 4))
        self.error_text = tk.Text(left, width=40, height=8, wrap="word", state="disabled")
        self.error_text.grid(row=7, column=0, sticky="ew")

        self.canvas = tk.Canvas(center, bg="#f8fafb", highlightthickness=1, highlightbackground="#c6ccd2")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        ttk.Label(center, text="VLM Request Image", font=("TkDefaultFont", 11, "bold")).grid(
            row=1, column=0, sticky="w", pady=(10, 4)
        )
        self.request_image_label = tk.Label(
            center,
            bg="#0f172a",
            fg="#e5e7eb",
            text="No VLM request image yet",
            anchor="center",
            compound="top",
            height=10,
        )
        self.request_image_label.grid(row=2, column=0, sticky="nsew")

        ttk.Label(right, text="Task Command", font=("TkDefaultFont", 13, "bold")).grid(row=0, column=0, sticky="w")
        self.task_text = tk.Text(right, width=42, height=6, wrap="word")
        self.task_text.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        self.task_text.insert("1.0", "Pick up the box on the ground and place it on the table.")

        buttons = ttk.Frame(right)
        buttons.grid(row=2, column=0, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        ttk.Button(buttons, text="Start", command=self.start_planner).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(buttons, text="Stop", command=self.stop_planner).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ttk.Label(right, text="Retargeted Keyframe", font=("TkDefaultFont", 11, "bold")).grid(row=3, column=0, sticky="w", pady=(12, 4))
        self.retarget_text = tk.Text(right, width=42, height=8, wrap="word", state="disabled")
        self.retarget_text.grid(row=4, column=0, sticky="nsew")

        ttk.Label(right, text="Planner Output", font=("TkDefaultFont", 11, "bold")).grid(row=5, column=0, sticky="w", pady=(12, 4))
        self.output_text = tk.Text(right, width=42, height=10, wrap="word", state="disabled")
        self.output_text.grid(row=6, column=0, sticky="nsew")

        ttk.Label(bottom, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(bottom, textvariable=self.bottom_status).grid(row=0, column=1, sticky="ew", padx=(8, 0))

    def _start_tunnel(self) -> None:
        if _port_open("127.0.0.1", self.args.local_port):
            self.tunnel_status.set(f"Tunnel: port {self.args.local_port} already open")
            self._append_status("tunnel", f"localhost:{self.args.local_port} already accepts connections")
            return

        cmd = [
            "ssh",
            "-N",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-L",
            f"{self.args.local_port}:localhost:{self.args.remote_port}",
            f"{self.args.user}@{self.args.host}",
        ]
        try:
            self.tunnel_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.tunnel_status.set("Tunnel: failed to start")
            self._append_status("tunnel", f"failed to start ssh: {exc}")
            return

        self.tunnel_status.set(f"Tunnel: starting localhost:{self.args.local_port}")
        self._append_status("tunnel", "started ssh tunnel command")
        self._read_stream(self.tunnel_proc.stderr, "tunnel")
        self.root.after(1000, self._check_tunnel_started)

    def _check_tunnel_started(self) -> None:
        if self.tunnel_proc is None:
            return
        if self.tunnel_proc.poll() is not None:
            self.tunnel_status.set(f"Tunnel: exited ({self.tunnel_proc.returncode})")
            return
        if _port_open("127.0.0.1", self.args.local_port):
            self.tunnel_status.set(f"Tunnel: connected localhost:{self.args.local_port}")
        else:
            self.tunnel_status.set(f"Tunnel: process running, waiting on port {self.args.local_port}")
            self.root.after(1000, self._check_tunnel_started)

    def start_planner(self) -> None:
        task = self.task_text.get("1.0", "end").strip()
        if not task:
            self._append_status("ui", "task command is empty")
            return
        if self.proc is not None and self.proc.poll() is None:
            self._append_status("ui", "planner is already running")
            return

        cmd = [
            sys.executable,
            "-m",
            "lm.vml",
            "--task",
            task,
            "--service",
            self.args.service,
            "--poll-period",
            str(self.args.poll_period),
        ]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            self._append_status("planner", f"failed to start planner: {exc}")
            return

        self.planner_status.set("Planner: running")
        self.bottom_status.set("Planner process started")
        self._append_status("planner", f"started task: {task}")
        self._read_stream(self.proc.stdout, "planner")
        self._read_stream(self.proc.stderr, "planner")

    def stop_planner(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            self._append_status("ui", "planner is not running")
            return
        self.proc.terminate()
        self.planner_status.set("Planner: stopping")
        self.bottom_status.set("Stopping planner process")

    def _read_stream(self, stream, source: str) -> None:
        if stream is None:
            return

        def worker() -> None:
            for line in stream:
                self.event_queue.put((source, line.rstrip()))

        threading.Thread(target=worker, daemon=True).start()

    def _refresh(self) -> None:
        self._drain_events()
        snap = self.node.snapshot()
        self._draw_scene(snap)
        self._update_request_image(snap)
        self._update_side_panels(snap)
        self._update_process_status()
        self.root.after(100, self._refresh)

    def _drain_events(self) -> None:
        while True:
            try:
                source, text = self.event_queue.get_nowait()
            except queue.Empty:
                return
            if text:
                self._append_status(source, text)
                if source == "planner":
                    self._append_output(text)

    def _update_process_status(self) -> None:
        if self.proc is not None and self.proc.poll() is not None:
            code = self.proc.returncode
            self.planner_status.set(f"Planner: exited ({code})")
            self.proc = None

    def _update_side_panels(self, snap: dict) -> None:
        status = snap["planner_status"]
        if status:
            state = str(status.get("state", "unknown"))
            msg = str(status.get("message", ""))
            self.bottom_status.set(f"{state}: {msg}")
            self.planner_status.set(f"Planner: {state}")
            self._set_text(self.decision_text, _format_json(snap["last_decision"]))
        self._set_text(self.error_text, _format_tracking_summary(snap))
        self._set_text(self.retarget_text, _format_json(snap["retargeted_info"]))

    def _update_request_image(self, snap: dict) -> None:
        image_rgb = snap.get("request_image_rgb")
        stamp = str(snap.get("request_image_stamp", ""))
        if image_rgb is None:
            self.request_image_label.configure(image="", text="No VLM request image yet")
            self.request_photo = None
            self._request_photo_cache_key = None
            return
        if PILImage is None or ImageTk is None:
            self.request_image_label.configure(image="", text="Pillow is not available for image preview")
            return

        label_width = max(int(self.request_image_label.winfo_width()), 1)
        label_height = max(int(self.request_image_label.winfo_height()), 1)
        if label_width < 20 or label_height < 20:
            return

        image_height, image_width = image_rgb.shape[:2]
        text_height = 24
        scale = min(label_width / max(image_width, 1), (label_height - text_height) / max(image_height, 1))
        scale = max(scale, 0.05)
        preview_size = (
            max(1, int(image_width * scale)),
            max(1, int(image_height * scale)),
        )
        cache_key = (stamp, preview_size, image_width, image_height)
        if cache_key == self._request_photo_cache_key:
            return

        resample = PILImage.Resampling.BILINEAR if hasattr(PILImage, "Resampling") else PILImage.BILINEAR
        pil_image = PILImage.fromarray(image_rgb.astype(np.uint8), mode="RGB").resize(preview_size, resample)
        self.request_photo = ImageTk.PhotoImage(pil_image)
        self._request_photo_cache_key = cache_key
        label_text = f"stamp {stamp}" if stamp else ""
        self.request_image_label.configure(image=self.request_photo, text=label_text, compound="top")

    def _draw_scene(self, snap: dict) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 200)
        height = max(canvas.winfo_height(), 200)

        robot_yaw = _quat_wxyz_to_yaw(snap["robot_quat"]) if snap["robot_quat"] is not None else 0.0

        points = []
        for key in ("robot_pos", "box_pos", "keyframe_object"):
            val = snap[key]
            if val is not None:
                points.append(np.asarray(val[:2], dtype=np.float64))
        for pt in snap["keyframe_points"]:
            points.append(np.asarray(pt[:2], dtype=np.float64))

        if points:
            arr = np.asarray(points, dtype=np.float64)
            min_xy = arr.min(axis=0)
            max_xy = arr.max(axis=0)
            center = 0.5 * (min_xy + max_xy)
            span = np.maximum(max_xy - min_xy, np.array([1.0, 1.0]))
        else:
            center = np.array([0.0, 0.0], dtype=np.float64)
            span = np.array([2.0, 2.0], dtype=np.float64)

        scale = 0.80 * min(width / max(span[0], 1.0), height / max(span[1], 1.0))
        scale = float(np.clip(scale, 80.0, 260.0))

        def to_px(xy: np.ndarray) -> tuple[float, float]:
            dx, dy = xy - center
            return width * 0.5 + dx * scale, height * 0.5 - dy * scale

        self._draw_grid(canvas, width, height, scale)

        goal_points = snap["keyframe_points"]
        for i, j in GOAL_BODY_LINKS:
            if i < len(goal_points) and j < len(goal_points):
                x0, y0 = to_px(goal_points[i][:2])
                x1, y1 = to_px(goal_points[j][:2])
                canvas.create_line(x0, y0, x1, y1, fill="#60a5fa", width=2)

        for idx, pt in enumerate(goal_points):
            x, y = to_px(pt[:2])
            radius = 7 if idx == 0 else 4
            fill = "#1d4ed8" if idx == 0 else "#2563eb"
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline="#eff6ff", width=1)

        if goal_points:
            x, y = to_px(goal_points[0][:2])
            canvas.create_oval(x - 14, y - 14, x + 14, y + 14, outline="#1d4ed8", width=2)
            canvas.create_text(x + 18, y + 2, text="keyframe pelvis", anchor="w", fill="#1e3a8a")

        if snap["keyframe_object"] is not None:
            x, y = to_px(snap["keyframe_object"][:2])
            half_x = 0.5 * DEFAULT_BOX_SIZE_XYZ[0] * scale
            half_y = 0.5 * DEFAULT_BOX_SIZE_XYZ[1] * scale
            canvas.create_rectangle(x - half_x, y - half_y, x + half_x, y + half_y, outline="#f59e0b", width=3)
            canvas.create_text(x + 14, y - 14, text="keyframe object", anchor="w", fill="#92400e")

        if snap["box_pos"] is not None:
            x, y = to_px(snap["box_pos"][:2])
            half_x = 0.5 * DEFAULT_BOX_SIZE_XYZ[0] * scale
            half_y = 0.5 * DEFAULT_BOX_SIZE_XYZ[1] * scale
            canvas.create_rectangle(x - half_x, y - half_y, x + half_x, y + half_y, fill="#ef4444", outline="#991b1b", width=2)
            canvas.create_text(x + half_x + 8, y, text="box", anchor="w", fill="#7f1d1d")

        if snap["robot_pos"] is not None:
            x, y = to_px(snap["robot_pos"][:2])
            canvas.create_oval(x - 12, y - 12, x + 12, y + 12, fill="#111827", outline="#111827")
            tip = np.array([math.cos(robot_yaw), math.sin(robot_yaw)], dtype=np.float64) * 0.35 + snap["robot_pos"][:2]
            tx, ty = to_px(tip)
            canvas.create_line(x, y, tx, ty, fill="#10b981", width=4, arrow=tk.LAST)
            canvas.create_text(x + 16, y - 16, text="robot", anchor="w", fill="#111827")

        canvas.create_text(
            14,
            14,
            text="Top-down view: current robot and published keyframe goal",
            anchor="nw",
            fill="#334155",
            font=("TkDefaultFont", 12, "bold"),
        )
        canvas.create_text(
            14,
            36,
            text="XY projection in the world frame",
            anchor="nw",
            fill="#64748b",
        )
        self._draw_scene_legend(canvas, width)

    def _draw_grid(self, canvas: tk.Canvas, width: int, height: int, scale: float) -> None:
        spacing = max(40, int(0.5 * scale))
        for x in range(0, width, spacing):
            canvas.create_line(x, 0, x, height, fill="#e2e8f0")
        for y in range(0, height, spacing):
            canvas.create_line(0, y, width, y, fill="#e2e8f0")

    def _draw_scene_legend(self, canvas: tk.Canvas, width: int) -> None:
        x0 = max(width - 185, 20)
        y0 = 14
        items = [
            ("#111827", "current robot"),
            ("#ef4444", "current box"),
            ("#2563eb", "keyframe body"),
            ("#f59e0b", "keyframe object"),
        ]
        for i, (color, label) in enumerate(items):
            y = y0 + 20 * i
            canvas.create_oval(x0, y, x0 + 10, y + 10, fill=color, outline="")
            canvas.create_text(x0 + 16, y + 5, text=label, anchor="w", fill="#334155")

    def _append_status(self, source: str, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self._append_text(self.status_text, f"[{stamp}] {source}: {text}\n", max_chars=12000)

    def _append_output(self, text: str) -> None:
        self._append_text(self.output_text, text + "\n", max_chars=18000)

    def _append_text(self, widget: tk.Text, text: str, max_chars: int) -> None:
        widget.configure(state="normal")
        widget.insert("end", text)
        content = widget.get("1.0", "end-1c")
        if len(content) > max_chars:
            widget.delete("1.0", f"1.0+{len(content) - max_chars}c")
        widget.see("end")
        widget.configure(state="disabled")

    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def shutdown(self) -> None:
        self.stop_planner()
        if self.tunnel_proc is not None and self.tunnel_proc.poll() is None:
            self.tunnel_proc.terminate()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=0.25):
            return True
    except OSError:
        return False


def _format_json(data: dict) -> str:
    if not data:
        return ""
    return json.dumps(data, indent=2, sort_keys=True)


def _float_from_dict(data: dict, key: str) -> float | None:
    try:
        value = data.get(key)
    except AttributeError:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _format_metric(name: str, value: float | None, unit: str) -> str:
    if value is None:
        return f"{name}: n/a"
    return f"{name}: {value:.4f} {unit}"


def _format_tracking_summary(snap: dict) -> str:
    tracking = snap.get("tracking_errors", {})
    root_pos_error = _float_from_dict(tracking, "root_position_error_m")
    root_ori_error = _float_from_dict(tracking, "root_orientation_error_rad")
    mean_body_error = _float_from_dict(tracking, "mean_body_position_error_m")

    object_pos_error = None
    object_ori_error = None
    box_pos = snap.get("box_pos")
    box_quat = snap.get("box_quat")
    target_obj = snap.get("keyframe_object")
    target_obj_quat = snap.get("keyframe_object_quat")
    if box_pos is not None and target_obj is not None and target_obj_quat is not None:
        target_quat_norm = float(np.linalg.norm(target_obj_quat))
        if target_quat_norm > 1e-9:
            object_pos_error = float(np.linalg.norm(np.asarray(box_pos) - np.asarray(target_obj)))
            if box_quat is not None:
                object_ori_error = _quat_angle_error(np.asarray(box_quat), np.asarray(target_obj_quat))

    lines = [
        _format_metric("mean body pose error", mean_body_error, "m"),
        _format_metric("root position error", root_pos_error, "m"),
        _format_metric("root orientation error", root_ori_error, "rad"),
        _format_metric("object position error", object_pos_error, "m"),
        _format_metric("object orientation error", object_ori_error, "rad"),
    ]
    if tracking:
        lines.extend(["", "raw tracking_errors:", _format_json(tracking)])
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLM planner GUI")
    parser.add_argument("--service", default="/vlm/query", help="VLM query service used by spawned planner client")
    parser.add_argument("--poll-period", type=float, default=0.1, help="Readiness poll period for spawned planner client")
    parser.add_argument("--no-tunnel", action="store_true", help="Do not start the SSH tunnel")
    parser.add_argument("--local-port", type=int, default=11434, help="Local tunnel port")
    parser.add_argument("--remote-port", type=int, default=8001, help="Remote tunnel target port")
    parser.add_argument("--host", default="case.inf.ethz.ch", help="SSH tunnel host")
    parser.add_argument("--user", default="sitchen", help="SSH tunnel user")
    return parser


def main(args: list[str] | None = None) -> None:
    if args is None:
        args = sys.argv[1:]
    ros_filtered_args = rclpy.utilities.remove_ros_args(args)
    parsed = build_arg_parser().parse_args(args=ros_filtered_args)

    rclpy.init(args=None)
    node = PlannerAppNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    app = VLMPlannerApp(node, parsed)
    try:
        app.run()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
