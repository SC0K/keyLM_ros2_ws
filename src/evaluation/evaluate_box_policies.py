"""Paired, headless policy evaluation using production VLM goals and observations.

No VLM queries, hardware, ROS executor, or control-topic publishing. ROS nodes
are constructed only to reuse their configured production methods; calls are
routed in-process. Physics runs synchronously, without transport/wall-time jitter.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from crl_humanoid_msgs.msg import Monitor
from std_srvs.srv import Trigger
import yaml

from crl_g1_goalcontroller_python import g1_keyframe_controller as control
from lm import vml
from lm.box_config import SIM_TARGET_BOX_GEOMETRY, SOURCE_BOX_GEOMETRY, format_box_size_xyz
from lm.keyframe_retargeter_node import KeyframeRetargeterNode
from lm_interfaces.srv import RetargetKeyframe, VLMQuery

SRC = Path(__file__).resolve().parents[1]
PACKAGE = SRC / "crl-humanoid-ros/crl_g1_goalcontroller_py/crl_g1_goalcontroller_python"
COMMONS = SRC / "crl-humanoid-ros/crl_humanoid_commons"
SCENE = COMMONS / "data/robots/g1_description/scene_crl_with_box.xml"
CONFIG = PACKAGE / "config/g1_keyframe_tracking_obj.yaml"
PHASES = (
    ("approach", "approach_box"),
    ("pick", "stand_before_pick_box"),
    ("pick", "crouch_to_pick_box"),
    ("pick", "stand_after_pick_box"),
    ("place", "stand_before_place_box"),
    ("place", "crouch_to_place_box"),
    ("place", "stand_after_place_box"),
)
# Measure the requested keyframe operations, not the final standing goal whose
# XY is anchored at the robot's current root and would understate placement error.
STAGE_ENDPOINTS = {"approach_box", "crouch_to_pick_box", "crouch_to_place_box"}
THRESHOLD_PARAMETERS = (
    "stationary_hold_sec", "min_action_duration_sec", "robot_linear_stationary_threshold_mps",
    "robot_angular_stationary_threshold_radps", "object_linear_stationary_threshold_mps",
    "object_angular_stationary_threshold_radps", "mean_body_success_threshold_m",
    "root_position_success_threshold_m", "root_orientation_success_threshold_rad",
    "object_position_success_threshold_m", "task_object_position_threshold_m", "pick_max_horizontal_distance_m",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_box_scene(output, size_xyz):
    """Copy the flat-hand scene with evaluation-only physical box dimensions.

    Keep mass, inertia, contact settings and retargeting geometry unchanged.
    Save the robot XML as well, resolving its mesh directory for later replay.
    """
    size = np.asarray(size_xyz, dtype=float)
    if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
        raise ValueError("Physical box size must be three positive finite dimensions")
    output = Path(output).resolve()
    scene = ET.parse(SCENE)
    include = scene.getroot().find("include")
    robot_source = SCENE.parent / include.attrib["file"]
    robot = ET.parse(robot_source)
    compiler = robot.getroot().find("compiler")
    compiler.set("meshdir", str((robot_source.parent / compiler.get("meshdir", ".")).resolve()))
    robot_path = output / "robot_snapshot.xml"
    robot.write(robot_path, encoding="unicode")
    include.set("file", str(robot_path))
    for name in ("box_geom", "target_object_geom"):
        scene.getroot().find(f".//geom[@name='{name}']").set("size", " ".join(f"{v:.15g}" for v in size / 2))
    scene_path = output / "scene_snapshot.xml"
    scene.write(scene_path, encoding="unicode")
    return scene_path


def perturbations(trials, seed, initial_root_pos=(-2., 0., .8), place_noise_xy_m=0.):
    if not np.isfinite(place_noise_xy_m) or place_noise_xy_m < 0:
        raise ValueError("Placement XY noise must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    # A separate stream preserves all original reset and walking-noise samples.
    place_rng = np.random.default_rng(np.random.SeedSequence([seed, 1701]))
    result = []
    for index in range(trials):
        root = np.array(initial_root_pos, dtype=np.float64)
        box = np.array([.35, 0., .15])
        root[:2] += rng.uniform(-.02, .02, 2)
        box[:2] += rng.uniform(-.02, .02, 2)
        yaws = rng.uniform(-np.deg2rad(2.), np.deg2rad(2.), 2)
        result.append(dict(trial=index + 1, root_pos=root.tolist(), box_pos=box.tolist(),
                           root_quat=vml._yaw_to_quat_wxyz(yaws[0]).tolist(),
                           box_quat=vml._yaw_to_quat_wxyz(yaws[1]).tolist(),
                           walking_noise_seed=seed + index,
                           place_offset_xy_m=place_rng.uniform(-place_noise_xy_m, place_noise_xy_m, 2).tolist()))
    return result


def simulator_torque_limits():
    """Read the C++ simulator's canonical G1 limits rather than assuming MJCF limits."""
    header = (COMMONS / "include/crl_humanoid_commons/RobotParameters.h").read_text()
    section = header.split("{RobotModelType::UNITREE_G1,", 1)[1].split("// Limx", 1)[0]
    arrays = re.findall(r"\{([^{}]+)\}", section)
    names = re.findall(r'"([^"]+)"', arrays[0])
    limits = np.array([float(value) for value in arrays[5].split(",") if value.strip()])
    if names != list(control.POLICY_JOINT_NAMES) or len(limits) != 29:
        raise ValueError("Cannot verify simulator torque-limit order")
    return limits


def pose_msg(position, quaternion):
    msg = PoseStamped()
    msg.header.frame_id = "world"
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, position)
    msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z = map(float, quaternion)
    return msg


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summarize(trials):
    result = {}
    for policy in dict.fromkeys(row["policy"] for row in trials):
        rows = [row for row in trials if row["policy"] == policy]
        successes = [row for row in rows if row["success"]]
        stages = {}
        for stage in ("approach", "pick", "place"):
            samples = [row["stage_errors"][stage] for row in successes if stage in row["stage_errors"]]
            stages[stage] = {}
            if samples:
                for metric in samples[0]:
                    values = [sample[metric] for sample in samples if sample[metric] is not None]
                    stages[stage][metric] = dict(n=len(values), mean=float(np.mean(values)) if values else None,
                                                std=float(np.std(values, ddof=1)) if len(values) > 1 else None)
        result[policy] = dict(trials=len(rows), successes=len(successes), success_rate=len(successes) / len(rows),
                              successful_trial_stage_errors=stages,
                              failures=[dict(trial=row["trial"], reason=row["reason"]) for row in rows if not row["success"]])
    return result


class Evaluator:
    def __init__(self, config, policy, output, args):
        self.output, self.args = output, args
        self.model = mujoco.MjModel.from_xml_path(str(getattr(args, "scene", SCENE)))
        self.model.opt.timestep = .001667  # g1_simulator.yaml, ten physics steps per policy tick
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.array([self.model.joint(name).id for name in control.POLICY_JOINT_NAMES])
        self.qadr = self.model.jnt_qposadr[self.joint_ids]
        self.vadr = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.array([self.model.actuator(name).id for name in control.POLICY_JOINT_NAMES])
        self.box_adr = int(self.model.joint("box_freejoint").qposadr[0])
        self.box_vel_adr = int(self.model.joint("box_freejoint").dofadr[0])
        self.torque_limits = simulator_torque_limits()
        self.root_id, self.box_id = self.model.body("pelvis").id, self.model.body("box").id
        self.floor_id = self.model.geom("floor").id
        self.box_geom_id = self.model.geom("box_geom").id
        self._real_planner_time = vml.time
        vml.time = SimpleNamespace(monotonic=lambda: float(self.data.time))
        overrides = {
            "config_file": str(config), "policy_onnx": str(policy), "local_keyframes": False,
            "manual_goal_advance": True, "fix_default_goal_fallback": True,
            "mocap_object_selection": False, "supervised_mode": False,
            "retarget_ik_enabled": False, "prepend_default_goal_frame": False, "append_default_goal_frame": False,
            "library_dir": str(SRC / "lm/keyframes"),
            "box_size_xyz": format_box_size_xyz(SIM_TARGET_BOX_GEOMETRY.size_xyz),
            "source_box_size_xyz": format_box_size_xyz(SOURCE_BOX_GEOMETRY.size_xyz),
        }
        ros_args = ["--ros-args", "--log-level", "error"]
        for name, value in overrides.items():
            ros_args += ["-p", f"{name}:={str(value).lower() if isinstance(value, bool) else value}"]
        rclpy.init(args=ros_args)
        self.controller = control.G1KeyframeController()
        self.controller.timer.cancel()
        self.controller.keyframe_visualization_publishers = []
        self.retargeter = KeyframeRetargeterNode()
        self.planner = None
        self.logs = []
        self.goal_blob = None
        self.controller.tracking_error_publisher = SimpleNamespace(publish=self._tracking)
        self.set_policy(policy)

    def set_policy(self, path):
        options = control.ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = control.ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        if session.get_inputs()[0].shape != [1, 1480] or session.get_outputs()[0].shape != [1, 29]:
            raise ValueError(f"Expected ver3 1480->29 policy, got {path}")
        self.controller.session = session
        self.controller.input_name = session.get_inputs()[0].name
        self.controller.output_name = session.get_outputs()[0].name
        self.policy = Path(path).name

    def _tracking(self, msg):
        self.metrics = json.loads(msg.data)
        if self.planner is not None:
            self.planner._on_tracking_errors(msg)

    def _retarget(self, **kwargs):
        request = RetargetKeyframe.Request()
        for name, value in kwargs.items():
            setattr(request, name, value)
        response = self.retargeter._on_retarget_keyframe_request(request, RetargetKeyframe.Response())
        if not response.success:
            raise RuntimeError(f"Retargeting failed: {response.error_message}")
        self.goal_blob = bytes(response.retargeted_keyframe)
        return response

    def sync(self):
        mujoco.mj_forward(self.model, self.data)
        c, d = self.controller, self.data
        c.root_pos[:] = d.qpos[:3]
        c.root_quat[:] = d.qpos[3:7]
        c.root_lin_vel_w[:] = d.qvel[:3]
        c.root_ang_vel_b[:] = d.qvel[3:6]
        c.joint_pos[:] = d.qpos[self.qadr]
        c.joint_vel[:] = d.qvel[self.vadr]
        c.current_object_pos_w[:] = d.qpos[self.box_adr:self.box_adr + 3]
        c.current_object_quat_w[:] = d.qpos[self.box_adr + 3:self.box_adr + 7]
        c.have_monitor = c.have_root_pose = c.have_object_pose = True
        monitor = Monitor()
        monitor.sensor.joint.name = list(control.POLICY_JOINT_NAMES)
        monitor.sensor.joint.position = c.joint_pos.astype(float).tolist()
        self.retargeter._on_standing_monitor(monitor)
        if self.planner is not None:
            p = self.planner
            p._current_robot_center = np.array(c.root_pos, dtype=np.float64)
            p._current_robot_quat_wxyz = np.array(c.root_quat, dtype=np.float64)
            p._has_monitor = p._has_robot_root_pose = True
            p._robot_linear_speed = float(np.linalg.norm(d.qvel[:3]))
            p._robot_angular_speed = float(np.linalg.norm(d.qvel[3:6]))
            p._on_actual_box_pose(pose_msg(c.current_object_pos_w, c.current_object_quat_w))

    def reset(self, trial):
        if self.planner is not None:
            self.planner.destroy_node()
            self.planner = None
        mujoco.mj_resetData(self.model, self.data)
        c, d = self.controller, self.data
        d.qpos[:3], d.qpos[3:7] = trial["root_pos"], trial["root_quat"]
        d.qpos[self.qadr] = c.default_angles
        d.qpos[self.box_adr:self.box_adr + 3] = trial["box_pos"]
        d.qpos[self.box_adr + 3:self.box_adr + 7] = trial["box_quat"]
        c._reset_policy_history()
        c.have_goal = False
        c.fixed_default_goal = None
        c.fixed_inserted_default_goals = {}
        c._pending_keyframe = None
        c.walking_goal_rng = np.random.default_rng(trial["walking_noise_seed"])
        c.current_fsm_state = control.GOAL_FSM_STATE
        self.sync()
        # Match the GUI's idle GOAL policy, not a different locomotion policy.
        c.object_to_manipulate = False
        self.warmup_goal = c._default_goal_frame_at_current_root()
        for _ in range(round(self.args.warmup_seconds / (.001667 * 10))):
            self.tick(self.warmup_goal)
            if self.fallen():
                break
        self.planner = vml.VLMClientNode("/unused_offline_vlm_query")
        self.planner.publish_status = lambda *a, **kw: None
        self.planner._retargeted_keyframe_pub = SimpleNamespace(publish=c._on_keyframe)
        self.planner._retargeted_info_pub = SimpleNamespace(publish=lambda msg: None)
        self.planner.request_retargeted_keyframe = self._retarget
        self.retargeter._on_reset_task(Trigger.Request(), Trigger.Response())
        self.sync()
        self.planner._default_place_distance_m = trial.get("place_distance_m", 1.0)
        self.planner.initialize_task_target_once("box")
        # Set once before publishing any goal: retargeting and completion checks
        # then use this same perturbed destination throughout the trial.
        self.planner._task_target_box_center[:2] += trial["place_offset_xy_m"]
        self.initial_box_height = float(d.qpos[self.box_adr + 2])
        self.max_box_height = self.initial_box_height
        self.trace = []

    def tick(self, goal):
        c = self.controller
        c.obs = c._build_observation(goal)
        c.obs_history = np.concatenate([c.obs_history[1:], c.obs.reshape(1, -1)], axis=0)
        c.action = c.session.run([c.output_name], {c.input_name: c.obs_history.reshape(1, -1).astype(np.float32)})[0].reshape(-1).astype(np.float32)
        if not np.all(np.isfinite(c.action)):
            raise RuntimeError("Policy produced non-finite actions")
        target = c.default_angles + c.action * c.action_scale
        for _ in range(10):
            tau = c.kps * (target - self.data.qpos[self.qadr]) - c.kds * self.data.qvel[self.vadr]
            self.data.ctrl[self.actuator_ids] = np.clip(tau, -self.torque_limits, self.torque_limits)
            mujoco.mj_step(self.model, self.data)
        self.sync()

    def fallen(self):
        return (not np.all(np.isfinite(self.data.qpos)) or self.data.qpos[2] < .30
                or self.data.xmat[self.root_id].reshape(3, 3)[2, 2] < .25)

    def errors(self, goal):
        c = self.controller
        c._publish_tracking_errors(goal)
        root_pos = float(self.metrics["root_position_error_m"])
        root_ori = float(self.metrics["root_orientation_error_rad"])
        joint_delta = c.joint_pos - (c.default_angles + goal[9:38])
        obj_pos, obj_ori = self.planner._object_error_to_last_target()
        object_enabled = c._goal_object_observation_mask(goal) > 0
        return dict(root_position_m=root_pos, root_xy_m=float(np.linalg.norm(c.root_pos[:2] - goal[:2])),
                    root_orientation_rad=root_ori, joint_mae_rad=float(np.mean(np.abs(joint_delta))),
                    joint_rmse_rad=float(np.sqrt(np.mean(joint_delta ** 2))),
                    joint_max_rad=float(np.max(np.abs(joint_delta))),
                    object_position_m=obj_pos if object_enabled else None,
                    object_orientation_rad=obj_ori if object_enabled else None,
                    mean_body_position_m=float(self.metrics["mean_body_position_error_m"]))

    def contacts(self):
        floor, robot = False, False
        for contact in self.data.contact:
            if self.box_geom_id not in (contact.geom1, contact.geom2) or contact.dist > .002:
                continue
            other = contact.geom2 if contact.geom1 == self.box_geom_id else contact.geom1
            floor |= other == self.floor_id
            robot |= self.model.geom_bodyid[other] not in (0, self.box_id)
        return dict(box_floor_contact=bool(floor), box_robot_contact=bool(robot))

    def run_trial(self, trial, reset_state=True):
        if reset_state:
            self.reset(trial)
        else:
            # Continue from an independently scored approach without a reset.
            self.trace = []
        result = dict(policy=self.policy, trial=trial["trial"], initial_conditions=trial, success=False,
                      reason="", phases=[], stage_errors={},
                      task_target_object_pos=self.planner._task_target_box_center.tolist(),
                      task_target_object_quat_wxyz=self.planner._task_target_box_quat_wxyz.tolist())
        if self.fallen():
            result["reason"] = "fall_during_idle_warmup"
        else:
            for phase_index, (stage, name) in enumerate(self.args.phases):
                response = VLMQuery.Response()
                response.success = True
                response.next_keyframe = name
                response.object_in_manipulation = name != "approach_box"
                if not self.planner.publish_planner_outputs(response):
                    raise RuntimeError(f"Production planner rejected {name}")
                if not self.controller.have_goal or len(self.controller.goal_sequence) != 1:
                    raise RuntimeError("Controller did not accept exactly one world-space goal")
                goal = self.controller._select_goal().copy()
                tag = f"{Path(self.policy).stem}_trial{trial['trial']:02d}_{name}"
                (self.output / "goals" / (tag + ".npz")).write_bytes(self.goal_blob)
                np.save(self.output / "goals" / (tag + "_policy_goal.npy"), goal)
                start = float(self.data.time)
                ready, passed = False, False
                metrics = self.errors(goal)
                for step in range(int(np.ceil(self.args.goal_timeout / (.001667 * 10)))):
                    self.tick(goal)
                    self.max_box_height = max(self.max_box_height, float(self.data.qpos[self.box_adr + 2]))
                    metrics = self.errors(goal)
                    passed = bool(self.planner.evaluate_last_action_success())
                    ready = self.planner.ready_for_next_request()
                    if step % 6 == 0:
                        self.trace.append(np.concatenate(([self.data.time, phase_index], self.data.qpos, self.data.qvel)))
                    if self.fallen() or (ready and passed):
                        break
                phase_success = bool(passed and ready and not self.fallen())
                result["phases"].append(dict(stage=stage, keyframe=name, success=phase_success,
                    duration_s=float(self.data.time - start), stationary_ready=bool(ready), errors=metrics,
                    success_checks=self.planner._action_success_checks, **self.contacts()))
                print(f"{self.policy} trial {trial['trial']:02d} {name}: {'PASS' if phase_success else 'FAIL'} "
                      f"t={self.data.time-start:.2f}s root={metrics['root_position_m']:.3f}m "
                      f"joint={metrics['joint_rmse_rad']:.3f}rad obj={metrics['object_position_m']}", flush=True)
                if not phase_success:
                    result["reason"] = f"{name}: {'fall' if self.fallen() else 'timeout'}"
                    break
                if name in STAGE_ENDPOINTS:
                    result["stage_errors"][stage] = metrics
            else:
                result["success"] = bool(self.planner.measured_task_completion())
                result["reason"] = "success" if result["success"] else "final_object_outside_task_tolerance"
        result["sim_time_s"] = float(self.data.time)
        result["max_box_lift_m"] = float(self.max_box_height - self.initial_box_height)
        result["final_task_object_error_m"] = self.planner._object_error_to_task_target()[0]
        result["final_contacts"] = self.contacts()
        np.savez_compressed(self.output / "traces" / f"{Path(self.policy).stem}_trial{trial['trial']:02d}.npz",
                            samples=np.asarray(self.trace), nq=self.model.nq, nv=self.model.nv)
        return result

    def close(self):
        vml.time = self._real_planner_time
        if self.planner is not None:
            self.planner.destroy_node()
        self.retargeter.destroy_node()
        self.controller.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", nargs="+", type=Path, default=[PACKAGE / "model" / name for name in
                        ("model_101000.onnx", "model_29000_uniform.onnx", "model_28000.onnx")])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--goal-timeout", type=float, default=30.)
    parser.add_argument("--warmup-seconds", type=float, default=2.)
    parser.add_argument("--skip-approach", action="store_true",
                        help="Begin with stand_before_pick_box; does not move the reset pose automatically")
    parser.add_argument("--initial-root-pos", nargs=3, type=float, default=[-2., 0., .8],
                        metavar=("X", "Y", "Z"), help="Nominal root reset position before paired XY perturbations")
    parser.add_argument("--place-noise-xy-m", type=float, default=.10,
                        help="Uniform placement destination noise per XY axis in metres (default ±0.10; 0 reproduces earlier protocol)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1 or args.goal_timeout <= 0 or args.warmup_seconds < 0:
        parser.error("Trials/timeout must be positive and warmup nonnegative")
    if not np.all(np.isfinite(args.initial_root_pos)):
        parser.error("Initial root position must be finite")
    if not np.isfinite(args.place_noise_xy_m) or args.place_noise_xy_m < 0:
        parser.error("Placement XY noise must be finite and nonnegative")
    args.phases = PHASES[1:] if args.skip_approach else PHASES
    if mujoco.__version__ != "3.3.5":
        raise RuntimeError(f"Use MuJoCo 3.3.5 to match /opt/mujoco ROS simulator; got {mujoco.__version__}")
    for policy in args.policies:
        if not policy.is_file():
            raise FileNotFoundError(policy)
    # All production ROS interfaces are inert (no executor); isolate discovery
    # too. All retargeting/control/state calls below remain inside this process.
    os.environ["ROS_DOMAIN_ID"] = "217"
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ["ROS_LOG_DIR"] = str(args.output / "ros_logs")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("goals", "traces"):
        (args.output / name).mkdir()
    config = yaml.safe_load(CONFIG.read_text())
    config.update(policy_observation_layout="ver3", num_obs=148, num_history=10, goal_state_file="")
    config_path = args.output / "controller_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    initial_conditions = perturbations(args.trials, args.seed, args.initial_root_pos, args.place_noise_xy_m)
    metadata = dict(created=datetime.now().isoformat(), seed=args.seed, initial_conditions=initial_conditions,
                    phases=args.phases, physics_timestep=.001667, control_decimation=10,
                    skip_approach=args.skip_approach, nominal_initial_root_pos=args.initial_root_pos,
                    stage_error_keyframes=sorted(STAGE_ENDPOINTS.intersection(name for _, name in args.phases)),
                    mujoco_version=mujoco.__version__, onnxruntime_version=control.ort.__version__,
                    goal_timeout_s=args.goal_timeout, warmup_seconds=args.warmup_seconds,
                    perturbation_bounds=dict(xy_m=.02, yaw_deg=2., heights="unchanged",
                                             place_xy_m=args.place_noise_xy_m),
                    policy_sha256={str(path): sha256(path) for path in args.policies},
                    source_sha256={str(path): sha256(path) for path in [SCENE, CONFIG, Path(__file__),
                        Path(control.__file__), Path(vml.__file__), COMMONS / "data/robots/g1_description/g1_29dof_crl.xml",
                        SRC / "lm/lm/keyframe_retargeter_node.py", *sorted((SRC / "lm/keyframes").glob("*_box.npz"))]},
                    caveats=["Numerical VLM success, not image-verified grasp/release.",
                             "Synchronous headless physics bypasses ROS transport/FSM startup timing.",
                             "No automatic recovery/retries; each phase waits for success+stationary or timeout.",
                             "Approach object goals are masked: object error is null, not measured against zero.",
                             "Errors are endpoint measurements; summary includes only fully successful trials."])
    write_json(args.output / "manifest.json", metadata)
    evaluator = Evaluator(config_path, args.policies[0], args.output, args)
    trials = []
    started = time.monotonic()
    try:
        for policy in args.policies:
            evaluator.set_policy(policy)
            for initial in initial_conditions:
                result = evaluator.run_trial(initial)
                trials.append(result)
                metadata["vlm_thresholds"] = {name: evaluator.planner.get_parameter(name).value
                                               for name in THRESHOLD_PARAMETERS}
                metadata["effective_retargeting"] = {
                    "ik_enabled": False, "box_geometry": SIM_TARGET_BOX_GEOMETRY.__dict__,
                    "source_geometry": SOURCE_BOX_GEOMETRY.__dict__,
                    "standing_waist_pitch_deg": evaluator.retargeter.get_parameter("standing_waist_pitch_deg").value,
                    "target_box_orientation_offset_rpy_deg": evaluator.retargeter.get_parameter("target_box_orientation_offset_rpy_deg").value,
                }
                with (args.output / "trials.jsonl").open("a") as stream:
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                write_json(args.output / "summary.json", summarize(trials))
                metadata["completed_trials"] = len(trials)
                write_json(args.output / "manifest.json", metadata)
                print(f"COMPLETED {len(trials)}/{len(args.policies)*args.trials}: {result['reason']}", flush=True)
        rows = [dict(policy=row["policy"], trial=row["trial"], stage=stage, **metrics)
                for row in trials if row["success"] for stage, metrics in row["stage_errors"].items()]
        with (args.output / "successful_stage_errors.csv").open("w") as stream:
            fields = list(rows[0]) if rows else ["policy", "trial", "stage"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        phase_rows = [dict(policy=row["policy"], trial=row["trial"], trial_success=row["success"],
                           keyframe=phase["keyframe"], phase_success=phase["success"],
                           duration_s=phase["duration_s"], **phase["errors"])
                      for row in trials for phase in row["phases"]]
        with (args.output / "all_phase_errors.csv").open("w") as stream:
            fields = list(phase_rows[0]) if phase_rows else ["policy", "trial", "trial_success", "keyframe"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(phase_rows)
        metadata["wall_time_seconds"] = time.monotonic() - started
        metadata["completed_trials"] = len(trials)
        write_json(args.output / "manifest.json", metadata)
    finally:
        evaluator.close()
    print(json.dumps(summarize(trials), indent=2), flush=True)


if __name__ == "__main__":
    main()
