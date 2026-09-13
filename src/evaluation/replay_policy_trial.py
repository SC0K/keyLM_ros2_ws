"""Render a recorded evaluator trial without rerunning physics or the policy.

Use MUJOCO_GL=osmesa for headless software rendering. The MP4 holds recorded
~10 Hz states at the requested output frame rate; it does not invent motion.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np
import yaml

from crl_g1_goalcontroller_python import g1_keyframe_controller as control
from lm.video_recording import VideoRecording


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--policy", default="model_29000_uniform.onnx")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.experiment / "trials.jsonl").read_text().splitlines()]
    trial = next(row for row in rows if row["policy"] == args.policy and row["trial"] == args.trial)
    tag = f"{Path(args.policy).stem}_trial{args.trial:02d}"
    with np.load(args.experiment / "traces" / f"{tag}.npz") as trace:
        samples, nq, nv = trace["samples"], int(trace["nq"]), int(trace["nv"])
    if len(samples) == 0:
        raise ValueError("Trial has no recorded states")
    src = Path(__file__).resolve().parents[1]
    scene = src / "crl-humanoid-ros/crl_humanoid_commons/data/robots/g1_description/scene_crl_with_box_video.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    if (model.nq, model.nv) != (nq, nv):
        raise ValueError("Replay scene coordinates differ from recorded trace")
    data, goal_data = mujoco.MjData(model), mujoco.MjData(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = args.width, args.height
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    config = yaml.safe_load((args.experiment / "controller_config.yaml").read_text())
    joint_ids = np.array([model.joint(name).id for name in control.POLICY_JOINT_NAMES])
    # Reuse production compact-goal decoding, including the prepared locomotion
    # joint override saved by the evaluator, without constructing any ROS node.
    decoder = SimpleNamespace(model=model, num_actions=29,
                              default_angles=np.asarray(config["default_angles"], dtype=np.float32),
                              joint_qpos_adr=model.jnt_qposadr[joint_ids], policy_to_mujoco=np.arange(29))
    body_ids = [model.body(name).id for name in control.FEATURE_BODY_NAMES]
    marker_ids = [int(model.body(f"target_kp_{index:02d}").mocapid[0]) for index in range(len(body_ids))]
    goals = []
    for phase in trial["phases"]:
        goal = np.load(args.experiment / "goals" / f"{tag}_{phase['keyframe']}_policy_goal.npy")
        goal_data.qpos[:] = control.G1KeyframeController._compact_goal_to_qpos(decoder, goal)
        mujoco.mj_forward(model, goal_data)
        goals.append(goal_data.xpos[body_ids].copy())
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    recording = None
    try:
        recording = VideoRecording(args.output, args.fps, args.width, args.height)
        start, end = float(samples[0, 0]), float(trial["sim_time_s"])
        frames = int(np.ceil((end - start) * args.fps)) + 1
        last_index, frame = -1, None
        for index in range(frames):
            t = min(start + index / args.fps, end)
            sample_index = int(np.clip(np.searchsorted(samples[:, 0], t, side="right") - 1, 0, len(samples) - 1))
            if sample_index != last_index:
                sample = samples[sample_index]
                phase_index = int(sample[1])
                data.qpos[:] = sample[2:2 + nq]
                data.qvel[:] = sample[2 + nq:2 + nq + nv]
                data.time = sample[0]
                data.mocap_pos[marker_ids] = goals[phase_index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="experiment_video", scene_option=option)
                frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                scale = args.width / 1920
                labels = [f"Uniform policy | Trial {args.trial} | Recorded experiment replay",
                          f"{trial['phases'][phase_index]['keyframe']} | sim t = {sample[0]:.2f} s",
                          "Green points: policy keyframe goal | Recorded states ~10 Hz"]
                for line, label in enumerate(labels):
                    cv2.putText(frame, label, (int(28*scale), int((42+38*line)*scale)),
                                cv2.FONT_HERSHEY_SIMPLEX, .8*scale, (35, 35, 35), max(1, int(2*scale)), cv2.LINE_AA)
                last_index = sample_index
            recording.write(frame, now=index / args.fps)
        print(f"Saved {args.output}: {frames} output frames, {len(samples)} recorded states; trial success={trial['success']}")
    finally:
        if recording is not None:
            recording.close()
        renderer.close()


if __name__ == "__main__":
    main()
