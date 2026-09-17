"""Distance sweep with separately scored approach and reset-assisted manipulation."""

import argparse
from copy import deepcopy
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import time

import numpy as np
import yaml

from evaluation import evaluate_box_policies as base


def sweep_initial_conditions(approach_range=None, repetitions=1, seed=20260911):
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("Repetitions must be a positive integer")
    initials = base.perturbations(10 * repetitions, seed, place_noise_xy_m=0.)
    distances = np.linspace(*approach_range, 10) if approach_range is not None else None
    for i, initial in enumerate(initials):
        case = i % 10
        initial["place_distance_m"] = round(.2 * (case + 1), 1)
        nominal_x = .35 - float(distances[case]) if distances is not None else -2.
        initial["root_pos"][0] += nominal_x + 2.
        initial["nominal_root_x_m"] = nominal_x
        initial["approach_distance_m"] = .35 - nominal_x
        initial["actual_initial_xy_distance_m"] = float(np.linalg.norm(
            np.asarray(initial["root_pos"][:2]) - initial["box_pos"][:2]))
    return initials


def evaluate_distance(evaluator, initial, output):
    evaluator.output = output / "approach"
    evaluator.args.phases = base.PHASES[:1]
    approach = evaluator.run_trial(initial)
    approach_ok = bool(approach["phases"] and approach["phases"][0]["success"])
    # A standalone approach is scored by its own gate, not final placement.
    approach["success"] = approach_ok
    if approach_ok:
        approach["reason"] = "success"
    recovery = not approach_ok
    manipulation_initial = deepcopy(initial)
    if recovery:
        # Retain reset noise while moving any nominal starting X to zero.
        manipulation_initial["root_pos"][0] -= initial.get("nominal_root_x_m", -2.)
        manipulation_initial["nominal_root_x_m"] = 0.
        manipulation_initial["actual_initial_xy_distance_m"] = float(np.linalg.norm(
            np.asarray(manipulation_initial["root_pos"][:2]) - manipulation_initial["box_pos"][:2]))
    evaluator.output = output / "manipulation"
    evaluator.args.phases = base.PHASES[1:]
    manipulation = evaluator.run_trial(manipulation_initial, reset_state=recovery)
    pick_ok = any(p["keyframe"] == "stand_after_pick_box" and p["success"] for p in manipulation["phases"])
    return dict(policy=approach["policy"], trial=initial["trial"],
                approach_distance_m=initial.get("approach_distance_m", 2.35),
                distance_m=initial["place_distance_m"], approach_success=approach_ok,
                recovery_reset=recovery, pick_success=pick_ok,
                place_goal_success=any(p["keyframe"] == "crouch_to_place_box" and p["success"] for p in manipulation["phases"]),
                pick_place_success=manipulation["success"],
                end_to_end_success=bool(approach_ok and manipulation["success"]),
                approach=approach, manipulation=manipulation)


def summary(rows):
    result = {}
    for policy in dict.fromkeys(r["policy"] for r in rows):
        selected = [r for r in rows if r["policy"] == policy]
        n = len(selected)
        rates = {key: dict(successes=sum(bool(r[key]) for r in selected), trials=n,
                           rate=sum(bool(r[key]) for r in selected) / n)
                 for key in ("approach_success", "pick_success", "place_goal_success", "pick_place_success", "end_to_end_success")}
        approach_stats = base.summarize([r["approach"] for r in selected])[policy]
        manipulation_stats = base.summarize([r["manipulation"] for r in selected])[policy]
        endpoint_stats = {}
        for stage, name in (("approach", "approach_box"), ("pick", "crouch_to_pick_box"), ("place", "crouch_to_place_box")):
            samples = [p.get("errors", {}) for r in selected for segment in ("approach", "manipulation")
                       for p in r[segment]["phases"] if p["keyframe"] == name and p["success"]]
            endpoint_stats[stage] = {}
            for metric in samples[0] if samples else []:
                values = [sample[metric] for sample in samples if sample[metric] is not None]
                endpoint_stats[stage][metric] = dict(n=len(values), mean=float(np.mean(values)) if values else None,
                    std=float(np.std(values, ddof=1)) if len(values)>1 else None)
        result[policy] = dict(**rates, recovery_resets=sum(r["recovery_reset"] for r in selected),
                             successful_keyframe_errors=endpoint_stats,
                             successful_approach_errors=approach_stats["successful_trial_stage_errors"]["approach"],
                             successful_pick_place_errors=manipulation_stats["successful_trial_stage_errors"],
                             manipulation_failures=manipulation_stats["failures"])
    return result


def save_csv(output, rows):
    fields = ["policy", "trial", "approach_distance_m", "distance_m", "approach_success", "recovery_reset",
              "pick_success", "place_goal_success", "pick_place_success", "end_to_end_success"]
    with (output / "outcomes.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
    phases = [dict(policy=r["policy"], approach_distance_m=r["approach_distance_m"], distance_m=r["distance_m"], trial=r["trial"], segment=segment,
                   recovery_reset=r["recovery_reset"], keyframe=p["keyframe"], phase_success=p["success"],
                   duration_s=p["duration_s"], **p["errors"])
              for r in rows for segment in ("approach", "manipulation") for p in r[segment]["phases"]]
    with (output / "all_phase_errors.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(phases[0]) if phases else ["policy", "distance_m"])
        writer.writeheader()
        writer.writerows(phases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1,
                        help="Repetitions of the ten distance pairs with independent reset samples (10 gives 100 trials per policy)")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--policies", nargs="+", type=Path, default=[base.PACKAGE / "model" / name for name in
                        ("model_101000.onnx", "model_29000_uniform.onnx")],
                        help="Checkpoints evaluated on identical paired initial conditions")
    parser.add_argument("--approach-range", type=float, nargs=2, metavar=("MIN_M", "MAX_M"),
                        help="Ten nominal starting XY distances from the box; leaves the approach stop offset unchanged")
    parser.add_argument("--physical-box-size", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Evaluation-only physical dimensions; mass/inertia and retargeting sizes stay unchanged")
    args = parser.parse_args()
    if args.repetitions < 1 or args.seed < 0:
        parser.error("Repetitions must be positive and seed must be nonnegative")
    if args.physical_box_size is not None and (not np.all(np.isfinite(args.physical_box_size))
                                              or np.any(np.asarray(args.physical_box_size) <= 0)):
        parser.error("Physical box dimensions must be finite and positive")
    policies = [path.resolve() for path in args.policies]
    if any(not path.is_file() for path in policies):
        parser.error("Every policy path must point to an existing checkpoint")
    if len({path.name for path in policies}) != len(policies):
        parser.error("Policy filenames must be unique to avoid overwriting trial artifacts")
    if args.approach_range is not None and (not np.all(np.isfinite(args.approach_range))
            or args.approach_range[0] <= 0 or args.approach_range[1] <= args.approach_range[0]):
        parser.error("Approach range must be finite, positive and increasing")
    args.warmup_seconds, args.goal_timeout, args.phases = 2., 30., base.PHASES
    if base.mujoco.__version__ != "3.3.5":
        raise RuntimeError("Use MuJoCo 3.3.5 as in the original benchmark")
    args.output.mkdir(parents=True, exist_ok=False)
    for segment in ("approach", "manipulation"):
        for name in ("goals", "traces"):
            (args.output / segment / name).mkdir(parents=True)
    os.environ.update(ROS_DOMAIN_ID="217", ROS_LOCALHOST_ONLY="1", ROS_LOG_DIR=str(args.output / "ros_logs"))
    config = yaml.safe_load(base.CONFIG.read_text())
    if args.physical_box_size is not None:
        args.scene = base.snapshot_box_scene(args.output, args.physical_box_size)
        config["xml_path"] = str(args.scene)
    config.update(policy_observation_layout="ver3", num_obs=148, num_history=10, goal_state_file="")
    config_path = args.output / "controller_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    for segment in ("approach", "manipulation"):
        (args.output / segment / "controller_config.yaml").write_text(config_path.read_text())
    initials = sweep_initial_conditions(args.approach_range, args.repetitions, args.seed)
    sources = [Path(__file__), Path(base.__file__), Path(base.control.__file__), base.CONFIG, base.SCENE,
               base.COMMONS / "data/robots/g1_description/g1_29dof_crl.xml",
               base.COMMONS / "include/crl_humanoid_commons/RobotParameters.h",
               *sorted((base.SRC / "lm/lm").glob("*.py")),
               *sorted((base.SRC / "lm/keyframes").glob("*_box.npz"))]
    if args.physical_box_size is not None:
        sources += [args.scene, args.scene.parent / "robot_snapshot.xml"]
    manifest = dict(created=datetime.now().isoformat(), initial_conditions=initials, seed=args.seed,
                    repetitions_per_distance=args.repetitions, trials_per_policy=len(initials),
                    phases=base.PHASES, mujoco_version=base.mujoco.__version__,
                    onnxruntime_version=base.control.ort.__version__, physics_timestep=.001667,
                    control_decimation=10, goal_timeout_s=30., warmup_seconds=2.,
                    reset_perturbations=dict(xy_m=.02, yaw_deg=2.), placement_noise_xy_m=0.,
                    approach_range_m=args.approach_range,
                    physical_box_size_override_xyz=args.physical_box_size,
                    simulation_scene=str(getattr(args, "scene", base.SCENE)),
                    recovery="On approach failure: reset robot nominal X to 0, restore initial box and all state/history; retry manipulation only once.",
                    error_cohorts="successful_keyframe_errors: each passed crouching pick/place or approach endpoint, irrespective of later failure; legacy full-manipulation statistics also retained.",
                    policy_sha256={str(p):base.sha256(p) for p in policies},
                    source_sha256={str(p):base.sha256(p) for p in sources}, completed_trials=0)
    base.write_json(args.output / "manifest.json", manifest)
    rows = []
    evaluator = base.Evaluator(config_path, policies[0], args.output, args)
    started = time.monotonic()
    try:
        for policy in policies:
            evaluator.set_policy(policy)
            for initial in initials:
                row = evaluate_distance(evaluator, initial, args.output)
                rows.append(row)
                with (args.output / "trials.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                for segment in ("approach", "manipulation"):
                    with (args.output / segment / "trials.jsonl").open("a") as stream:
                        stream.write(json.dumps(row[segment], allow_nan=False) + "\n")
                manifest["vlm_thresholds"] = {name:evaluator.planner.get_parameter(name).value for name in base.THRESHOLD_PARAMETERS}
                manifest["physical_box_size_xyz"] = (2 * evaluator.model.geom("box_geom").size).tolist()
                manifest["effective_retargeting"] = dict(ik_enabled=False,
                    source_geometry=base.SOURCE_BOX_GEOMETRY.__dict__, target_geometry=base.SIM_TARGET_BOX_GEOMETRY.__dict__,
                    standing_waist_pitch_deg=evaluator.retargeter.get_parameter("standing_waist_pitch_deg").value,
                    target_box_orientation_offset_rpy_deg=evaluator.retargeter.get_parameter("target_box_orientation_offset_rpy_deg").value)
                manifest["completed_trials"] = len(rows)
                manifest["wall_time_seconds"] = time.monotonic() - started
                base.write_json(args.output / "manifest.json", manifest)
                base.write_json(args.output / "summary.json", summary(rows))
                save_csv(args.output, rows)
                print(f"COMPLETED {len(rows)}/{len(policies) * len(initials)} {policy.name} distance={row['distance_m']:.1f}m "
                      f"approach={row['approach_success']} reset={row['recovery_reset']} "
                      f"pick/place={row['pick_place_success']} end-to-end={row['end_to_end_success']}", flush=True)
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
