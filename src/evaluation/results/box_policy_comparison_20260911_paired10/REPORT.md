# Box policy comparison — 2026-09-11

Completed 30 headless, synchronous MuJoCo trials: 10 paired randomized trials per policy. These are simulation results using the production controller observation builder, keyframe retargeter and VLM numerical success methods, not live ROS deployment trials or VLM decision-quality measurements.

## Success rates

| Policy | Successful trials | Success rate | Failures |
| --- | ---: | ---: | --- |
| `model_101000.onnx` | 10/10 | 100% | None |
| `model_29000_uniform.onnx` | 3/10 | 30% | Five falls at `stand_before_pick_box`; one fall and one timeout at `stand_after_pick_box` |
| `model_28000.onnx` | 0/10 | 0% | All ten timed out at `approach_box` |

Successful trials for `model_29000_uniform` were 1, 9 and 10. All 13 successful trials ended with box–floor contact and no detected box–robot contact. Their maximum box lifts were 0.765–0.810 m for `model_101000` and 0.829–0.852 m for `model_29000_uniform`. Contact and lift are diagnostics, not additional success thresholds.

`model_101000` was more reliable in this cohort and had smaller joint errors. Among successful trials, `model_29000_uniform` placed the object closer to the target, but with larger root and joint errors. Successful-only error averages compare different surviving subsets (10 versus 3); they should not be read as an unbiased paired accuracy ranking. Ten trials are a small sample, not evidence of guaranteed reliability.

## Keyframe errors on fully successful trials

Mean ± sample standard deviation. Root and object errors are Euclidean 3D position errors in metres; joint error is RMSE over all 29 joints in radians.

| Policy | Stage | n | Root position (m) | Joint RMSE (rad) | Object position (m) |
| --- | --- | ---: | ---: | ---: | ---: |
| `model_101000` | Approach | 10 | 0.0999 ± 0.0363 | 0.0777 ± 0.0050 | N/A |
| `model_101000` | Pick | 10 | 0.1119 ± 0.0172 | 0.1161 ± 0.0028 | 0.040745 ± 0.000001 |
| `model_101000` | Place | 10 | 0.0690 ± 0.0107 | 0.1187 ± 0.0023 | 0.2073 ± 0.0152 |
| `model_29000_uniform` | Approach | 3 | 0.0984 ± 0.0510 | 0.1024 ± 0.0034 | N/A |
| `model_29000_uniform` | Pick | 3 | 0.1057 ± 0.0237 | 0.1360 ± 0.0036 | 0.0413 ± 0.0008 |
| `model_29000_uniform` | Place | 3 | 0.1855 ± 0.0101 | 0.1357 ± 0.0012 | 0.1009 ± 0.0057 |
| `model_28000` | All stages | 0 | N/A | N/A | N/A |

Measurements are taken at the first success-and-stationary endpoint of `approach_box`, `crouch_to_pick_box`, and `crouch_to_place_box`, respectively. “Pick” here measures the crouching grasp keyframe, not the later lifted standing keyframe. Every intermediate and final standing keyframe also has its own endpoint errors in `all_phase_errors.csv`. This avoids reporting the final standing frame's root error as the placement error: that standing frame is anchored at the robot's current root.

Approach is locomotion mode, so object goals are masked and object errors are recorded as null/N/A rather than measured against zero. Root/object orientation errors, joint MAE, maximum joint error, root XY error and mean body position error are also saved in the CSV/JSON files.

## Protocol

Nominal reset poses (quaternions WXYZ):

```yaml
initial_root_pos: [-2.0, 0.0, 0.8]
initial_root_quat_wxyz: [1.0, 0.0, 0.0, 0.0]
initial_object_pos: [0.35, 0.0, 0.15]
initial_object_quat_wxyz: [1.0, 0.0, 0.0, 0.0]
```

Robot and box XY are independently perturbed uniformly by ±0.02 m per coordinate; yaw independently by ±2°. Heights are unchanged. Seed: `20260911`. The exact same ten initial conditions and walking-goal noise seeds are used for all three policies and saved in `manifest.json`.

There is a two-second idle GOAL-policy warmup after reset. Placement is 1 m along world +X from the starting observed box position, using the production planner's task-target construction. The manual sequence is:

```text
approach_box → stand_before_pick_box → crouch_to_pick_box → stand_after_pick_box
             → stand_before_place_box → crouch_to_place_box → stand_after_place_box
```

Each goal must pass the current VLM numerical action check and stationary gate before advancing. Each has a 30-second simulation-time timeout. There are no retries or recovery actions. Full success requires all seven goals to pass, the final numerical task-completion check to pass, and no detected robot fall. Fall guard: pelvis height below 0.30 m, root-up world-Z component below 0.25, or nonfinite state.

Current VLM thresholds, unchanged:

| Check | Threshold |
| --- | ---: |
| Minimum action duration | 1.0 s |
| Continuous stationary hold | 0.5 s |
| Robot linear / angular speed | 0.10 m/s / 0.15 rad/s |
| Object linear / angular speed | 0.15 m/s / 0.30 rad/s |
| Mean body position error | 0.30 m |
| Root position / orientation error | 0.30 m / 0.80 rad |
| Per-keyframe object position error | 0.45 m |
| Approach / pre-pick alternative XY reach check | 0.45 m |
| Final task object position error | 0.60 m |

As in production, approach excludes object error; approach and pre-pick can pass via the XY reach check instead of the generic tracking check. Object orientation is not a success criterion. The numerical checks do not prove grasp/release, and the 0.60 m final tolerance is generous; this report does not reinterpret it as precise placement success.

## Configuration and fidelity

- Physics: MuJoCo **3.3.5**, matching the installed C++ simulator version; normal `scene_crl_with_box.xml`, unchanged physical properties. Timestep 0.001667 s, ten physics steps per policy inference (~60 Hz).
- Three ONNX checkpoints each have `[1, 1480]` inputs and `[1, 29]` outputs. CPU inference uses the current `ver3` observation layout, ten history samples, production COM-velocity correction, action scaling and PD gains. Canonical torque limits are read from the C++ simulator's `RobotParameters.h`.
- Production keyframe library and retargeting paths are used. IK is off, generated standing lean is 5°, and policy-only object orientation offset is zero.
- The existing retargeting configuration uses a **0.30 m source cube and 0.35 m target cube**, while the physical MuJoCo box is **0.345 × 0.250 × 0.285 m**. This existing difference was preserved, not corrected for the benchmark. The production task target consequently uses a nominal box-center height of 0.175 m, while authored keyframe target heights follow the existing retargeter behavior.
- Physics, observations and planner callbacks run synchronously in one process. ROS nodes are constructed for their configured helpers but no executor, VLM query, hardware command, or live control-topic publisher is used. ROS latency, scheduling jitter, real FSM startup and camera-based decision quality are not evaluated.
- No active deployment controller configuration was edited to switch policies. `controller_config.yaml` is an evaluation-local snapshot.

## Artifacts and validation

- `manifest.json`: seeds/poses, effective thresholds and retargeting settings, versions and source/checkpoint SHA-256 hashes.
- `summary.json`: success counts and successful-only metric means/sample standard deviations.
- `trials.jsonl`: all 30 outcomes, phase checks, errors, durations, lift/contact diagnostics and failure reasons.
- `successful_stage_errors.csv`: 39 rows (13 successful trials × 3 stages).
- `all_phase_errors.csv`: every attempted phase, including failures.
- `goals/`: serialized retargeter output and compact controller goal for each attempted phase.
- `traces/`: sampled simulation time, phase index, full qpos and qvel at roughly 10 Hz.

Validated exactly ten trials per policy, identical paired initial conditions, matching source/checkpoint hashes, and the expected successful-stage CSV row count. Four evaluator tests cover perturbations, successful-only aggregation/masking, canonical torque limits, and agreement of the production COM-velocity observation with MuJoCo. Pilot integration runs are excluded from these results.

## Repeat the experiment

From this workspace, with ROS Humble, built workspace interfaces and Python MuJoCo 3.3.5 available:

```bash
cd /home/sitongchen/keyLM_ros2_ws/src
source /opt/ros/humble/setup.bash
source /home/sitongchen/keyLM_ros2_ws/install/setup.bash
# This isolated directory was used for the recorded run; recreate it if /tmp was cleared.
export PYTHONPATH=/tmp/keyframe-eval-deps.YnSpFF:/home/sitongchen/keyLM_ros2_ws/src/lm:/home/sitongchen/keyLM_ros2_ws/src/crl-humanoid-ros/crl_g1_goalcontroller_py/crl_g1_goalcontroller_python:$PYTHONPATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
/usr/bin/python3 -m lm.evaluate_box_policies --trials 10 --seed 20260911 \
  --output /home/sitongchen/keyLM_ros2_ws/src/lm/evaluation_results/box_policy_repeat
```

Choose a new output directory; existing results are never overwritten. If the isolated dependency directory is absent, create a new one with `mktemp -d`, install `mujoco==3.3.5` there with `/usr/bin/python3 -m pip install --target <directory> --no-deps mujoco==3.3.5`, and substitute that directory in `PYTHONPATH`. The remaining dependencies come from the existing ROS/workspace environment. Add `--policies <path> ...` to select checkpoints; the default is the three evaluated here.
