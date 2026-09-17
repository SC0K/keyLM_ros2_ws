# Offline policy evaluation

All offline experiment runners, replay tooling, tests, and recorded results live here, outside the live `lm` ROS package. `COLCON_IGNORE` prevents ROS builds from discovering anything under this directory. Nothing here is a ROS launch file or installed production entry point.

```text
evaluation/
  evaluate_box_policies.py    # Fixed/manual sequence benchmark
  evaluate_box_distances.py   # Placement and starting-distance sweeps
  replay_policy_trial.py      # MP4 replay of recorded states and goal markers
  tests/                     # Evaluation-only tests
  results/                   # Trial logs, traces, goals, reports, CSV, LaTeX, video
```

The tools reuse the production controller and retargeter to match deployment behavior. This dependency is one-way: the live pipeline does not import evaluation code. The shared `lm.video_recording` helper stays in `lm` because the live experiment camera uses it too. Models and keyframe libraries also remain in their original locations.

## Environment

Run from the workspace source root (not from inside the evaluation directory):

```bash
cd /home/sitongchen/keyLM_ros2_ws/src
source /opt/ros/humble/setup.bash
source /home/sitongchen/keyLM_ros2_ws/install/setup.bash
export PYTHONPATH=/home/sitongchen/keyLM_ros2_ws/src:/tmp/keyframe-eval-deps.YnSpFF:/home/sitongchen/keyLM_ros2_ws/src/lm:/home/sitongchen/keyLM_ros2_ws/src/crl-humanoid-ros/crl_g1_goalcontroller_py/crl_g1_goalcontroller_python:$PYTHONPATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

The recorded runs used Python MuJoCo **3.3.5**, matching the C++ simulator. `/tmp/keyframe-eval-deps.YnSpFF` is the isolated installation used for these runs. If it has been cleared, create another isolated directory, install `mujoco==3.3.5` there with `/usr/bin/python3 -m pip install --target <directory> --no-deps mujoco==3.3.5`, and substitute its path above. The remaining dependencies come from the existing workspace environment.

## Run

Latest starting-distance and placement-distance sweep:

```bash
/usr/bin/python3 -m evaluation.evaluate_box_distances --approach-range 0.5 2.3 \
  --output evaluation/results/approach_place_repeat
```

Placement-distance sweep with the original nominal robot X=-2 start:

```bash
/usr/bin/python3 -m evaluation.evaluate_box_distances \
  --output evaluation/results/place_distance_repeat
```

Fixed 1 m placement benchmark (the original three-policy comparison had no extra placement noise):

```bash
/usr/bin/python3 -m evaluation.evaluate_box_policies --trials 10 --place-noise-xy-m 0 \
  --output evaluation/results/fixed_place_repeat
```

This runner also accepts `--policies <onnx-path> ...`, `--skip-approach`, `--initial-root-pos X Y Z` and `--place-noise-xy-m`. Its current placement-noise default is ±0.10 m per XY axis; specify zero to reproduce the original fixed-target benchmark. The distance-sweep runner uses no extra placement noise.

The distance-sweep runner also accepts `--policies <onnx-path> ...` to compare other checkpoints on the same paired trials.

Add `--repetitions 10` for 100 trials per policy: ten repetitions of each distance pair with independent reset perturbations. The default seed is 20260911 (`--seed` overrides it); the first ten samples remain identical to the original one-repetition sweep.

Use `--physical-box-size 0.3 0.3 0.3` for a physical 0.30 m cube. This saves evaluation-only scene/robot XML snapshots in the result directory; live scenes, flat-hand geometry, mass/inertia, and retargeting dimensions are unchanged. Omit it to use the normal box scene.

All output directories must be new; existing results are never overwritten.

## Replay

```bash
MUJOCO_GL=osmesa /usr/bin/python3 -m evaluation.replay_policy_trial \
  evaluation/results/box_policy_comparison_20260911_paired10 \
  --policy model_29000_uniform.onnx --trial 1 \
  --output evaluation/results/uniform_trial01_repeat.mp4
```

For sweep trials, point the replay tool at the run's `approach/` or `manipulation/` subdirectory. Replay uses recorded ~10 Hz states, not a new inference/physics run.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest -q evaluation/tests
```

## Historical paths

The complete former `lm/evaluation_results/` tree was moved intact to `evaluation/results/`. Historical manifests, controller snapshots and reports intentionally retain the paths/commands recorded at experiment time; their bytes were not rewritten. For current commands, use this README. Translate:

- `lm/evaluation_results/...` → `evaluation/results/...`
- `python -m lm.evaluate_box_policies` → `python -m evaluation.evaluate_box_policies`
- `python -m lm.evaluate_box_distances` → `python -m evaluation.evaluate_box_distances`
- `python -m lm.replay_policy_trial` → `python -m evaluation.replay_policy_trial`

Relative links between historical reports still work. Source-file hashes in old manifests describe the original code at execution time, not its subsequently relocated/updated version. No compatibility wrapper is left in the live `lm` package.
