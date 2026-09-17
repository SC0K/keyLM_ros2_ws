# 0.30 m box: 101000 vs Uniform — 2026-09-15

Completed ten fresh paired trials per policy with a physical **0.300 × 0.300 × 0.300 m cube**. Both use the earlier **flat-hand G1**, not the newly added ball-hand test launch. Live simulation files and existing results were not changed.

## Goal errors

Mean ± sample standard deviation over passed keyframe endpoints, including endpoints in trials that later fail. Pick is `crouch_to_pick_box`; place is `crouch_to_place_box`. Measurements are at goal completion, not time-averaged. Root and object position errors are 3D Euclidean distances; joint RMSE covers all 29 policy joints.

| Policy | Stage | n | Root position (m) | Joint RMSE (rad) | Object position (m) |
| --- | --- | ---: | ---: | ---: | ---: |
| 101000 | Approach | 10 | 0.117473 ± 0.048179 | 0.075857 ± 0.005261 | Masked |
| Uniform (29000_uniform) | Approach | 10 | 0.124113 ± 0.040157 | 0.100778 ± 0.007200 | Masked |
| 101000 | Pick | 10 | 0.126322 ± 0.020287 | 0.114689 ± 0.002112 | 0.033246 ± 0.000002 |
| Uniform (29000_uniform) | Pick | 8 | 0.152161 ± 0.057807 | 0.152437 ± 0.029052 | 0.086933 ± 0.096870 |
| 101000 | Place | 9 | 0.070104 ± 0.020164 | 0.121933 ± 0.004185 | 0.250210 ± 0.028122 |
| Uniform (29000_uniform) | Place | 6 | 0.191177 ± 0.071007 | 0.148239 ± 0.009634 | 0.076697 ± 0.027023 |

## Success rates

| Policy | Approach | Lifted pickup goal | Place goal | Full task |
| --- | ---: | ---: | ---: | ---: |
| 101000 | 10/10 (100%) | 10/10 (100%) | 9/10 (90%) | 9/10 (90%) |
| Uniform | 10/10 (100%) | 6/10 (60%) | 6/10 (60%) | 2/10 (20%) |

Full task includes the final standing frame without falling and the production final object-position check. Lifted pickup is the `stand_after_pick_box` goal gate, not an independent lift-height test. All approaches succeeded, so no near-box recovery resets were needed.

101000's failure was a timeout at `stand_before_place_box`. Uniform passed six place endpoints, but four subsequently fell during `stand_after_place_box`. Thus its lower object-position error on passed place endpoints does not imply higher task success; the endpoint cohorts differ.

## Per-case outcomes

Both policies passed approach in every case. Outcomes below refer to full pick/place.

| Starting distance (m) | Placement displacement (m) | 101000 | Uniform |
| ---: | ---: | --- | --- |
| 0.5 | 0.2 | Pass | Crouching pickup timeout |
| 0.7 | 0.4 | Pass | Fall in final standing frame |
| 0.9 | 0.6 | Pre-place timeout | Fall during pickup-standing |
| 1.1 | 0.8 | Pass | Fall in final standing frame |
| 1.3 | 1.0 | Pass | Fall before pickup |
| 1.5 | 1.2 | Pass | Pass |
| 1.7 | 1.4 | Pass | Pickup-standing timeout |
| 1.9 | 1.6 | Pass | Pass |
| 2.1 | 1.8 | Pass | Fall in final standing frame |
| 2.3 | 2.0 | Pass | Fall in final standing frame |

## Protocol and scope

- Same ten paired distance cases as before: starting XY distance 0.5–2.3 m, placement displacement 0.2–2.0 m along world +X, both in 0.2 m increments. This is not a Cartesian grid or ten repetitions at each distance.
- Initial conditions exactly match the earlier sweep: seed 20260911, independent ±0.02 m XY and ±2° yaw perturbations, nominal box [0.35, 0, 0.15], root [0.35 - starting_distance, 0, 0.8]. No extra placement noise. Approach stopping offset stays 0.30 m.
- Only physical box dimensions were intentionally varied. Mass 0.6 kg, explicit body inertia [0.002, 0.002, 0.002], friction/contact settings and flat-hand robot dynamics stay unchanged. These inertias are the existing tuned values, not recalculated uniform-cube inertias. Hidden target marker dimensions also match the cube.
- Retargeting geometry stays at **0.30 m source / 0.35 m target** as in previous experiments. IK is disabled; generated standing lean is 5°. This experiment does not test changing retargeting dimensions alongside the physical cube.
- Manual production goal sequence: approach, stand-before-pick, crouch-to-pick, stand-after-pick, stand-before-place, crouch-to-place, stand-after-place. Failed approach would be recorded and followed by a near-box reset and one manipulation attempt; no retry after manipulation failure.
- Current production controller and retargeter, ver3 1480-input/29-action observations, MuJoCo 3.3.5, ONNX Runtime 1.23.2. Physics step 0.001667 s, ten steps per control tick, 2 s idle warmup, 30 s per-goal timeout. Headless in-process execution: no VLM query, ROS executor, or hardware commands.
- Unchanged box success thresholds: root/mean-body position ≤0.30 m, root orientation ≤0.80 rad, object action error ≤0.45 m, alternative approach/pre-pick XY reach ≤0.45 m; final task object error ≤0.60 m. Object orientation and joint RMSE are diagnostics, not success gates. Stationarity requires 0.5 s hold, robot speeds ≤0.10 m/s and 0.15 rad/s, object speeds ≤0.15 m/s and 0.30 rad/s, minimum action duration 1 s.
- Rates describe this small distance sweep. Sample standard deviations mix distance and reset variation. Passed-endpoint statistics have selection bias. Permissive final task tolerance does not imply exact placement or independently verified object release.

## Artifacts and verification

[summary.json](summary.json) contains full-precision statistics and separate full-success-only cohorts. [outcomes.csv](outcomes.csv) contains per-case outcomes; [all_phase_errors.csv](all_phase_errors.csv) includes every attempted endpoint, including failed ones. [manifest.json](manifest.json) records sampled poses, configuration, versions, and source/checkpoint hashes. `approach/` and `manipulation/` contain exact goal payloads and state traces.

`scene_snapshot.xml` and `robot_snapshot.xml` preserve the evaluation-only cube/flat-hand model; controller config points to this scene. The existing video replay defaults to the old rectangular-box video scene, so a replay must account for these saved cube dimensions rather than blindly using that default.

Verified 20 completed trials, paired initial conditions, exact agreement with previous sampled starts and thresholds, source/model hashes, and all reported mean/std values recomputed from raw passed endpoints. All 25 evaluation tests passed. Execution time was approximately 89 s. Sandbox DDS transport warnings did not affect the in-process data path.

## Reproduce

Use the environment setup in [the evaluation README](../../README.md), with isolated MuJoCo 3.3.5 first on PYTHONPATH, then run:

```bash
/home/sitongchen/miniconda3/envs/keyLM_ros310/bin/python -m evaluation.evaluate_box_distances \
  --approach-range 0.5 2.3 --physical-box-size 0.3 0.3 0.3 \
  --output evaluation/results/box_030m_101000_vs_uniform_repeat
```

Default checkpoints are `model_101000.onnx` and `model_29000_uniform.onnx`. Use a new output directory to preserve existing results.
