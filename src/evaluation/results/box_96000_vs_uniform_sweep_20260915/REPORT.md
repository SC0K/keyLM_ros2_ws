# Box goal-error comparison: 96000_bucket vs Uniform

Completed 20 fresh headless trials on 2026-09-15: ten per checkpoint. Despite the checkpoint name, both policies were evaluated on the **box** task, as requested. This report does not measure bucket performance.

Starting distances are 0.5–2.3 m and corresponding placement displacements are 0.2–2.0 m, both in 0.2 m increments. These are ten paired cases, not a Cartesian grid. Initial conditions exactly match the previous 2026-09-12 approach/place sweep: seed 20260911, independent ±0.02 m XY and ±2° yaw perturbations, nominal box [0.35, 0, 0.15], root [0.35 - starting_distance, 0, 0.8]. No additional placement noise. Both checkpoints receive identical conditions per case.

## Goal errors

Mean ± sample standard deviation across **passed keyframe endpoints**, including endpoints in trials that subsequently failed. Pick means `crouch_to_pick_box`, not the lifted `stand_after_pick_box`; place means `crouch_to_place_box`, not the final standing frame. Errors are measured at each endpoint, not averaged over its trajectory. Root/object position errors are 3D Euclidean distances; joint RMSE includes all 29 policy joints. Approach masks object observations/goals.

| Policy | Stage | n | Root position (m) | Joint RMSE (rad) | Object position (m) |
| --- | --- | ---: | ---: | ---: | ---: |
| 96000_bucket | Approach | 10 | 0.076920 ± 0.045552 | 0.110601 ± 0.008696 | Masked |
| Uniform (29000_uniform) | Approach | 10 | 0.124112 ± 0.040156 | 0.100777 ± 0.007199 | Masked |
| 96000_bucket | Pick | 10 | 0.095501 ± 0.034288 | 0.136591 ± 0.020430 | 0.094942 ± 0.087831 |
| Uniform (29000_uniform) | Pick | 9 | 0.136813 ± 0.040578 | 0.137524 ± 0.021321 | 0.138655 ± 0.094033 |
| 96000_bucket | Place | 8 | 0.039364 ± 0.017980 | 0.117309 ± 0.002543 | 0.113254 ± 0.045835 |
| Uniform (29000_uniform) | Place | 6 | 0.196697 ± 0.051707 | 0.147675 ± 0.012981 | 0.091394 ± 0.018748 |

## Success rates

| Policy | Approach | Lifted pickup goal | Place goal | Full pick/place | End-to-end |
| --- | ---: | ---: | ---: | ---: | ---: |
| 96000_bucket | 10/10 (100%) | 8/10 (80%) | 8/10 (80%) | 8/10 (80%) | 8/10 (80%) |
| Uniform | 10/10 (100%) | 6/10 (60%) | 6/10 (60%) | 3/10 (30%) | 3/10 (30%) |

The fallback protocol resets near the box after failed approach, preserving reset perturbations and restoring the initial box and policy history. No recovery resets were needed in this run. Lifted pickup is scored by the production `stand_after_pick_box` goal gate, not an independent lift-height criterion.

## Per-case outcomes

Both policies passed every approach. Listed outcomes are full pick/place outcomes.

| Starting distance (m) | Placement displacement (m) | 96000_bucket | Uniform |
| ---: | ---: | --- | --- |
| 0.5 | 0.2 | Pass | Pickup-standing timeout |
| 0.7 | 0.4 | Pickup-standing timeout | Pickup-standing timeout |
| 0.9 | 0.6 | Pass | Fall in final standing frame |
| 1.1 | 0.8 | Pass | Fall in final standing frame |
| 1.3 | 1.0 | Pass | Fall before pickup |
| 1.5 | 1.2 | Pickup-standing timeout | Pass |
| 1.7 | 1.4 | Pass | Fall during pickup-standing |
| 1.9 | 1.6 | Pass | Pass |
| 2.1 | 1.8 | Pass | Fall in final standing frame |
| 2.3 | 2.0 | Pass | Pass |

96000_bucket has lower mean root error at all stages, lower place joint RMSE, and higher full-task success in this sweep. Uniform has lower approach joint RMSE and lower object error among its passed place endpoints, but three of those six trials fall during final standing. The endpoint cohorts differ; their means are not matched-trial estimates of policy superiority.

## Configuration and interpretation

- Current production controller, planner goal generation, and retargeter; manual phase order with no actual VLM queries, ROS executor, hardware, or control-topic publishing. MuJoCo 3.3.5 and ONNX Runtime 1.23.2; synchronous 0.001667 s physics, ten physics steps per policy tick; ver3 1480-input/29-action checkpoints.
- Box thresholds remain unchanged: root/mean-body position ≤0.30 m, root orientation ≤0.80 rad, object action error ≤0.45 m, alternative approach/pre-pick XY reach ≤0.45 m; final object task error ≤0.60 m. The bucket's new 0.60 m reach threshold does not apply to this box task.
- Stationarity: 0.5 s hold; robot linear/angular speeds ≤0.10 m/s and 0.15 rad/s, object speeds ≤0.15 m/s and 0.30 rad/s. Minimum action duration 1 s, timeout 30 s, idle warmup 2 s. Fall guard checks root height, root-up direction, and finite state.
- IK disabled; 5° generated standing lean; source 0.30 m cube and retargeting target 0.35 m cube, physical box unchanged. Joint/root/object errors are computed against the goal actually sent to the policy.
- Uniform was rerun on the current pipeline; these numbers are not copied from the earlier report. No live pipeline files were changed for this experiment. The offline distance runner gained `--policies` selection.
- One trial per distance pair. Standard deviations mix distance and reset variation. Starting and placement distances vary together. Passed-endpoint statistics have selection bias. These production success gates use permissive tolerances and do not independently verify visual release or exact placement.

## Artifacts and validation

- [summary.json](summary.json): full precision mean/std and success counts, with separate full-success-only statistics.
- [all_phase_errors.csv](all_phase_errors.csv): errors for all attempted phases, including failures.
- [outcomes.csv](outcomes.csv): per-case rates and resets.
- [manifest.json](manifest.json): initial conditions, configuration, versions, and source/model SHA-256 hashes.
- `approach/` and `manipulation/`: saved exact goals, traces, and trial logs for replay.

Validated all 20 trials, paired initial conditions, source/model hashes, and mean/std values recomputed from raw passed endpoints. All 20 evaluation tests passed. Run wall time was approximately 105 s. ROS emitted sandbox transport warnings, but the in-process evaluation completed; no DDS transport is used for its data path.

## Reproduce

Use the environment setup in [the evaluation README](../../README.md), with the Python environment providing ONNX Runtime and isolated MuJoCo 3.3.5 first on PYTHONPATH:

```bash
/home/sitongchen/miniconda3/envs/keyLM_ros310/bin/python -m evaluation.evaluate_box_distances \
  --approach-range 0.5 2.3 \
  --policies \
  crl-humanoid-ros/crl_g1_goalcontroller_py/crl_g1_goalcontroller_python/model/model_96000_bucket.onnx \
  crl-humanoid-ros/crl_g1_goalcontroller_py/crl_g1_goalcontroller_python/model/model_29000_uniform.onnx \
  --output evaluation/results/box_96000_vs_uniform_repeat
```

Use a new output directory; existing results are never overwritten.
