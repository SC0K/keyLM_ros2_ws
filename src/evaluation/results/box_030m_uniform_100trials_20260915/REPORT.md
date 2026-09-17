# Uniform vs 101000: matched 100-trial cube comparison

Completed 100 trials of `model_29000_uniform.onnx` on 2026-09-15. These match the 100 initial conditions, shared source hashes, thresholds and retargeting settings of the [101000 experiment](../box_030m_101000_100trials_20260915/REPORT.md). Both use a 0.30 m physical cube and flat hands. No live pipeline files changed.

## Success rates

| Policy | Approach | Lifted pickup goal | Place goal | Pick/place with recovery | Uninterrupted full task |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 97/100 | 63/100 | 63/100 | 47/100 | 44/100 |
| 101000 | 99/100 | 95/100 | 93/100 | 93/100 | 92/100 |

Uniform's approach timeouts were trials 33, 83 and 93. All three completed manipulation after the prescribed near-box reset, counting toward 47 reset-assisted successes but not 44 uninterrupted successes. Of 63 passed place goals, 16 fell during final standing. Pick success is the production `stand_after_pick_box` gate, not an independent lift-height test.

Uniform manipulation failures: 22 timeouts at `stand_after_pick_box`, 16 falls at `stand_after_place_box`, 9 falls at `stand_before_pick_box`, 3 timeouts at `crouch_to_pick_box`, and 3 falls at `stand_after_pick_box`.

## Goal errors

Mean ± sample standard deviation across passed keyframe endpoints, including later-failed and reset-assisted trials. Pick is `crouch_to_pick_box`, place is `crouch_to_place_box`; errors are at endpoint completion, not averaged over motion. Root/object errors are 3D position norms; joint RMSE covers 29 policy joints.

| Policy | Stage | n | Root (m) | Joint RMSE (rad) | Object (m) |
| --- | --- | ---: | ---: | ---: | ---: |
| Uniform | Approach | 97 | 0.099577 ± 0.050245 | 0.099486 ± 0.006903 | Masked |
| 101000 | Approach | 99 | 0.112888 ± 0.046756 | 0.074951 ± 0.004421 | Masked |
| Uniform | Pick | 88 | 0.149263 ± 0.043160 | 0.149670 ± 0.019972 | 0.170206 ± 0.093693 |
| 101000 | Pick | 100 | 0.118875 ± 0.034188 | 0.116199 ± 0.007154 | 0.042868 ± 0.035279 |
| Uniform | Place | 63 | 0.162375 ± 0.047503 | 0.139365 ± 0.009029 | 0.081667 ± 0.025951 |
| 101000 | Place | 93 | 0.066812 ± 0.018286 | 0.121097 ± 0.003881 | 0.226570 ± 0.043239 |

Uniform has lower approach root error and lower passed-place object error, but substantially lower uninterrupted task success. Passed-endpoint cohorts differ between policies; the error means are not paired estimates over identical successful trials. No statistical significance claim is made.

## Uniform by distance pair

Ten reset samples per row; starting and placement distances vary together.

| Starting XY (m) | Placement (m) | Approach | Lifted pickup | Place goal | Pick/place with recovery | Uninterrupted task |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.2 | 10/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| 0.7 | 0.4 | 10/10 | 2/10 | 2/10 | 1/10 | 1/10 |
| 0.9 | 0.6 | 7/10 | 6/10 | 6/10 | 6/10 | 3/10 |
| 1.1 | 0.8 | 10/10 | 9/10 | 9/10 | 6/10 | 6/10 |
| 1.3 | 1.0 | 10/10 | 7/10 | 7/10 | 3/10 | 3/10 |
| 1.5 | 1.2 | 10/10 | 10/10 | 10/10 | 9/10 | 9/10 |
| 1.7 | 1.4 | 10/10 | 9/10 | 9/10 | 7/10 | 7/10 |
| 1.9 | 1.6 | 10/10 | 9/10 | 9/10 | 9/10 | 9/10 |
| 2.1 | 1.8 | 10/10 | 10/10 | 10/10 | 6/10 | 6/10 |
| 2.3 | 2.0 | 10/10 | 1/10 | 1/10 | 0/10 | 0/10 |

## Protocol, verification, and artifacts

Same protocol as the 101000 report: seed 20260911, ten repetitions of ten distance pairs, independent ±0.02 m XY and ±2° yaw reset perturbations, no additional placement noise. The first ten samples rerun the earlier Uniform experiment exactly; this is 100 total trials, not 100 additional trials. MuJoCo 3.3.5, ONNX Runtime 1.23.2, ver3 1480-input/29-action policy, 0.001667 s physics step, ten steps per control tick, 2 s idle warmup, 30 s goal timeout. Manual production goal sequence; no VLM query, ROS executor, or hardware commands.

Retargeting source/target cubes remain 0.30/0.35 m; IK off, generated standing lean 5°. Physical cube mass, explicit inertia, friction/contact settings and flat-hand robot remain unchanged. Box production thresholds remain root/mean-body error 0.30 m, root orientation 0.80 rad, object action error 0.45 m, alternative approach/pre-pick XY reach 0.45 m, final object tolerance 0.60 m. Stationarity hold is 0.5 s, minimum action duration 1 s. These permissive gates do not independently prove precise placement or release. Error sample standard deviations mix distance and reset variation and are success-conditioned.

Validated 100 unique trial IDs, ten per distance, matched initial conditions and configuration, unchanged source/checkpoint hashes, exact reproduction of earlier first-ten phase results, and all Uniform mean/std values recomputed from raw passed endpoints. No evaluator code changes were needed for this run. Runtime was approximately 396 s. Sandbox DDS warnings did not affect the in-process data path.

- [Updated comparison LaTeX](comparison_tables.tex): both requested tables, 100 trials per policy.
- [summary.json](summary.json): full-precision Uniform statistics, including separate full-success-only cohorts.
- [outcomes.csv](outcomes.csv) and [all_phase_errors.csv](all_phase_errors.csv): all outcomes and attempted endpoint errors.
- [manifest.json](manifest.json): configuration, sampled resets, versions and source/checkpoint hashes.
- `approach/` and `manipulation/`: exact goal payloads and recorded states.
- `scene_snapshot.xml` and `robot_snapshot.xml`: evaluation-only physical cube and flat-hand model. The existing video replay defaults to a rectangular-box scene and needs to account for these saved cube dimensions.

## Reproduce

Use the [evaluation environment setup](../../README.md), then run from the workspace source root with a new output directory:

```bash
/home/sitongchen/miniconda3/envs/keyLM_ros310/bin/python -m evaluation.evaluate_box_distances \
  --approach-range 0.5 2.3 --physical-box-size 0.3 0.3 0.3 --repetitions 10 \
  --policies crl-humanoid-ros/crl_g1_goalcontroller_py/crl_g1_goalcontroller_python/model/model_29000_uniform.onnx \
  --output evaluation/results/box_030m_uniform_100trials_repeat
```
