# Four-policy box comparison — 2026-09-11

Completed ten new trials for `model_40000.onnx` and combined them with the 30 previous trials. Previous results are reused unchanged, not rerun. All four policies use the same ten paired initial perturbations (±0.02 m robot/box XY, ±2° yaw; seed 20260911), nominal robot position [-2, 0, 0.8], nominal box position [0.35, 0, 0.15], and a placement destination 1 m forward along world +X.

This is a headless, synchronous MuJoCo benchmark using production observation, retargeting and VLM numerical success methods, not a live ROS or camera/VLM decision-quality experiment. Source hashes, thresholds, timing, geometry and initial conditions match between the two experiment manifests. Model 40000 accepted the same ver3 1480-input/29-action interface; this alone does not verify its original training observation semantics.

## Success rates

| Policy | Success | Rate | Failure breakdown |
| --- | ---: | ---: | --- |
| `model_101000.onnx` | 10/10 | 100% | None |
| `model_29000_uniform.onnx` | 3/10 | 30% | 5 × stand_before_pick_box: fall; 1 × stand_after_pick_box: timeout; 1 × stand_after_pick_box: fall |
| `model_28000.onnx` | 0/10 | 0% | 10 × approach_box: timeout |
| `model_40000.onnx` | 0/10 | 0% | 10 × approach_box: fall |

## Successful-trial errors: mean ± sample standard deviation

Only fully successful trials contribute. Standard deviation is across trials, with denominator n−1; joint RMSE is first computed over the 29 joints at each endpoint. Root/object position errors are 3D Euclidean distances. Approach object goals are masked, so their errors are N/A. Policies with no successes have no successful-trial error statistics, not zero errors.

| Policy | Stage | n | Root position (m) | Joint RMSE (rad) | Object position (m) |
| --- | --- | ---: | ---: | ---: | ---: |
| `model_101000` | approach | 10 | 0.099865 ± 0.036285 | 0.077719 ± 0.005028 | N/A |
| `model_101000` | pick | 10 | 0.111944 ± 0.017151 | 0.116088 ± 0.002840 | 0.040745 ± 0.000001 |
| `model_101000` | place | 10 | 0.068998 ± 0.010721 | 0.118698 ± 0.002320 | 0.207267 ± 0.015195 |
| `model_29000_uniform` | approach | 3 | 0.098444 ± 0.050970 | 0.102449 ± 0.003426 | N/A |
| `model_29000_uniform` | pick | 3 | 0.105732 ± 0.023715 | 0.135979 ± 0.003586 | 0.041310 ± 0.000794 |
| `model_29000_uniform` | place | 3 | 0.185507 ± 0.010067 | 0.135732 ± 0.001156 | 0.100897 ± 0.005653 |
| `model_28000` | approach | 0 | N/A | N/A | N/A |
| `model_28000` | pick | 0 | N/A | N/A | N/A |
| `model_28000` | place | 0 | N/A | N/A | N/A |
| `model_40000` | approach | 0 | N/A | N/A | N/A |
| `model_40000` | pick | 0 | N/A | N/A | N/A |
| `model_40000` | place | 0 | N/A | N/A | N/A |

Stage endpoints are `approach_box`, `crouch_to_pick_box`, and `crouch_to_place_box`. Pick therefore means the crouching grasp pose, not the later lifted standing pose. All attempted intermediate standing goals retain their own errors in each run's `all_phase_errors.csv`.

## Root and object orientation errors

Mean ± sample standard deviation, radians; these use the same successful-trial subset.

| Policy | Stage | n | Root orientation (rad) | Object orientation (rad) |
| --- | --- | ---: | ---: | ---: |
| `model_101000` | approach | 10 | 0.060136 ± 0.008554 | N/A |
| `model_101000` | pick | 10 | 0.048323 ± 0.015202 | 0.000026 ± 0.000031 |
| `model_101000` | place | 10 | 0.051427 ± 0.002003 | 0.116943 ± 0.033953 |
| `model_29000_uniform` | approach | 3 | 0.058217 ± 0.026716 | N/A |
| `model_29000_uniform` | pick | 3 | 0.045967 ± 0.005184 | 0.001140 ± 0.000847 |
| `model_29000_uniform` | place | 3 | 0.055180 ± 0.008309 | 0.172705 ± 0.009438 |
| `model_28000` | approach | 0 | N/A | N/A |
| `model_28000` | pick | 0 | N/A | N/A |
| `model_28000` | place | 0 | N/A | N/A |
| `model_40000` | approach | 0 | N/A | N/A |
| `model_40000` | pick | 0 | N/A | N/A |
| `model_40000` | place | 0 | N/A | N/A |

## Interpretation and limits

`model_40000` fell during approach in all ten trials, 1.40–1.72 simulation seconds after that goal was sent (after the common two-second idle warmup). It never reached pick or place. This run establishes failure under the current deployment configuration, not its underlying cause.

For debugging only, its approach errors at failure were:

- root_position_m: 2.942456 ± 0.040474
- root_orientation_rad: 0.906150 ± 0.186833
- joint_rmse_rad: 0.325059 ± 0.027191

These failure-endpoint measurements are deliberately excluded from the successful-trial comparison above. Earlier `model_28000` trials all timed out during approach instead of falling.

`model_101000` remains the most reliable in this small cohort. Successful-only averages compare different surviving subsets (10 for 101000 versus 3 for 29000_uniform), so apparent accuracy differences are subject to selection bias. All 13 successful trials in the earlier run ended with box–floor contact and no detected box–robot contact.

Current VLM thresholds are unchanged, including the 0.60 m final task object-position tolerance, 0.45 m per-keyframe object-position tolerance, 0.45 m alternative approach/pre-pick XY reach check and 0.5 s stationary hold. A phase has a 30 s timeout, and any detected fall fails the trial. There are no retries. Numerical success does not prove precise placement or grasp/release. IK stays off. Existing 0.35 m retargeting target cube versus physical 0.345 × 0.250 × 0.285 m box configuration is preserved.

## Artifacts

- `combined_summary.json`: all four policies, all recorded error metrics, mean/std/n and failure lists.
- `combined_error_statistics.csv`: long-form policy/stage/metric/n/mean/std across all four policies, including null results.
- `manifest.json`, `trials.jsonl`, `all_phase_errors.csv`, `goals/`, `traces/`: new model 40000 run only.
- [Previous report and complete protocol](../box_policy_comparison_20260911_paired10/REPORT.md).
- [Previous per-trial successful errors](../box_policy_comparison_20260911_paired10/successful_stage_errors.csv).

The model 40000 run used the unchanged evaluator with `--trials 10 --seed 20260911 --policies /home/sitongchen/keyLM_ros2_ws/src/crl-humanoid-ros/crl_g1_goalcontroller_py/crl_g1_goalcontroller_python/model/model_40000.onnx`. See the previous report for the ROS/Python environment setup. The output directory must be new to prevent overwriting results.
