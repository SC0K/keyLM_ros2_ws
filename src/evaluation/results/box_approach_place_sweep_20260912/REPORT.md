# Varied starting and placement distances — 2026-09-12

Completed twenty headless MuJoCo trials, ten per policy. Starting root-to-box nominal XY distances are 0.5–2.3 m in 0.2 m increments. Corresponding placement displacements are 0.2–2.0 m along world +X from the starting observed box position. These are paired distance cases, not a Cartesian grid. The approach goal's 0.30 m stopping offset remains unchanged.

## Success rates

| Method | Approach | Pickup with lift | Place goal | Pick-and-place (reset allowed) | End-to-end | Recovery resets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 9/10 | 5/10 | 5/10 | 4/10 | 3/10 | 1 |
| Saliency-based (ours) | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 0 |

Uniform timed out at approach for the 1.1 m starting-distance / 0.8 m placement-distance case. The robot was reset to nominal X=0 with the same XY/yaw perturbations, the initial box pose was restored, policy history and physics were reset, and manipulation was attempted once. That pick-and-place attempt succeeded. It counts toward 4/10 reset-allowed pick-and-place completions, but not toward 3/10 uninterrupted end-to-end completions. All other manipulation attempts continued directly from approach without resetting.

Pickup success requires the lifted `stand_after_pick_box` goal. Place-goal success is `crouch_to_place_box`. Full manipulation additionally requires `stand_after_place_box`, final task completion and no fall. Every success rate uses all ten distance cases as denominator.

## Successful-keyframe endpoint errors

Mean ± sample standard deviation. Include a keyframe whenever its own success and stationary checks passed, even if a later goal fails. Pick/place errors refer to the crouching goals, not lifted standing or final standing. Recovery-assisted endpoints are included. Root/object position is 3D Euclidean distance in metres; joint RMSE is across 29 joints in radians. Approach object goals are masked.

| Method | Stage | n | Root position (m) | Joint RMSE (rad) | Object position (m) |
| --- | --- | ---: | ---: | ---: | ---: |
| Uniform | approach | 9 | 0.125262 ± 0.043326 | 0.098917 ± 0.006068 | Masked |
| Saliency-based (ours) | approach | 10 | 0.113973 ± 0.048265 | 0.076387 ± 0.004981 | Masked |
| Uniform | pick | 9 | 0.146208 ± 0.045402 | 0.141529 ± 0.018078 | 0.176976 ± 0.078811 |
| Saliency-based (ours) | pick | 10 | 0.127503 ± 0.022257 | 0.114464 ± 0.002029 | 0.040745 ± 0.000001 |
| Uniform | place | 5 | 0.193580 ± 0.038395 | 0.143977 ± 0.009588 | 0.094331 ± 0.007196 |
| Saliency-based (ours) | place | 10 | 0.072547 ± 0.017885 | 0.121839 ± 0.002207 | 0.230435 ± 0.021429 |

Uniform has 9 passed crouching-pick endpoints, but only 5 completed lifted pickups. This explains the different error sample count and pickup success count. Its five passed place goals include a trial that fell afterward; they are not five successful full tasks. Quaternions/orientation errors are recorded in JSON/CSV but object orientation is not used for success.

## Per-case results

| Start distance (m) | Placement displacement (m) | Selected end-to-end | Uniform approach | Uniform manipulation | Uniform reset | Uniform manipulation outcome |
| ---: | ---: | --- | --- | --- | --- | --- |
| 0.5 | 0.2 | Pass | Pass | Fail | No | stand_after_pick_box: timeout |
| 0.7 | 0.4 | Pass | Pass | Fail | No | stand_after_pick_box: fall |
| 0.9 | 0.6 | Pass | Pass | Fail | No | stand_after_pick_box: fall |
| 1.1 | 0.8 | Pass | Fail | Pass | Yes | success |
| 1.3 | 1.0 | Pass | Pass | Fail | No | stand_before_pick_box: fall |
| 1.5 | 1.2 | Pass | Pass | Pass | No | success |
| 1.7 | 1.4 | Pass | Pass | Fail | No | stand_after_pick_box: fall |
| 1.9 | 1.6 | Pass | Pass | Pass | No | success |
| 2.1 | 1.8 | Pass | Pass | Pass | No | success |
| 2.3 | 2.0 | Pass | Pass | Fail | No | stand_after_place_box: fall |

## Configuration and limits

- Seed 20260911. Nominal box `[0.35, 0, 0.15]`. Nominal root `[0.35 - starting_distance, 0, 0.8]`. Nominal quaternions identity WXYZ. The original independent ±0.02 m per XY axis and ±2° yaw perturbations remain; heights unchanged. Actual initial XY distances are recorded and differ slightly from nominal distances. Both policies receive identical initial conditions per case.
- No extra placement-position noise. Placement destination is the observed starting box XY plus the exact requested +X displacement. Existing target height/orientation handling unchanged. Failed-approach recovery resets nominal root X to 0 while retaining perturbations and the corresponding placement displacement.
- Current VLM thresholds unchanged: 0.30 m root/mean-body position error, 0.80 rad root orientation, 0.45 m object action error, alternative 0.45 m approach/pre-pick XY reach; final task object position tolerance 0.60 m. Object orientation is ignored for success. Stationarity: 0.5 s hold with robot speeds ≤0.10 m/s and 0.15 rad/s, object speeds ≤0.15 m/s and 0.30 rad/s. Minimum goal duration 1 s; timeout 30 s.
- Fall guard: pelvis Z<0.30 m, root-up world Z<0.25, or nonfinite state. Two-second idle warmup after each reset. No retry after a manipulation failure.
- MuJoCo 3.3.5, synchronous headless physics at 0.001667 s with ten steps per control tick. Production ver3 observations, controller PD/torque limits and retargeter; IK off, standing lean 5°. Source cube 0.30 m, retargeting target cube 0.35 m, physical box 0.345 × 0.250 × 0.285 m. No live ROS timing, VLM queries or real robot commands.
- One trial per distance pair: rates summarize this small sweep, not reliability at each distance. Standard deviations mix distance and reset variation. Starting and placement distances vary together, so this is not an experiment that isolates their individual effects. Success-only samples have selection bias and may include recovered trials. The generous final 0.60 m tolerance is particularly permissive for short placement requests; success does not imply precise placement or image-verified release.

## Files and verification

`summary.json` contains `successful_keyframe_errors` (the requested endpoint cohort), plus explicitly separate legacy full-manipulation statistics. `error_statistics.csv` and `tracking_errors.tex` use only the requested successful-keyframe cohort. `success_rates.tex`, `outcomes.csv`, and `all_phase_errors.csv` provide paper tables and detailed outcomes. `approach/` and `manipulation/` contain separate replayable traces/goals/trial logs. `manifest.json` stores full parameters, sampled poses and source/model SHA-256 hashes.

Validated 20 completed trials, identical paired initial conditions, all requested starting/placement distances, target displacement, recovery reset preservation, independent success accounting, source/model hashes, and every mean/std recomputed from raw passed endpoints. Eighteen evaluator tests passed, and the recovery path was exercised once in physics. Previous result directories are unchanged.

## Reproduce

Using the ROS Humble workspace/Python environment and isolated MuJoCo 3.3.5 installation described in the earlier reports:

```bash
/usr/bin/python3 -m lm.evaluate_box_distances --approach-range 0.5 2.3 \
  --output /home/sitongchen/keyLM_ros2_ws/src/lm/evaluation_results/box_approach_place_sweep_repeat
```

Use a new output directory to preserve previous results.
