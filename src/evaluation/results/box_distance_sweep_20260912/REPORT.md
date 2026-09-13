# Box placement-distance sweep — 2026-09-12

Twenty headless MuJoCo trials: Selected (`model_101000`) and Uniform (`model_29000_uniform`), one paired trial per distance from 0.2 to 2.0 m. Placement distance is relative to the starting observed box center along world +X, not an absolute world coordinate.

## Success rates

| Policy | Approach | Pick | Place keyframe | Full pick-and-place | End-to-end | Recovery resets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model_101000.onnx | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 0 |
| model_29000_uniform.onnx | 10/10 | 3/10 | 3/10 | 0/10 | 0/10 | 0 |

Pick success means passing through `stand_after_pick_box`. Place-keyframe success means passing `crouch_to_place_box`. Full pick-and-place additionally requires `stand_after_place_box`, final task completion and no fall. End-to-end requires successful approach too. Every percentage uses all ten trials as its denominator, not only trials that reached the stage.

Both policies passed every approach. The requested failure fallback was available but was never triggered in this sweep. Uniform passed pickup and the place keyframe at 0.2, 1.8 and 2.0 m, then fell during the final standing frame in all three cases; these are full-task failures. Its other seven trials failed before placement, so placement at those seven distances was not exercised.

## Per-distance full-task outcomes

| Distance (m) | Selected | Uniform | Uniform failure |
| ---: | --- | --- | --- |
| 0.2 | Pass | Fail | stand_after_place_box: fall |
| 0.4 | Pass | Fail | stand_before_pick_box: fall |
| 0.6 | Pass | Fail | stand_before_pick_box: fall |
| 0.8 | Pass | Fail | stand_before_pick_box: fall |
| 1.0 | Pass | Fail | stand_before_pick_box: fall |
| 1.2 | Pass | Fail | stand_before_pick_box: fall |
| 1.4 | Pass | Fail | stand_after_pick_box: timeout |
| 1.6 | Pass | Fail | stand_after_pick_box: fall |
| 1.8 | Pass | Fail | stand_after_place_box: fall |
| 2.0 | Pass | Fail | stand_after_place_box: fall |

## Errors: mean ± sample standard deviation

Approach errors use all successful approaches (n=10 per policy). Pick/place errors use fully successful manipulation sequences only (Selected n=10; Uniform n=0). This differs from the earlier report's approach averages, which were restricted to fully successful entire trials. Pick/place measurements remain at `crouch_to_pick_box` and `crouch_to_place_box`, not at the final standing frame. Root/object position is 3D distance in metres; joint RMSE is over all 29 joints in radians.

| Policy | Stage | n | Root position (m) | Joint RMSE (rad) | Object position (m) |
| --- | --- | ---: | ---: | ---: | ---: |
| model_101000 | approach | 10 | 0.099865 ± 0.036285 | 0.077719 ± 0.005028 | Masked |
| model_101000 | pick | 10 | 0.111944 ± 0.017151 | 0.116088 ± 0.002840 | 0.040745 ± 0.000001 |
| model_101000 | place | 10 | 0.084630 ± 0.044145 | 0.122246 ± 0.003907 | 0.220127 ± 0.020265 |
| model_29000_uniform | approach | 10 | 0.057707 ± 0.037713 | 0.109936 ± 0.007426 | Masked |
| model_29000_uniform | pick | 0 | N/A | N/A | N/A |
| model_29000_uniform | place | 0 | N/A | N/A | N/A |

For diagnostic context only, Uniform's three passed place keyframes had the following errors even though every trial subsequently failed:

| Cohort | n | Root position (m) | Joint RMSE (rad) | Object position (m) |
| --- | ---: | ---: | ---: | ---: |
| Uniform: passed place goal, later fell | 3 | 0.210398 ± 0.022871 | 0.154653 ± 0.004265 | 0.079269 ± 0.005402 |

These diagnostic values must not be reported as successful-full-task errors. All ten successful Selected trials ended with box–floor contact and no detected box–robot contact.

## Protocol and interpretation

- Robot nominal reset `[-2, 0, 0.8]`, box `[0.35, 0, 0.15]`, nominal identity WXYZ quaternions. Independently sampled ±0.02 m XY and ±2° yaw perturbations; unchanged heights. Seed 20260911. Corresponding distance trials use identical perturbations across policies.
- Ten exact placement offsets: 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0 m. Additional placement noise is disabled. Existing target height/orientation handling is unchanged.
- Approach is scored first. If successful, manipulation continues from the resulting simulation state without a reset or extra warmup. On failure, approach remains failed and manipulation is attempted once from nominal robot X=0 with the same perturbations, restored initial box pose and reset policy history/physics. Such assisted success is never counted as end-to-end success. No manipulation-stage retry is performed.
- Same production VLM numerical gates as before: 0.5 s stationary hold, minimum action duration 1 s, 0.30 m root/mean-body error, 0.80 rad root orientation error, 0.45 m object action error and alternative approach/pre-pick XY reach, 0.60 m final task object-position error. Stationarity thresholds: robot 0.10 m/s and 0.15 rad/s, object 0.15 m/s and 0.30 rad/s. Object orientation does not determine success.
- Phase timeout 30 simulation seconds. Fall guard: pelvis Z below 0.30 m or root-up world Z below 0.25 (or nonfinite state). Common initial idle warmup 2 s.
- Synchronous headless MuJoCo 3.3.5, timestep 0.001667 s, 10 steps/control tick, production ver3 observations, ONNX policies, PD gains/torque limits and retargeting. No VLM queries or live robot commands. IK remains off and generated standing lean is 5 degrees.
- Existing retargeting target cube is 0.35 m; source cube 0.30 m; physical box remains 0.345 × 0.250 × 0.285 m. No production configuration was changed for the sweep.
- **One trial per distance is not an estimate of repeatability at that distance.** Standard deviations combine distance variation and initial-pose variation. Pre-placement failures cannot be attributed to the requested placement distance. The generous 0.60 m task tolerance is especially permissive for short moves; these are numerical policy/VLM successes, not proof of exact placement or image-verified release.

## Artifacts and checks

- `outcomes.csv`: per-distance approach/pick/full-task outcomes and reset flag.
- `error_statistics.csv`: mean/std/n for all recorded metrics, with explicit successful-approach versus successful-pick-and-place cohorts.
- `all_phase_errors.csv`: every attempted keyframe endpoint, including failed trials.
- `trials.jsonl`, `summary.json`: complete trial details and aggregate statistics.
- `manifest.json`: exact initial poses, distances, thresholds, geometry and source/checkpoint hashes.
- `approach/` and `manipulation/`: separate traces, goal payloads, controller configuration and trial logs; compatible with the existing replay script.

Verified 20 completed trials; both policies cover all ten distances; paired initial conditions; actual target XY equals observed starting box XY plus the requested displacement; no recovery reset triggered; consistent task targets across approach/manipulation; source/model hashes unchanged during execution. Twelve evaluator tests passed, including mocked success-continuation and failed-approach reset paths. The fallback branch was unit-tested, not exercised by these twenty physics trials.

## Reproduce

```bash
cd /home/sitongchen/keyLM_ros2_ws/src
source /opt/ros/humble/setup.bash
source /home/sitongchen/keyLM_ros2_ws/install/setup.bash
export PYTHONPATH=/tmp/keyframe-eval-deps.YnSpFF:/home/sitongchen/keyLM_ros2_ws/src/lm:/home/sitongchen/keyLM_ros2_ws/src/crl-humanoid-ros/crl_g1_goalcontroller_py/crl_g1_goalcontroller_python:$PYTHONPATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
/usr/bin/python3 -m lm.evaluate_box_distances \
  --output /home/sitongchen/keyLM_ros2_ws/src/lm/evaluation_results/box_distance_sweep_repeat
```

Use a new output directory. The isolated `/tmp/keyframe-eval-deps.YnSpFF` contains MuJoCo 3.3.5; recreate an isolated installation and adjust PYTHONPATH if it has been cleared. Earlier result directories remain unchanged.
