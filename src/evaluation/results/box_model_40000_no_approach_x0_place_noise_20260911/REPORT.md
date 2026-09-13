# Model 40000: no approach, root X = 0, noisy placement

## Result

**0/10 successes.** All ten trials fell at `stand_before_pick_box`, 1.50–1.57 simulation seconds after that goal was sent. The common two-second idle warmup preceded this goal. No trial reached `crouch_to_pick_box` or any placement goal.

Skipping approach and moving the nominal start to X = 0 did not prevent failure under this evaluation configuration. This does not identify the underlying cause; no policy/controller fix or training-compatibility claim follows from this test alone.

## Protocol

- Policy: `model_40000.onnx`, current ver3 1480-input/29-action deployment interface.
- Ten trials; seed 20260911.
- Nominal robot reset: `[0.0, 0.0, 0.8]`, identity WXYZ quaternion. Existing independent ±0.02 m XY and ±2° yaw perturbations retained.
- Nominal box reset: `[0.35, 0.0, 0.15]`, identity WXYZ quaternion, with the same reset perturbation bounds.
- `approach_box` omitted. Planned sequence: `stand_before_pick_box`, `crouch_to_pick_box`, `stand_after_pick_box`, `stand_before_place_box`, `crouch_to_place_box`, `stand_after_place_box`.
- Placement destination: starting observed box XY plus 1 m world +X, with independent uniform ±0.10 m per XY coordinate, sampled once per trial. Target height/orientation unchanged. Both the planner's retargeting requests and task-completion check refer to this same destination.
- Placement noise was configured and validated, but **its effect on placement performance was not tested**, because every trial failed before pickup.
- Current VLM numerical thresholds, stationary gate, 30 s goal timeout, fall guard, 2 s idle warmup, physics, PD control and retargeting settings unchanged from the earlier tests. No retries or recovery. Headless synchronous MuJoCo 3.3.5, not live ROS deployment.
- The changed initial position, omitted approach and placement noise make this a different protocol from the earlier four-policy comparison. Earlier results are not overwritten or pooled with this run.

## Error statistics

Successful-trial pick/place errors are **N/A**, because there were no successful trials. Approach is not part of this protocol.

The following are **failure-endpoint diagnostics only**, measured at `stand_before_pick_box` when each fall was detected, and are not successful pick/place tracking statistics. Values are mean ± sample standard deviation across ten trials.

| Metric | Mean ± sample SD |
| --- | ---: |
| root_position_m | 0.930746 ± 0.018867 |
| root_xy_m | 0.789462 ± 0.020946 |
| root_orientation_rad | 0.641681 ± 0.026089 |
| joint_mae_rad | 0.264347 ± 0.003097 |
| joint_rmse_rad | 0.342583 ± 0.002536 |
| joint_max_rad | 0.709480 ± 0.000679 |
| object_position_m | 0.038674 ± 0.000000 |
| object_orientation_rad | 0.000000 ± 0.000000 |
| mean_body_position_m | 0.741582 ± 0.018692 |

## Files and validation

`manifest.json` stores the initial conditions, offsets, configuration and source/model hashes. `trials.jsonl` stores actual task target poses and outcomes; `all_phase_errors.csv` stores the ten attempted pre-pick goal errors. `goals/` and `traces/` retain goal payloads and sampled simulation states. `successful_stage_errors.csv` has no data rows, correctly reflecting zero successes.

Validated ten trials, no approach phase, X/Y reset perturbation bounds, placement XY targets against the sampled offsets, unchanged target height, and source/model hashes. [Earlier four-policy comparison with mean and standard deviation](../box_model_40000_20260911_paired10/COMPARISON.md).

## Run command

After the ROS/Python environment setup described in the [original report](../box_policy_comparison_20260911_paired10/REPORT.md):

```bash
/usr/bin/python3 -m lm.evaluate_box_policies \
  --trials 10 --seed 20260911 --skip-approach \
  --initial-root-pos 0 0 0.8 --place-noise-xy-m 0.10 \
  --policies /home/sitongchen/keyLM_ros2_ws/src/crl-humanoid-ros/crl_g1_goalcontroller_py/crl_g1_goalcontroller_python/model/model_40000.onnx \
  --output /home/sitongchen/keyLM_ros2_ws/src/lm/evaluation_results/box_40000_no_approach_repeat
```

Use a new output directory for each run.
