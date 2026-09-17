# 101000 with a 0.30 m cube — 100 trials

Completed 100 trials of `model_101000.onnx` on 2026-09-15. The requested `10100` was interpreted as the previously evaluated `101000` checkpoint; there is no `model_10100.onnx` in the model directory. Uniform was not rerun in this experiment.

## Success rates

| Metric | Successes | Rate |
| --- | ---: | ---: |
| Approach | 99/100 | 99% |
| Lifted pickup goal | 95/100 | 95% |
| Place goal | 93/100 | 93% |
| Full pick/place, approach reset allowed | 93/100 | 93% |
| Uninterrupted full task | 92/100 | 92% |

Trial 42 timed out at approach, then completed manipulation after the prescribed near-box reset. It counts as reset-assisted pick/place success, not uninterrupted task success. Five trials timed out at `stand_after_pick_box` (22, 32, 62, 71, 72), and two at `stand_before_place_box` (3, 61). These are the only manipulation failures. Lifted pickup means passing the production `stand_after_pick_box` goal gate, not an independently measured minimum lift height.

## Goal errors

Mean ± sample standard deviation over **passed keyframe endpoints**, including endpoints from trials that later failed or used an approach-reset recovery. Pick means `crouch_to_pick_box`, not lifted pickup; place means `crouch_to_place_box`, not final standing. Root/object position errors are 3D Euclidean distances; joint RMSE includes all 29 policy joints. Errors are measured at endpoint completion, not averaged over motion duration.

| Stage | n | Root position (m) | Joint RMSE (rad) | Object position (m) |
| --- | ---: | ---: | ---: | ---: |
| Approach | 99 | 0.112888 ± 0.046756 | 0.074951 ± 0.004421 | Masked |
| Pick | 100 | 0.118875 ± 0.034188 | 0.116199 ± 0.007154 | 0.042868 ± 0.035279 |
| Place | 93 | 0.066812 ± 0.018286 | 0.121097 ± 0.003881 | 0.226570 ± 0.043239 |

All 100 crouching pickup endpoints passed, but only 95 lifted pickup goals passed. The different counts are intentional. `summary.json` also retains separate full-manipulation-success-only statistics; these are not mixed into the table above.

## Results by distance pair

Ten independently perturbed resets per row. Starting and placement distances vary together, so their separate effects cannot be inferred from this experiment.

| Starting XY distance (m) | Placement displacement (m) | Approach | Lifted pickup | Place goal | Pick/place with recovery | Uninterrupted task |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.2 | 10/10 | 9/10 | 8/10 | 8/10 | 8/10 |
| 0.7 | 0.4 | 9/10 | 6/10 | 6/10 | 6/10 | 5/10 |
| 0.9 | 0.6 | 10/10 | 10/10 | 9/10 | 9/10 | 9/10 |
| 1.1 | 0.8 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| 1.3 | 1.0 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| 1.5 | 1.2 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| 1.7 | 1.4 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| 1.9 | 1.6 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| 2.1 | 1.8 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| 2.3 | 2.0 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |

## Unchanged protocol

- Ten repetitions of the previous ten distance pairs, cycling through the pairs in ascending order. Seed 20260911 generates 100 distinct reset samples. Trials 1–10 rerun the previous ten samples and reproduce their phase outcomes and errors exactly; trials 11–100 use new perturbations. This is 100 total trials, not 100 additional trials beyond the earlier ten.
- Physical box 0.30 m cube, **flat hands**, mass 0.6 kg, explicit diagonal inertia [0.002, 0.002, 0.002], existing friction/contact settings unchanged. Evaluation-only scene and robot snapshots; live pipeline unchanged. Retargeting geometry remains **0.30 m source / 0.35 m target**, IK disabled, generated standing lean 5°.
- Nominal box [0.35, 0, 0.15], root [0.35 - starting_distance, 0, 0.8], identity nominal quaternions. Independent ±0.02 m per XY axis and ±2° yaw perturbations; no extra placement noise. Approach stopping offset remains 0.30 m.
- Headless MuJoCo 3.3.5, ONNX Runtime 1.23.2, production ver3 observations and manual production goal order. Physics timestep 0.001667 s with ten steps per control tick; 2 s idle warmup, 30 s per-goal timeout. No VLM queries, ROS executor, or hardware commands.
- Root/mean-body position tolerance 0.30 m, root orientation 0.80 rad, object action error 0.45 m, alternative approach/pre-pick XY reach 0.45 m, final object task tolerance 0.60 m. Joint RMSE and object orientation are diagnostic only. Stationarity hold 0.5 s; robot speeds ≤0.10 m/s and 0.15 rad/s, object speeds ≤0.15 m/s and 0.30 rad/s; minimum action duration 1 s.
- Failed approach triggers a reset at nominal robot X=0 while retaining perturbations, restoring the initial box and policy history, then attempts manipulation once. No retry after manipulation failure. Fall guard unchanged.

Standard deviations combine distance and reset variation. Success-conditioned error samples are subject to selection bias. Generous production tolerances do not imply exact placement or independently verified release. The 100-trial result should not be treated as a paired comparison against Uniform's previous ten trials.

## Artifacts and verification

[summary.json](summary.json): full precision errors and success rates. [outcomes.csv](outcomes.csv): 100 per-trial results. [all_phase_errors.csv](all_phase_errors.csv): all attempted endpoint errors, including failures. [manifest.json](manifest.json): full initial conditions, parameters, source/model hashes and timing. Exact goal payloads and state traces are under `approach/` and `manipulation/`. Scene and flat-hand robot snapshots are saved at the result root; the controller config uses the snapshot.

Validated all 100 unique trial IDs, ten samples per distance pair, first-ten reproduction, source/model hashes, and every displayed mean/std recomputed from raw passed endpoints. All 29 evaluation tests passed. Runtime: approximately 283 s. The existing video replay defaults to the rectangular-box video scene; replay must account for the saved cube dimensions rather than blindly using that default.

## Reproduce

Use the [evaluation environment setup](../../README.md), then run from the workspace source root:

```bash
/home/sitongchen/miniconda3/envs/keyLM_ros310/bin/python -m evaluation.evaluate_box_distances \
  --approach-range 0.5 2.3 --physical-box-size 0.3 0.3 0.3 --repetitions 10 \
  --policies crl-humanoid-ros/crl_g1_goalcontroller_py/crl_g1_goalcontroller_python/model/model_101000.onnx \
  --output evaluation/results/box_030m_101000_100trials_repeat
```

Use a new output directory to avoid overwriting existing results.
