# Policy evaluation protocols

The evaluator now supports:

- `--skip-approach`: omit `approach_box`; the first goal becomes `stand_before_pick_box`. This does not automatically move the robot's reset position.
- `--initial-root-pos X Y Z`: change the nominal reset position; the existing ±0.02 m XY and ±2° yaw perturbations remain.
- `--place-noise-xy-m 0.10`: independently perturb the placement destination by up to ±0.10 m per world XY coordinate. This is the new default. Target height/orientation stay unchanged. Noise is sampled once per trial using an independent seeded stream and reused for matching trial indices across policies.

Both retargeted placement goals and the task-completion check use the same perturbed destination. Sampled offsets are stored in `manifest.json` / trial initial conditions, and each trial records the actual task target pose.

The original 30-trial comparison and the following model 40000 run had **no placement destination noise**. To reproduce their protocol with the current evaluator, explicitly add `--place-noise-xy-m 0` to their recorded commands. Existing result artifacts are historical and remain unchanged. Runs using a near-box start or placement noise constitute a different protocol and should not silently replace the original results.
