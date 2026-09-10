# TARS and CASE VLM connections

## One-command experiment launch (recommended)

Start the robot simulator/monitor, VLM camera/server/retargeter, GUI and managed
SSH tunnel together:

```bash
ros2 launch lm vlm_experiment_launch.py mode:=sim server:=tars
```

For the real robot and USB camera:

```bash
ros2 launch lm vlm_experiment_launch.py mode:=real server:=tars
```

**Real mode starts the hardware controller and OptiTrack stack.** Check the
hardware configuration/calibration before launching. Neither mode starts a
task automatically: enter the task in the GUI and press **Start** when ready.
The GUI owns the single planner subprocess; do not run another planner or GUI.
This uses the VLM robot launches, not the fixed-sequence test launches.

Use `server:=case` for CASE. The GUI's **VLM server** dropdown also switches between
TARS and CASE: stop the planner first, then select the server. It replaces only
the GUI-owned tunnel, retaining the local port and SSH user. **Connect / retry**
retries a failed connection. Selecting a different profile clears any custom
CLI SSH host/remote-port overrides. External tunnels are never stopped or
switched by the GUI; change those manually.

With a tunnel already running, add
`manage_tunnel:=false`. If the robot/controller/monitor already runs, add
`start_robot:=false`. Shared VLM settings such as `camera_device`, `model_name`,
`retarget_ik_enabled`, `task_text`, and the object/retargeter topics remain
available. `local_port:=11435` changes both the managed tunnel and default
Ollama endpoint. Real OptiTrack options such as `start_optitrack:=false` and
`optitrack_object_pose_topic:=...` are passed through to the hardware launch.

Ctrl-C in the launch terminal stops the launched stack and the GUI-owned
planner/tunnel; an externally managed tunnel is not stopped. GUI **Stop** stops
planning only, not the robot controller; use the robot's normal safety controls
to stop motion. A display and working noninteractive SSH login are required.

## Box and bucket tasks

The library in `lm/keyframes/` now contains six `*_box.npz` frames and six
`*_bucket.npz` frames. The box files were renamed without changing their
contents. Bucket files were converted from `resource/test_sequence_bucket`:
full robot FK and 29 named joints replace the raw export format, retaining
authored joint poses, heights and robot/object relative geometry. The original
bucket exports are unchanged.

The VLM receives both libraries, the camera image and task text. It selects one
of seven phases with an explicit object suffix, e.g. `crouch_to_pick_box` or
`crouch_to_pick_bucket`. `approach_box` and `approach_bucket` reuse the respective
pickup stand, retaining the 0.30 m XY offset and locomotion/object masking.
The other six phases remain object-aware. The JSON fields are unchanged:
`next_keyframe`, `object_in_manipulation`, `task_completion`.

The selected suffix automatically chooses the retargeter behavior: box semantic
axes/two-hand grasp or bucket physical axes/right-hand grasp without scaling.
IK remains disabled by default for both. Root/object heights stay authored;
setup/final stands use the existing generated standing behavior. Once a task
has an accepted goal, changing object families mid-task is rejected. Start a
new task to manipulate a different object.

For `stand_before_pick_bucket`, placement comes from the library's root-to-bucket
offset and heading: the bucket is approximately 0.407 m forward and 0.164 m to
the robot's right. This offset rotates with the bucket; it is not a fixed world-Y
shift. The generated standing joint posture/lean and authored heights remain
unchanged. Box pickup still uses the configured centred stance. `approach_bucket`
and final standing placement are unchanged.

For a simulated bucket experiment:

```bash
ros2 launch lm vlm_experiment_launch.py mode:=sim server:=tars scene_object:=bucket
```

Enter a bucket task in the GUI and press Start. `scene_object` selects the actual
simulation/monitor/camera mesh and its initial pose; it does not decide the VLM
keyframe. `scene_object:=box` is the default. Both support the supervision checkbox.

For the real robot with a bucket and USB camera:

```bash
ros2 launch lm vlm_experiment_launch.py mode:=real server:=tars scene_object:=bucket
```

Real mode starts hardware. `mocap_object_selection:=true` is retained as the
compatibility parameter for local pose routing, not a separate VLM selection step.
The first normal VLM action chooses the object from image+text; its `_box` or
`_bucket` suffix routes the matching mocap stream for retargeting and policy
observations. That action follows the normal execution/supervised-approval path.
No mocap availability list or candidate-object poses are sent to the VLM. Later
requests include the selected object's measured distances and the previously
chosen type. Both objects may be tracked simultaneously; the selected family
stays fixed for that task. Before the first choice, startup waits for the robot
to be stationary without requiring an object pose; missing selected tracking
still blocks goal execution.

| Object | Raw mocap input parameter/default | Converted pose topic |
| --- | --- | --- |
| Box | `optitrack_box_pose_topic:=/optitrack_dispatcher/rigidbodies/carton_box` | `/mocap/box_pose` |
| Bucket | `optitrack_bucket_pose_topic:=/optitrack_dispatcher/rigidbodies/bucket` | `/mocap/bucket_pose` |

The dispatcher configuration currently contains `carton_box` but no `bucket`
mapping. Configure the actual bucket rigid-body ID/name in the OptiTrack config,
or override `optitrack_bucket_pose_topic` with its existing topic. Never map the
box and bucket to the same tracked object. Both poses must already be calibrated
into the robot's world coordinates; bucket poses use the mesh-base origin, not
a box-centre origin. The bridges reject invalid NatNet tracking flags.

The policy controller chooses its object observations from the accepted goal's
`object_type`. During supervised preview it retains the previous active object's
observations; switching occurs only on approval. The controller also forwards
that active pose on `actual_box_pose_topic` (legacy name, default
`/actual_box_pose`) for the GUI/monitor. It continues forwarding after the planner
stops. Missing/stale selected tracking blocks new goals/approval instead of
falling back to the other object (`tracked_object_timeout_sec`, default 1 s).

`scene_object` still controls the monitor mesh, not real mocap selection. The
GUI's 2D outline is schematic. Simulation remains single-object by default, with
its simulator pose input (`mocap_object_selection:=false`). For externally
converted poses, set `start_object_pose_bridge:=false` and configure
`tracked_box_pose_topic` and `tracked_bucket_pose_topic` consistently.

Only the 14 explicit `_box`/`_bucket` action names are allowed by the VLM schema,
planner goal output and public retargeter service. Unsuffixed names are rejected.

## Supervised goal approval

Add `supervised_mode:=true` to the combined launch in either mode:

```bash
ros2 launch lm vlm_experiment_launch.py mode:=sim server:=tars supervised_mode:=true
ros2 launch lm vlm_experiment_launch.py mode:=real server:=tars supervised_mode:=true
```

The GUI also has a **Supervised mode** checkbox below Start/Stop, initialized
from the launch argument. Stop planning to change it, then press **Start**.
Start synchronizes the controller mode before launching the planner; it refuses
to start if the controller cannot confirm the setting. A mode change discards
any unapproved preview without executing it and leaves the active goal alone.
No robot-stack relaunch is needed to switch modes after installing this update.

Each proposed goal appears in the monitor first. The GUI reports
`awaiting_approval`. Focus the **robot monitor window** and press **N**, or on
the real robot hold **R1** and press **A**, to activate that exact preview.
The robot must already be in **GOAL** mode; approval does not switch the FSM.
The monitor's existing target-overlay visibility setting still applies.

Until approval, the policy continues its previous goal (or its idle stance
before the first goal); it does not receive the preview as a policy input.
Tracking errors still describe the active goal, while the monitor shows the
pending goal. The planner waits and starts the new action timer only after
controller acknowledgement. Goal JSON and manipulation masks are unchanged.

Repeated presses with no pending goal do nothing. Stopping/crashing the planner
invalidates an unapproved preview within 3 seconds; it does **not** stop an
already executing goal. Use the robot's normal safety controls to stop motion.
Supervision defaults to `false`, preserving automatic operation.

For separate processes or `start_robot:=false`, the GUI still synchronizes the
updated controller through the service associated with `retargeted_keyframe_topic`.
A standalone client without the GUI requires matching `supervised_mode:=true`
on the robot launch and `-p supervised_mode:=true` in the client's ROS arguments.
Preview-only topics cannot execute on a controller without supervision enabled.

## Separate-process launches

Both profiles expose the remote Ollama API on local port **11434**. Keep the
same VLM launch for either server:

```bash
ros2 launch lm vlm_launch.py
```

Start the GUI with the server whose tunnel you want it to manage:

```bash
ros2 run lm vlm_planner_app --server tars
# or
ros2 run lm vlm_planner_app --server case
```

TARS is the default. Profiles are defined in `lm/vlm_connection.py`:

| Profile | SSH destination | Forwarding |
| --- | --- | --- |
| tars | sitchen@tars | 11434:localhost:11434 |
| case | sitchen@case.inf.ethz.ch | 11434:localhost:8001 |

The GUI does not launch the VLM service; run the ROS launch separately. SSH
authentication must already work non-interactively for the GUI-managed tunnel.
For password authentication, start a tunnel yourself in another terminal:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 11434:localhost:11434 sitchen@tars
# or
ssh -N -o ExitOnForwardFailure=yes -L 11434:localhost:8001 sitchen@case.inf.ethz.ch
```

Then start the GUI with `ros2 run lm vlm_planner_app --no-tunnel`.
Stop the old tunnel before switching servers: only one can listen on port
11434. An existing open port is reused but shown as **server unverified**;
selecting a new profile cannot change another process's tunnel.

The service defaults to `http://localhost:11434` and model `qwen3.6:27b`.
Both remote ports must serve the Ollama API. If the model name differs on a
server, pass `model_name:=THE_MODEL_NAME` to the launch. For a different local
port, use matching settings in both processes:

```bash
ros2 launch lm vlm_launch.py ollama_host:=http://localhost:11435
ros2 run lm vlm_planner_app --server case --local-port 11435
```

Explicit `--host`, `--remote-port`, and `--user` GUI arguments override the
selected profile. Connection parameters are read at startup; restart the VLM
node after changing its endpoint/model. Neither command automatically installs
or downloads models on the server.

## Real robot with the computer's USB camera

The real VLM stack captures a V4L2 camera directly (no MuJoCo rendering and no
separate ROS camera driver):

```bash
ros2 launch lm vlm_real_launch.py
```

It defaults to `/dev/video0`, publishes `bgr8` images on `/camera/image_raw` at
2 Hz, and uses `/g1_hardware/monitor` and the real target geometry. Camera
capture runs continuously at 30 FPS so driver buffering does not build up
old images. No image is republished when capture stalls. For the Logitech C922
tested here, a stable device path is:

```bash
ros2 launch lm vlm_real_launch.py \
  camera_device:=/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_E63E731F-video-index0
```

The stack starts camera/VLM/retargeter services, **not hardware or a task**.
Use the existing VLM hardware launch (not a fixed-sequence test) when ready:

```bash
ros2 launch crl_g1_goalcontroller g1_keyframe.py
```

Check the hardware network interface, mocap calibration and policy checkpoint
before enabling GOAL. This hardware launch bridges
`/optitrack_dispatcher/rigidbodies/carton_box` to `/actual_box_pose`; override
`optitrack_object_pose_topic:=...` to match the physical rigid body. Use
`start_optitrack:=false` if the adaptor already runs. If a calibrated object
PoseStamped is already available, set `start_object_pose_bridge:=false` and
use matching `current_object_pose_topic` (hardware) / `actual_box_pose_topic`
(VLM and GUI) overrides. Camera images do not replace calibrated robot/object
pose tracking. This remains the box-library VLM planner, not a bucket planner.

Start the GUI/tunnel in real mode, then enter a task and press Start explicitly:

```bash
ros2 run lm vlm_planner_app --real --server tars
# With a manually started SSH tunnel:
ros2 run lm vlm_planner_app --real --no-tunnel
```

`--real` selects hardware monitor data and real box geometry in both the GUI
and its spawned planner. For a custom object topic, add
`--ros-args -p actual_box_pose_topic:=/YOUR_OBJECT_POSE` to the GUI command.
Do not also enable `start_client:=true` in the VLM launch when using GUI Start.
For a pre-existing ROS camera instead, use
`camera_backend:=real real_image_topic:=/YOUR_CAMERA/image_raw`.

Camera-only check (no VLM inference or robot nodes):

```bash
ros2 run lm scene_camera --ros-args -p backend:=usb -p camera_device:=/dev/video0
ros2 topic hz /camera/image_raw
```

The camera-only node keeps images local. A VLM query sends a fresh image to the
configured model server through the tunnel. After unplugging/replugging the
camera, restart the camera/stack if fresh images do not resume.

The controller's idle waiting stand (before any VLM goal) uses default joint
angles with **no added lean**, in no-object mode. This is independent of the
standing-lean setting below.

The VLM `stand_before_pick_box` / `stand_after_place_box` and their `_bucket` counterparts
goals use `default_angles` from `g1_keyframe_tracking_obj.yaml` plus
`standing_waist_pitch_deg` (default **5 degrees forward**, defined by
`VLM_STANDING_LEAN_DEG` in `lm/generated_stand.py`) and upright root yaw.
This VLM-only setting does not change the fixed-sequence tests' configured lean.
Override it on the combined launch, e.g. `standing_waist_pitch_deg:=5.0`.
When using separate VLM and robot launches, pass the override to both.
VLM stands keep the root
height from their library keyframe; the controller's waiting/test stands retain
their separately configured standing height.
These two VLM stands no longer use the library's authored joint pose. Object
goals remain present in manipulation mode. `stand_before_pick_box` has
`object_to_manipulate=true`, with its pickup stance 0.4 m from the box centre.
The bucket pickup stand instead retains its library's left-offset stance.

The separate **`approach_box` / `approach_bucket`** actions always have `object_to_manipulate=false`.
They reuse `stand_before_pick_box.npz` / `stand_before_pick_bucket.npz` (no separate NPZ to edit).
The retargeter places that library robot pose **0.30 m from the current object origin in XY, on the
robot-facing side**, preserves its root
height, faces the object, and zeroes the target object pose. The controller then
overwrites the robot posture through its existing walking-goal
path: default joint posture with configured walking-goal noise, measured and
target object inputs masked out. The offset is `APPROACH_XY_OFFSET_M` in
`lm/vml.py`; it is a centre-to-root XY distance, not clearance from the box
surface. This is not a collision-avoiding path or the separate 0.4 m pickup stance.
The VLM is instructed to choose the suffixed approach when too far away, then
the matching pickup stand once within 0.45 m XY reach, then the normal pickup sequence.
Never use approach while holding an object. Mode is fixed locally in
`lm/keyframe_modes.py`, independent of the model's returned object flag; all six
pick/place phases remain manipulation goals. JSON fields stay unchanged; only
the 14 suffixed action names are allowed. Locomotion success checks omit object
tracking and accept reaching the pickup range.
Set the lean to 0 for upright manipulation stands. If the controller uses
a different policy config, pass that same path as `standing_config_file:=...`
to the VLM launch.

### Rigid retargeting experiment

Both `vlm_launch.py` and `vlm_real_launch.py` default to
`retarget_ik_enabled:=false`. Pickup/carry/place motion frames undergo only an
XY translation and yaw rotation; robot joints, body Z coordinates and object
Z are preserved from the source. There is no size scaling or IK. Semantic
approach axes still select the robot's planar alignment. The object goal uses
the physical target orientation, as in IK mode, rather than retaining the
source box's forward/up convention. Object-only orientation offsets still
apply afterwards and do not rotate the robot.
The generated VLM stands preserve both root Z and object Z from their source
library frame, while generating the configured standing joint pose and lean.

Use `retarget_ik_enabled:=true` to restore the grasp-IK path for motion frames.
All manipulation keyframes preserve their source library root Z and object Z, with or
without IK. Observed/requested heights do not lift or lower these goals; only
XY placement and orientation are retargeted. With IK enabled, joints may change
but root Z is locked. The separate `stand_before_place_height_m`,
`stand_after_pick_height_m`, and VLM `default_goal_root_height_m` overrides have
been removed.
Box/bucket sim/real fixed-sequence test launches also expose
`retarget_ik_enabled` and default it to false; set it on the test launch
independently when testing IK.
Success checks now read the object target from the actual serialized goal,
rather than guessing a lift/placement height. Planner status and VLM context
include `action_success_checks` with each value, threshold and pass/fail result.
A failed or missing check can cause `stand_before_place` to repeat; do not
force placement just to escape a repeat while grasp/tracking is unsuccessful.
