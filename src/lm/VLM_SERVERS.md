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

The controller's waiting stand and the VLM `stand_before_pick` / `stand_after_place`
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
goals remain present. Set the lean to 0 for upright stands. If the controller uses
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
All VLM keyframes preserve their source library root Z and object Z, with or
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
