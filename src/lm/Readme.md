## Tunnel to VLM server

For TARS:
```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 11434:localhost:11434 sitchen@tars
```
For CASE:
```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 11434:localhost:8001 sitchen@case.inf.ethz.ch
```
Then always start the GUI with:
```bash
ros2 run lm vlm_planner_app --no-tunnel
```