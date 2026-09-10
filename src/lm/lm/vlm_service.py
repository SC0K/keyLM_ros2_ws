from __future__ import annotations

import base64
import json
import threading
import time
from typing import Literal

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

from ollama import Client
from pydantic import BaseModel, ValidationError

from lm.keyframe_modes import MANIPULATION_KEYFRAMES, PLANNER_KEYFRAMES, keyframe_phase, keyframe_object_type
from lm.vlm_connection import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST
from lm_interfaces.srv import VLMQuery

try:
    import cv2
    from cv_bridge import CvBridge
except ImportError:
    cv2 = None
    CvBridge = None


AllowedKeyframe = Literal[
    "approach_box", "stand_before_pick_box", "crouch_to_pick_box", "stand_after_pick_box",
    "stand_before_place_box", "crouch_to_place_box", "stand_after_place_box",
    "approach_bucket", "stand_before_pick_bucket", "crouch_to_pick_bucket", "stand_after_pick_bucket",
    "stand_before_place_bucket", "crouch_to_place_bucket", "stand_after_place_bucket",
]


class KeyframeDecision(BaseModel):
    next_keyframe: AllowedKeyframe
    object_in_manipulation: bool
    task_completion: bool


SYSTEM_PROMPT = """
You are a high-level planner for a robot performing a physical pick-and-place task.
Choose exactly ONE next keyframe from the allowed library. Use the measured
planner_context and current image together. Do not assume an action executed
successfully merely because it was selected, published, or the robot stopped.

OBJECT AND LIBRARY SELECTION
Use the task text to identify the requested object and the image to confirm
whether it is a box or bucket. Boxes use two-hand grasps; buckets use the right
hand at the handle. Never require a two-hand grasp for a bucket.
All phase names below are shorthand: append _box or _bucket to EVERY selected
keyframe, including approach and standing poses. For example, a bucket placement
uses stand_before_place_bucket -> crouch_to_place_bucket -> stand_after_place_bucket.
The JSON fields do not change; the keyframe suffix identifies the object type.
Never mix object libraries during a task. If planner_context.selected_object_type
is set, retain it, including recovery. Do not switch objects while holding one.
Choose the object from the task text and image, not from tracking availability.
Every response selects a normal action; there is no separate object-selection
query. The action's _box or _bucket suffix routes the matching measured pose
locally for retargeting and policy observations. On the first real-deployment
request, object distances may be unknown; follow the normal approach/reach rules.
This is not an image-based 3D pose estimator. If the requested object is absent/ambiguous,
do not invent a grasp or claim completion. Select only a visually safe
setup/recovery phase, keeping task_completion=false.

KEYFRAME MEANINGS
- approach: locomotion toward the box when outside pickup reach. No grasp or lift;
  object inputs are masked by the robot node. Never use while holding the box.
- stand_before_pick: object-aware preparation at the pickup stance. Does NOT grasp or lift.
- crouch_to_pick: lower the robot and establish the grasp. Does NOT complete lifting.
- stand_after_pick: stand and lift the grasped object. Does NOT carry it to its destination.
- stand_before_place: carry/position the object ABOVE the placement location,
  still holding it. Does NOT lower, release, or place the object.
- crouch_to_place: lower the object onto the intended supporting surface at the
  destination. This is the required placement action.
- stand_after_place: withdraw from the placed object and return to standby.
  This is NOT a substitute for crouch_to_place.

REQUIRED PROGRESSION
Normal successful order:
approach (if initially too far) -> stand_before_pick -> crouch_to_pick -> stand_after_pick -> stand_before_place
-> crouch_to_place -> stand_after_place.
Select only the next step, never skip an intermediate manipulation step.
In particular, NEVER transition directly from stand_before_place to
stand_after_place, even if measured_task_completion is true.

DECISION PROCEDURE (apply in this order)
1. Read previous_action, previous_action_finished, previous_action_success,
   stationary, distance_context, tracking_errors, and measured_task_completion.
   Treat missing/null evidence as unknown, not success.
2. On the first request, if pick_within_horizontal_reach is false or unknown,
   choose approach. If already within reach choose stand_before_pick, or
   crouch_to_pick only if the image supports a safe immediate pickup.
   Do not jump to carry/place/finish.
3. For an existing previous action, if previous_action_finished is false, do not
   advance. Select the same action only if safe; otherwise a safe recovery/setup
   action. task_completion must be false.
4. If previous_action_success is false or null, or the image contradicts success,
   do not advance or claim completion. Apply the recovery rules below.
5. Only for a finished, successful action with consistent visual evidence:
   - approach -> stand_before_pick when within horizontal pickup reach;
     otherwise repeat approach. Do not jump from approach directly to pickup.
   - stand_before_pick -> crouch_to_pick, but ONLY when
     distance_context.pick_within_horizontal_reach is true.
     Otherwise select approach if not holding the object.
   - crouch_to_pick -> stand_after_pick, only with the grasp established.
   - stand_after_pick -> stand_before_place, only with the object held securely.
   - stand_before_place -> crouch_to_place, only when safely positioned to lower it.
   - crouch_to_place -> stand_after_place, only when the object is supported at
     the intended destination, not suspended above it.
   - stand_after_place -> stand_after_place; evaluate completion as specified below.

INTERPRET THE MEASUREMENTS CORRECTLY
previous_action_finished means stationary long enough; it does NOT prove success.
previous_action_success checks configured tracking/object errors, with a horizontal
reach exception for approach and stand_before_pick. Verify the image is consistent with it.
measured_task_completion means object POSITION is inside a tolerance around the
destination. It does NOT prove that crouch_to_place happened, that the object was
released, that the support surface carries its weight, or that the final stand
finished. A held object near the target is NOT a completed placement.
Small body/root tracking errors only show pose tracking; they do not prove grasp,
transport, placement, or release. Ignore object orientation error for completion.
Use the configured XY reach test, not 3D robot-to-object distance, for pickup.
If the object/support/grasp is occluded or ambiguous, do not claim visual success.

RECOVERY
- Failed approach: repeat approach while out of reach; once within reach,
  select stand_before_pick when safe.
- Failed pick/lift or lost grasp: approach if out of reach and not holding the
  object, otherwise stand_before_pick; retry crouch_to_pick only after setup.
- Failed stand_before_place/crouch_to_place while safely holding the object:
  retry that action if safe, or use stand_before_place to re-establish placement
  setup, then crouch_to_place. NEVER recover by skipping to stand_after_place.
- If the object dropped or is no longer controlled, do not continue as if carrying
  it. Recover through stand_before_pick and pickup; proximity to the target alone
  must not be counted as successful placement.
- Failed final standby with the object still properly placed: retry stand_after_place.
- Safety/recovery can move backward or repeat, but cannot skip placement.

TASK COMPLETION: OBSERVED, NOT PREDICTED
task_completion must remain false for every pick/carry/place decision AND for
the first selection of stand_after_place after crouch_to_place.
Set task_completion=true ONLY when ALL are true:
- previous_action is stand_after_place;
- previous_action_finished is true;
- previous_action_success is true;
- measured_task_completion is true;
- the image confirms the robot is standing without holding the object and the
  object rests on its intended supporting surface at the destination.
When all checks pass, return next_keyframe="stand_after_place_box" or
"stand_after_place_bucket" for the selected object and task_completion=true.
Otherwise keep task_completion=false and choose the appropriate retry/recovery.
Never predict that a newly selected action will finish the task: the caller uses
task_completion=true to stop planning immediately.

OBJECT OBSERVATION FLAG
Always set object_in_manipulation=true for the six pick/place keyframes, including
both setup and final standing poses. This flag enables object-aware retargeting
and current-object/goal observations; it does NOT mean the hands currently hold
the object.

The separate approach action is forced to no-object locomotion by the robot
node, regardless of the returned object_in_manipulation flag.

EXAMPLES (conditions in each example must actually be observed)
Finished/successful stand_before_place, safely holding above the destination:
{"next_keyframe":"crouch_to_place_box","object_in_manipulation":true,"task_completion":false}
Finished/successful crouch_to_place, object supported at the destination:
{"next_keyframe":"stand_after_place_bucket","object_in_manipulation":true,"task_completion":false}
Failed stand_before_place, object still safely held and a retry is safe:
{"next_keyframe":"stand_before_place_bucket","object_in_manipulation":true,"task_completion":false}
Finished/successful stand_after_place, measured completion true, object visibly
placed and no longer held:
{"next_keyframe":"stand_after_place_box","object_in_manipulation":true,"task_completion":true}

OUTPUT
Return only a JSON object matching the provided schema: next_keyframe (one allowed
name), object_in_manipulation (boolean), task_completion (boolean).
No markdown, explanations, extra fields, invented actions, or multiple keyframes.
""".strip()


def build_user_prompt(task_text: str, planner_context: str, allowed_keyframes: list[str]) -> str:
    return f"""
Task:
{task_text}

Planner context JSON:
{planner_context}

Allowed keyframes:
{", ".join(allowed_keyframes)}
""".strip()


class VLMServiceNode(Node):
    def __init__(self) -> None:
        super().__init__("vlm_service_node")

        self.declare_parameter("service_name", "/vlm/query")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("request_image_topic", "/vlm/request_image")
        self.declare_parameter("image_wait_timeout_sec", 10.0)
        self.declare_parameter("ollama_host", DEFAULT_OLLAMA_HOST)
        self.declare_parameter("model_name", DEFAULT_MODEL)
        self._ollama_host = str(self.get_parameter("ollama_host").value).strip()
        self._model_name = str(self.get_parameter("model_name").value).strip()
        if not self._ollama_host or not self._model_name:
            raise ValueError("ollama_host and model_name must not be empty")
        self._ollama_client = Client(host=self._ollama_host)

        service_name = self.get_parameter("service_name").get_parameter_value().string_value
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        request_image_topic = self.get_parameter("request_image_topic").get_parameter_value().string_value
        self._image_wait_timeout_sec = float(self.get_parameter("image_wait_timeout_sec").value)

        self._allowed_keyframes = list(PLANNER_KEYFRAMES)
        self._latest_image_bgr = None
        self._latest_image_stamp = None
        self._latest_image_frame_id = ""
        self._latest_image_sequence = 0
        self._image_condition = threading.Condition()
        self._callback_group = ReentrantCallbackGroup()
        self._bridge = CvBridge() if CvBridge is not None else None
        self._cv_bridge_error_logged = False

        self._image_sub = self.create_subscription(
            Image,
            image_topic,
            self._image_callback,
            10,
            callback_group=self._callback_group,
        )
        self._request_image_pub = self.create_publisher(Image, request_image_topic, 10)
        self._srv = self.create_service(
            VLMQuery,
            service_name,
            self._handle_query,
            callback_group=self._callback_group,
        )

        self.get_logger().info(f"VLM service ready at {service_name}")
        self.get_logger().info(f"Subscribed to image topic: {image_topic}")
        self.get_logger().info(f"Publishing request images to: {request_image_topic}")
        self.get_logger().info(
            f"Waiting up to {self._image_wait_timeout_sec:.1f}s for a fresh camera frame per request"
        )
        self.get_logger().info(f"Using Ollama endpoint: {self._ollama_host}, model: {self._model_name}")

    def _image_callback(self, msg: Image) -> None:
        if self._bridge is None or cv2 is None:
            if not self._cv_bridge_error_logged:
                self.get_logger().error("cv_bridge/OpenCV not available. Install ROS cv_bridge and OpenCV.")
                self._cv_bridge_error_logged = True
            return

        try:
            image_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert/store camera frame: {exc}")
            return

        with self._image_condition:
            self._latest_image_bgr = image_bgr
            self._latest_image_stamp = msg.header.stamp
            self._latest_image_frame_id = msg.header.frame_id
            self._latest_image_sequence += 1
            self._image_condition.notify_all()

    def _current_image_sequence(self) -> int:
        with self._image_condition:
            return self._latest_image_sequence

    def _copy_next_image_after(self, image_sequence: int, timeout_sec: float):
        deadline = time.monotonic() + max(0.0, timeout_sec)
        with self._image_condition:
            while (
                self._latest_image_sequence <= image_sequence
                or self._latest_image_stamp is None
                or self._latest_image_bgr is None
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None, None, ""
                self._image_condition.wait(timeout=remaining)
            return self._latest_image_bgr.copy(), self._latest_image_stamp, self._latest_image_frame_id

    def _publish_request_image(self, image_bgr, image_stamp, frame_id: str) -> None:
        if self._bridge is None:
            return
        try:
            msg = self._bridge.cv2_to_imgmsg(image_bgr, encoding="bgr8")
            msg.header.stamp = image_stamp
            msg.header.frame_id = frame_id
            self._request_image_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to publish VLM request image: {exc}")

    def _query_vlm(self, image_bgr, task_text: str, planner_context: str) -> tuple[KeyframeDecision, str, float]:
        user_prompt = build_user_prompt(
            task_text=task_text,
            planner_context=planner_context,
            allowed_keyframes=self._allowed_keyframes,
        )

        ok, encoded = cv2.imencode(".png", image_bgr)
        if not ok:
            raise RuntimeError("Failed to PNG-encode image from camera topic")
        image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

        start_time = time.perf_counter()
        schema = KeyframeDecision.model_json_schema()
        schema["properties"]["next_keyframe"]["enum"] = self._allowed_keyframes
        response = self._ollama_client.chat(
            model=self._model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt, "images": [image_b64]},
            ],
            format=schema,
            think=False,
            options={
                "temperature": 0.0,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "repeat_penalty": 1.0,
            },
        )
        latency_sec = time.perf_counter() - start_time
        raw_content = response.message.content
        decision = KeyframeDecision.model_validate_json(raw_content)
        if decision.next_keyframe not in self._allowed_keyframes:
            raise ValueError("VLM must choose an explicit _box or _bucket keyframe from the allowed library")
        context = json.loads(planner_context) if planner_context.strip() else {}
        selected_type = context.get("selected_object_type") if isinstance(context, dict) else None
        if selected_type in ("box", "bucket") and keyframe_object_type(decision.next_keyframe) != selected_type:
            raise ValueError("VLM attempted to switch object libraries during an active task")

        # decision = KeyframeDecision(
        #     next_keyframe="crouch_to_pick",
        #     object_in_manipulation=True,
        #     task_completion=False,
        # )
        raw_content = decision.model_dump_json()
        latency_sec = 0.123
        return decision, raw_content, latency_sec

    def _handle_query(self, request: VLMQuery.Request, response: VLMQuery.Response) -> VLMQuery.Response:
        task_text = request.task_text.strip()
        planner_context = request.planner_context.strip()

        if not task_text:
            response.success = False
            response.error_message = "task_text cannot be empty"
            return response

        request_start_image_sequence = self._current_image_sequence()
        request_image_bgr, request_image_stamp, request_image_frame_id = self._copy_next_image_after(
            request_start_image_sequence,
            self._image_wait_timeout_sec,
        )
        if request_image_stamp is None or request_image_bgr is None:
            response.success = False
            response.error_message = (
                "No fresh camera image received after VLM request started "
                f"after waiting {self._image_wait_timeout_sec:.1f}s"
            )
            return response

        try:
            self._publish_request_image(request_image_bgr, request_image_stamp, request_image_frame_id)
            decision, raw_json, latency_sec = self._query_vlm(
                image_bgr=request_image_bgr,
                task_text=task_text,
                planner_context=planner_context,
            )
            response.success = True
            response.error_message = ""
            response.next_keyframe = decision.next_keyframe
            response.object_in_manipulation = (
                keyframe_phase(decision.next_keyframe) != "approach"
                and (decision.object_in_manipulation or decision.next_keyframe in MANIPULATION_KEYFRAMES)
            )
            response.task_completion = decision.task_completion
            response.raw_json = raw_json
            response.latency_sec = float(latency_sec)
            response.image_stamp = request_image_stamp
            self.get_logger().info(f"VLM decision={decision.next_keyframe} latency={latency_sec:.3f}s")
        except ValidationError as exc:
            response.success = False
            response.error_message = f"Model output schema validation failed: {exc}"
        except Exception as exc:
            response.success = False
            response.error_message = f"VLM request failed: {exc}"
        return response


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VLMServiceNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
