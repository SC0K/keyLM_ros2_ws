from __future__ import annotations

import base64
import threading
import time
from typing import Literal

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

from ollama import chat
from pydantic import BaseModel, ValidationError

from lm_interfaces.srv import VLMQuery

try:
    import cv2
    from cv_bridge import CvBridge
except ImportError:
    cv2 = None
    CvBridge = None


MODEL_NAME = "qwen3.6:27b"

AllowedKeyframe = Literal[
    "crouch_to_pick",
    "crouch_to_place",
    "stand_after_pick",
    "stand_after_place",
    "stand_before_pick",
    "stand_before_place",
]


class KeyframeDecision(BaseModel):
    next_keyframe: AllowedKeyframe
    object_in_manipulation: bool
    task_completion: bool


SYSTEM_PROMPT = """
You are a high-level robot planner.
Your job is to choose exactly one next action keyframe from the allowed motion library.
A image of the current scene is provided to help you understand the environment, but you must choose from the allowed keyframes.
You need to decide if the task is completed after executing the chosen keyframe, and whether the selected keyframe should use the object-aware retargeting and policy mask.
The user prompt includes planner_context as JSON. Treat that JSON as measured execution state.
The planner_context.previous_action field is "none" on the first request; otherwise it is the keyframe selected by the previous VLM response.
The planner_context.previous_action_finished field is true only after the robot and object have been stationary below configured thresholds.
The planner_context.previous_action_success field is true only when the tracked mean body error, root pose error, and object position error are below their configured thresholds.
The planner_context.measured_task_completion field is true only when the actual object position is within threshold of the target object position. Ignore object orientation for success and completion.
The planner_context.distance_context field contains robot-to-object and object-to-target distances; use those distances when deciding whether to approach, pick, place, retry, or finish.
Use the image to infer whether the robot is doing the task correctly. Check whether the box appears held with two hands during carry/place keyframes, whether it has slipped or been dropped, and whether the visible robot/object state contradicts the expected phase.
If previous_action_finished is false, do not advance to a new semantic phase.
If previous_action_success is false, do not advance to the next semantic phase and do not mark the task complete.
If the image suggests the object is dropped, not between the hands, or not controlled during an object-aware keyframe, treat the previous action as unsafe/failed even if the numeric context is ambiguous, and choose a recovery or retry keyframe.
On failure, choose the safest retry/recovery keyframe from the allowed list: retry the previous keyframe if the robot/object state still matches it; otherwise choose a safe standing/setup keyframe before retrying.
For failed pick attempts, especially crouch_to_pick or stand_after_pick, recover with stand_before_pick first, then retry crouch_to_pick on the next request.
For failed place attempts, especially stand_before_place or crouch_to_place, retry the failed place keyframe if the robot is still safely holding the object; otherwise recover with stand_before_place before retrying crouch_to_place.
For failed final standby, retry stand_after_place.
At the end of the task, the robot should be in a "stand" keyframe with the object placed at the target location. After placing the object, the robot can only stand up without the object.
Your response must follow exactly the JSON schema provided, and only include the allowed keyframes.
The JSON format is:
{
    "next_keyframe": string,  // one of the allowed keyframes
    "object_in_manipulation": boolean,  // same as object_to_manipulate: true means retargeting and policy should consider the object
    "task_completion": boolean  // whether the task is completed after executing the chosen keyframe
}

Normally after successfully placing the object, choose the final stand keyframe and set task_completion true only if planner_context.measured_task_completion is true and the selected keyframe leaves the robot in the final standby state.
The object_in_manipulation boolean is the same effective flag as object_to_manipulate. It controls whether the retargeter and policy should consider object target/current-object observations.
Set object_in_manipulation true for keyframes that need object-aware hand, object target retargeting, or policy object observations: crouch_to_pick, stand_after_pick, stand_before_place, and crouch_to_place.
stand_before_place is not root-only: set object_in_manipulation true because the robot should still hold the box at the target x/y position with the configured default hold height.
Set object_in_manipulation false for pure standing/root/standby keyframes that do not need the object mask, especially stand_before_pick and the final stand_after_place after the object has been placed.
For standing keyframes that still carry or position the object, keep object_in_manipulation true.

Rules:
- Return only valid JSON matching the provided schema.
- Do not output markdown, explanations, or code fences.
- Do not invent actions outside the allowed keyframes.
- Use only planner_context and the image; do not assume an action succeeded if planner_context.previous_action_success is false or null.
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

        service_name = self.get_parameter("service_name").get_parameter_value().string_value
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        request_image_topic = self.get_parameter("request_image_topic").get_parameter_value().string_value
        self._image_wait_timeout_sec = float(self.get_parameter("image_wait_timeout_sec").value)

        self._allowed_keyframes = [
            "crouch_to_pick",
            "crouch_to_place",
            "stand_after_pick",
            "stand_after_place",
            "stand_before_pick",
            "stand_before_place",
        ]
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
        self.get_logger().info(f"Using model: {MODEL_NAME}")

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
        response = chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt, "images": [image_b64]},
            ],
            format=KeyframeDecision.model_json_schema(),
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
            response.object_in_manipulation = decision.object_in_manipulation
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
