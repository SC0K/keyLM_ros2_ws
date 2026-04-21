from __future__ import annotations

import base64
import time
from typing import Literal

import rclpy
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


MODEL_NAME = "qwen3.5:9b"

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
You need to decide if the task is completed after executing the chosen keyframe, and whether the robot is currently manipulating the object.
at the end of the task, the robot should be in a "stand" keyframe with the object placed at the target location. After placing the object the robot can only stand up witout the object.    
Your response must follow exactly the JSON schema provided, and only include the allowed keyframes.
The JSON format is:
{
    "next_keyframe": string,  // one of the allowed keyframes
    "object_in_manipulation": boolean,  // whether the robot is currently holding/manipulating the object
    "task_completion": boolean  // whether the task is completed after executing the chosen keyframe
}

Normally after successfully placing the object the task is completed and the robot return to standing pose for stand by.

Rules:
- Return only valid JSON matching the provided schema.
- Do not output markdown, explanations, or code fences.
- Do not invent actions outside the allowed keyframes.
- Prefer safer actions like inspection or repositioning if the scene is uncertain.
""".strip()


def build_user_prompt(task_text: str, planner_context: str, allowed_keyframes: list[str]) -> str:
    return f"""
Task:
{task_text}

Planner context:
{planner_context}

Allowed keyframes:
{", ".join(allowed_keyframes)}
""".strip()


class VLMServiceNode(Node):
    def __init__(self) -> None:
        super().__init__("vlm_service_node")

        self.declare_parameter("service_name", "/vlm/query")
        self.declare_parameter("image_topic", "/camera/image_raw")

        service_name = self.get_parameter("service_name").get_parameter_value().string_value
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value

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
        self._bridge = CvBridge() if CvBridge is not None else None
        self._cv_bridge_error_logged = False

        self._image_sub = self.create_subscription(Image, image_topic, self._image_callback, 10)
        self._srv = self.create_service(VLMQuery, service_name, self._handle_query)

        self.get_logger().info(f"VLM service ready at {service_name}")
        self.get_logger().info(f"Subscribed to image topic: {image_topic}")
        self.get_logger().info(f"Using model: {MODEL_NAME}")

    def _image_callback(self, msg: Image) -> None:
        if self._bridge is None or cv2 is None:
            if not self._cv_bridge_error_logged:
                self.get_logger().error("cv_bridge/OpenCV not available. Install ROS cv_bridge and OpenCV.")
                self._cv_bridge_error_logged = True
            return

        try:
            self._latest_image_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._latest_image_stamp = msg.header.stamp
        except Exception as exc:
            self.get_logger().error(f"Failed to convert/store camera frame: {exc}")

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
        #     next_keyframe="stand_after_place",
        #     object_in_manipulation=False,
        #     task_completion=True,
        # )
        # raw_content = decision.model_dump_json()
        # latency_sec = 0.123
        return decision, raw_content, latency_sec

    def _handle_query(self, request: VLMQuery.Request, response: VLMQuery.Response) -> VLMQuery.Response:
        task_text = request.task_text.strip()
        planner_context = request.planner_context.strip()

        if not task_text:
            response.success = False
            response.error_message = "task_text cannot be empty"
            return response
        if self._latest_image_stamp is None:
            response.success = False
            response.error_message = "No camera image received yet from subscribed topic"
            return response

        try:
            decision, raw_json, latency_sec = self._query_vlm(
                image_bgr=self._latest_image_bgr,
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
            response.image_stamp = self._latest_image_stamp
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
