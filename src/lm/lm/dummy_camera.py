from __future__ import annotations

from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class DummyCameraNode(Node):
    def __init__(self) -> None:
        super().__init__("dummy_camera_node")

        self.declare_parameter("topic", "/camera/image_raw")
        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("frame_id", "dummy_camera_frame")
        self.declare_parameter("image_path", "/home/sitongchen/pics/placebox.png")

        topic = self.get_parameter("topic").get_parameter_value().string_value
        rate_hz = self.get_parameter("rate_hz").get_parameter_value().double_value
        self._width = self.get_parameter("width").get_parameter_value().integer_value
        self._height = self.get_parameter("height").get_parameter_value().integer_value
        self._frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self._image_path = Path(self.get_parameter("image_path").get_parameter_value().string_value)

        if self._width <= 0 or self._height <= 0:
            raise ValueError("width and height must be > 0")
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be > 0")

        if not self._image_path.exists():
            raise RuntimeError(f"Image file not found: {self._image_path}")

        self._pub = self.create_publisher(Image, topic, 10)
        self._image_data = self._load_constant_image_bgr8()
        self._timer = self.create_timer(1.0 / rate_hz, self._publish_image)

        self.get_logger().info(
            f"Dummy camera publishing constant image {self._image_path} to {topic} at {rate_hz:.2f} Hz"
        )

    def _load_constant_image_bgr8(self) -> bytes:
        img = cv2.imread(str(self._image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {self._image_path}")
        img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return img.tobytes()

    def _publish_image(self) -> None:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.height = self._height
        msg.width = self._width
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = self._width * 3
        msg.data = self._image_data
        self._pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DummyCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
