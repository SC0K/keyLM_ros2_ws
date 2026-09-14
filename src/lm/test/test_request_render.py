"""Request rendering must never fall back to cached camera frames."""
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock
import time

import pytest
from sensor_msgs.msg import Image
from lm_interfaces.srv import RenderImage
from lm.scene_camera import SceneCameraNode
from lm.vlm_service import VLMServiceNode


def server_with_render(result):
    node = VLMServiceNode.__new__(VLMServiceNode)
    node._image_wait_timeout_sec = 0.1
    node._render_client = Mock()
    node._render_client.wait_for_service.return_value = True
    future = Future()
    future.set_result(result)
    node._render_client.call_async.return_value = future
    node._bridge = Mock()
    node._copy_next_image_after = Mock(side_effect=AssertionError("Used streaming cache"))
    return node


def test_request_uses_render_response_even_with_old_cached_image():
    response = RenderImage.Response(success=True)
    response.image.header.frame_id = "current_box"
    response.image.header.stamp.sec = 123
    node = server_with_render(response)
    node._latest_image_bgr = "old bucket"
    node._bridge.imgmsg_to_cv2.return_value = "new box"
    image, stamp, frame = node._capture_request_image()
    assert (image, stamp.sec, frame) == ("new box", 123, "current_box")
    node._render_client.call_async.assert_called_once()


def test_render_failure_does_not_use_cached_frame():
    node = server_with_render(RenderImage.Response(success=False, error_message="stale state"))
    with pytest.raises(RuntimeError, match="stale state"):
        node._capture_request_image()
    node._copy_next_image_after.assert_not_called()


def test_render_timeout_does_not_use_cached_frame():
    node = server_with_render(None)
    future = Future()
    node._render_client.call_async.return_value = future
    with pytest.raises(RuntimeError, match="Timed out"):
        node._capture_request_image()
    assert future.cancelled()


@pytest.mark.parametrize("age,have_state,success", [(0.0, True, True), (5.0, True, False), (0.0, False, False)])
def test_camera_renders_only_with_recent_robot_and_object_state(age, have_state, success):
    camera = SceneCameraNode.__new__(SceneCameraNode)
    camera._have_monitor = camera._have_object_pose = have_state
    camera._last_monitor_time = time.monotonic()
    camera._last_object_time = time.monotonic() - age
    camera.get_parameter = Mock(return_value=SimpleNamespace(value=1.0))
    camera._render_mujoco_image = Mock(return_value=Image())
    response = camera._on_render_image(RenderImage.Request(), RenderImage.Response())
    assert response.success is success
    assert camera._render_mujoco_image.call_count == int(success)
