"""USB publication tests without camera access, ROS discovery or robot motion."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from builtin_interfaces.msg import Time

from lm.scene_camera import SceneCameraNode
from lm.vlm_planner_app import PLANNER_EXTRA_DEFAULTS, build_arg_parser, planner_ros_arguments


def camera_stub():
    return SimpleNamespace(
        _usb_lock=threading.Lock(), _usb_latest=None, _usb_published_sequence=0,
        _usb_last_warning=0.0, _published_first_image=False,
        _frame_id="usb_camera_optical_frame", _pub=Mock(), get_logger=lambda: Mock(),
    )


def test_usb_publishes_actual_dimensions_and_capture_stamp_once():
    node = camera_stub()
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    node._usb_latest = (1, frame, Time(sec=123), time.monotonic())
    SceneCameraNode._publish_usb_image(node)
    msg = node._pub.publish.call_args.args[0]
    assert (msg.width, msg.height, msg.step, msg.encoding) == (3, 2, 9, "bgr8")
    assert bytes(msg.data) == frame.tobytes()
    assert msg.header.stamp.sec == 123
    assert msg.header.frame_id == "usb_camera_optical_frame"
    SceneCameraNode._publish_usb_image(node)
    assert node._pub.publish.call_count == 1


def test_usb_does_not_publish_missing_or_stale_frames():
    node = camera_stub()
    SceneCameraNode._publish_usb_image(node)
    node._usb_latest = (1, np.zeros((2, 3, 3), dtype=np.uint8), Time(), time.monotonic() - 3)
    SceneCameraNode._publish_usb_image(node)
    node._pub.publish.assert_not_called()


def test_capture_failure_clears_frame_and_releases_device():
    node = camera_stub()
    node._usb_stop = threading.Event()
    node._usb_latest = "old frame"
    capture = Mock()
    def fail_read():
        node._usb_stop.set()
        return False, None
    capture.read.side_effect = fail_read
    SceneCameraNode._capture_usb_images(node, capture)
    assert node._usb_latest is None
    capture.release.assert_called_once()


def test_real_gui_forwards_topics_and_geometry_to_planner():
    assert build_arg_parser().parse_args(["--real", "--no-tunnel"]).real
    assert not build_arg_parser().parse_args([]).real
    params = dict(monitor_topic="/g1_hardware/monitor", actual_box_pose_topic="/tracked_box",
                  tracking_error_topic="/tracking_errors", box_size_xyz=[0.3, 0.3, 0.3],
                  retargeted_info_topic="/custom/info", **PLANNER_EXTRA_DEFAULTS)
    node = SimpleNamespace(get_parameter=lambda key: SimpleNamespace(value=params[key]))
    args = planner_ros_arguments(node)
    assert 'monitor_topic:="/g1_hardware/monitor"' in args
    assert 'actual_box_pose_topic:="/tracked_box"' in args
    assert 'box_size_xyz:=[0.3, 0.3, 0.3]' in args
