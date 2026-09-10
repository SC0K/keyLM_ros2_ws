"""Moving the object before pickup refreshes goals without moving the destination."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from builtin_interfaces.msg import Time

from lm.keyframe_retargeter_node import KeyframeRetargeterNode
from lm.vml import VLMClientNode


def pose(center, quat=(1., 0., 0., 0.)):
    return VLMClientNode._pose_stamped_from(np.array(center), np.array(quat), Time(), 'world')


@pytest.mark.parametrize('kind', ['box', 'bucket'])
@pytest.mark.parametrize('phase', ['stand_before_pick', 'crouch_to_pick', 'stand_after_pick'])
def test_planner_sends_live_pose_only_before_lift(kind, phase):
    start = np.array([.4, 0., .15])
    live = np.array([.7, .2, .15])
    quat = np.array([1., 0., 0., 0.])
    node = SimpleNamespace(
        _selected_object_type=kind, mocap_object_selection=False,
        _has_actual_box_pose=True, _current_box_pose_stamp=Time(),
        _current_box_frame_id='world', _current_box_center=live,
        _current_box_quat_wxyz=quat, _task_target_box_center=np.array([1.4, 0., .15]),
        _task_target_box_quat_wxyz=quat, _default_target_box_quat_wxyz=quat,
        _default_target_root_center=np.zeros(3), _default_target_root_quat_wxyz=quat,
        box_forward_axis='x', get_logger=Mock(), publish_status=Mock(),
        _effective_object_to_manipulate=lambda _: True,
        _update_box_forward_axis_from_robot_once=Mock(),
        _fixed_start_box_pose=lambda: (start.copy(), quat.copy()),
        _pose_stamped_from=VLMClientNode._pose_stamped_from,
        initialize_task_target_once=lambda _: True,
        _stand_before_pick_root_pose=lambda: (np.zeros(3), quat.copy()),
        request_retargeted_keyframe=Mock(return_value=None),
    )
    response = SimpleNamespace(next_keyframe=f'{phase}_{kind}', object_in_manipulation=True)
    VLMClientNode.publish_planner_outputs(node, response)
    sent = node.request_retargeted_keyframe.call_args.kwargs
    p = sent['current_box_pose'].pose.position
    np.testing.assert_allclose([p.x, p.y, p.z], start if phase == 'stand_after_pick' else live)
    p = sent['target_box_pose'].pose.position
    np.testing.assert_allclose([p.x, p.y, p.z], [1.4, 0., .15])


@pytest.mark.parametrize('kind', ['box', 'bucket'])
def test_retargeter_refreshes_each_pre_pick_request_but_preserves_lift_and_destination(kind):
    node = KeyframeRetargeterNode.__new__(KeyframeRetargeterNode)
    node._select_keyframe_object = Mock()
    node.get_logger = Mock()
    node._process_keyframe = Mock(return_value=(b'', '{}'))
    node._fixed_start_box_center = node._fixed_start_box_quat_wxyz = None
    node._fixed_target_box_center = node._fixed_target_box_quat_wxyz = None
    node._fixed_box_hold_forward_axis = None
    for phase, x in [('stand_before_pick', .4), ('stand_before_pick', .6),
                     ('crouch_to_pick', .7), ('stand_after_pick', .9)]:
        request = SimpleNamespace(
            keyframe_name=f'{phase}_{kind}', object_to_manipulate=True,
            current_box_pose=pose([x, .2, .15]), target_box_pose=pose([x + 1., 0., .15]),
            target_root_pose=pose([0., 0., .8]), box_forward_axis='x',
        )
        response = node._on_retarget_keyframe_request(request, SimpleNamespace())
        assert response.success, response.error_message
        np.testing.assert_allclose(node._current_box_center, [.4 if phase == 'stand_after_pick' else x, .2, .15])
        np.testing.assert_allclose(node._target_box_center, [1.4, 0., .15])


def test_standing_root_follows_live_box_instead_of_start_pose():
    node = SimpleNamespace(
        _has_actual_box_pose=True, _has_robot_root_pose=True, _has_monitor=False,
        _current_box_center=np.array([1., .2, .15]),
        _current_box_quat_wxyz=np.array([1., 0., 0., 0.]),
        _current_robot_center=np.array([0., .2, .8]),
        _box_size_xyz=np.array([.3, .3, .3]),
        _stand_before_pick_distance_m=.4, _min_stand_root_height_m=.7,
        _default_target_root_center=np.array([0., 0., .8]),
        _fixed_start_box_pose=lambda: (np.array([.4, 0., .15]), np.array([1., 0., 0., 0.])),
    )
    first, _ = VLMClientNode._stand_before_pick_root_pose(node)
    np.testing.assert_allclose(first, [.6, .2, .8])
    node._current_box_center[0] += .2
    second, _ = VLMClientNode._stand_before_pick_root_pose(node)
    np.testing.assert_allclose(second, [.8, .2, .8])
