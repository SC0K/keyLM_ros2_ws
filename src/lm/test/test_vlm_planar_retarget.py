"""VLM rigid placement preserves reference motion; standing goals share test defaults."""

from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
import yaml

from lm.box_config import SOURCE_BOX_GEOMETRY, REAL_TARGET_BOX_GEOMETRY
from lm.box_orientation import apply_target_box_orientation_offset
from lm.keyframe_box_retarget import matched_box_rotation, _quat_wxyz_to_rotmat
from lm.generated_stand import POLICY_JOINT_NAMES, VLM_STANDING_LEAN_DEG, generated_stand_joint_delta
from lm.keyframe_retargeter_node import KeyframeRetargeterNode, _yaw_to_quat_wxyz
from lm.vml import VLMClientNode, published_goal_targets

SRC = Path(__file__).resolve().parents[2]
LIBRARY = SRC / "lm/keyframes"
PYTHON_PACKAGE = SRC / "crl-humanoid-ros/crl_g1_goalcontroller_py/crl_g1_goalcontroller_python"


@pytest.fixture
def retargeter():
    node = KeyframeRetargeterNode.__new__(KeyframeRetargeterNode)
    node._ik_model = mujoco.MjModel.from_xml_path(str(
        SRC / "crl-humanoid-ros/crl_humanoid_commons/data/robots/g1_description/g1_29dof_crl.xml"))
    node._ik_data = mujoco.MjData(node._ik_model)
    with open(PYTHON_PACKAGE / "config/g1_keyframe_tracking_obj.yaml") as stream:
        config = yaml.safe_load(stream)
    node._standing_default_angles = np.asarray(config["default_angles"], dtype=np.float32)
    node._standing_joint_delta = generated_stand_joint_delta(np.deg2rad(VLM_STANDING_LEAN_DEG))
    node._retarget_ik_enabled = False
    node._retarget_object_type = "box"
    node._source_box_forward_axis, node._source_box_up_axis = "y", "-z"
    node._box_hold_forward_axis, node._box_hold_up_axis = "x", "z"
    node._target_box_orientation_offset_rpy_deg = np.zeros(3)
    node._current_box_center = np.array([1.3, .1, 3.5])
    node._current_box_quat_wxyz = _yaw_to_quat_wxyz(.4)
    node._target_box_center = np.array([2.3, .6, 4.4])
    node._target_box_quat_wxyz = _yaw_to_quat_wxyz(.8)
    node._target_root_center = np.array([1., .4, .25])
    node._target_root_quat_wxyz = _yaw_to_quat_wxyz(.8)
    node._fixed_start_box_center = node._fixed_target_box_center = None
    node._has_current_box_pose = True
    node._library_dir = LIBRARY
    node.get_logger = lambda: Mock()
    node._apply_box_ik = Mock(side_effect=AssertionError("IK must not run"))
    return node


@pytest.mark.parametrize("ik_enabled", [False, True])
@pytest.mark.parametrize("name", [
    "stand_before_pick", "crouch_to_pick", "stand_after_pick",
    "stand_before_place", "crouch_to_place", "stand_after_place",
])
def test_all_vlm_goals_preserve_library_root_and_object_height(retargeter, name, ik_enabled):
    from lm.keyframe_box_retarget import _pick_existing_default_ee, _pick_existing_default_feet

    retargeter._retarget_ik_enabled = ik_enabled
    retargeter._source_box_size_xyz = np.asarray(SOURCE_BOX_GEOMETRY.size_xyz)
    retargeter._box_size_xyz = np.asarray(REAL_TARGET_BOX_GEOMETRY.size_xyz)
    retargeter._ik_ee_body_ids = [retargeter._ik_model.body(n).id for n in _pick_existing_default_ee(retargeter._ik_model)]
    retargeter._ik_foot_body_ids = [retargeter._ik_model.body(n).id for n in _pick_existing_default_feet(retargeter._ik_model)]
    retargeter._ik_foot_constraint_weight = 6.0
    retargeter._ik_max_foot_residual_m = .001
    retargeter._ik_max_residual_m = .01
    if ik_enabled:
        retargeter._apply_box_ik = KeyframeRetargeterNode._apply_box_ik.__get__(retargeter)
    original = retargeter._load_payload(name)
    pelvis = list(original["body_names"]).index("pelvis")
    source_root_z = original["body_positions"].reshape(-1, 3)[pelvis, 2]
    source_object_z = original["object_position_xyz"].reshape(-1, 3)[0, 2]
    blob, _ = retargeter._process_keyframe(name, True)
    with np.load(BytesIO(blob), allow_pickle=True) as result:
        assert result["body_positions"].reshape(-1, 3)[pelvis, 2] == source_root_z
        assert result["object_position_xyz"].reshape(-1, 3)[0, 2] == source_object_z


@pytest.mark.parametrize("ik_enabled", [False, True])
def test_root_only_placement_does_not_change_body_heights(retargeter, ik_enabled):
    retargeter._retarget_ik_enabled = ik_enabled
    payload = retargeter._load_payload("stand_after_pick")
    source_z = payload["body_positions"][..., 2].copy()
    retargeter._retarget_root_only(payload)
    np.testing.assert_array_equal(payload["body_positions"][..., 2], source_z)


@pytest.mark.parametrize("name", ["crouch_to_pick", "stand_after_pick", "stand_before_place", "crouch_to_place"])
def test_planar_preserves_joints_heights_and_relative_geometry(retargeter, name):
    original = retargeter._load_payload(name)
    blob, _ = retargeter._process_keyframe(name, True)
    with np.load(BytesIO(blob), allow_pickle=True) as result:
        np.testing.assert_array_equal(result["dof_positions"], original["dof_positions"])
        np.testing.assert_array_equal(result["body_positions"][..., 2], original["body_positions"][..., 2])
        np.testing.assert_array_equal(result["object_position_xyz"][..., 2], original["object_position_xyz"][..., 2])
        target = retargeter._current_box_center if name in ("crouch_to_pick", "stand_after_pick") else retargeter._target_box_center
        np.testing.assert_allclose(result["object_position_xyz"][:2], target[:2], atol=1e-6)
        # One rigid transform: all distances from bodies to the object unchanged.
        np.testing.assert_allclose(
            np.linalg.norm(result["body_positions"] - result["object_position_xyz"], axis=-1),
            np.linalg.norm(original["body_positions"] - original["object_position_xyz"], axis=-1), atol=1e-6)
    retargeter._apply_box_ik.assert_not_called()
    obj, _, _, _ = published_goal_targets(blob)
    assert obj[2] == original["object_position_xyz"][2]


@pytest.mark.parametrize("name", ["stand_before_pick", "stand_after_place"])
def test_vlm_stand_matches_test_generated_goal(retargeter, name):
    from crl_g1_goalcontroller_python.g1_keyframe_controller import (
        G1KeyframeController, FEATURE_BODY_NAMES, MUJOCO_JOINT_NAMES,
        goal_state_from_payload,
    )
    original = retargeter._load_payload(name)
    source_root_z = original["body_positions"].reshape(-1, 3)[list(original["body_names"]).index("pelvis"), 2]
    blob, _ = retargeter._process_keyframe(name, True)
    with np.load(BytesIO(blob), allow_pickle=True) as payload:
        actual = goal_state_from_payload(payload, retargeter._standing_default_angles).reshape(-1, 171)[0]
    controller = G1KeyframeController.__new__(G1KeyframeController)
    controller.model = retargeter._ik_model
    controller.data = mujoco.MjData(controller.model)
    controller.default_angles = retargeter._standing_default_angles
    controller.num_actions = 29
    controller.feature_body_ids = [controller.model.body(n).id for n in FEATURE_BODY_NAMES]
    controller.joint_qpos_adr = np.array([controller.model.joint(n).qposadr[0] for n in MUJOCO_JOINT_NAMES])
    controller.policy_to_mujoco = np.array([POLICY_JOINT_NAMES.index(n) for n in MUJOCO_JOINT_NAMES])
    expected = controller._default_goal_frame_from_reference(
        actual, joint_delta=retargeter._standing_joint_delta, root_height_m=float(source_root_z))
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(actual[:3], [1., .4, source_root_z])
    assert VLM_STANDING_LEAN_DEG == 5.0
    np.testing.assert_allclose(actual[9 + POLICY_JOINT_NAMES.index("waist_pitch_joint")], np.deg2rad(5.0), atol=1e-6)


@pytest.mark.parametrize("target_axis", ["x", "-x", "y", "-y"])
@pytest.mark.parametrize("name", ["crouch_to_pick", "stand_after_pick", "stand_before_place", "crouch_to_place"])
def test_no_ik_keeps_axis_alignment_and_physical_object_orientation(retargeter, name, target_axis):
    retargeter._box_hold_forward_axis = target_axis
    original = retargeter._load_payload(name)
    target_quat = (retargeter._current_box_quat_wxyz if name in ("crouch_to_pick", "stand_after_pick")
                   else retargeter._target_box_quat_wxyz)
    source_forward = matched_box_rotation(original["object_quat_wxyz"], "y", "-z")[:2, 0]
    target_forward = matched_box_rotation(target_quat, target_axis, "z")[:2, 0]
    yaw = np.arctan2(target_forward[1], target_forward[0]) - np.arctan2(source_forward[1], source_forward[0])
    expected_rotation = _quat_wxyz_to_rotmat(_yaw_to_quat_wxyz(yaw))
    for offset in (np.zeros(3), np.array([7., -11., 23.])):
        retargeter._target_box_orientation_offset_rpy_deg = offset
        blob, _ = retargeter._process_keyframe(name, True)
        with np.load(BytesIO(blob), allow_pickle=True) as result:
            np.testing.assert_array_equal(result["dof_positions"], original["dof_positions"])
            for before, after in zip(original["body_rotations"].reshape(-1, 4), result["body_rotations"].reshape(-1, 4)):
                np.testing.assert_allclose(_quat_wxyz_to_rotmat(after), expected_rotation @ _quat_wxyz_to_rotmat(before), atol=2e-6)
            np.testing.assert_allclose(
                _quat_wxyz_to_rotmat(result["object_quat_wxyz"]),
                _quat_wxyz_to_rotmat(apply_target_box_orientation_offset(target_quat, offset)), atol=2e-6)
    retargeter._apply_box_ik.assert_not_called()


def test_failure_diagnostics_identify_why_placement_is_repeated():
    node = VLMClientNode.__new__(VLMClientNode)
    node._last_action_name = "stand_before_place"
    node._tracking_errors = dict(mean_body_position_error_m=.01, root_position_error_m=.02,
                                root_orientation_error_rad=.03)
    node._mean_body_success_threshold_m = node._root_position_success_threshold_m = .3
    node._root_orientation_success_threshold_rad = .8
    node._object_position_success_threshold_m = .45
    node._object_error_to_last_target = lambda: (.6, 0.)
    node._distance_context = lambda: {"pick_within_horizontal_reach": True}
    assert node.evaluate_last_action_success() is False
    assert [k for k,v in node._action_success_checks.items() if not v["passed"]] == ["object_position_error_m"]
    node._object_error_to_last_target = lambda: (.01, 0.)
    assert node.evaluate_last_action_success() is True
