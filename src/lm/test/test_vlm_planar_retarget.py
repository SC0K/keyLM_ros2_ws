"""VLM rigid placement preserves reference motion; standing goals share test defaults."""

from io import BytesIO
import json
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


@pytest.mark.parametrize("kind", ["box", "bucket"])
@pytest.mark.parametrize("phase", ["approach", "stand_before_pick", "crouch_to_pick", "stand_after_pick",
                                  "stand_before_place", "crouch_to_place", "stand_after_place"])
def test_named_object_libraries_route_and_preserve_authored_geometry(retargeter, kind, phase):
    from lm.keyframe_modes import keyframe_phase, keyframe_object_type
    name = f"{phase}_{kind}"
    source = retargeter._load_payload(name)
    blob, info = retargeter._process_keyframe(name, True)
    assert keyframe_phase(name) == phase and keyframe_object_type(name) == kind
    assert json.loads(info)["object_type"] == kind
    hands = [retargeter._ik_model.body(i).name for i in retargeter._ik_ee_body_ids]
    assert len(hands) == (1 if kind == "bucket" else 2)
    if kind == "bucket":
        assert hands[0].startswith("right_")
    with np.load(BytesIO(blob), allow_pickle=True) as result:
        assert bool(result["object_to_manipulate"][0]) == (phase != "approach")
        names = list(source["body_names"])
        assert result["body_positions"][names.index("pelvis"), 2] == source["body_positions"][names.index("pelvis"), 2]
        if phase != "approach":
            assert result["object_position_xyz"][2] == source["object_position_xyz"][2]
        if phase not in ("stand_before_pick", "stand_after_place"):
            np.testing.assert_array_equal(result["dof_positions"], source["dof_positions"])
    retargeter._apply_box_ik.assert_not_called()


def test_bucket_conversion_keeps_original_joint_pose_and_heights(retargeter):
    for path in sorted((PYTHON_PACKAGE / "resource/test_sequence_bucket").glob("*.npz")):
        converted = retargeter._load_payload(path.stem)
        with np.load(path, allow_pickle=True) as raw:
            np.testing.assert_allclose(converted["dof_positions"], raw["qpos"][7:36], atol=1e-7)
            assert converted["object_position_xyz"][2] == raw["object_pos_w"][2]
            assert np.isclose(converted["body_positions"][list(converted["body_names"]).index("pelvis"), 2], raw["qpos"][2])
        assert len(converted["dof_names"]) == 29
        assert len(converted["body_names"]) > 14


def test_switching_library_resets_old_object_latches_and_hand_selection(retargeter):
    retargeter._fixed_start_box_center = np.ones(3)
    retargeter._select_keyframe_object("crouch_to_pick_bucket")
    assert retargeter._fixed_start_box_center is None
    assert len(retargeter._ik_ee_body_ids) == 1
    retargeter._fixed_target_box_center = np.ones(3)
    retargeter._select_keyframe_object("crouch_to_pick_box")
    assert retargeter._fixed_target_box_center is None
    assert len(retargeter._ik_ee_body_ids) == 2


@pytest.mark.parametrize("ik_enabled", [False, True])
@pytest.mark.parametrize("yaw", [0., .7, -1.4, np.pi])
def test_bucket_pickup_stand_keeps_library_lateral_offset_and_heading(retargeter, yaw, ik_enabled):
    retargeter._retarget_ik_enabled = ik_enabled
    retargeter._current_box_center = np.array([1.6, -.8, .05])
    retargeter._current_box_quat_wxyz = _yaw_to_quat_wxyz(yaw)
    # This box-style planner hint must not override the bucket stance.
    retargeter._target_root_center = np.array([10., 20., 30.])
    retargeter._target_root_quat_wxyz = _yaw_to_quat_wxyz(-2.)
    source = retargeter._load_payload("stand_before_pick_bucket")
    pelvis = list(source["body_names"]).index("pelvis")
    source_root = source["body_positions"][pelvis]
    source_root_rotation = _quat_wxyz_to_rotmat(source["body_rotations"][pelvis])
    source_root_yaw = np.arctan2(source_root_rotation[1, 0], source_root_rotation[0, 0])
    source_heading = _quat_wxyz_to_rotmat(_yaw_to_quat_wxyz(source_root_yaw))
    # Stance placement is planar; the generated stand keeps its own roll/pitch.
    source_offset = source_heading[:2, :2].T @ (source["object_position_xyz"] - source_root)[:2]
    source_box_rotation = _quat_wxyz_to_rotmat(source["object_quat_wxyz"])
    source_box_yaw = np.arctan2(source_box_rotation[1, 0], source_box_rotation[0, 0])
    blob, info = retargeter._process_keyframe("stand_before_pick_bucket", True)
    with np.load(BytesIO(blob), allow_pickle=True) as result:
        root = result["body_positions"][pelvis]
        root_rotation = _quat_wxyz_to_rotmat(result["body_rotations"][pelvis])
        offset = root_rotation[:2, :2].T @ (result["object_position_xyz"] - root)[:2]
        np.testing.assert_allclose(offset, source_offset, atol=1e-6)
        np.testing.assert_allclose(offset, [.40746197, -.16375582], atol=1e-6)
        expected_rotation = _quat_wxyz_to_rotmat(_yaw_to_quat_wxyz(yaw - source_box_yaw))
        np.testing.assert_allclose(root[:2], retargeter._current_box_center[:2]
                                   + expected_rotation[:2, :2] @ (source_root - source["object_position_xyz"])[:2], atol=1e-6)
        np.testing.assert_allclose(result["object_position_xyz"][:2], retargeter._current_box_center[:2])
        assert root[2] == source_root[2]
        assert result["object_position_xyz"][2] == source["object_position_xyz"][2]
        expected_angles = dict(zip(POLICY_JOINT_NAMES, retargeter._standing_default_angles + retargeter._standing_joint_delta))
        np.testing.assert_allclose(result["dof_positions"], [expected_angles[n] for n in result["dof_names"]])
        assert bool(result["object_to_manipulate"][0])
    assert json.loads(info)["mode"] == "generated_stationary_stand_bucket_library_placement"
    retargeter._apply_box_ik.assert_not_called()


def test_bucket_right_hand_target_ignores_box_dimensions(retargeter):
    from lm.keyframe_box_retarget import BoxFrame, grasp_body_names, _pick_existing_default_feet, retarget_qpos_for_box_grasp
    payload = retargeter._load_payload("crouch_to_pick_bucket")
    model, data = retargeter._ik_model, retargeter._ik_data
    qpos = retargeter._build_qpos_from_payload(payload)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    hands = [model.body(n).id for n in grasp_body_names(model, "bucket")]
    feet = [model.body(n).id for n in _pick_existing_default_feet(model)]
    current_hand = data.xpos[hands].copy()
    source = BoxFrame(center=payload["object_position_xyz"], quat_wxyz=payload["object_quat_wxyz"], size=np.array([.3, .3, .3]))
    target = BoxFrame(center=source.center.copy(), quat_wxyz=source.quat_wxyz.copy(), size=np.array([1.2, .05, .8]))
    result = retarget_qpos_for_box_grasp(model, data, qpos=qpos, source_box=source, target_box=target,
                                        hand_body_ids=hands, foot_body_ids=feet, object_type="bucket", preserve_root_height=True,
                                        source_forward_axis="y", source_up_axis="-z", target_forward_axis="x", target_up_axis="z")
    np.testing.assert_allclose(result.qpos, qpos)
    np.testing.assert_allclose(result.hand_targets, current_hand)


@pytest.mark.parametrize("yaw", [0., .7, -1.4, np.pi])
@pytest.mark.parametrize("target_xy", [[2., -1.], [-3., 4.]])
def test_bucket_before_place_heading_follows_bucket_not_bearing(retargeter, yaw, target_xy):
    source = retargeter._load_payload("stand_before_place_bucket")
    pelvis = list(source["body_names"]).index("pelvis")
    source_box_rotation = _quat_wxyz_to_rotmat(source["object_quat_wxyz"])
    source_yaw = np.arctan2(source_box_rotation[1, 0], source_box_rotation[0, 0])
    retargeter._target_box_quat_wxyz = _yaw_to_quat_wxyz(yaw)
    retargeter._target_box_center[:2] = target_xy
    # A conflicting root hint must not make the robot look toward the bucket.
    retargeter._target_root_quat_wxyz = _yaw_to_quat_wxyz(yaw + 1.2)
    blob, _ = retargeter._process_keyframe("stand_before_place_bucket", True)
    rotation = _quat_wxyz_to_rotmat(_yaw_to_quat_wxyz(yaw - source_yaw))
    with np.load(BytesIO(blob), allow_pickle=True) as result:
        np.testing.assert_allclose(
            _quat_wxyz_to_rotmat(result["body_rotations"][pelvis]),
            rotation @ _quat_wxyz_to_rotmat(source["body_rotations"][pelvis]), atol=1e-6)
        np.testing.assert_allclose(
            (result["body_positions"][pelvis] - result["object_position_xyz"])[:2],
            (rotation @ (source["body_positions"][pelvis] - source["object_position_xyz"]))[:2], atol=1e-6)
        np.testing.assert_array_equal(result["dof_positions"], source["dof_positions"])
        np.testing.assert_array_equal(result["body_positions"][:, 2], source["body_positions"][:, 2])
        assert result["object_position_xyz"][2] == source["object_position_xyz"][2]


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


@pytest.mark.parametrize("ik_enabled", [False, True])
def test_locomotion_approach_keeps_root_target_and_masks_goal_object(retargeter, ik_enabled):
    from crl_g1_goalcontroller_python.g1_keyframe_controller import (
        G1KeyframeController, FEATURE_BODY_NAMES, MUJOCO_JOINT_NAMES, goal_state_from_payload,
    )
    retargeter._retarget_ik_enabled = ik_enabled
    original = retargeter._load_payload("stand_before_pick")
    retargeter._target_root_center = retargeter._current_box_center.copy()
    retargeter._target_root_center[:2] -= .30 * _quat_wxyz_to_rotmat(retargeter._target_root_quat_wxyz)[:2, 0]
    blob, info = retargeter._process_keyframe("approach", True)
    with np.load(BytesIO(blob), allow_pickle=True) as payload:
        assert not bool(payload["object_to_manipulate"][0])
        np.testing.assert_array_equal(payload["dof_positions"], original["dof_positions"])
        np.testing.assert_array_equal(payload["object_position_xyz"], np.zeros(3))
        np.testing.assert_array_equal(payload["object_quat_wxyz"], np.zeros(4))
        goal = goal_state_from_payload(payload, retargeter._standing_default_angles)
    np.testing.assert_allclose(goal[0, :2], retargeter._target_root_center[:2], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(goal[0, :2] - retargeter._current_box_center[:2]), .30, atol=1e-6)
    assert goal[0, 2] == original["body_positions"][list(original["body_names"]).index("pelvis"), 2]
    assert json.loads(info)["object_to_manipulate"] is False
    controller = G1KeyframeController.__new__(G1KeyframeController)
    controller.model = retargeter._ik_model
    controller.data = mujoco.MjData(controller.model)
    controller.default_angles = retargeter._standing_default_angles
    controller.num_actions = 29
    controller.object_to_manipulate = False
    controller.walking_goal_joint_noise_std = 0.0
    controller.feature_body_ids = [controller.model.body(n).id for n in FEATURE_BODY_NAMES]
    controller.joint_qpos_adr = np.array([controller.model.joint(n).qposadr[0] for n in MUJOCO_JOINT_NAMES])
    controller.policy_to_mujoco = np.array([POLICY_JOINT_NAMES.index(n) for n in MUJOCO_JOINT_NAMES])
    walking_goal = controller._replace_walking_goal_pose_with_default_noise(goal)[0]
    np.testing.assert_allclose(walking_goal[:3], goal[0, :3], atol=1e-6)
    # The existing locomotion override makes the root upright, preserving yaw.
    np.testing.assert_allclose(walking_goal[3:9],
                               _quat_wxyz_to_rotmat(retargeter._target_root_quat_wxyz)[:2, :].reshape(-1), atol=1e-6)
    np.testing.assert_array_equal(walking_goal[9:38], np.zeros(29))
    np.testing.assert_array_equal(walking_goal[-7:], np.zeros(7))
    assert controller._goal_object_observation_mask(walking_goal) == 0.0
    retargeter._apply_box_ik.assert_not_called()


def test_stand_before_pick_is_restored_to_manipulation(retargeter):
    blob, _ = retargeter._process_keyframe("stand_before_pick", False)
    with np.load(BytesIO(blob), allow_pickle=True) as payload:
        assert bool(payload["object_to_manipulate"][0])
        assert np.linalg.norm(payload["object_quat_wxyz"]) > .99


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
