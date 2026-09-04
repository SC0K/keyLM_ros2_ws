"""Focused tests for semantic, grounded box-grasp retargeting."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from lm.box_config import (
    REAL_TARGET_BOX_GEOMETRY,
    SIM_TARGET_BOX_GEOMETRY,
    SOURCE_BOX_GEOMETRY,
)
from lm.keyframe_box_retarget import (
    BoxFrame,
    _pick_existing_default_ee,
    _pick_existing_default_feet,
    _quat_wxyz_normalize,
    _quat_wxyz_to_rotmat,
    box_size_in_matched_frame,
    map_points_by_closest_box_corner,
    matched_box_rotation,
    retarget_qpos_for_box_grasp,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _body_ids(model: mujoco.MjModel, names: list[str]) -> list[int]:
    return [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in names
    ]


def _qpos_from_payload(model: mujoco.MjModel, payload: object) -> np.ndarray:
    body_names = [str(name) for name in payload["body_names"].tolist()]
    pelvis_index = body_names.index("pelvis")
    body_positions = np.asarray(payload["body_positions"])
    body_rotations = np.asarray(payload["body_rotations"])
    if body_positions.ndim == 3:
        body_positions = body_positions[0]
        body_rotations = body_rotations[0]

    qpos = np.zeros(model.nq, dtype=np.float64)
    qpos[0:3] = body_positions[pelvis_index]
    root_quat = body_rotations[pelvis_index]
    qpos[3:7] = root_quat / np.linalg.norm(root_quat)
    dof_positions = np.asarray(payload["dof_positions"])
    if dof_positions.ndim == 2:
        dof_positions = dof_positions[0]
    qpos[7:7 + dof_positions.size] = dof_positions
    return qpos


def test_deployment_uses_stored_source_quaternion_without_training_offset() -> None:
    """Frame metadata and legacy markers must not rotate deployment inputs."""
    pytest.importorskip("rclpy")
    pytest.importorskip("lm_interfaces.srv")
    from lm.keyframe_retargeter_node import KeyframeRetargeterNode

    raw = np.array(
        [-0.03970517, 0.80428827, 0.59291060, 0.00023665],
        dtype=np.float64,
    )
    expected = raw / np.linalg.norm(raw)
    node = KeyframeRetargeterNode.__new__(KeyframeRetargeterNode)

    payloads = (
        {"object_quat_wxyz": raw.copy()},
        {
            "object_quat_wxyz": 2.0 * raw,
            "object_quat_frame": np.asarray("dataset_mesh"),
        },
        {
            "object_quat_wxyz": raw.copy(),
            "object_quat_frame": np.asarray("physical_box"),
            "object_mesh_offset_removed_rpy_deg": np.array([-6.0, -9.0, 26.0]),
        },
    )
    for payload in payloads:
        source_quat, _, _ = node._source_object_quat(payload)
        np.testing.assert_allclose(source_quat, expected, atol=1e-12)


def test_matched_frame_respects_axis_semantics_and_reorders_dimensions() -> None:
    """Signed semantic axes select the correct physical dimensions."""
    yaw = np.deg2rad(31.0)
    quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    box_rot = _quat_wxyz_to_rotmat(quat)
    matched = matched_box_rotation(quat, "-y", "z")

    np.testing.assert_allclose(matched[:, 0], -(box_rot[:, 1]), atol=1e-12)
    np.testing.assert_allclose(matched[:, 2], box_rot[:, 2], atol=1e-12)
    np.testing.assert_allclose(
        box_size_in_matched_frame(SIM_TARGET_BOX_GEOMETRY.size_xyz, "-y", "z"),
        np.asarray(SIM_TARGET_BOX_GEOMETRY.size_xyz)[[1, 0, 2]],
        atol=0.0,
    )


def test_retargeter_infers_reference_pickup_axis_and_clears_legacy_marker() -> None:
    """Physical output must be single-converted and use the reference approach axis."""
    pytest.importorskip("rclpy")
    pytest.importorskip("lm_interfaces.srv")
    from lm.keyframe_retargeter_node import KeyframeRetargeterNode

    node = KeyframeRetargeterNode.__new__(KeyframeRetargeterNode)
    node._source_box_up_axis = "-z"

    # The reference robot faces world +x. With a box yawed -90 degrees, its
    # local +y axis faces world +x and is therefore the reference pickup axis.
    source_box_quat = np.array(
        [np.cos(-np.pi / 4.0), 0.0, 0.0, np.sin(-np.pi / 4.0)],
        dtype=np.float64,
    )
    assert node._infer_source_forward_axis(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        source_box_quat,
    ) == "y"

    payload = {
        "object_position_xyz": np.zeros(3, dtype=np.float32),
        "object_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "object_quat_frame": np.asarray("physical_box"),
        "object_mesh_offset_removed_rpy_deg": np.array([-6.0, -9.0, 26.0]),
    }
    target_center = np.array([0.4, -0.1, 0.1425], dtype=np.float64)
    target_quat = np.array([0.9, 0.0, 0.0, 0.1], dtype=np.float64)
    node._write_physical_object_pose(payload, target_center, target_quat)

    np.testing.assert_allclose(payload["object_position_xyz"], target_center)
    np.testing.assert_allclose(payload["object_quat_wxyz"], target_quat)
    assert str(payload["object_quat_frame"]) == "physical_box"
    assert "object_mesh_offset_removed_rpy_deg" not in payload


@pytest.mark.parametrize(
    "target_geometry",
    (REAL_TARGET_BOX_GEOMETRY, SIM_TARGET_BOX_GEOMETRY),
    ids=("deployment_cube", "simulation_box"),
)
def test_pickup_retarget_keeps_both_feet_grounded(target_geometry) -> None:
    """Both deployment and simulation boxes preserve grounded foot contacts."""
    model = mujoco.MjModel.from_xml_path(
        str(PACKAGE_ROOT / "models" / "g1" / "g1_29dof.xml")
    )
    data = mujoco.MjData(model)
    with np.load(PACKAGE_ROOT / "keyframes" / "crouch_to_pick.npz", allow_pickle=True) as npz:
        payload = {key: npz[key] for key in npz.files}

    qpos = _qpos_from_payload(model, payload)
    source_box = BoxFrame(
        center=np.asarray(payload["object_position_xyz"], dtype=np.float64),
        size=np.asarray(SOURCE_BOX_GEOMETRY.size_xyz, dtype=np.float64),
        quat_wxyz=_quat_wxyz_normalize(payload["object_quat_wxyz"]),
    )
    target_box = BoxFrame(
        center=np.array([0.35, 0.0, 0.15], dtype=np.float64),
        size=np.asarray(target_geometry.size_xyz, dtype=np.float64),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    hand_names = _pick_existing_default_ee(model)
    foot_names = _pick_existing_default_feet(model)
    assert foot_names is not None
    hand_ids = _body_ids(model, hand_names)
    foot_ids = _body_ids(model, foot_names)

    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    source_hand_positions = np.asarray(data.xpos[hand_ids], dtype=np.float64).copy()
    source_foot_positions = np.asarray(data.xpos[foot_ids], dtype=np.float64).copy()

    no_op = retarget_qpos_for_box_grasp(
        model,
        data,
        qpos=qpos,
        source_box=source_box,
        target_box=source_box,
        hand_body_ids=hand_ids,
        foot_body_ids=foot_ids,
        source_forward_axis=SOURCE_BOX_GEOMETRY.forward_axis,
        source_up_axis=SOURCE_BOX_GEOMETRY.up_axis,
        target_forward_axis=SOURCE_BOX_GEOMETRY.forward_axis,
        target_up_axis=SOURCE_BOX_GEOMETRY.up_axis,
    )
    np.testing.assert_array_equal(no_op.qpos, qpos)
    np.testing.assert_array_equal(no_op.hand_targets, source_hand_positions)
    np.testing.assert_array_equal(no_op.foot_targets, source_foot_positions)

    result = retarget_qpos_for_box_grasp(
        model,
        data,
        qpos=qpos,
        source_box=source_box,
        target_box=target_box,
        hand_body_ids=hand_ids,
        foot_body_ids=foot_ids,
        source_forward_axis=SOURCE_BOX_GEOMETRY.forward_axis,
        source_up_axis=SOURCE_BOX_GEOMETRY.up_axis,
        target_forward_axis=target_geometry.forward_axis,
        target_up_axis=target_geometry.up_axis,
    )

    np.testing.assert_allclose(result.source_matched_size, SOURCE_BOX_GEOMETRY.size_xyz)
    np.testing.assert_allclose(result.target_matched_size, target_geometry.size_xyz)
    assert result.hand_residual_m < 1e-3
    assert result.foot_residual_m < 1e-4
    np.testing.assert_allclose(result.foot_positions, result.foot_targets, atol=1e-5)
    np.testing.assert_allclose(
        result.foot_targets[:, 2],
        source_foot_positions[:, 2],
        atol=1e-12,
    )
    assert np.all(result.foot_positions[:, 2] >= 0.0)

    source_yaw = -result.yaw_delta_rad
    source_ground_box = BoxFrame(
        center=source_box.center,
        size=result.source_matched_size,
        quat_wxyz=np.array(
            [np.cos(source_yaw / 2.0), 0.0, 0.0, np.sin(source_yaw / 2.0)]
        ),
    )
    target_ground_box = BoxFrame(
        center=target_box.center,
        size=result.target_matched_size,
        quat_wxyz=target_box.quat_wxyz,
    )
    source_normalized = (
        source_ground_box.world_to_local(source_hand_positions)
        / source_ground_box.half_extents
    )
    target_normalized = (
        target_ground_box.world_to_local(result.hand_targets)
        / target_ground_box.half_extents
    )
    np.testing.assert_allclose(target_normalized, source_normalized, atol=1e-12)

    root_delta = _quat_wxyz_to_rotmat(result.qpos[3:7]) @ _quat_wxyz_to_rotmat(qpos[3:7]).T
    expected_delta = np.array(
        [
            [np.cos(result.yaw_delta_rad), -np.sin(result.yaw_delta_rad), 0.0],
            [np.sin(result.yaw_delta_rad), np.cos(result.yaw_delta_rad), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(root_delta, expected_delta, atol=1e-6)

    mapped_root, _ = map_points_by_closest_box_corner(
        source_ground_box,
        target_ground_box,
        qpos[0:3],
    )
    np.testing.assert_allclose(result.qpos[0:2], mapped_root[0:2], atol=1e-12)
    rigid_foot_targets = (
        (source_foot_positions - qpos[0:3]) @ expected_delta.T
        + np.array([mapped_root[0], mapped_root[1], qpos[2]])
    )
    np.testing.assert_allclose(result.foot_targets, rigid_foot_targets, atol=1e-12)
