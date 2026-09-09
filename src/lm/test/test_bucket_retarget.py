"""Rigid, right-hand-only retargeting of the actual bucket test sequence."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from lm.keyframe_box_retarget import (
    BoxFrame, grasp_body_names, retarget_qpos_for_box_grasp,
    _quat_wxyz_multiply, _quat_wxyz_to_rotmat,
)

ROS_ROOT = Path(__file__).resolve().parents[2] / "crl-humanoid-ros"
RESOURCE = ROS_ROOT / "crl_g1_goalcontroller_py/crl_g1_goalcontroller_python/resource"


@pytest.fixture
def bucket_frame(request):
    model = mujoco.MjModel.from_xml_path(str(
        ROS_ROOT / "crl_humanoid_commons/data/robots/g1_description/scene_crl_with_bucket.xml"
    ))
    with np.load(RESOURCE / "test_sequence_bucket_local" / f"{getattr(request, 'param', 2)}.npz",
                 allow_pickle=True) as payload:
        qpos = model.qpos0.copy()
        pelvis = payload["body_names"].tolist().index("pelvis")
        qpos[:3] = payload["body_positions"][pelvis]
        qpos[3:7] = payload["body_rotations"][pelvis]
        for name, value in zip(payload["dof_names"], payload["dof_positions"]):
            qpos[model.joint(str(name)).qposadr[0]] = value
        source = BoxFrame(payload["object_position_xyz"].copy(), np.full(3, 0.3),
                          payload["object_quat_wxyz"].copy())
    return model, qpos, source


def run(bucket_frame, target, **kwargs):
    model, qpos, source = bucket_frame
    for key, value in dict(source_forward_axis="x", source_up_axis="z",
                           target_forward_axis="x", target_up_axis="z").items():
        kwargs.setdefault(key, value)
    return retarget_qpos_for_box_grasp(
        model, mujoco.MjData(model), qpos=qpos, source_box=source, target_box=target,
        hand_body_ids=kwargs.pop("hand_body_ids", [model.body("right_flat_hand").id]),
        foot_body_ids=[model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")],
        object_type="bucket", **kwargs,
    )


def test_bucket_selects_only_right_hand(bucket_frame):
    model, _, source = bucket_frame
    assert grasp_body_names(model, "bucket") == ["right_flat_hand"]
    assert grasp_body_names(model, "box") == ["left_flat_hand", "right_flat_hand"]
    with pytest.raises(ValueError, match="right-hand"):
        run(bucket_frame, source, hand_body_ids=[model.body("left_flat_hand").id])
    with pytest.raises(ValueError, match="Expected 1 hand"):
        run(bucket_frame, source, hand_body_ids=[model.body(n).id for n in grasp_body_names(model, "box")])


@pytest.mark.parametrize("bucket_frame", [2, 3, 4, 5], indirect=True)
def test_bucket_identity_ignores_sizes_and_box_axes(bucket_frame):
    _, qpos, source = bucket_frame
    target = BoxFrame(source.center, np.array([0.9, 0.1, 0.8]), source.quat_wxyz)
    result = run(bucket_frame, target, source_forward_axis="y", source_up_axis="-z")
    np.testing.assert_array_equal(result.qpos, qpos)
    assert result.hand_targets.shape == (1, 3)
    assert result.source_forward_axis == result.target_forward_axis == "x"
    assert result.source_up_axis == result.target_up_axis == "z"


@pytest.mark.parametrize("bucket_frame", [2, 3, 4, 5], indirect=True)
@pytest.mark.parametrize("tilt", [0.0, 0.025])
def test_bucket_rigid_grasp_and_grounded_feet(bucket_frame, tilt):
    model, qpos, source = bucket_frame
    yaw = 0.35
    delta = _quat_wxyz_multiply(
        np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
        np.array([np.cos(tilt / 2), np.sin(tilt / 2), 0, 0]),
    )
    target = BoxFrame(source.center + [0.15, -0.1, 0.01 if tilt else 0], source.size,
                      _quat_wxyz_multiply(delta, source.quat_wxyz))
    result = run(bucket_frame, target)
    resized = run(bucket_frame, BoxFrame(target.center, np.array([0.8, 0.12, 0.6]), target.quat_wxyz))
    np.testing.assert_array_equal(result.qpos, resized.qpos)
    np.testing.assert_array_equal(result.hand_targets, resized.hand_targets)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    expected = (_quat_wxyz_to_rotmat(delta) @
                (data.xpos[model.body("right_flat_hand").id] - source.center) + target.center)
    np.testing.assert_allclose(result.hand_targets[0], expected, atol=1e-10)
    assert result.hand_residual_m < 0.001
    assert result.foot_residual_m < 0.001
    for joint in range(model.njnt):
        if model.joint(joint).name.startswith(("left_shoulder_", "left_elbow_", "left_wrist_")):
            adr = model.jnt_qposadr[joint]
            np.testing.assert_allclose(result.qpos[adr], qpos[adr], atol=1e-12)
