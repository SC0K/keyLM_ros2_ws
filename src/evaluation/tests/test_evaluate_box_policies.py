"""Protocol checks for the paired offline evaluation (no live ROS nodes)."""

import numpy as np
import pytest

from evaluation.evaluate_box_policies import perturbations, summarize, simulator_torque_limits, SCENE
from crl_g1_goalcontroller_python.g1_keyframe_controller import root_com_linear_velocity_b
import mujoco


def test_box_size_snapshot_preserves_physics_and_live_scene(tmp_path):
    from evaluation.evaluate_box_policies import snapshot_box_scene, sha256
    digest = sha256(SCENE)
    scene = snapshot_box_scene(tmp_path, [.3, .3, .3])
    original = mujoco.MjModel.from_xml_path(str(SCENE))
    cube = mujoco.MjModel.from_xml_path(str(scene))
    assert sha256(SCENE) == digest
    assert (original.nq, original.nv, original.nu) == (cube.nq, cube.nv, cube.nu)
    for name in ("box_geom", "target_object_geom"):
        np.testing.assert_allclose(cube.geom(name).size, [.15, .15, .15])
    # The resized target mocap marker has inferred mass but no dynamic DOFs.
    dynamic = original.body_mocapid < 0
    for attr in ("body_mass", "body_inertia"):
        np.testing.assert_allclose(getattr(cube, attr)[dynamic], getattr(original, attr)[dynamic])
    for attr in ("geom_friction", "geom_solref", "geom_solimp"):
        np.testing.assert_allclose(getattr(cube, attr), getattr(original, attr))
    assert cube.body("left_flat_hand").id > 0


@pytest.mark.parametrize("size", [[0, .3, .3], [-.3, .3, .3], [.3, .3], [float("nan"), .3, .3]])
def test_box_size_snapshot_rejects_invalid_dimensions(tmp_path, size):
    from evaluation.evaluate_box_policies import snapshot_box_scene
    with pytest.raises(ValueError):
        snapshot_box_scene(tmp_path, size)


def test_paired_perturbations_reproducible_bounded_and_keep_heights():
    trials = perturbations(10, 20260911)
    assert trials == perturbations(10, 20260911)
    assert trials != perturbations(10, 20260912)
    for trial in trials:
        assert np.max(np.abs(np.asarray(trial["root_pos"])[:2] - [-2., 0.])) <= .02
        assert np.max(np.abs(np.asarray(trial["box_pos"])[:2] - [.35, 0.])) <= .02
        assert trial["root_pos"][2] == .8 and trial["box_pos"][2] == .15
        for name in ("root_quat", "box_quat"):
            q = np.asarray(trial[name])
            assert abs(2 * np.arctan2(q[3], q[0])) <= np.deg2rad(2.)


def test_summary_excludes_failed_trials_and_masked_object_error():
    trials = []
    for index, success, error in [(1, True, .1), (2, True, .3), (3, False, 100.)]:
        trials.append(dict(policy="a", trial=index, success=success, reason="test", stage_errors={
            stage: dict(root_position_m=error, object_position_m=None if stage == "approach" else error)
            for stage in ("approach", "pick", "place")}))
    summary = summarize(trials)["a"]
    assert summary["successes"] == 2 and summary["success_rate"] == 2/3
    assert summary["successful_trial_stage_errors"]["pick"]["root_position_m"]["mean"] == .2
    assert summary["successful_trial_stage_errors"]["approach"]["object_position_m"]["mean"] is None
    assert summarize([dict(policy="b", trial=1, success=False, reason="fall", stage_errors={})])["b"]["successes"] == 0


def test_torque_limits_match_simulator_and_have_correct_canonical_order():
    limits = simulator_torque_limits()
    np.testing.assert_array_equal(limits[:6], [88., 88., 88., 139., 139., 50.])
    np.testing.assert_array_equal(limits[-4:], [5., 5., 5., 5.])


def test_near_box_reset_preserves_paired_perturbations():
    original = perturbations(10, 20260911)
    near = perturbations(10, 20260911, [-.05, 0., .8])
    for a, b in zip(original, near):
        np.testing.assert_allclose(np.asarray(b["root_pos"]) - a["root_pos"], [1.95, 0., 0.])
        for key in ("box_pos", "root_quat", "box_quat", "walking_noise_seed"):
            assert a[key] == b[key]


def test_summary_handles_successful_trial_without_approach():
    result = summarize([dict(policy="a", trial=1, success=True, reason="success",
                             stage_errors={"pick": {"root_position_m": .1},
                                           "place": {"root_position_m": .2}})])["a"]
    assert result["successes"] == 1
    assert result["successful_trial_stage_errors"]["approach"] == {}
    assert result["successful_trial_stage_errors"]["pick"]["root_position_m"] == dict(n=1, mean=.1, std=None)


def test_place_noise_is_paired_bounded_and_independent_of_reset_noise():
    original = perturbations(10, 20260911)
    noisy = perturbations(10, 20260911, place_noise_xy_m=.10)
    assert noisy == perturbations(10, 20260911, place_noise_xy_m=.10)
    offsets = []
    for a, b in zip(original, noisy):
        assert a["place_offset_xy_m"] == [0., 0.]
        for key in a.keys() - {"place_offset_xy_m"}:
            assert a[key] == b[key]
        offsets.append(b["place_offset_xy_m"])
    offsets = np.asarray(offsets)
    assert np.max(np.abs(offsets)) <= .10
    assert np.max(np.abs(offsets)) > .02
    assert len(np.unique(offsets, axis=0)) == 10


@pytest.mark.parametrize("bound", [-.1, float("nan"), float("inf")])
def test_place_noise_rejects_invalid_bounds(bound):
    with pytest.raises(ValueError, match="Placement XY noise"):
        perturbations(10, 20260911, place_noise_xy_m=bound)


def test_offline_origin_velocity_reproduces_mujoco_pelvis_com_observation():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    data.qpos[3:7] = [.92387953, 0., 0., .38268343]
    data.qvel[:6] = [.1, -.3, .2, .4, -.6, .7]
    mujoco.mj_forward(model, data)
    root = model.body("pelvis").id
    velocity = np.zeros(6)
    # mjOBJ_BODY local velocity uses the inertial frame, which has a small
    # rotation relative to the pelvis link. Convert world COM velocity instead.
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, root, velocity, 0)
    actual = root_com_linear_velocity_b(data.qpos[3:7], data.qvel[:3], data.qvel[3:6], model.body_ipos[root])
    expected = data.xmat[root].reshape(3, 3).T @ velocity[3:]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
