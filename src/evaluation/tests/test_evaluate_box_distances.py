from copy import deepcopy
from types import SimpleNamespace

import pytest

from evaluation.evaluate_box_distances import evaluate_distance, summary, sweep_initial_conditions
from evaluation.evaluate_box_policies import perturbations
import numpy as np


@pytest.mark.parametrize("invalid_kind", ["missing", "duplicate_name"])
def test_policy_cli_rejects_invalid_paths_before_creating_output(tmp_path, monkeypatch, invalid_kind):
    from evaluation.evaluate_box_distances import main
    first = tmp_path / "model.onnx"
    paths = [first]
    if invalid_kind == "duplicate_name":
        first.touch()
        second = tmp_path / "other" / first.name
        second.parent.mkdir()
        second.touch()
        paths.append(second)
    output = tmp_path / "results"
    monkeypatch.setattr("sys.argv", ["evaluate_box_distances", "--output", str(output),
                                    "--policies", *map(str, paths)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("approach_ok", [False, True])
@pytest.mark.parametrize("nominal_x", [-2., -.15, -1.95])
def test_failed_approach_resets_but_successful_approach_continues(tmp_path, approach_ok, nominal_x):
    initial = perturbations(1, 20260911)[0]
    initial["root_pos"][0] += nominal_x + 2.
    initial["nominal_root_x_m"] = nominal_x
    initial["place_distance_m"] = 1.2
    calls = []

    def run(trial, reset_state=True):
        calls.append((deepcopy(trial), reset_state))
        first = len(calls) == 1
        return dict(policy="test", trial=1, success=not first, reason="test",
                    phases=[dict(keyframe="approach_box" if first else "stand_after_pick_box",
                                 success=approach_ok if first else True)],
                    stage_errors={})

    evaluator = SimpleNamespace(args=SimpleNamespace(phases=None), run_trial=run)
    row = evaluate_distance(evaluator, initial, tmp_path)
    assert row["approach_success"] == approach_ok
    assert row["end_to_end_success"] == approach_ok
    assert row["pick_place_success"]
    assert row["recovery_reset"] != approach_ok
    assert calls[1][1] != approach_ok
    assert calls[1][0]["root_pos"][0] == initial["root_pos"][0] - (0 if approach_ok else nominal_x)
    assert calls[1][0]["box_pos"] == initial["box_pos"]
    assert calls[1][0]["place_distance_m"] == 1.2
    result = summary([row])["test"]
    assert result["end_to_end_success"]["successes"] == int(approach_ok)
    assert result["pick_place_success"]["successes"] == 1


def test_approach_range_preserves_noise_and_pairing():
    original = perturbations(10, 20260911)
    varied = sweep_initial_conditions([.5, 2.3])
    assert varied == sweep_initial_conditions([.5, 2.3])
    for i, (a, b) in enumerate(zip(original, varied)):
        np.testing.assert_allclose(b["approach_distance_m"], .5+.2*i)
        np.testing.assert_allclose(b["root_pos"][0]-b["nominal_root_x_m"], a["root_pos"][0]+2)
        assert b["place_distance_m"] == round(.2*(i+1),1)
        for key in ("box_pos", "root_quat", "box_quat", "walking_noise_seed", "place_offset_xy_m"):
            assert a[key] == b[key]


def test_repeated_sweep_has_100_unique_resets_and_ten_of_each_distance():
    trials = sweep_initial_conditions([.5, 2.3], repetitions=10)
    assert len(trials) == 100
    assert trials[:10] == sweep_initial_conditions([.5, 2.3])
    assert trials == sweep_initial_conditions([.5, 2.3], repetitions=10)
    assert trials != sweep_initial_conditions([.5, 2.3], repetitions=10, seed=20260915)
    assert [t["trial"] for t in trials] == list(range(1, 101))
    assert len({tuple(t["box_pos"]) for t in trials}) == 100
    assert len({t["walking_noise_seed"] for t in trials}) == 100
    for case in range(10):
        for trial in trials[case::10]:
            np.testing.assert_allclose(trial["approach_distance_m"], .5 + .2 * case)
            assert trial["place_distance_m"] == round(.2 * (case + 1), 1)
            assert abs(trial["root_pos"][0] - trial["nominal_root_x_m"]) <= .02


@pytest.mark.parametrize("repetitions", [0, -1, 1.5])
def test_repeated_sweep_rejects_invalid_repetition_count(repetitions):
    with pytest.raises(ValueError):
        sweep_initial_conditions(repetitions=repetitions)


def test_endpoint_errors_include_later_failed_trials():
    row = dict(policy="test", trial=1, approach_success=True, recovery_reset=False,
               pick_success=False, place_goal_success=False, pick_place_success=False, end_to_end_success=False,
               approach=dict(policy="test", trial=1, success=True, reason="success", stage_errors={}, phases=[]),
               manipulation=dict(policy="test", trial=1, success=False, reason="later fall", stage_errors={},
                    phases=[dict(keyframe="crouch_to_pick_box", success=True, errors={"root_position_m": .12})]))
    s = summary([row])["test"]
    assert s["successful_keyframe_errors"]["pick"]["root_position_m"] == dict(n=1, mean=.12, std=None)
    assert s["successful_pick_place_errors"]["pick"] == {}
