from copy import deepcopy
from types import SimpleNamespace

import pytest

from evaluation.evaluate_box_distances import evaluate_distance, summary, sweep_initial_conditions
from evaluation.evaluate_box_policies import perturbations
import numpy as np


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


def test_endpoint_errors_include_later_failed_trials():
    row = dict(policy="test", trial=1, approach_success=True, recovery_reset=False,
               pick_success=False, place_goal_success=False, pick_place_success=False, end_to_end_success=False,
               approach=dict(policy="test", trial=1, success=True, reason="success", stage_errors={}, phases=[]),
               manipulation=dict(policy="test", trial=1, success=False, reason="later fall", stage_errors={},
                    phases=[dict(keyframe="crouch_to_pick_box", success=True, errors={"root_position_m": .12})]))
    s = summary([row])["test"]
    assert s["successful_keyframe_errors"]["pick"]["root_position_m"] == dict(n=1, mean=.12, std=None)
    assert s["successful_pick_place_errors"]["pick"] == {}
