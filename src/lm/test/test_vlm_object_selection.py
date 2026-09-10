"""Both object families use image+text selection and the same action protocol."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lm.keyframe_modes import PLANNER_KEYFRAMES, keyframe_phase, source_keyframe_name
from lm.vlm_service import VLMServiceNode, SYSTEM_PROMPT
from lm.vml import VLMClientNode


@pytest.mark.parametrize("kind", ["box", "bucket"])
def test_image_text_and_both_libraries_are_sent_to_model(kind):
    node = VLMServiceNode.__new__(VLMServiceNode)
    node._allowed_keyframes = list(PLANNER_KEYFRAMES)
    node._model_name = "test-model"
    node._ollama_client = Mock()
    decision = dict(next_keyframe=f"crouch_to_pick_{kind}", object_in_manipulation=True, task_completion=False)
    node._ollama_client.chat.return_value.message.content = json.dumps(decision)
    result, _, _ = node._query_vlm(np.zeros((16, 16, 3), dtype=np.uint8), f"Pick up the {kind}", "{}")
    assert result.next_keyframe == decision["next_keyframe"]
    args = node._ollama_client.chat.call_args.kwargs
    assert args["messages"][1]["images"]
    assert f"Pick up the {kind}" in args["messages"][1]["content"]
    assert set(args["format"]["properties"]["next_keyframe"]["enum"]) == set(PLANNER_KEYFRAMES)
    assert len(PLANNER_KEYFRAMES) == 14
    assert set(args["format"]["properties"]) == {"next_keyframe", "object_in_manipulation", "task_completion"}
    assert "right" in SYSTEM_PROMPT and "_bucket" in SYSTEM_PROMPT


def test_model_cannot_change_object_family_during_task():
    node = VLMServiceNode.__new__(VLMServiceNode)
    node._allowed_keyframes = list(PLANNER_KEYFRAMES)
    node._model_name = "test"
    node._ollama_client = Mock()
    node._ollama_client.chat.return_value.message.content = json.dumps(dict(
        next_keyframe="stand_after_place_box", object_in_manipulation=True, task_completion=True))
    with pytest.raises(ValueError, match="switch object libraries"):
        node._query_vlm(np.zeros((16, 16, 3), dtype=np.uint8), "Move bucket", '{"selected_object_type":"bucket"}')


@pytest.mark.parametrize("name", PLANNER_KEYFRAMES)
def test_policy_mask_and_approach_alias_for_each_family(name):
    assert VLMClientNode._effective_object_to_manipulate(
        SimpleNamespace(next_keyframe=name, object_in_manipulation=False)) == (keyframe_phase(name) != "approach")
    if name.startswith("approach_"):
        assert source_keyframe_name(name) == name.replace("approach_", "stand_before_pick_")


def test_planner_also_rejects_cross_family_goal_before_retargeting():
    node = VLMClientNode.__new__(VLMClientNode)
    node._selected_object_type = "bucket"
    node.publish_status = Mock()
    assert not node.publish_planner_outputs(SimpleNamespace(next_keyframe="crouch_to_pick_box"))
    assert node.publish_status.call_args.args[0] == "object_type_mismatch"
