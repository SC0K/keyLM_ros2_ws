"""Connection selection tests; do not open SSH connections or send VLM requests."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from lm.vlm_connection import ssh_tunnel_command
from lm.vlm_planner_app import build_arg_parser
from lm.vlm_service import KeyframeDecision, SYSTEM_PROMPT, VLMServiceNode


def test_tars_is_default():
    args = build_arg_parser().parse_args([])
    assert args.server == "tars"
    cmd = ssh_tunnel_command(args.server, host=args.host, remote_port=args.remote_port)
    assert cmd[-2:] == ["11434:localhost:11434", "sitchen@tars"]


def test_case_profile_and_explicit_overrides():
    args = build_arg_parser().parse_args(["--server", "case"])
    assert ssh_tunnel_command(args.server)[-2:] == [
        "11434:localhost:8001", "sitchen@case.inf.ethz.ch"
    ]
    cmd = ssh_tunnel_command("case", host="custom-host", remote_port=12345,
                             local_port=11435, user="tester")
    assert cmd[-2:] == ["11435:localhost:12345", "tester@custom-host"]
    assert build_arg_parser().parse_args(["--no-tunnel"]).no_tunnel


def test_tailscale_profile_uses_its_own_user_and_ollama_port():
    args = build_arg_parser().parse_args(["--server", "tailscale"])
    assert args.user is None
    assert ssh_tunnel_command(args.server, user=args.user)[-2:] == [
        "11434:localhost:11434", "sitongchen@100.99.254.46"]
    assert ssh_tunnel_command("tailscale", user="custom", local_port=11435)[-2:] == [
        "11435:localhost:11434", "custom@100.99.254.46"]
    # Selecting another profile must restore its user, not retain sitongchen.
    args.server = "tars"
    assert ssh_tunnel_command(args.server, user=args.user)[-1] == "sitchen@tars"


def test_query_uses_configured_client_and_model():
    node = VLMServiceNode.__new__(VLMServiceNode)
    node._model_name = "server-specific-model"
    node._allowed_keyframes = ["stand_before_pick_box"]
    node._ollama_client = Mock()
    node._ollama_client.chat.return_value = SimpleNamespace(message=SimpleNamespace(
        content='{"next_keyframe":"stand_before_pick_box",'
                '"object_in_manipulation":true,"task_completion":false}'
    ))
    decision, _, _ = node._query_vlm(np.zeros((8, 8, 3), dtype=np.uint8), "pick", "{}")
    assert decision.next_keyframe == "stand_before_pick_box"
    assert node._ollama_client.chat.call_args.kwargs["model"] == "server-specific-model"
    messages = node._ollama_client.chat.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "Planner context JSON:\n{}" in messages[1]["content"]


def test_prompt_examples_require_placement_before_final_stand():
    examples = [KeyframeDecision.model_validate_json(line)
                for line in SYSTEM_PROMPT.splitlines() if line.startswith('{"next_keyframe"')]
    assert [(example.next_keyframe, example.task_completion) for example in examples] == [
        ("crouch_to_place_box", False),
        ("stand_after_place_bucket", False),
        ("stand_before_place_bucket", False),
        ("stand_after_place_box", True),
    ]
    assert all(example.object_in_manipulation for example in examples)
    assert "NEVER transition directly from stand_before_place to" in SYSTEM_PROMPT
    assert "previous_action is stand_after_place" in SYSTEM_PROMPT
