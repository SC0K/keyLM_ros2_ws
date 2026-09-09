"""GUI connection selection without a display, network, or subprocesses."""

from unittest.mock import Mock

import pytest

from lm.vlm_planner_app import VLMPlannerApp, build_arg_parser


def app_stub():
    app = VLMPlannerApp.__new__(VLMPlannerApp)
    app.args = build_arg_parser().parse_args([])
    app.proc = None
    app.tunnel_proc = Mock()
    app.tunnel_proc.poll.return_value = None
    app.tunnel_host, app.tunnel_remote_port = "tars", 11434
    app.server_selection = Mock()
    app.server_selection.get.return_value = "case"
    app.bottom_status, app.root, app._append_status, app._start_tunnel = Mock(), Mock(), Mock(), Mock()
    app._tunnel_check_id = "pending-check"
    return app


def test_dropdown_replaces_only_owned_tunnel_and_uses_case_profile():
    app = app_stub()
    old_tunnel = app.tunnel_proc
    app.args.host, app.args.remote_port = "custom", 1234
    app.args.local_port, app.args.user = 11435, "custom-user"
    app._select_server()
    old_tunnel.terminate.assert_called_once()
    old_tunnel.wait.assert_called_once_with(timeout=2.0)
    app.root.after_cancel.assert_called_once_with("pending-check")
    assert app.args.server == "case"
    assert (app.tunnel_host, app.tunnel_remote_port) == ("case.inf.ethz.ch", 8001)
    assert app.args.host is None and app.args.remote_port is None
    assert (app.args.local_port, app.args.user) == (11435, "custom-user")
    app._start_tunnel.assert_called_once()


@pytest.mark.parametrize("blocked_by", ["planner", "external_tunnel", "disabled"])
def test_dropdown_does_not_interrupt_planner_or_external_tunnel(monkeypatch, blocked_by):
    app = app_stub()
    old_tunnel = app.tunnel_proc
    if blocked_by == "planner":
        app.proc = Mock()
        app.proc.poll.return_value = None
    elif blocked_by == "external_tunnel":
        app.tunnel_proc = None
        monkeypatch.setattr("lm.vlm_planner_app._port_open", lambda *_: True)
    else:
        app.args.no_tunnel = True
    app._select_server()
    assert app.args.server == "tars"
    app.server_selection.set.assert_called_once_with("tars")
    old_tunnel.terminate.assert_not_called()
    app._start_tunnel.assert_not_called()


def test_retry_same_server_after_tunnel_failure(monkeypatch):
    app = app_stub()
    app.tunnel_proc.poll.return_value = 255
    app.server_selection.get.return_value = "tars"
    monkeypatch.setattr("lm.vlm_planner_app._port_open", lambda *_: False)
    app._select_server()
    app._start_tunnel.assert_called_once()
    assert (app.tunnel_host, app.tunnel_remote_port) == ("tars", 11434)


def test_selecting_active_server_does_not_restart_it():
    app = app_stub()
    app.server_selection.get.return_value = "tars"
    app._select_server()
    app.tunnel_proc.terminate.assert_not_called()
    app._start_tunnel.assert_not_called()
