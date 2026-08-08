from __future__ import annotations

from pathlib import Path

from swaag.config import load_config


def test_default_runtime_session_state_is_under_data_var() -> None:
    config = load_config(env={})
    assert config.sessions.root == Path("/data/var/swaag/sessions")


def test_wakeup_dispatcher_systemd_unit_matches_runtime_store() -> None:
    text = Path("deploy/systemd/swaag-wakeup-dispatcher.service").read_text(encoding="utf-8")
    assert "swaag.wakeup_dispatcher --poll-seconds 1" in text
    assert "SWAAG__SESSIONS__ROOT=/data/var/swaag/sessions" in text
    assert "Restart=always" in text
