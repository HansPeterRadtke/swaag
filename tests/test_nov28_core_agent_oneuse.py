"""Offline source/authority contracts for fresh read-only core-agent demo."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts.demos.nov28_core_agent_oneuse import (
    BROKER_OWNER, MISSING, SOURCES, ALLOWED_TOOLS,
    broker_admitted, check_preflight, prepare, run_once, sha,
)
from scripts.demos.verify_nov28_core_agent_oneuse import read_recorded


def _fresh_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    for name, func in SOURCES.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"def {func}():\n    assert 1 == 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "tests"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=ReadOnlyTest",
                    "-c", "user.email=readonly@example.invalid", "commit", "-qm", "source"], check=True)
    return repo


def test_prepare_freezes_new_private_task_source_and_readonly_authority(tmp_path):
    repo = _fresh_repo(tmp_path)
    folder = tmp_path / "private"
    data = prepare(repo, folder)
    assert data["tool_names"] == ALLOWED_TOOLS
    assert data["native_transport"] == "/chat/completions"
    assert data["allow_side_effect_tools"] is False
    assert data["source_revision"]
    assert folder.stat().st_mode & 0o077 == 0
    assert (folder / "manifest.json").stat().st_mode & 0o077 == 0
    assert check_preflight(repo, folder) == data
    assert "read_file" in (folder / "fixed-task.txt").read_text()
    assert MISSING in (folder / "fixed-task.txt").read_text()
    assert "allow_side_effect_tools = false" in (folder / "read-only-config.toml").read_text()
    with pytest.raises(ValueError, match="Fresh absolute private"):
        prepare(repo, folder)
    assert not (folder / "sessions").exists()


def test_changed_source_and_already_started_run_are_refused(tmp_path):
    repo = _fresh_repo(tmp_path)
    folder = tmp_path / "private"
    prepare(repo, folder)
    (repo / next(iter(SOURCES))).write_text("def tampered(): pass\n")
    with pytest.raises(ValueError, match="uncommitted changes"):
        check_preflight(repo, folder)
    subprocess.run(["git", "-C", str(repo), "checkout", "--", "tests"], check=True)
    marker = folder / ".one_use_started.json"
    marker.write_text("{}")
    with pytest.raises(ValueError, match="run may have occurred"):
        check_preflight(repo, folder)
    assert not (folder / "result.json").exists()


def test_broker_verification_never_promotes_queued_or_other_owner(monkeypatch):
    lease = "rb_fresh_test_123"
    import time
    states = [
        {"request_id": lease, "owner": BROKER_OWNER, "state": "active", "expires_at": time.time() + 3600},
        {"request_id": lease, "owner": BROKER_OWNER, "state": "queued", "expires_at": time.time() + 3600},
        {"request_id": lease, "owner": "someone-else", "state": "active", "expires_at": time.time() + 3600},
        {"request_id": lease, "owner": BROKER_OWNER, "state": "active", "expires_at": time.time() - 1},
    ]
    def fake_status(record):
        return json.dumps({"machines": {"thor": {"requests": [record]}}}).encode()
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: fake_status(states[0]).decode())
    assert broker_admitted(lease)["lease_id"] == lease
    for denied in states[1:]:
        monkeypatch.setattr(subprocess, "check_output", lambda *args, obj=denied, **kwargs: fake_status(obj).decode())
        with pytest.raises(ValueError, match="not active|expired"):
            broker_admitted(lease)


def test_independent_oracle_cannot_run_on_prepared_untouched_source(tmp_path):
    repo = _fresh_repo(tmp_path)
    folder = tmp_path / "private"
    prepare(repo, folder)
    with pytest.raises(ValueError, match="has not finished"):
        read_recorded(repo, folder)


def test_independent_verifier_rejects_tampered_durable_source_event(tmp_path):
    """A finished-looking process receipt cannot excuse a corrupted journal."""
    from dataclasses import asdict
    from swaag.events import create_event
    repo = _fresh_repo(tmp_path)
    folder = tmp_path / "private"
    manifest = prepare(repo, folder)
    (folder / ".one_use_started.json").write_text(json.dumps({
        "owner": BROKER_OWNER, "lease_id": "rb_synthetic_unit_only"}))
    (folder / "real-cli-console.log").write_text("synthetic test, not an actual run\n")
    (folder / "result.json").write_text(json.dumps({
        "source_revision": manifest["source_revision"],
        "console_sha256": sha((folder / "real-cli-console.log").read_bytes()),
        "child_returncode": 0,
        "terminal_turns": 1,
    }))
    log = folder / "sessions" / "session_synthetic" / "complete_history.jsonl"
    log.parent.mkdir(parents=True)
    event = create_event(session_id="session_synthetic", sequence=1,
        event_type="message_added", prev_hash=None,
        payload={"message": {"role": "user", "content": "actual original",
                             "created_at": "2026-11-28T00:00:00Z"}})
    tampered = asdict(event)
    tampered["payload"]["message"]["content"] = "forged source while retaining event hash"
    log.write_text(json.dumps(tampered) + "\n")
    with pytest.raises(Exception, match="hash|schema|integrity|Event"):
        read_recorded(repo, folder)
