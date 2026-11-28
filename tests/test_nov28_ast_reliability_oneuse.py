"""Offline safety and tamper tests for a distinct real AST-agent experiment."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import pytest

from scripts.demos.nov28_ast_reliability_oneuse import (
    ALLOWLIST, MISSING, SOURCES, prepare, preflight, sha,
)
from scripts.demos.verify_nov28_ast_reliability_oneuse import verify
from swaag.events import create_event


def source_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    for index, name in enumerate(SOURCES):
        source = repo / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("\n".join(f"def test_generated_{i}(): pass" for i in range(index+2)) + "\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "tests"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=OneUseTest",
                    "-c", "user.email=oneuse@example.invalid", "commit", "-qm", "initial"], check=True)
    return repo


def test_new_ast_task_preparation_freezes_all_authority_without_model(make_config, tmp_path):
    repo = source_repo(tmp_path)
    folder = tmp_path / "private"
    manifest = prepare(repo, folder)
    assert manifest["schema"] == "swaag-core-ast-holdout-oneuse-v1"
    assert manifest["tool_names"] == ALLOWLIST
    assert manifest["source_revision"]
    assert manifest["model_call_count_during_prepare"] == 0
    assert manifest["tool_side_effects_enabled"] is False
    assert list(manifest["sources"]) == list(SOURCES)
    assert MISSING in (folder / "task.txt").read_text()
    assert (folder / "config.toml").is_file()
    assert folder.stat().st_mode & 0o077 == 0
    assert preflight(repo, folder) == manifest
    assert not (folder / ".one_use_started.json").exists()
    assert not (folder / "sessions").exists()
    with pytest.raises(ValueError, match="fresh private"):
        prepare(repo, folder)
    with pytest.raises((OSError, ValueError, FileNotFoundError)):
        verify(repo, folder)  # A frozen manifest is NOT a finished agent test.


def test_modified_git_or_preexisting_model_effect_blocks_oneuse(tmp_path):
    repo = source_repo(tmp_path)
    folder = tmp_path / "private"
    prepare(repo, folder)
    changed=repo / SOURCES[0]
    changed.write_text("def tampered_original(): pass\n")
    with pytest.raises(ValueError, match="clean absolute"):
        preflight(repo, folder)
    subprocess.run(["git", "-C", str(repo), "checkout", "--", "tests"], check=True)
    (folder / ".one_use_started.json").write_text("{}")
    with pytest.raises(ValueError, match="effects may have occurred"):
        preflight(repo, folder)


def test_oracle_cannot_accept_forged_event_when_run_result_claims_success(tmp_path):
    repo = source_repo(tmp_path)
    folder = tmp_path / "private"
    manifest = prepare(repo, folder)
    (folder / ".one_use_started.json").write_text(json.dumps({
        "owner": "owner-20261128-swaag-core", "lease_id": "rb_synthetic_test_only"}))
    (folder / "console.log").write_text("synthetic unit test, no model execution")
    (folder / "run-result.json").write_text(json.dumps({
        "source_revision": manifest["source_revision"],
        "console_sha256": sha((folder / "console.log").read_bytes()),
        "exit_code": 0, "turn_finished": 1,
    }))
    target = folder / "sessions" / "session_fixture" / "complete_history.jsonl"
    target.parent.mkdir(parents=True)
    created = create_event(session_id="session_fixture",sequence=1,
        event_type="message_added",prev_hash=None,
        payload={"message": {"role":"user","created_at":"2026-11-28T00:00:00Z",
                             "content": "Original prompt"}})
    data = asdict(created)
    data["payload"]["message"]["content"] = "Forged tool result"
    target.write_text(json.dumps(data)+"\n")
    with pytest.raises(Exception, match="hash|schema|integrity"):
        verify(repo, folder)


def test_public_provider_endpoint_is_rejected_before_preparation(monkeypatch, tmp_path):
    import scripts.demos.nov28_ast_reliability_oneuse as scenario
    repo=source_repo(tmp_path)
    folder=tmp_path / "new_private"
    monkeypatch.setattr(scenario, "MODEL_URL", "https://public.example/v1")
    with pytest.raises(ValueError, match="numeric private IP"):
        scenario.prepare(repo,folder)
    assert not folder.exists()
