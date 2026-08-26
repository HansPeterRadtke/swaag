from __future__ import annotations

from pathlib import Path

import pytest

from swaag.environment.artifacts import TextArtifactStore
from swaag.tools.base import ToolValidationError
from swaag.tools.registry import ToolRegistry
from swaag.types import SessionState
from swaag.utils import sha256_text


def _state(session_id: str = "session_artifacts") -> SessionState:
    return SessionState(
        session_id=session_id,
        created_at="now",
        updated_at="now",
        config_fingerprint="cfg",
        model_base_url="http://model",
    )


def test_text_artifact_store_round_trip(tmp_path: Path) -> None:
    store = TextArtifactStore(tmp_path, "session_a")
    text = "alpha\nbeta\ngamma\n"
    artifact = store.create(text, kind="shell_stdout")

    first = store.read(artifact.artifact_id, start_offset=0, max_chars=6)
    second = store.read(artifact.artifact_id, start_offset=first["next_offset"], max_chars=100)

    assert artifact.sha256 == sha256_text(text)
    assert first["text"] == "alpha\n"
    assert first["finished"] is False
    assert second["text"] == "beta\ngamma\n"
    assert second["finished"] is True
    assert first["total_chars"] == len(text)


def test_text_artifact_store_rejects_path_escape(tmp_path: Path) -> None:
    store = TextArtifactStore(tmp_path, "session_a")
    with pytest.raises(ValueError):
        store.get("../outside")


def test_text_artifact_store_rejects_tampered_metadata_path(tmp_path: Path) -> None:
    store = TextArtifactStore(tmp_path, "session_a")
    artifact = store.create("safe", kind="test")
    metadata = Path(artifact.metadata_path)
    import json
    payload = json.loads(metadata.read_text())
    payload["path"] = str(tmp_path / "outside.txt")
    metadata.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        store.get(artifact.artifact_id)


def test_text_artifact_store_archive_is_exact_and_read_only(tmp_path: Path) -> None:
    store = TextArtifactStore(tmp_path, "session_a")
    artifact = store.create("durable exact artifact", kind="test")

    assert store.archive() == 1
    assert store.archive() == 1
    archived = store.get(artifact.artifact_id)

    assert store.read(artifact.artifact_id, max_chars=100)["text"] == "durable exact artifact"
    assert Path(archived.path).is_relative_to(tmp_path / "archives" / "artifacts")
    assert Path(archived.path).stat().st_mode & 0o222 == 0
    with pytest.raises(RuntimeError, match="archived session"):
        store.create("late", kind="test")


def test_text_artifact_store_rejects_tampered_content(tmp_path: Path) -> None:
    store = TextArtifactStore(tmp_path, "session_a")
    artifact = store.create("safe", kind="test")
    Path(artifact.path).write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity"):
        store.get(artifact.artifact_id)


def test_read_artifact_tool_is_bounded_by_reader_config(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.reader.max_chunk_chars = 5
    state = _state()
    artifact = TextArtifactStore(config.sessions.root, state.session_id).create("0123456789", kind="test")

    _, result = ToolRegistry().dispatch(
        "read_artifact",
        {"artifact_id": artifact.artifact_id, "start_offset": 2, "max_chars": 100},
        config,
        state,
    )

    assert result.output["text"] == "23456"
    assert result.output["next_offset"] == 7
    assert result.output["finished"] is False
    assert [event.event_type for event in result.generated_events] == ["artifact_read"]


def test_read_artifact_validation(make_config) -> None:
    tool = ToolRegistry().get("read_artifact")
    with pytest.raises(ToolValidationError):
        tool.validate({"artifact_id": ""})
    with pytest.raises(ToolValidationError):
        tool.validate({"artifact_id": "x", "start_offset": -1})


def test_read_artifact_enabled_by_default(make_config) -> None:
    assert "read_artifact" in ToolRegistry().tool_names(make_config())
