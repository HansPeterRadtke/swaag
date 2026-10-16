from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from swaag.environment.artifacts import TextArtifactStore
from swaag.environment.capture import capture_command
from swaag.tools.base import ToolExecutionError, ToolValidationError
from swaag.tools.registry import ToolRegistry
from tests.test_tools import _empty_state


def workspace_call(config, operation, **inputs):
    return ToolRegistry().dispatch("agent_workspace", {"operation": operation, **inputs}, config, _empty_state())[1]


def test_sandbox_retains_output_tails_both_streams_and_nonzero_exit(make_config, tmp_path):
    config = make_config(agent_data__root=tmp_path / "agent", agent_data__max_capture_chars=23)
    result = workspace_call(config, "shell", command="printf '%0200d\\nTAIL_OUT\\n' 0; printf '%0200d\\nTAIL_ERR\\n' 0 >&2; exit 7")
    assert result.output["return_code"] == 7
    assert result.output["status"] == "failed" and result.output["capture_complete"]
    store = TextArtifactStore(config.sessions.root, _empty_state().session_id)
    for stream, tail in [("stdout", "TAIL_OUT"), ("stderr", "TAIL_ERR")]:
        assert result.output[stream + "_truncated"]
        artifact = store.get(result.output[stream + "_artifact_id"])
        raw = Path(artifact.path).read_bytes()
        assert raw == ("0" * 200 + "\n" + tail + "\n").encode()
        assert artifact.sha256 == hashlib.sha256(raw).hexdigest()
        page = store.read(artifact.artifact_id, start_offset=201, max_chars=50)
        assert page["text"] == tail + "\n" and page["finished"]
    assert sum(e.event_type == "artifact_created" for e in result.generated_events) == 2


def test_artifact_streaming_preserves_raw_bytes_and_unicode_offsets(tmp_path, monkeypatch):
    raw = ("ä中🙂\r\n" * 17000).encode() + b"\xff\r\nEND"
    source = tmp_path / "source.raw"; source.write_bytes(raw)
    store = TextArtifactStore(tmp_path / "sessions", "test")
    old_read_text = Path.read_text
    def no_full_artifact_read(path, *args, **kwargs):
        if path.suffix == ".txt":
            raise AssertionError("Artifact payload must be streamed, not read wholesale")
        return old_read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", no_full_artifact_read)
    artifact = store.create_from_file(source, kind="binary-output")
    assert Path(artifact.path).read_bytes() == raw
    text = raw.decode("utf-8", errors="replace")
    page = store.read(artifact.artifact_id, start_offset=len(text) - 8, max_chars=50)
    assert page["text"] == text[-8:] and page["finished"]
    assert page["sha256"] == hashlib.sha256(raw).hexdigest()
    with Path(artifact.path).open("r+b") as out:
        out.seek(17); out.write(b"x")
    with pytest.raises(ValueError, match="integrity"):
        store.read(artifact.artifact_id)


@pytest.mark.parametrize("bytes_to_write,complete", [(4096, True), (4097, False)])
def test_capture_byte_limit_is_explicit_and_exact(tmp_path, bytes_to_write, complete):
    store = TextArtifactStore(tmp_path, "capture")
    output, _ = capture_command(
        [sys.executable, "-c", f"import sys;sys.stdout.buffer.write(b'x'*{bytes_to_write})"],
        store=store, timeout_seconds=5, max_output_bytes=4096, preview_chars=13,
        env={}, kind="test",
    )
    assert output["capture_complete"] is complete
    assert output["status"] == ("completed" if complete else "output_limit_exceeded")
    artifact = store.get(output["stdout_artifact_id"])
    assert Path(artifact.path).read_bytes() == b"x" * 4096


def test_timeout_preserves_output_and_prevents_later_child_effect(tmp_path):
    store = TextArtifactStore(tmp_path, "capture")
    marker = tmp_path / "must-not-exist"
    script = "import os,time;print('before timeout',flush=True);time.sleep(2);open(%r,'w').write('bad')" % str(marker)
    output, _ = capture_command([sys.executable, "-c", script], store=store,
        timeout_seconds=0.1, max_output_bytes=4096, preview_chars=100, env={}, kind="test")
    assert output["status"] == "timed_out"
    assert output["stdout"] == "before timeout\n"
    assert not marker.exists()


def test_output_limit_is_a_tool_failure_with_recovery_evidence(make_config, tmp_path):
    config = make_config(agent_data__root=tmp_path / "agent", agent_data__max_output_bytes=73)
    with pytest.raises(ToolExecutionError) as caught:
        workspace_call(config, "shell", command="printf '%0200d' 0")
    assert caught.value.error_type == "output_limit_exceeded"
    assert caught.value.evidence["capture_complete"] is False
    assert caught.value.evidence["stdout_bytes"] == 73
    assert any(e.event_type == "agent_workspace_operation" for e in caught.value.generated_events)


def test_failed_artifact_publication_keeps_raw_evidence(tmp_path, monkeypatch):
    store = TextArtifactStore(tmp_path, "capture")
    def fail(*args, **kwargs):
        raise OSError("simulated metadata storage failure")
    monkeypatch.setattr(store, "create_from_file", fail)
    with pytest.raises(ToolExecutionError) as caught:
        capture_command([sys.executable, "-c", "print('retain me')"], store=store,
            timeout_seconds=5, max_output_bytes=4096, preview_chars=20, env={}, kind="test")
    recovery = Path(caught.value.evidence["recovery_directory"])
    assert (recovery / "stdout.raw").read_bytes() == b"retain me\n"
    assert caught.value.evidence["capture_complete"] is False


def test_text_paging_is_complete_and_rejects_changed_revision(make_config, tmp_path):
    config = make_config(agent_data__root=tmp_path / "agent", agent_data__max_capture_chars=7)
    text = "ä中🙂\r\n" * 9 + "last line"
    first = workspace_call(config, "write_text", path="unicode.txt", text=text)
    pieces = []; offset = 0
    while True:
        page = workspace_call(config, "read_text", path="unicode.txt", start_offset=offset,
            expected_sha256=first.output["sha256"]).output
        pieces.append(page["text"]); offset = page["next_offset"]
        if page["finished"]:break
    assert "".join(pieces) == text
    workspace_call(config, "write_text", path="unicode.txt", text="new revision")
    with pytest.raises(ToolValidationError, match="changed"):
        workspace_call(config, "read_text", path="unicode.txt", start_offset=offset, expected_sha256=first.output["sha256"])
    with pytest.raises(ToolValidationError, match="not applied"):
        workspace_call(config, "write_text", path="unicode.txt", text="stale overwrite", expected_sha256=first.output["sha256"])
    assert (config.agent_data.root / "workspace/unicode.txt").read_text() == "new revision"


def test_directory_paging_recovers_all_files_beyond_old_cap(make_config, tmp_path):
    config = make_config(agent_data__root=tmp_path / "agent", agent_data__max_list_files=137)
    root = config.agent_data.root / "workspace/catalog";root.mkdir(parents=True)
    expected = [f"catalog/{i:04d}.txt" for i in range(1007)]
    for name in reversed(expected):
        (root.parent / name).write_text("x")
    seen = []; cursor = None
    while True:
        page = workspace_call(config, "list_files", path="catalog", after_path=cursor).output
        seen.extend(row["path"] for row in page["files"])
        assert page["total_files"] == len(expected)
        if page["finished"]:break
        cursor = page["next_path"]
    assert seen == expected
