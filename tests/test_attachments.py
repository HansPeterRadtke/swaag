from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from swaag.attachments import AttachmentStore
from swaag.runtime import AgentRuntime
from swaag.task_api import TaskApi
from swaag.tokens import ConservativeEstimator
from swaag.workers import WorkerManager


def test_raw_attachment_survives_session_archival_with_exact_lineage(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    reference = runtime.add_attachment(
        b"raw attachment bytes\n",
        original_name="evidence.txt",
        source="test",
        session_id=state.session_id,
    )
    session_id = state.session_id
    state = runtime.history.rebuild_from_history(session_id, write_projections=False)

    assert state.attachments[0].attachment_id == reference.attachment_id
    assert state.attachments[0].metadata["source_event_hash"]
    archived = runtime.history.archive_session(session_id, remove_active=True)
    rebuilt = runtime.history.rebuild_from_history(session_id, write_projections=False)
    stored = AttachmentStore(config.sessions.root, max_upload_bytes=config.attachments.max_upload_bytes)

    assert archived["event_count"] >= 2
    assert stored.read_bytes(rebuilt.attachments[0]) == b"raw attachment bytes\n"
    assert not (config.sessions.root / session_id).exists()
    with pytest.raises(RuntimeError, match="archived session"):
        runtime.add_attachment(b"late", original_name="late.txt", session_id=session_id)


def test_attachment_prompt_contains_references_but_not_raw_content(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    runtime.add_attachment(
        b"secret raw payload must not be injected",
        original_name="payload.bin",
        session_id=state.session_id,
    )
    state = runtime.history.rebuild_from_history(state.session_id, write_projections=False)

    components = runtime._runtime_context_components(state, ConservativeEstimator())
    attachment = next(item for item in components if item.name == "attachment_references")

    assert "payload.bin" in attachment.text
    assert state.attachments[0].sha256 in attachment.text
    assert "secret raw payload" not in attachment.text
    assert attachment.category == "attachments"


def test_read_attachment_is_model_selected_bounded_and_provenanced(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.attachments.preview_chars = 5
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    reference = runtime.add_attachment(
        b"abcdefghij",
        original_name="plain.txt",
        session_id=state.session_id,
    )

    result = runtime.execute_tool_once(
        "read_attachment",
        {"attachment_id": reference.attachment_id, "max_chars": None},
        session_id=state.session_id,
    ).tool_result

    assert result is not None
    assert result.output["text"] == "abcde"
    assert result.output["truncated"] is True
    assert result.output["source_event_references"][0]["event_type"] == "attachment_added"


def _write_fake_all2text(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
from pathlib import Path
import sys
source = Path(sys.argv[-2])
target = Path(sys.argv[-1])
target.mkdir(parents=True, exist_ok=True)
name = next(item.name for item in source.iterdir() if item.is_file())
output = target / (name + '.txt')
output.write_text('derived text from ' + name + '\\n', encoding='utf-8')
manifest = {
    'schema': 'all2text.conversion_manifest.v1',
    'summary': {'converted': 1},
    'entries': [{
        'relative_path': name,
        'output_path': str(output),
        'converter_used': 'fake_text',
        'extraction_methods_used': ['fake'],
        'errors': [], 'warnings': [], 'limitations': [],
        'llm_used': False, 'ocr_used': False, 'vlm_used': False,
    }],
}
(target / '_conversion_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_extract_attachment_retains_complete_derived_artifact(make_config, tmp_path: Path) -> None:
    command = tmp_path / "fake-all2text"
    _write_fake_all2text(command)
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.attachments.all2text_command = str(command)
    config.attachments.preview_chars = 8
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    reference = runtime.add_attachment(b"source", original_name="report.txt", session_id=state.session_id)

    result = runtime.execute_tool_once(
        "extract_attachment",
        {"attachment_id": reference.attachment_id, "profile": "core"},
        session_id=state.session_id,
    ).tool_result
    artifact = runtime.execute_tool_once(
        "read_artifact",
        {"artifact_id": result.output["artifact_id"], "start_offset": 0, "max_chars": None},
        session_id=state.session_id,
    ).tool_result

    assert result.output["text"] == "derived "
    assert result.output["truncated"] is True
    assert artifact.output["text"] == "derived text from report.txt\n"
    history = runtime.history.read_history(state.session_id)
    assert any(event.event_type == "attachment_extracted" for event in history)
    assert any(event.event_type == "artifact_created" for event in history)


def test_task_api_accepts_attachments_before_worker_start(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    manager = WorkerManager(AgentRuntime(config, model_client=object()))
    api = TaskApi(manager)

    created = api.execute(
        "create",
        {
            "objective": "Inspect the supplied evidence only if needed.",
            "attachments": [
                {
                    "original_name": "evidence.txt",
                    "media_type": "text/plain",
                    "content_base64": base64.b64encode(b"evidence").decode("ascii"),
                }
            ],
        },
    )
    worker_id = created["worker"]["worker_id"]
    listed = api.execute("attachment.list", {"worker_id": worker_id})
    inspected = api.execute("get", {"worker_id": worker_id})
    manager.shutdown()

    assert listed["attachments"][0]["original_name"] == "evidence.txt"
    assert "storage_ref" not in listed["attachments"][0]
    assert inspected["attachments"][0]["size_bytes"] == len(b"evidence")
