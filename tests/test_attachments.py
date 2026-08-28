from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from swaag.attachments import AttachmentStore
from swaag.environment.artifacts import TextArtifactStore
from swaag.runtime import AgentRuntime
from swaag.task_api import TaskApi
from swaag.tokens import ConservativeEstimator
from swaag.utils import sha256_text
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
    assert result.output["start_offset"] == 0
    assert result.output["next_offset"] == 5
    assert result.output["finished"] is False
    assert result.output["source_event_references"][0]["event_type"] == "attachment_added"

    continued = runtime.execute_tool_once(
        "read_attachment",
        {
            "attachment_id": reference.attachment_id,
            "start_offset": result.output["next_offset"],
            "max_chars": None,
        },
        session_id=state.session_id,
    ).tool_result
    assert continued is not None
    assert continued.output["text"] == "fghij"
    assert continued.output["start_offset"] == 5
    assert continued.output["finished"] is True


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
print('conversion completed for ' + name)
print('specialist warning for ' + name, file=sys.stderr)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_failing_all2text(path: Path) -> tuple[str, str]:
    stdout = "complete stdout evidence\n" * 300
    stderr = "complete stderr evidence\n" * 300
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"sys.stdout.write({stdout!r})\n"
        f"sys.stderr.write({stderr!r})\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return stdout, stderr


def _write_fake_capabilities(path: Path) -> tuple[str, str, dict]:
    payload = {
        "profile": {"name": "auto", "allow_external_tools": True},
        "summary": {"available_external_tools": ["tesseract"]},
        "optional_python_libraries": [
            {
                "name": "docling",
                "module": "docling",
                "extra": "documents",
                "enabled_by_profile": True,
                "available": False,
                "implemented_in_core": False,
                "error": "missing",
            }
        ],
        "external_tools": [
            {
                "name": "tesseract",
                "enabled": True,
                "available": True,
                "source": "/usr/bin/tesseract",
                "used_by_core": True,
                "error": None,
            }
        ],
        "provider_statuses": [
            {
                "name": "ocr",
                "kind": "ocr",
                "enabled": True,
                "available": True,
                "source": "/usr/bin/tesseract",
                "error": None,
                "details": {"execution_status": "implemented"},
                "lifecycle": {"configured": True, "used": False},
            }
        ],
        "provider_family_statuses": [
            {
                "name": "docling",
                "kind": "document_parser",
                "enabled": True,
                "available": False,
                "source": None,
                "error": "dependency missing",
                "details": {
                    "candidate": {
                        "family": "document_ocr_layout",
                        "execution_status": "contract_only",
                        "notes": "Document parser",
                    },
                    "exact_unprojected_detail": "x" * 5000,
                },
                "lifecycle": {"configured": True, "missing": True},
            },
            {
                "name": "faster_whisper",
                "kind": "speech_to_text",
                "enabled": True,
                "available": True,
                "source": "/data/models/whisper",
                "error": None,
                "details": {
                    "candidate": {
                        "family": "speech_transcription",
                        "execution_status": "implemented",
                        "notes": "Local speech model",
                    }
                },
                "lifecycle": {"configured": True, "dependency_found": True},
            },
        ],
    }
    stdout = json.dumps(payload, sort_keys=True)
    stderr = "capability probe note\n"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"sys.stdout.write({stdout!r})\n"
        f"sys.stderr.write({stderr!r})\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return stdout, stderr, payload


def test_inspect_attachment_capabilities_indexes_every_family_and_retains_raw_response(
    make_config, tmp_path: Path
) -> None:
    command = tmp_path / "fake-capabilities"
    expected_stdout, expected_stderr, payload = _write_fake_capabilities(command)
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.attachments.all2text_command = str(command)
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()

    run = runtime.execute_tool_once(
        "inspect_attachment_capabilities", {}, session_id=state.session_id
    )

    assert run.error is None
    result = run.tool_result
    assert result is not None
    assert result.output["detected_profile"] == payload["profile"]
    assert result.output["extraction_profiles"] == [
        "core",
        "pip",
        "tools",
        "local-models",
        "full",
    ]
    assert [item["name"] for item in result.output["provider_families"]] == [
        "docling",
        "faster_whisper",
    ]
    assert result.output["provider_families"][0]["family"] == "document_ocr_layout"
    assert result.output["provider_families"][0]["lifecycle"] == [
        "configured",
        "missing",
    ]
    assert result.output["providers"][0]["execution_status"] == "implemented"
    assert "exact_unprojected_detail" not in result.display_text

    store = TextArtifactStore(config.sessions.root, state.session_id)
    capabilities = store.read(
        result.output["capabilities_artifact_id"],
        max_chars=len(expected_stdout) + 1,
    )
    stderr = store.read(
        result.output["stderr_artifact_id"], max_chars=len(expected_stderr) + 1
    )
    assert capabilities["text"] == expected_stdout
    assert stderr["text"] == expected_stderr
    events = runtime.history.read_history(state.session_id)
    event_types = [event.event_type for event in events]
    assert event_types.count("artifact_created") == 2
    assert event_types.index("artifact_created") < event_types.index("tool_result")


def test_inspect_attachment_capabilities_failure_retains_exact_streams(
    make_config, tmp_path: Path
) -> None:
    command = tmp_path / "failing-capabilities"
    expected_stdout, expected_stderr = _write_failing_all2text(command)
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.attachments.all2text_command = str(command)
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()

    run = runtime.execute_tool_once(
        "inspect_attachment_capabilities", {}, session_id=state.session_id
    )

    assert run.tool_result is None
    assert run.error is not None
    assert run.error["error_type"] == "All2TextCapabilityError"
    evidence = run.error["evidence"]
    store = TextArtifactStore(config.sessions.root, state.session_id)
    stdout = store.read(
        evidence["stdout_artifact_id"], max_chars=len(expected_stdout) + 1
    )
    stderr = store.read(
        evidence["stderr_artifact_id"], max_chars=len(expected_stderr) + 1
    )
    assert stdout["text"] == expected_stdout
    assert stderr["text"] == expected_stderr


def test_inspect_attachment_capabilities_rejects_malformed_success_as_evidence(
    make_config, tmp_path: Path
) -> None:
    command = tmp_path / "malformed-capabilities"
    expected_stdout = '{"provider_statuses": "not-a-list"}'
    expected_stderr = "probe diagnostic\n"
    command.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"sys.stdout.write({expected_stdout!r})\n"
        f"sys.stderr.write({expected_stderr!r})\n",
        encoding="utf-8",
    )
    command.chmod(0o755)
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.attachments.all2text_command = str(command)
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()

    run = runtime.execute_tool_once(
        "inspect_attachment_capabilities", {}, session_id=state.session_id
    )

    assert run.tool_result is None
    assert run.error is not None
    assert run.error["error_type"] == "ValueError"
    assert "must be a list of objects" in run.error["error"]
    store = TextArtifactStore(config.sessions.root, state.session_id)
    stdout = store.read(
        run.error["evidence"]["stdout_artifact_id"],
        max_chars=len(expected_stdout) + 1,
    )
    stderr = store.read(
        run.error["evidence"]["stderr_artifact_id"],
        max_chars=len(expected_stderr) + 1,
    )
    assert stdout["text"] == expected_stdout
    assert stderr["text"] == expected_stderr


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

    archived = runtime.history.archive_session(state.session_id, remove_active=True)
    archived_store = TextArtifactStore(config.sessions.root, state.session_id)
    archived_text = archived_store.read(
        result.output["artifact_id"], start_offset=0, max_chars=1000
    )
    archived_manifest = archived_store.read(
        result.output["manifest_artifact_id"], start_offset=0, max_chars=10_000
    )
    archived_stdout = archived_store.read(
        result.output["stdout_artifact_id"], start_offset=0, max_chars=1000
    )
    archived_stderr = archived_store.read(
        result.output["stderr_artifact_id"], start_offset=0, max_chars=1000
    )

    assert archived["artifact_count"] == 4
    assert archived_text["text"] == "derived text from report.txt\n"
    assert json.loads(archived_manifest["text"])["schema"] == "all2text.conversion_manifest.v1"
    assert archived_stdout["text"] == "conversion completed for report.txt\n"
    assert archived_stderr["text"] == "specialist warning for report.txt\n"
    assert result.output["stdout_sha256"] == sha256_text(archived_stdout["text"])
    assert result.output["stderr_sha256"] == sha256_text(archived_stderr["text"])
    assert not (config.sessions.root / state.session_id).exists()


def test_extract_attachment_failure_retains_complete_stdout_and_stderr(
    make_config, tmp_path: Path
) -> None:
    command = tmp_path / "failing-all2text"
    expected_stdout, expected_stderr = _write_failing_all2text(command)
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.attachments.all2text_command = str(command)
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    reference = runtime.add_attachment(
        b"source",
        original_name="report.txt",
        session_id=state.session_id,
    )

    run = runtime.execute_tool_once(
        "extract_attachment",
        {"attachment_id": reference.attachment_id, "profile": "core"},
        session_id=state.session_id,
    )

    assert run.tool_result is None
    assert run.error is not None
    assert run.error["error_type"] == "All2TextProcessError"
    assert len(run.error["error"]) < 1200
    evidence = run.error["evidence"]
    store = TextArtifactStore(config.sessions.root, state.session_id)
    stdout = store.read(
        evidence["stdout_artifact_id"], max_chars=len(expected_stdout) + 1
    )
    stderr = store.read(
        evidence["stderr_artifact_id"], max_chars=len(expected_stderr) + 1
    )
    assert stdout["text"] == expected_stdout
    assert stderr["text"] == expected_stderr
    events = runtime.history.read_history(state.session_id)
    event_types = [event.event_type for event in events]
    assert event_types.count("artifact_created") == 2
    assert event_types.index("artifact_created") < event_types.index("tool_error")


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
            "attachment_source": "upload_transport",
        },
    )
    worker_id = created["worker"]["worker_id"]
    listed = api.execute("attachment.list", {"worker_id": worker_id})
    inspected = api.execute("get", {"worker_id": worker_id})
    manager.shutdown()

    assert listed["attachments"][0]["original_name"] == "evidence.txt"
    assert listed["attachments"][0]["source"] == "upload_transport"
    assert "storage_ref" not in listed["attachments"][0]
    assert inspected["attachments"][0]["size_bytes"] == len(b"evidence")
