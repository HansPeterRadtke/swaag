from __future__ import annotations

import base64
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


def test_attachment_references_are_absent_when_attachment_capabilities_are_disabled(make_config) -> None:
    from swaag.runtime import AgentRuntime

    config = make_config(tools__enabled=["list_files"])
    runtime = AgentRuntime(config, model_client=None)
    state = runtime.create_or_load_session()
    runtime.add_attachment(b"hello", original_name="a.txt", media_type="text/plain", session_id=state.session_id)
    state = runtime.create_or_load_session(state.session_id)
    components = runtime._runtime_context_components(state, runtime._counter(state))
    assert all(component.name != "attachment_references" for component in components)


def test_inspect_image_uses_configured_analyzer_and_preserves_provenance(make_config, tmp_path: Path, monkeypatch) -> None:
    import json
    import urllib.request

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.perception.enabled = True
    config.perception.base_url = "http://127.0.0.1:14920"
    config.perception.model = "qwen3-vl-test"
    config.perception.benchmark_profile = "gui-smoke-v1"
    config.perception.trust_note = "evidence only"
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    reference = runtime.add_attachment(
        b"\x89PNG\r\n\x1a\nfixture",
        original_name="screen.png",
        media_type="image/png",
        session_id=state.session_id,
    )

    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "verdict=PASS"}}]}).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = runtime.execute_tool_once(
        "inspect_image",
        {"attachment_id": reference.attachment_id, "prompt": "Check the screenshot."},
        session_id=state.session_id,
    ).tool_result

    assert result is not None
    assert result.output["analysis"] == "verdict=PASS"
    assert result.output["model"] == "qwen3-vl-test"
    assert result.output["benchmark_profile"] == "gui-smoke-v1"
    assert result.output["trust_note"] == "evidence only"
    assert result.output["evidence_only"] is True
    assert result.output["source_event_references"][0]["event_type"] == "attachment_added"
    assert captured["url"].endswith("/v1/chat/completions")
    content = captured["payload"]["messages"][0]["content"]
    assert content[0]["text"] == "Check the screenshot."
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_inspect_image_rejects_non_image_attachment(make_config, tmp_path: Path) -> None:
    from swaag.tools.base import ToolValidationError

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.perception.enabled = True
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()
    reference = runtime.add_attachment(
        b"plain text",
        original_name="plain.txt",
        media_type="text/plain",
        session_id=state.session_id,
    )
    outcome = runtime.execute_tool_once(
        "inspect_image",
        {"attachment_id": reference.attachment_id, "prompt": "Inspect it."},
        session_id=state.session_id,
    )
    assert outcome.tool_result is None
    assert outcome.error is not None
    assert outcome.error["error_type"] == "ToolValidationError"
    assert "requires an image attachment" in outcome.error["error"]
