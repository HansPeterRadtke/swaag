from __future__ import annotations

import json

from swaag.environment import browser as browser_module
from swaag.environment.artifacts import TextArtifactStore
from swaag.environment.browser import (
    AubroCommandResult,
    AubroInvocation,
    BrowserAutomationError,
)
from swaag.environment.environment import AgentEnvironment
from swaag.environment.process import ProcessResult
from swaag.environment.state import ProcessRecord
from swaag.history import HistoryStore
from swaag.runtime import AgentRuntime
from swaag.tools.registry import ToolRegistry
from swaag.utils import sha256_text


def _state(config, session_id: str):
    return HistoryStore(config.sessions.root).create(
        config_fingerprint=config.config_fingerprint(),
        model_base_url=config.model.base_url,
        session_id=session_id,
    )


def _aubro_result(payload: dict, *, stderr: str = "") -> AubroCommandResult:
    raw = json.dumps(payload, sort_keys=True)
    record = ProcessRecord(
        process_id="proc_browser",
        command=["aubro", "test"],
        cwd="/tmp",
        status="completed",
        return_code=0,
        stdout=raw,
        stderr=stderr,
        started_at="2026-08-26T10:00:00+00:00",
        ended_at="2026-08-26T10:00:01+00:00",
        metadata={"kind": "aubro"},
    )
    return AubroCommandResult(
        payload=payload,
        process_result=ProcessResult(record=record, stdout=raw, stderr=stderr),
        invocation=AubroInvocation(
            command_prefix=["aubro"], env_overrides={}, source_path=None
        ),
    )


def test_browser_search_preserves_exact_raw_evidence_behind_bounded_preview(
    make_config, tmp_path, monkeypatch
) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.environment.max_capture_chars = 32
    config.environment.aubro_max_results = 1
    config.environment.aubro_max_text_chars = 8
    payload = {
        "query": "durable research",
        "engine": "test",
        "url": "https://search.example/",
        "results": [
            {
                "title": "first",
                "url": "https://example.test/one",
                "snippet": "first exact snippet is longer",
            },
            {
                "title": "second",
                "url": "https://example.test/two",
                "snippet": "second exact snippet",
            },
        ],
        "attempts": [
            {"engine": "one", "url": "https://one", "results": 0, "blocked": True},
            {"engine": "two", "url": "https://two", "results": 2, "blocked": False},
        ],
        "provider_metadata": {"exact": "retained only in raw evidence"},
    }
    invocation = _aubro_result(payload, stderr="diagnostic evidence")
    monkeypatch.setattr(
        "swaag.environment.environment.run_aubro_command",
        lambda **_kwargs: invocation,
    )
    state = _state(config, "session_browser_search")

    result = AgentEnvironment(config, state).browser_search(
        query="durable research", engine="auto", limit=5
    )
    ToolRegistry().get("browser_search").validate_output(result.output)
    artifact = TextArtifactStore(config.sessions.root, state.session_id).read(
        result.output["artifact_id"], max_chars=100_000
    )
    completed = next(
        event
        for event in result.generated_events
        if event.event_type == "process_completed"
    )

    assert result.output["result_count"] == 2
    assert result.output["returned_result_count"] == 1
    assert result.output["results_truncated"] is True
    assert result.output["results"][0]["snippet"] == "first ex"
    assert result.output["results"][0]["snippet_truncated"] is True
    assert result.output["attempts_truncated"] is True
    assert artifact["text"] == invocation.process_result.stdout
    assert artifact["finished"] is True
    assert len(completed.payload["stdout"]) == 32
    assert completed.payload["output_artifacts"]["artifact_id"] == result.output["artifact_id"]
    assert result.output["stderr_artifact_id"]
    sources = [
        event.payload
        for event in result.generated_events
        if event.event_type == "external_source_observed"
    ]
    assert sources == [
        {
            "source_id": "source_" + sha256_text("https://example.test/one")[:16],
            "name": "first",
            "url": "https://example.test/one",
            "document": "first ex",
            "document_truncated": True,
            "tool_name": "browser_search",
        }
    ]


def test_browser_browse_preserves_full_text_and_links_for_reexpansion(
    make_config, tmp_path, monkeypatch
) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.environment.max_capture_chars = 40
    config.environment.aubro_max_links = 1
    config.environment.aubro_max_text_chars = 10
    payload = {
        "url": "https://example.test/page",
        "title": "Exact page",
        "backend": "test",
        "blocked": False,
        "text": "complete authoritative page text that exceeds the preview",
        "links": [
            {"text": "first complete link text", "href": "https://example.test/one"},
            {"text": "second link", "href": "https://example.test/two"},
        ],
        "forms": [{"name": "exact form"}],
        "buttons": [{"name": "exact button"}],
    }
    invocation = _aubro_result(payload)
    monkeypatch.setattr(
        "swaag.environment.environment.run_aubro_command",
        lambda **_kwargs: invocation,
    )
    state = _state(config, "session_browser_browse")

    result = AgentEnvironment(config, state).browser_browse(
        url="https://example.test/page"
    )
    ToolRegistry().get("browser_browse").validate_output(result.output)
    artifact = TextArtifactStore(config.sessions.root, state.session_id).read(
        result.output["artifact_id"], max_chars=100_000
    )

    assert result.output["text_excerpt"] == "complete a"
    assert result.output["text_chars"] == len(payload["text"])
    assert result.output["text_truncated"] is True
    assert result.output["link_count"] == 2
    assert result.output["returned_link_count"] == 1
    assert result.output["links_truncated"] is True
    assert result.output["links"][0]["text_truncated"] is True
    assert artifact["text"] == invocation.process_result.stdout
    assert artifact["sha256"] == result.output["artifact_sha256"]
    assert sum(
        event.event_type == "artifact_created" for event in result.generated_events
    ) == 1
    source = next(
        event.payload
        for event in result.generated_events
        if event.event_type == "external_source_observed"
    )
    assert source["url"] == "https://example.test/page"
    assert source["document"] == "complete a"
    assert source["document_truncated"] is True


def test_browser_failure_commits_complete_raw_process_evidence(
    make_config, tmp_path, monkeypatch
) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    config.tools.enabled = ["browser_search", "read_artifact"]
    config.environment.max_capture_chars = 32
    raw_stdout = "not-json-" + ("complete-provider-output-" * 300)
    raw_stderr = "provider diagnostic\n" * 200
    process_result = _aubro_result({}, stderr=raw_stderr).process_result
    process_result.stdout = raw_stdout
    process_result.record.stdout = raw_stdout

    def fail(**_kwargs):
        raise BrowserAutomationError(
            f"aubro returned invalid JSON: {raw_stdout[:32]!r}",
            process_result=process_result,
        )

    monkeypatch.setattr("swaag.environment.environment.run_aubro_command", fail)
    monkeypatch.setattr("swaag.tools.builtin.aubro_available", lambda _config: True)
    runtime = AgentRuntime(config, model_client=object())
    state = runtime.create_or_load_session()

    run = runtime.execute_tool_once(
        "browser_search",
        {"query": "durable errors", "engine": "auto", "limit": 2},
        session_id=state.session_id,
    )

    assert run.tool_result is None
    assert run.error is not None
    assert run.error["error_type"] == "BrowserAutomationError"
    evidence = run.error["evidence"]
    stdout = TextArtifactStore(config.sessions.root, state.session_id).read(
        evidence["artifact_id"], max_chars=len(raw_stdout) + 1
    )
    stderr = TextArtifactStore(config.sessions.root, state.session_id).read(
        evidence["stderr_artifact_id"], max_chars=len(raw_stderr) + 1
    )
    assert stdout["text"] == raw_stdout
    assert stderr["text"] == raw_stderr
    history = runtime.history.read_history(state.session_id)
    event_types = [event.event_type for event in history]
    assert event_types.index("artifact_created") < event_types.index("tool_error")
    assert "process_completed" in event_types


def test_browser_capabilities_are_exposed_only_when_backend_is_available(
    make_config, monkeypatch
) -> None:
    config = make_config()
    config.tools.enabled = ["calculator", "browser_search", "browser_browse"]
    registry = ToolRegistry()

    monkeypatch.setattr("swaag.tools.builtin.aubro_available", lambda _config: False)
    unavailable = {name for name, _description, _guidance in registry.capability_index(config)}
    monkeypatch.setattr("swaag.tools.builtin.aubro_available", lambda _config: True)
    available = {name for name, _description, _guidance in registry.capability_index(config)}

    assert unavailable == {"calculator"}
    assert available == {"calculator", "browser_search", "browser_browse"}


def test_aubro_discovery_supports_nested_repository_collections(
    make_config, tmp_path, monkeypatch
) -> None:
    repository = tmp_path / "repositories" / "swaag"
    (repository / ".git").mkdir(parents=True)
    fake_module = repository / "src" / "swaag" / "environment" / "browser.py"
    fake_module.parent.mkdir(parents=True)
    nested = tmp_path / "repositories" / "utilities" / "aubro" / "src"
    (nested / "aubro").mkdir(parents=True)
    (nested / "aubro" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(browser_module, "__file__", str(fake_module))
    monkeypatch.chdir(repository)
    config = make_config()
    config.environment.aubro_src = ""

    assert browser_module.discover_aubro_src(config) == nested.resolve()


def test_aubro_availability_handles_missing_parent_package(
    make_config, monkeypatch
) -> None:
    config = make_config()
    config.environment.aubro_src = ""
    monkeypatch.delenv("SWAAG_AUBRO_SRC", raising=False)
    monkeypatch.setattr(browser_module, "_repo_aubro_src_candidates", lambda: [])

    def missing(_name: str):
        raise ModuleNotFoundError("aubro")

    monkeypatch.setattr(browser_module.importlib.util, "find_spec", missing)

    assert browser_module.aubro_available(config) is False
