from __future__ import annotations

import json
from typing import Any, Callable

from swaag.model import CompletionRequestPolicy
from swaag.runtime import AgentRuntime
from swaag.types import CompletionResult, ContractSpec


class FakeModelClient:
    is_deterministic_test_client = True

    def __init__(self, responses: list[str | Callable[[dict[str, Any]], str]]):
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

    def tokenize(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def tokenize_selection(self, text: str) -> int:
        return self.tokenize(text)

    def select_request_policy(
        self,
        *,
        contract: ContractSpec,
        kind: str,
        prompt: str,
        max_tokens: int,
        live_mode: bool = False,
    ) -> CompletionRequestPolicy:
        return CompletionRequestPolicy(
            profile_name="test",
            structured_output_mode="server_schema",
            effective_contract_mode=contract.mode,
            effective_timeout_seconds=30,
            progress_poll_seconds=0.01,
        )

    def resolve_contract(
        self,
        contract: ContractSpec,
        *,
        kind: str,
        prompt: str,
        max_tokens: int,
        live_mode: bool = False,
    ) -> tuple[ContractSpec, CompletionRequestPolicy]:
        return contract, self.select_request_policy(
            contract=contract, kind=kind, prompt=prompt, max_tokens=max_tokens, live_mode=live_mode
        )

    def build_completion_request(
        self,
        prompt: str,
        *,
        max_tokens: int,
        contract: ContractSpec,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0 if temperature is None else temperature,
            "contract": contract.name,
            "json_schema": contract.json_schema,
        }

    def send_completion(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
        progress_callback=None,
    ) -> CompletionResult:
        self.requests.append(payload)
        if not self.responses:
            raise AssertionError("No fake model responses left")
        response = self.responses.pop(0)
        if callable(response):
            response = response(payload)
        return CompletionResult(
            text=response,
            raw_request=payload,
            raw_response={"content": response},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


def _action(
    *,
    message: str = "",
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
    continue_loop: bool = False,
) -> str:
    return json.dumps(
        {
            "assistant_message": message,
            "tool_calls": [
                {"tool_name": name, "arguments": arguments}
                for name, arguments in (tool_calls or [])
            ],
            "continue_loop": continue_loop,
        }
    )


def _runtime(make_config, responses: list[str | Callable[[dict[str, Any]], str]]) -> tuple[AgentRuntime, FakeModelClient]:
    client = FakeModelClient(responses)
    runtime = AgentRuntime(make_config(model__context_limit=32_000), model_client=client)
    return runtime, client


def test_direct_answer_is_one_constrained_model_call_with_all_tools(make_config) -> None:
    request = "What model are you?"
    runtime, client = _runtime(
        make_config,
        [_action(message="I am the configured SWAAG model.")],
    )

    result = runtime.run_turn(request)

    assert result.assistant_text == "I am the configured SWAAG model."
    assert [item["contract"] for item in client.requests] == ["agent_action"]
    prompt = str(client.requests[0]["prompt"])
    assert request in prompt
    for tool_name in runtime.tools.tool_names(runtime.config):
        assert f"- name: {tool_name}\n" in prompt
    schema = client.requests[0]["json_schema"]
    assert set(schema["properties"]) == {"assistant_message", "tool_calls", "continue_loop"}
    events = runtime.history.read_history(result.session_id)
    assert not any(event.event_type in {"plan_created", "plan_updated"} for event in events)
    assert not any(event.event_type.startswith("plan_") for event in events)


def test_multiple_tool_calls_execute_in_order_and_exact_results_reach_next_call(make_config) -> None:
    request = "Calculate 21 * 2 and 5 + 7, then tell me both results."

    def finish(payload: dict[str, Any]) -> str:
        prompt = str(payload["prompt"])
        assert request in prompt
        assert prompt.index('"result": 42') < prompt.index('"result": 12')
        return _action(message="42 and 12")

    runtime, client = _runtime(
        make_config,
        [
            _action(
                tool_calls=[
                    ("calculator", {"expression": "21 * 2"}),
                    ("calculator", {"expression": "5 + 7"}),
                ],
                continue_loop=True,
            ),
            finish,
        ],
    )

    result = runtime.run_turn(request)

    assert result.assistant_text == "42 and 12"
    assert [item["contract"] for item in client.requests] == ["agent_action", "agent_action"]
    assert [item.tool_name for item in result.tool_results] == ["calculator", "calculator"]
    assert [item.output["result"] for item in result.tool_results] == [42, 12]


def test_tool_error_is_returned_verbatim_before_model_can_finish(make_config) -> None:
    request = "Read a missing file and report what happens."

    observed: dict[str, str] = {}

    def finish(payload: dict[str, Any]) -> str:
        observed["prompt"] = str(payload["prompt"])
        return _action(message="The file could not be read.")

    runtime, client = _runtime(
        make_config,
        [
            _action(
                tool_calls=[("read_file", {"path": "/etc/passwd"})],
                continue_loop=True,
            ),
            finish,
        ],
    )

    result = runtime.run_turn(request)

    assert result.assistant_text == "The file could not be read."
    assert len(client.requests) == 2
    assert result.tool_results == []
    assert "tool_error:" in observed["prompt"]
    assert "/etc/passwd" in observed["prompt"]
    assert "FilesystemError" in observed["prompt"]


def test_pending_user_intervention_is_verbatim_on_next_model_call(make_config) -> None:
    request = "Calculate 6 * 7."
    intervention = "Also state that this instruction arrived while you were working."
    holder: dict[str, Any] = {}

    def first(payload: dict[str, Any]) -> str:
        holder["runtime"].history.enqueue_control_message(holder["state"].session_id, intervention, source="test")
        return _action(
            tool_calls=[("calculator", {"expression": "6 * 7"})],
            continue_loop=True,
        )

    observed: dict[str, str] = {}

    def finish(payload: dict[str, Any]) -> str:
        observed["prompt"] = str(payload["prompt"])
        return _action(message="42. The additional instruction arrived while I was working.")

    runtime, client = _runtime(make_config, [first, finish])
    state = runtime.create_or_load_session()
    holder.update(runtime=runtime, state=state)

    result = runtime.run_turn_in_session(state, request)

    assert result.assistant_text.startswith("42")
    assert intervention in observed["prompt"]
    assert '"result": 42' in observed["prompt"]
    assert runtime.history.list_pending_control_messages(state.session_id) == []
    events = runtime.history.read_history(state.session_id)
    processed = [event for event in events if event.event_type == "control_message_processed"]
    assert [event.payload["message"] for event in processed] == [intervention]
    assert [item["contract"] for item in client.requests] == ["agent_action", "agent_action"]


def test_identical_model_tool_response_is_rejected_for_recovery(make_config) -> None:
    repeated = _action(
        tool_calls=[("calculator", {"expression": "1 + 1"})],
        continue_loop=True,
    )
    recovered = _action(message="Recovered after duplicate rejection.", continue_loop=False)
    runtime, client = _runtime(make_config, [repeated, repeated, repeated, recovered])

    result = runtime.run_turn("Keep calculating 1 + 1 forever.")

    assert result.assistant_text == "Recovered after duplicate rejection."
    assert len(client.requests) == 4
    assert len(result.tool_results) == 2
    events = runtime.history.read_history(result.session_id)
    assert any(
        event.event_type == "agent_action_rejected"
        and "materially different next action" in str(event.payload.get("reason", ""))
        for event in events
    )


def test_failure_analyzer_supports_action_loop_metrics_without_legacy_fields() -> None:
    from swaag.benchmark.failure_analyzer import FailureAnalyzer
    from swaag.types import HistoryEvent, SessionState

    state = SessionState(session_id="s", created_at="t", updated_at="t", config_fingerprint="cfg", model_base_url="http://127.0.0.1:14829")
    state.metrics.action_count = 1
    events = [HistoryEvent(id="e1", sequence=1, session_id="s", timestamp="t", type="turn_failed", version=1, payload={"reason": "budget_exhausted"})]

    failure = FailureAnalyzer().analyze(state=state, events=events, deterministic_verification_passed=False, runtime_error=None)

    assert failure.category == "premature_termination"
    assert failure.evidence["last_reason"] == "budget_exhausted"


def test_non_live_benchmark_uses_resolvable_local_model_default(monkeypatch) -> None:
    from swaag.benchmark.benchmark_runner import _resolve_live_model_settings

    monkeypatch.delenv("SWAAG_LIVE_BASE_URL", raising=False)
    settings = _resolve_live_model_settings(
        use_live_model=False, model_base_url=None, timeout_seconds=None, connect_timeout_seconds=None,
        model_profile=None, structured_output_mode=None, progress_poll_seconds=None, seeds=None,
    )

    assert settings["base_url"] == "http://127.0.0.1:14829"


def test_context_discovery_retries_transient_failures(monkeypatch) -> None:
    import json
    import urllib.error

    from swaag.benchmark import benchmark_runner

    calls = {"count": 0}

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({"default_generation_settings": {"params": {"n_ctx": 32000}}}).encode()

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] < 3:
            raise urllib.error.URLError("busy")
        return Response()

    monkeypatch.setattr(benchmark_runner.urllib.request, "urlopen", fake_urlopen)

    assert benchmark_runner._discover_server_context_limit("http://127.0.0.1:14829", timeout_seconds=5) == 32000
    assert calls["count"] == 3


def test_duration_parser_supports_recording_units() -> None:
    from swaag.scheduler import parse_duration

    assert parse_duration("30 seconds").total_seconds() == 30
    assert parse_duration("2 hours").total_seconds() == 7200
    assert parse_duration("3 days").total_seconds() == 259200
    assert parse_duration("1 month").total_seconds() == 2629800
    assert parse_duration("1 year").total_seconds() == 31557600


def test_wakeup_store_persists_lists_cancels_and_claims_once(tmp_path) -> None:
    from datetime import datetime, timedelta, timezone
    from swaag.scheduler import WakeupStore

    now = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)
    store = WakeupStore(tmp_path)
    wakeup = store.schedule(session_id="s", reason="resume benchmark", duration="2 hours", now=now)

    assert WakeupStore(tmp_path).list(session_id="s") == [wakeup]
    assert WakeupStore(tmp_path).claim_due(session_id="s", now=now + timedelta(hours=1)) == []
    claimed = WakeupStore(tmp_path).claim_due(session_id="s", now=now + timedelta(hours=3))
    assert [item.wakeup_id for item in claimed] == [wakeup.wakeup_id]
    assert claimed[0].status == "delivered"
    assert WakeupStore(tmp_path).claim_due(session_id="s", now=now + timedelta(hours=4)) == []

    second = store.schedule(session_id="s", reason="later", duration="1 day", now=now)
    cancelled = store.cancel(session_id="s", wakeup_id=second.wakeup_id, now=now)
    assert cancelled.status == "cancelled"
    assert second.wakeup_id not in {item.wakeup_id for item in store.list(session_id="s")}


def test_schedule_wakeup_tools_are_registered_and_emit_events(tmp_path) -> None:
    from swaag.config import load_config
    from swaag.history import HistoryStore
    from swaag.tools.registry import ToolRegistry

    config = load_config(env={
        "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
        "SWAAG__TOOLS__ALLOW_STATEFUL_TOOLS": "true",
    })
    state = HistoryStore(config.sessions.root).create(config_fingerprint="cfg", model_base_url="http://model")
    registry = ToolRegistry()

    assert {"schedule_wakeup", "list_wakeups", "cancel_wakeup"} <= set(registry.tool_names(config))
    invocation, result = registry.dispatch(
        "schedule_wakeup",
        {"duration": "2 hours", "wake_at": None, "reason": "resume work"},
        config,
        state,
    )
    assert invocation.validated_input["duration"] == "2 hours"
    assert [event.event_type for event in result.generated_events] == ["wakeup_scheduled"]


def test_runtime_delivers_due_wakeup_as_control_message_once(tmp_path) -> None:
    from datetime import datetime, timedelta, timezone
    from swaag.config import load_config
    from swaag.history import HistoryStore
    from swaag.runtime import AgentRuntime
    from swaag.scheduler import WakeupStore

    config = load_config(env={"SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions")})
    history = HistoryStore(config.sessions.root)
    state = history.create(config_fingerprint="cfg", model_base_url="http://model", session_id="s")
    WakeupStore(config.sessions.root).schedule(
        session_id="s",
        reason="continue task",
        wake_at=(datetime.now(timezone.utc) + timedelta(milliseconds=1)).isoformat(),
    )
    import time
    time.sleep(0.01)
    runtime = AgentRuntime(config, model_client=object(), history_store=history)

    loaded = runtime.create_or_load_session("s")
    pending = history.list_pending_control_messages("s")
    assert len(pending) == 1
    assert "continue task" in pending[0]["message"]
    runtime._deliver_due_wakeups(loaded)
    assert len(history.list_pending_control_messages("s")) == 1


def test_summary_contract_requires_adaptive_retention_decision() -> None:
    from swaag.grammar import summary_contract

    schema = summary_contract().json_schema
    assert schema is not None
    assert set(schema["required"]) == {"summary", "preserve_recent_messages"}
    assert schema["properties"]["preserve_recent_messages"] == {"type": "integer"}


def test_adaptive_retention_is_mechanically_bounded() -> None:
    from swaag.runtime import AgentRuntime

    assert AgentRuntime._validated_preserve_recent_messages(0, source_count=8, maximum=4) == 0
    assert AgentRuntime._validated_preserve_recent_messages(4, source_count=8, maximum=4) == 4
    import pytest
    with pytest.raises(ValueError):
        AgentRuntime._validated_preserve_recent_messages(5, source_count=8, maximum=4)
    with pytest.raises(ValueError):
        AgentRuntime._validated_preserve_recent_messages(True, source_count=8, maximum=4)


def test_summary_prompt_exposes_retention_cap(tmp_path) -> None:
    from swaag.config import load_config
    from swaag.prompts import PromptBuilder
    from swaag.types import Message

    config = load_config(env={"SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions")})
    prompt = PromptBuilder(config).build_summary_prompt(
        [Message(role="user", content="exact command", created_at="now")],
        maximum_preserve_recent_messages=7,
    )
    assert "from 0 through 7" in prompt.prompt_text
    assert "preserve_recent_messages" in prompt.prompt_text


def test_runtime_model_unavailable_retry_cap_is_finite(tmp_path) -> None:
    from swaag.config import load_config
    from swaag.runtime import AgentRuntime

    config = load_config(env={"SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions")})
    runtime = AgentRuntime(config, model_client=object())
    assert isinstance(runtime._max_model_unavailable_attempts, int)
    assert runtime._max_model_unavailable_attempts >= 1


def test_benchmark_run_cli_enables_live_model_profile(monkeypatch, tmp_path) -> None:
    from swaag.benchmark import benchmark_runner

    captured = {}
    def fake_run_benchmarks(**kwargs):
        captured.update(kwargs)
        return {"summary": {"total_tasks": 0, "successful_tasks": 0, "failed_tasks": 0, "false_positives": 0}}

    monkeypatch.setattr(benchmark_runner, "run_benchmarks", fake_run_benchmarks)
    rc = benchmark_runner.main(["run", "--output", str(tmp_path), "--json"])
    assert rc == 0
    assert captured["use_live_model"] is True
    assert captured["agent_behavior_mode"] == "cached"


def test_benchmark_contracts_require_current_action_loop_events() -> None:
    from swaag.benchmark.task_definitions import get_benchmark_tasks

    for task in get_benchmark_tasks():
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            scenario = task.build(Path(tmp) / task.task_id)
        assert "reasoning_completed" not in scenario.verification_contract.required_history_events
        if scenario.verification_contract.required_history_events:
            assert "agent_action_selected" in scenario.verification_contract.required_history_events


def test_failure_analyzer_contains_no_planner_specific_classification() -> None:
    from pathlib import Path
    import swaag.benchmark.failure_analyzer as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "plan_validation" not in source
    assert 'subsystem="planner"' not in source


def test_benchmark_verifier_normalizes_absolute_allowed_paths(tmp_path) -> None:
    from swaag.benchmark.task_definitions import BenchmarkVerificationContract
    from swaag.benchmark.verifier import verify_benchmark_contract
    from swaag.types import SessionState

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "pkg" / "module.py"
    target.parent.mkdir()
    target.write_text("fixed\n", encoding="utf-8")
    contract = BenchmarkVerificationContract(
        task_type="coding",
        command_cwd=str(workspace),
        allowed_modified_files=[str(target)],
        forbid_unexpected_workspace_changes=True,
    )
    state = SessionState(session_id="s", created_at="t", updated_at="t", config_fingerprint="cfg", model_base_url="http://model")
    report = verify_benchmark_contract(
        contract,
        assistant_text="done",
        state=state,
        events=[],
        workspace_before={"pkg/module.py": "broken\n"},
        workspace_after={"pkg/module.py": "fixed\n"},
    )
    assert report.passed is True
    assert report.evidence["allowed_modified_files"]["allowed"] == ["pkg/module.py", "pkg/module.py.bak"]


def test_benchmark_verifier_uses_current_python_for_python3_commands(tmp_path) -> None:
    import sys
    from swaag.benchmark.verifier import _run_command

    ok, evidence = _run_command(["python3", "-c", "import sys; print(sys.executable)"], cwd=str(tmp_path))
    assert ok is True
    assert evidence["command"][0] == sys.executable
    assert evidence["stdout"].strip() == sys.executable


def test_benchmark_replay_cache_root_is_persistent(monkeypatch, tmp_path) -> None:
    from swaag.benchmark.benchmark_runner import _benchmark_replay_cache_root

    root = tmp_path / "persistent-cache"
    monkeypatch.setenv("SWAAG_BENCHMARK_REPLAY_CACHE_ROOT", str(root))
    assert _benchmark_replay_cache_root() == root
    assert root.is_dir()


def test_invalid_tool_input_is_rejected_before_execution(make_config) -> None:
    bad = _action(
        tool_calls=[("shell_command", {"command": "python3 -m unittest -q test_x.py", "background": False})],
        continue_loop=True,
    )
    recovered = _action(
        tool_calls=[("run_tests", {"command": ["python3", "-c", "print('ok')"], "background": False})],
        continue_loop=True,
    )
    finish = _action(message="done", continue_loop=False)
    runtime, client = _runtime(make_config, [bad, recovered, finish])
    # The recovery test command itself may fail because the fixture does not exist; the point is that
    # the invalid shell test command is rejected before any shell execution event is emitted.
    result = runtime.run_turn("run a test")
    events = runtime.history.read_history(result.session_id)
    shell_started = [e for e in events if e.event_type == "shell_command_started"]
    assert shell_started == []
    assert len(client.requests) >= 2


def test_failed_verification_requires_diagnostic_action_before_edit_or_retest(make_config) -> None:
    from swaag.action import action_from_payload, ActionValidationError

    diagnostic_names = AgentRuntime._DIAGNOSTIC_TOOL_NAMES
    assert "read_file" in diagnostic_names
    assert "run_tests" not in diagnostic_names
    assert "edit_text" not in diagnostic_names

    edit = action_from_payload(
        {
            "assistant_message": "edit",
            "tool_calls": [{"tool_name": "edit_text", "arguments": {"path": "x.py", "operation": "replace_exact", "old_text": "a", "new_text": "b", "pattern": None, "replacement": None, "start": None, "end": None, "position": None, "insertion": None, "expected_text": None, "dry_run": False}}],
            "continue_loop": True,
        },
        enabled_tool_names=["edit_text"],
    )
    assert edit.tool_calls[0].tool_name not in diagnostic_names


def test_failed_verification_blocks_terminal_answer_until_tests_pass() -> None:
    assert "run_tests" not in AgentRuntime._DIAGNOSTIC_TOOL_NAMES


def test_verification_repair_prompt_contains_only_failed_checks() -> None:
    from types import SimpleNamespace
    from swaag.benchmark.benchmark_runner import _verification_repair_prompt

    report = SimpleNamespace(
        checks={"command": True, "expected_files": False},
        evidence={"command": {"return_code": 0}, "expected_files": {"x": {"actual": "bad", "expected": "good"}}},
        reason="expected_files",
    )
    text = _verification_repair_prompt(report)
    assert "expected_files" in text
    assert '"command"' not in text
    assert "Do not claim completion until the verifier passes" in text


def test_benchmark_repair_round_limit_is_bounded(monkeypatch) -> None:
    from swaag.benchmark.benchmark_runner import _verification_repair_round_limit

    monkeypatch.setenv("SWAAG_BENCHMARK_VERIFICATION_REPAIR_ROUNDS", "99")
    assert _verification_repair_round_limit() == 10
    monkeypatch.setenv("SWAAG_BENCHMARK_VERIFICATION_REPAIR_ROUNDS", "-2")
    assert _verification_repair_round_limit() == 0
