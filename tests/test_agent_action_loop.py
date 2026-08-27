from __future__ import annotations

import json
from typing import Any, Callable

from swaag.model import CompletionRequestPolicy
from swaag.runtime import AgentRuntime
from swaag.types import Message
from swaag.utils import stable_json_dumps
from swaag.grammar import agent_action_contract
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


class OutputLimitedFakeModelClient(FakeModelClient):
    def send_completion(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
        progress_callback=None,
    ) -> CompletionResult:
        if not self.requests:
            self.requests.append(payload)
            return CompletionResult(
                text="{",
                raw_request=payload,
                raw_response={"content": "{", "stop_type": "limit"},
                prompt_tokens=None,
                completion_tokens=payload["n_predict"],
                finish_reason="length",
            )
        return super().send_completion(
            payload,
            timeout_seconds=timeout_seconds,
            progress_callback=progress_callback,
        )


class CharacterCountSummaryClient(FakeModelClient):
    def __init__(self, marker: str):
        super().__init__([])
        self.marker = marker

    def tokenize(self, text: str) -> int:
        return len(text)

    def send_completion(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
        progress_callback=None,
    ) -> CompletionResult:
        del timeout_seconds, progress_callback
        self.requests.append(payload)
        summary = self.marker if self.marker in str(payload["prompt"]) else "fragment retained"
        response = json.dumps(
            {"summary": summary, "preserve_recent_messages": 0}
        )
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
    situation: str = "Working on the current request.",
    status_action: str = "Choose and execute the next useful action.",
    reason: str = "This advances the user's request using current evidence.",
    importance: str = "normal",
    silent_completion: bool = False,
) -> str:
    return json.dumps(
        {
            "assistant_message": message,
            "tool_calls": [
                {"tool_name": name, "arguments": arguments}
                for name, arguments in (tool_calls or [])
            ],
            "continue_loop": continue_loop,
            "silent_completion": silent_completion,
            "status": {
                "situation": situation,
                "action": status_action,
                "reason": reason,
                "importance": importance,
            },
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
    assert set(schema["properties"]) == {"assistant_message", "tool_calls", "continue_loop", "silent_completion", "status", "questions"}
    events = runtime.history.read_history(result.session_id)
    assert not any(event.event_type in {"plan_created", "plan_updated"} for event in events)
    assert not any(event.event_type.startswith("plan_") for event in events)
    sent = next(event for event in events if event.event_type == "model_request_sent")
    provenance = sent.payload["context_provenance"]
    context_reference = provenance["context_compiled"]
    prompt_reference = provenance["prompt_built"]
    by_sequence = {event.sequence: event for event in events}
    assert by_sequence[context_reference["sequence"]].hash == context_reference["event_hash"]
    assert by_sequence[prompt_reference["sequence"]].hash == prompt_reference["event_hash"]
    assert provenance["input_tokens"] == sent.payload["budget_report"]["input_tokens"]
    assert {item["category"] for item in provenance["components"]} >= {
        "instruction",
        "current_user",
        "constraint_schema",
    }


def test_output_limit_rebuilds_action_with_more_headroom(make_config) -> None:
    config = make_config(model__context_limit=12_000)
    client = OutputLimitedFakeModelClient([_action(message="done")])
    runtime = AgentRuntime(config, model_client=client)

    result = runtime.run_turn("Return a concise answer.")

    assert result.assistant_text == "done"
    assert len(client.requests) == 2
    assert client.requests[1]["n_predict"] > client.requests[0]["n_predict"]
    events = runtime.history.read_history(result.session_id)
    exhausted = [event for event in events if event.event_type == "model_output_budget_exhausted"]
    assert len(exhausted) == 1


def test_history_compaction_creates_replayable_summary_with_exact_sources(
    make_config,
) -> None:
    config = make_config(model__context_limit=32_000)
    client = FakeModelClient(
        [json.dumps({"summary": "Earlier facts summarized.", "preserve_recent_messages": 0})]
    )
    runtime = AgentRuntime(config, model_client=client)
    state = runtime.create_or_load_session()
    for role, content in (
        ("user", "first " * 200),
        ("assistant", "second"),
        ("user", "third"),
        ("assistant", "fourth"),
    ):
        runtime._record_message(state, Message(role=role, content=content, created_at="t"))

    assert runtime._compact_once(state) is True

    assert state.messages[0].role == "summary"
    assert state.messages[0].metadata["source_event_references"]
    assert state.messages[0].metadata["projection_event_sequence"] > 0
    compressed = next(
        event
        for event in runtime.history.read_history(state.session_id)
        if event.event_type == "history_compressed"
    )
    assert compressed.payload["candidate_source_message_count"] == 1
    assert compressed.payload["actual_recovered_tokens"] > 0
    assert compressed.payload["required_recovery_tokens"] == 1
    rebuilt = runtime.history.rebuild_from_history(state.session_id, prefer_checkpoint=False)
    assert rebuilt.messages == state.messages


def test_history_summary_recompiles_after_output_starvation(make_config) -> None:
    config = make_config(
        model__context_limit=32_000,
        model__max_retries=1,
    )
    client = OutputLimitedFakeModelClient(
        [json.dumps({"summary": "Earlier facts retained.", "preserve_recent_messages": 0})]
    )
    runtime = AgentRuntime(config, model_client=client)
    state = runtime.create_or_load_session()
    for role, content in (
        ("user", "first fact " * 200),
        ("assistant", "second fact"),
        ("user", "third fact"),
        ("assistant", "fourth fact"),
    ):
        runtime._record_message(
            state,
            Message(role=role, content=content, created_at="t"),
        )

    assert runtime._compact_once(state) is True
    assert len(client.requests) == 2
    assert client.requests[1]["n_predict"] > client.requests[0]["n_predict"]
    assert state.messages[0].content == "Earlier facts retained."


def test_history_compaction_target_recovers_only_the_measured_deficit(
    make_config,
) -> None:
    runtime = AgentRuntime(
        make_config(model__context_limit=32_000),
        model_client=FakeModelClient([]),
    )
    state = runtime.create_or_load_session()
    runtime._record_message(
        state,
        Message(role="user", content="durable evidence " * 400, created_at="t"),
    )
    source = state.messages[:1]

    small = runtime._history_compaction_target(
        state,
        source,
        required_recovery_tokens=50,
    )
    large = runtime._history_compaction_target(
        state,
        source,
        required_recovery_tokens=250,
    )

    assert small is not None and large is not None
    assert small[1] == large[1]
    assert small[0] - large[0] == 200


def test_oversized_single_history_message_is_hierarchically_summarized(
    make_config,
) -> None:
    marker = "critical-history-marker-731"
    config = make_config(
        model__context_limit=2_000,
        context__max_compaction_rounds=4,
        context__safety_margin_tokens=32,
    )
    client = CharacterCountSummaryClient(marker)
    runtime = AgentRuntime(config, model_client=client)
    state = runtime.create_or_load_session()
    for role, content in (
        ("user", "A" * 8_000 + marker),
        ("assistant", "second"),
        ("user", "third"),
        ("assistant", "fourth"),
    ):
        runtime._record_message(
            state,
            Message(role=role, content=content, created_at="t"),
        )
    raw_source = next(
        event
        for event in runtime.history.read_history(state.session_id)
        if event.event_type == "message_added" and marker in str(event.payload)
    )

    assert runtime._compact_once(state) is True

    assert state.messages[0].role == "summary"
    assert marker in state.messages[0].content
    assert state.messages[0].metadata["source_event_references"][0]["hash"] == raw_source.hash
    assert len(client.requests) > 1
    compressed = [
        event
        for event in runtime.history.read_history(state.session_id)
        if event.event_type == "history_compressed"
    ]
    assert compressed[-1].payload["hierarchical"] is True
    assert any(marker in str(event.payload) for event in runtime.history.read_history(state.session_id))


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
    assert len(result.tool_results) == 1
    events = runtime.history.read_history(result.session_id)
    assert any(
        event.event_type == "agent_action_rejected"
        and "materially different next action" in str(event.payload.get("reason", ""))
        for event in events
    )


def test_failure_analyzer_supports_current_action_loop_metrics_without_legacy_fields() -> None:
    from swaag.benchmark.failure_analyzer import FailureAnalyzer
    from swaag.types import HistoryEvent, SessionState

    state = SessionState(session_id="s", created_at="t", updated_at="t", config_fingerprint="cfg", model_base_url="http://127.0.0.1:14829")
    state.metrics.action_count = 1
    events = [
        HistoryEvent(
            id="e1", sequence=1, session_id="s", timestamp="t", type="agent_action_selected", version=1,
            payload={"action_index": 1, "action": {"assistant_message": "done", "tool_calls": [], "continue_loop": False}, "occurrence": 1},
        )
    ]

    failure = FailureAnalyzer().analyze(state=state, events=events, deterministic_verification_passed=False, runtime_error=None)

    assert failure.category == "premature_termination"
    assert failure.evidence["actions"] == 1


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


def test_context_discovery_reads_current_props_shape(monkeypatch) -> None:
    import json

    from swaag.benchmark import benchmark_runner

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"default_generation_settings": {"n_ctx": 22016}}).encode()

    monkeypatch.setattr(benchmark_runner.urllib.request, "urlopen", lambda request, timeout: Response())

    assert benchmark_runner._discover_server_context_limit(
        "http://127.0.0.1:14829", timeout_seconds=5
    ) == 22016


def test_duration_parser_supports_recording_units() -> None:
    from swaag.scheduler import parse_duration

    assert parse_duration("250 ms").total_seconds() == 0.25
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
    assert claimed[0].status == "claimed"
    delivered = WakeupStore(tmp_path).mark_delivered(wakeup_id=wakeup.wakeup_id, now=now + timedelta(hours=3))
    assert delivered.status == "delivered"
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


def test_failed_run_tests_is_evidence_not_permanent_completion_gate(make_config) -> None:
    failing = _action(
        tool_calls=[("run_tests", {"command": ["python3", "-c", "raise SystemExit(1)"], "background": False})],
        continue_loop=True,
    )
    finish = _action(message="The test command was irrelevant to the requested reading task; here is the answer.", continue_loop=False)
    runtime, client = _runtime(make_config, [failing, finish])
    result = runtime.run_turn("Read the supplied text and answer directly; no test suite is required.")
    assert "irrelevant" in result.assistant_text
    events = runtime.history.read_history(result.session_id)
    assert not any(
        event.event_type == "agent_action_rejected" and "verification" in str(event.payload.get("reason", "")).lower()
        for event in events
    )
    assert len(client.requests) == 2


def test_zero_tool_budget_removes_tools_from_action_schema_and_prompt(make_config) -> None:
    config = make_config(runtime__tool_call_budget=0, runtime__max_total_actions=1, model__context_limit=8192)
    seen = {}

    def capture(payload):
        seen["prompt"] = payload["prompt"]
        seen["schema"] = payload["json_schema"]
        return _action(message="done", continue_loop=False)

    client = FakeModelClient([capture])
    runtime = AgentRuntime(config, model_client=client)
    result = runtime.run_turn("Answer from the supplied context without tools.")
    assert result.assistant_text == "done"
    assert '"enum": []' in str(seen["schema"]) or "'enum': []" in str(seen["schema"])
    assert "If no tools are listed" in seen["prompt"]


def test_immediate_duplicate_action_is_rejected_before_second_execution(make_config, tmp_path) -> None:
    config = make_config(runtime__tool_call_budget=4, runtime__max_total_actions=3, model__context_limit=32_000)
    config.sessions.root = tmp_path / "sessions"
    config.tools.read_roots = [tmp_path]
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    repeated = _action(tool_calls=[("read_file", {"path": "a.txt"})], continue_loop=True)
    finish = _action(message="done", continue_loop=False)
    client = FakeModelClient([repeated, repeated, finish])
    runtime = AgentRuntime(config, model_client=client)
    result = runtime.run_turn("Read a.txt once, then answer.")
    events = runtime.history.read_history(result.session_id)
    called = [e for e in events if e.event_type == "tool_called" and e.payload.get("tool_name") == "read_file"]
    assert len(called) == 1
    assert any(e.event_type == "agent_action_rejected" and "immediately preceding action" in str(e.payload.get("reason", "")) for e in events)


def test_benchmark_answer_fragments_are_case_insensitive_and_support_alternatives(tmp_path) -> None:
    from swaag.benchmark.task_definitions import BenchmarkVerificationContract
    from swaag.benchmark.verifier import verify_benchmark_contract
    from swaag.types import SessionState

    contract = BenchmarkVerificationContract(
        task_type="failure",
        expected_answer_contains=["policy", "protected.log"],
        expected_answer_any_of=[["cannot", "refuse", "not allowed"]],
    )
    state = SessionState(session_id="s", created_at="t", updated_at="t", config_fingerprint="cfg", model_base_url="http://model")
    report = verify_benchmark_contract(
        contract,
        assistant_text="Policy requires preserving protected.log; this action is Not Allowed.",
        state=state,
        events=[],
        workspace_before={},
        workspace_after={},
        workspace_root=str(tmp_path),
    )
    assert report.passed is True
    assert report.checks["expected_answer_contains"] is True
    assert report.checks["expected_answer_any_of"] is True

def test_action_schema_disallows_silent_completion_by_default() -> None:
    from swaag.grammar import agent_action_contract
    schema = agent_action_contract([]).json_schema
    assert schema["properties"]["silent_completion"]["enum"] == [False]
    allowed = agent_action_contract([], allow_silent_completion=True).json_schema
    assert "enum" not in allowed["properties"]["silent_completion"]


def test_action_parser_rejects_empty_terminal_message_without_explicit_silence() -> None:
    import pytest
    from swaag.action import ActionValidationError, action_from_payload
    with pytest.raises(ActionValidationError, match="Terminal actions require a non-empty assistant_message"):
        action_from_payload(
            {"assistant_message": "", "tool_calls": [], "continue_loop": False, "silent_completion": False},
            enabled_tool_names=[],
        )


def test_action_parser_allows_explicit_silent_terminal_completion() -> None:
    from swaag.action import action_from_payload
    action = action_from_payload(
        {"assistant_message": "", "tool_calls": [], "continue_loop": False, "silent_completion": True},
        enabled_tool_names=[],
    )
    assert action.assistant_message == ""
    assert action.tool_calls == []
    assert action.continue_loop is False
    assert action.silent_completion is True


def test_runtime_can_finish_empty_after_successful_tool_result(make_config, tmp_path) -> None:
    config = make_config(runtime__tool_call_budget=2, runtime__max_total_actions=2, model__context_limit=32_000)
    config.sessions.root = tmp_path / "sessions"
    config.tools.read_roots = [tmp_path]
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    read = _action(tool_calls=[("read_file", {"path": "a.txt"})], continue_loop=True)
    finish = _action(message="", continue_loop=False, silent_completion=True)
    client = FakeModelClient([read, finish])
    runtime = AgentRuntime(config, model_client=client)
    result = runtime.run_turn(
        "Read a.txt. No prose response is required after the read succeeds.",
        allow_silent_completion=True,
    )
    assert result.assistant_text == ""
    assert len(result.tool_results) == 1
    events = runtime.history.read_history(result.session_id)
    assert not any(e.event_type == "agent_action_rejected" for e in events)


def test_action_prompt_explains_cross_action_tool_result_dependencies(make_config) -> None:
    config = make_config(model__context_limit=32_000)
    runtime = AgentRuntime(config, model_client=FakeModelClient([]))
    state = runtime.create_or_load_session()
    prepared = runtime._prepare_action_call(
        state,
        original_request="Inspect output then use its returned handle.",
        pending_messages=[],
        tool_specs=runtime.tools.prompt_tuples(config),
        contract=agent_action_contract(runtime.tools.prompt_tuples(config)),
        validation_feedback="",
    )
    prompt = prepared.assembly.prompt_text
    assert "All tool_calls in one action are selected before any of them execute" in prompt
    assert "issue the dependent call in the next action" in prompt
    assert "do not simply stop on the failed verification" in prompt


def test_environment_context_exposes_latest_mechanical_handles(make_config, tmp_path) -> None:
    config = make_config(model__context_limit=32_000)
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()
    runtime.history.record_event(
        state,
        "tool_result",
        {
            "tool_name": "shell_command",
            "raw_input": {},
            "validated_input": {},
            "output": {"stdout_artifact_id": "artifact_abc123"},
        },
    )
    runtime.history.record_event(
        state,
        "tool_result",
        {
            "tool_name": "terminal",
            "raw_input": {},
            "validated_input": {},
            "output": {"terminal_id": "terminal_xyz789"},
        },
    )
    components = runtime._runtime_context_components(state, runtime._counter(state))
    environment = next(item.text for item in components if item.name == "environment_state")
    assert '"latest_handles"' in environment
    assert '"stdout_artifact_id": "artifact_abc123"' in environment
    assert '"terminal_id": "terminal_xyz789"' in environment
    runtime.history.record_event(
        state,
        "tool_result",
        {
            "tool_name": "read_artifact",
            "raw_input": {},
            "validated_input": {},
            "output": {"artifact_id": "artifact_abc123", "next_offset": 4096, "finished": False},
        },
    )
    components = runtime._runtime_context_components(state, runtime._counter(state))
    environment = next(item.text for item in components if item.name == "environment_state")
    assert '"latest_artifact_cursor"' in environment
    assert '"next_offset": 4096' in environment


def test_action_prompt_requires_requested_side_effects_before_final_message(make_config) -> None:
    config = make_config(model__context_limit=32_000)
    runtime = AgentRuntime(config, model_client=FakeModelClient([]))
    state = runtime.create_or_load_session()
    specs = runtime.tools.prompt_tuples(config)
    prepared = runtime._prepare_action_call(
        state,
        original_request="Recover a value, write it to a file, and verify it.",
        pending_messages=[],
        tool_specs=specs,
        contract=agent_action_contract(specs),
        validation_feedback="",
    )
    prompt = prepared.assembly.prompt_text
    assert "never a substitute for an explicitly requested mechanical side effect" in prompt
    assert "apply the required state change and verify again before finishing" in prompt


def test_selected_action_persists_structured_status(make_config) -> None:
    runtime, _client = _runtime(
        make_config,
        [_action(
            message="done",
            situation="The requested fact is available in current evidence.",
            status_action="Return the grounded answer.",
            reason="No further tool work is required.",
            importance="minor",
        )],
    )
    result = runtime.run_turn("Answer the request.")
    events = runtime.history.read_history(result.session_id)
    status = next(event for event in events if event.event_type == "agent_status")
    assert status.payload == {
        "action_index": 1,
        "situation": "The requested fact is available in current evidence.",
        "action": "Return the grounded answer.",
        "reason": "No further tool work is required.",
        "importance": "minor",
        "importance_rank": 1,
    }


def test_action_and_validation_retries_use_reproducibly_distinct_seeds(make_config) -> None:
    # First response is malformed JSON, second is a valid action for the same action
    # index, then the next action should advance the deterministic seed again.
    runtime, client = _runtime(
        make_config,
        [
            '{"assistant_message":',
            _action(tool_calls=[("calculator", {"expression": "1 + 1"})], continue_loop=True),
            _action(message="2", continue_loop=False),
        ],
    )
    runtime.config.model.seed = 100
    result = runtime.run_turn("Calculate 1 + 1 and answer.")
    assert result.assistant_text == "2"
    assert [request["seed"] for request in client.requests] == [100, 101, 103]


def test_action_seed_schedule_is_deterministic_for_same_base_seed(make_config) -> None:
    def run_once() -> list[int]:
        runtime, client = _runtime(
            make_config,
            [
                _action(tool_calls=[("calculator", {"expression": "2 + 2"})], continue_loop=True),
                _action(message="4", continue_loop=False),
            ],
        )
        runtime.config.model.seed = 23
        runtime.run_turn("Calculate 2 + 2.")
        return [request["seed"] for request in client.requests]
    assert run_once() == run_once() == [23, 26]


def test_duplicate_tool_action_ignores_cosmetic_status_changes(make_config, tmp_path) -> None:
    config = make_config(runtime__tool_call_budget=4, runtime__max_total_actions=3, model__context_limit=32_000)
    config.sessions.root = tmp_path / "sessions"
    config.tools.read_roots = [tmp_path]
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    first = _action(
        tool_calls=[("read_file", {"path": "a.txt"})],
        continue_loop=True,
        situation="Need the file.",
        status_action="Read it.",
        reason="Gather evidence.",
    )
    repeated_with_new_status = _action(
        tool_calls=[("read_file", {"path": "a.txt"})],
        continue_loop=True,
        situation="I have read the file.",
        status_action="Read it again.",
        reason="Double-checking.",
    )
    finish = _action(message="alpha", continue_loop=False)
    client = FakeModelClient([first, repeated_with_new_status, finish])
    runtime = AgentRuntime(config, model_client=client)
    result = runtime.run_turn("Read a.txt once and answer with its content.")
    events = runtime.history.read_history(result.session_id)
    called = [e for e in events if e.event_type == "tool_called" and e.payload.get("tool_name") == "read_file"]
    assert len(called) == 1
    assert any(e.event_type == "agent_action_rejected" and "immediately preceding action" in str(e.payload.get("reason", "")) for e in events)


def test_repeated_pure_call_is_rejected_until_state_changes(make_config, tmp_path) -> None:
    config = make_config(runtime__tool_call_budget=5, runtime__max_total_actions=6, model__context_limit=32_000, tools__allow_side_effect_tools=True)
    config.sessions.root = tmp_path / "sessions"
    config.tools.read_roots = [tmp_path]
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bravo\n", encoding="utf-8")
    first_read = _action(tool_calls=[("read_file", {"path": "a.txt"})], continue_loop=True)
    different_read = _action(tool_calls=[("read_file", {"path": "b.txt"})], continue_loop=True)
    redundant_read = _action(
        tool_calls=[("read_file", {"path": "a.txt"})],
        continue_loop=True,
        situation="Checking again.", status_action="Reread.", reason="Double-check.",
    )
    write = _action(tool_calls=[("write_file", {"path": "a.txt", "content": "beta\n", "create": False})], continue_loop=True)
    reread_after_change = _action(tool_calls=[("read_file", {"path": "a.txt"})], continue_loop=True)
    finish = _action(message="beta", continue_loop=False)
    client = FakeModelClient([first_read, different_read, redundant_read, write, reread_after_change, finish])
    runtime = AgentRuntime(config, model_client=client)
    result = runtime.run_turn("Read a.txt and b.txt, change a.txt to beta, then reread a.txt.")
    events = runtime.history.read_history(result.session_id)
    reads = [e for e in events if e.event_type == "tool_called" and e.payload.get("tool_name") == "read_file"]
    assert len(reads) == 3
    assert any(e.event_type == "agent_action_rejected" and "repeats observation calls" in str(e.payload.get("reason", "")) for e in events)
    assert result.assistant_text == "beta"


def test_action_prompt_requires_explicit_decision_not_only_evidence(make_config) -> None:
    config = make_config(model__context_limit=32_000)
    runtime = AgentRuntime(config, model_client=FakeModelClient([]))
    state = runtime.create_or_load_session()
    specs = runtime.tools.prompt_tuples(config)
    prepared = runtime._prepare_action_call(
        state,
        original_request="Choose the only justified next move.",
        pending_messages=[],
        tool_specs=specs,
        contract=agent_action_contract(specs),
        validation_feedback="",
    )
    prompt = prepared.assembly.prompt_text
    assert "final assistant_message must explicitly state that decision/conclusion" in prompt


def test_environment_state_exposes_authoritative_active_session(make_config) -> None:
    config = make_config(model__context_limit=32_000)
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()
    components = runtime._runtime_context_components(state, runtime._counter(state))
    environment = next(item.text for item in components if item.name == "environment_state")
    assert '"active_session"' in environment
    assert f'"session_id": "{state.session_id}"' in environment


def test_rejected_duplicate_does_not_consume_accepted_action_budget(make_config, tmp_path) -> None:
    config = make_config(runtime__tool_call_budget=2, runtime__max_total_actions=2, model__context_limit=32_000)
    config.sessions.root = tmp_path / "sessions"
    config.tools.read_roots = [tmp_path]
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    read = _action(tool_calls=[("read_file", {"path": "a.txt"})], continue_loop=True)
    duplicate = _action(
        tool_calls=[("read_file", {"path": "a.txt"})],
        continue_loop=True,
        situation="Checking again.", status_action="Reread.", reason="Double-checking.",
    )
    finish = _action(message="alpha", continue_loop=False)
    client = FakeModelClient([read, duplicate, finish])
    runtime = AgentRuntime(config, model_client=client)
    result = runtime.run_turn("Read a.txt once and answer with its content.")
    assert result.assistant_text == "alpha"
    events = runtime.history.read_history(result.session_id)
    called = [e for e in events if e.event_type == "tool_called" and e.payload.get("tool_name") == "read_file"]
    assert len(called) == 1
    terminal = [e for e in events if e.event_type == "agent_action_terminal"]
    assert terminal[-1].payload["action_index"] == 2


def test_validation_retry_exhaustion_retries_same_semantic_action(make_config) -> None:
    config = make_config(runtime__tool_call_budget=0, runtime__max_total_actions=1, model__context_limit=32_000)
    bad = '{"assistant_message":'
    finish = _action(message="recovered", continue_loop=False)
    client = FakeModelClient([bad, bad, bad, finish])
    runtime = AgentRuntime(config, model_client=client)
    result = runtime.run_turn("Answer directly.")
    assert result.assistant_text == "recovered"
    events = runtime.history.read_history(result.session_id)
    terminal = [e for e in events if e.event_type == "agent_action_terminal"]
    assert terminal[-1].payload["action_index"] == 1
    assert len(client.requests) == 4


def test_duplicate_recovery_feedback_contains_exact_calls_and_edit_reread_guidance(make_config, tmp_path) -> None:
    config = make_config(runtime__tool_call_budget=2, runtime__max_total_actions=3, model__context_limit=32_000, tools__allow_side_effect_tools=True)
    config.sessions.root = tmp_path / "sessions"
    config.tools.read_roots = [tmp_path]
    (tmp_path / "a.py").write_text("value = 1\n", encoding="utf-8")
    first = _action(
        tool_calls=[("edit_text", {"path": "a.py", "operation": "replace_exact", "old_text": "missing", "new_text": "value = 2", "replace_all": False})],
        continue_loop=True,
    )
    duplicate = _action(
        tool_calls=[("edit_text", {"path": "a.py", "operation": "replace_exact", "old_text": "missing", "new_text": "value = 2", "replace_all": False})],
        continue_loop=True,
        situation="Retrying.", status_action="Retry edit.", reason="Try again.",
    )
    read = _action(tool_calls=[("read_file", {"path": "a.py"})], continue_loop=True)
    finish = _action(message="done", continue_loop=False)
    client = FakeModelClient([first, duplicate, read, finish])
    runtime = AgentRuntime(config, model_client=client)
    result = runtime.run_turn("Inspect a.py and fix it if needed.")
    assert result.assistant_text == "done"
    events = runtime.history.read_history(result.session_id)
    rejected = [e for e in events if e.event_type == "agent_action_rejected"]
    reason = "\n".join(str(e.payload.get("reason", "")) for e in rejected)
    assert '"tool_name":"edit_text"' in reason
    assert "reread the current target file" in reason
    assert "a.py" in reason


def test_repeated_observation_feedback_requires_synthesis_from_existing_evidence(make_config, tmp_path) -> None:
    config = make_config(runtime__tool_call_budget=3, runtime__max_total_actions=3, model__context_limit=32_000)
    config.sessions.root = tmp_path / "sessions"
    config.tools.read_roots = [tmp_path]
    (tmp_path / "facts.txt").write_text("owner=team-blue\n", encoding="utf-8")
    (tmp_path / "other.txt").write_text("status=green\n", encoding="utf-8")
    first = _action(tool_calls=[("read_file", {"path": "facts.txt"})], continue_loop=True)
    different = _action(tool_calls=[("read_file", {"path": "other.txt"})], continue_loop=True)
    repeated = _action(
        tool_calls=[("read_file", {"path": "facts.txt"})],
        continue_loop=True,
        situation="Need certainty.", status_action="Read again.", reason="Double-check.",
    )
    finish = _action(message='{"owner":"team-blue"}', continue_loop=False)
    client = FakeModelClient([first, different, repeated, finish])
    runtime = AgentRuntime(config, model_client=client)
    result = runtime.run_turn("Read facts.txt and other.txt and return the owner as JSON.")
    assert result.assistant_text == '{"owner":"team-blue"}'
    events = runtime.history.read_history(result.session_id)
    rejected = [e for e in events if e.event_type == "agent_action_rejected"]
    reason = "\n".join(str(e.payload.get("reason", "")) for e in rejected)
    assert "Already-observed calls" in reason
    assert "synthesize and return the final answer now" in reason
    assert 'facts.txt' in reason


def test_repeated_observation_is_allowed_after_compaction_removes_visible_result(make_config, tmp_path) -> None:
    config = make_config(runtime__tool_call_budget=4, runtime__max_total_actions=4, model__context_limit=32_000)
    config.sessions.root = tmp_path / "sessions"
    config.tools.read_roots = [tmp_path]
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    runtime, _client = _runtime(make_config, [_action(message="unused", continue_loop=False)])
    runtime.config = config
    state = runtime.create_or_load_session()
    signature = stable_json_dumps({"tool_name": "read_file", "arguments": {"path": "a.txt"}}, indent=None)
    runtime._record_message(state, Message(role="tool", name="read_file", content="alpha", created_at="t", metadata={"validated_input": {"path": "a.txt"}}))
    assert signature in runtime._visible_observation_signatures(state)
    state.messages = [Message(role="system", content="summary only", created_at="t")]
    assert signature not in runtime._visible_observation_signatures(state)


def test_duplicate_no_progress_limit_stops_boundedly(make_config, tmp_path) -> None:
    config = make_config(runtime__tool_call_budget=4, runtime__max_total_actions=10, runtime__max_repeated_action_occurrences=2, model__context_limit=32_000)
    config.sessions.root = tmp_path / "sessions"
    config.tools.read_roots = [tmp_path]
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    repeated = _action(tool_calls=[("read_file", {"path": "a.txt"})], continue_loop=True)
    runtime, client = _runtime(make_config, [repeated, repeated, repeated, repeated])
    runtime.config = config
    result = runtime.run_turn("Read a.txt and answer.")
    assert "no-progress limit" in result.assistant_text
    assert len(client.requests) == 4
