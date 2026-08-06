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
    assert not any(
        event.payload.get("kind") in {"analysis", "task_decision", "plan", "subagent_selection", "verification"}
        for event in events
    )


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


def test_identical_model_tool_response_is_cut_off_mechanically(make_config) -> None:
    repeated = _action(
        tool_calls=[("calculator", {"expression": "1 + 1"})],
        continue_loop=True,
    )
    runtime, client = _runtime(make_config, [repeated, repeated, repeated])

    result = runtime.run_turn("Keep calculating 1 + 1 forever.")

    assert "same constrained response repeated" in result.assistant_text
    assert len(client.requests) == 3
    assert len(result.tool_results) == 2
    assert all(item.output["result"] == 2 for item in result.tool_results)


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
