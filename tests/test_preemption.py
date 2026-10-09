from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from swaag.communication import CommunicationService
from swaag.model import CompletionRequestPolicy
from swaag.preemption import ModelCallPreempted
from swaag.runtime import AgentRuntime
from swaag.types import CompletionResult, ContractSpec, Message
from swaag.utils import stable_json_dumps, utc_now_iso


def _action(message: str) -> str:
    return json.dumps(
        {
            "assistant_message": message,
            "tool_calls": [],
            "continue_loop": False,
            "silent_completion": False,
            "status": {
                "situation": "Responding.",
                "action": "Return the response.",
                "reason": "The required evidence is available.",
                "importance": "normal",
            },
        }
    )


def _orchestrator_interaction(
    answer: str = "",
    *,
    route: str = "respond",
    reason: str = "This interaction does not require delegated work.",
) -> str:
    return json.dumps(
        {
            "route": route,
            "answer": answer,
            "reason": reason,
        }
    )


def _status(
    message: str,
    *,
    escalate: bool = False,
    escalation_reason: str = "",
) -> str:
    return json.dumps(
        {
            "answer": message,
            "situation": "The worker status was interpreted from durable evidence.",
            "action": "Report the evidence-backed snapshot.",
            "reason": "The communication operation is independent of worker action selection.",
            "importance": "normal",
            "evidence_sequences": [],
            "uncertainty": "No event citation was needed for this test response.",
            "escalate_to_stronger_model": escalate,
            "escalation_reason": escalation_reason,
        }
    )


class _BaseClient:
    is_deterministic_test_client = True

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

    def tokenize(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def tokenize_selection(self, text: str) -> int:
        return self.tokenize(text)

    def select_request_policy(self, *, contract: ContractSpec, kind: str, prompt: str, max_tokens: int, live_mode: bool = False) -> CompletionRequestPolicy:
        return CompletionRequestPolicy(
            profile_name="test",
            structured_output_mode="server_schema",
            effective_contract_mode=contract.mode,
            effective_timeout_seconds=5,
            progress_poll_seconds=0.01,
        )

    def resolve_contract(self, contract: ContractSpec, *, kind: str, prompt: str, max_tokens: int, live_mode: bool = False):
        return contract, self.select_request_policy(
            contract=contract,
            kind=kind,
            prompt=prompt,
            max_tokens=max_tokens,
            live_mode=live_mode,
        )

    def build_completion_request(self, prompt: str, *, max_tokens: int, contract: ContractSpec, temperature: float | None = None) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0 if temperature is None else temperature,
            "contract": contract.name,
            "json_schema": contract.json_schema,
        }

    @staticmethod
    def _result(payload: dict[str, Any], text: str) -> CompletionResult:
        return CompletionResult(
            text=text,
            raw_request=payload,
            raw_response={"content": text},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


class _PreemptReplayClient(_BaseClient):
    def __init__(self) -> None:
        super().__init__()
        self.main_started = threading.Event()
        self.first_main_request: dict[str, Any] | None = None
        self.replay_verified = False

    def send_completion(self, payload: dict[str, Any], *, timeout_seconds: int | None = None, progress_callback=None, cancel_check=None) -> CompletionResult:
        copied = json.loads(stable_json_dumps(payload, indent=None))
        self.requests.append(copied)
        if payload.get("contract") == "communication_status":
            return self._result(payload, _status("The main agent is still working."))
        if payload.get("contract") == "orchestrator_interaction":
            return self._result(
                payload,
                _orchestrator_interaction("Yes, I can hear you."),
            )
        if self.first_main_request is None:
            self.first_main_request = copied
            self.main_started.set()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if cancel_check is not None and cancel_check():
                    raise ModelCallPreempted("test preemption")
                time.sleep(0.005)
            raise AssertionError("main request was not preempted")
        assert copied == self.first_main_request
        self.replay_verified = True
        return self._result(payload, _action("main finished"))


class _InvalidationClient(_BaseClient):
    def __init__(self) -> None:
        super().__init__()
        self.main_started = threading.Event()
        self.blocked = False

    def send_completion(self, payload: dict[str, Any], *, timeout_seconds: int | None = None, progress_callback=None, cancel_check=None) -> CompletionResult:
        copied = json.loads(stable_json_dumps(payload, indent=None))
        self.requests.append(copied)
        prompt = str(payload.get("prompt", ""))
        if not self.blocked:
            self.blocked = True
            self.main_started.set()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if cancel_check is not None and cancel_check():
                    raise ModelCallPreempted("test state-changing preemption")
                time.sleep(0.005)
            raise AssertionError("main request was not preempted")
        assert "redirected objective" in prompt
        return self._result(payload, _action("continued after redirect"))


class _HoldClient(_BaseClient):
    def __init__(self, answer: str) -> None:
        super().__init__()
        self.answer = answer
        self.started = threading.Event()
        self.release = threading.Event()

    def send_completion(self, payload: dict[str, Any], *, timeout_seconds: int | None = None, progress_callback=None, cancel_check=None) -> CompletionResult:
        self.requests.append(json.loads(stable_json_dumps(payload, indent=None)))
        self.started.set()
        assert self.release.wait(timeout=5)
        return self._result(payload, _action(self.answer))


class _ImmediateClient(_BaseClient):
    def __init__(self, answer: str) -> None:
        super().__init__()
        self.answer = answer

    def send_completion(self, payload: dict[str, Any], *, timeout_seconds: int | None = None, progress_callback=None, cancel_check=None) -> CompletionResult:
        self.requests.append(json.loads(stable_json_dumps(payload, indent=None)))
        if payload.get("contract") == "communication_status":
            response = _status(self.answer)
        elif payload.get("contract") == "orchestrator_interaction":
            response = _orchestrator_interaction(self.answer)
        else:
            response = _action(self.answer)
        return self._result(payload, response)


class _EscalatingStatusClient(_ImmediateClient):
    def send_completion(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
        progress_callback=None,
        cancel_check=None,
    ) -> CompletionResult:
        self.requests.append(json.loads(stable_json_dumps(payload, indent=None)))
        if payload.get("contract") == "communication_status":
            return self._result(
                payload,
                _status(
                    self.answer,
                    escalate=True,
                    escalation_reason=(
                        "The question requires stronger semantic interpretation."
                    ),
                ),
            )
        return self._result(payload, _action(self.answer))


class _FailingStatusClient(_ImmediateClient):
    def send_completion(self, payload: dict[str, Any], **_kwargs) -> CompletionResult:
        self.requests.append(json.loads(stable_json_dumps(payload, indent=None)))
        if payload.get("contract") == "communication_status":
            raise RuntimeError("strong status backend unavailable")
        return self._result(payload, _action(self.answer))


def test_same_model_communication_preempts_and_exactly_replays_main_request(make_config) -> None:
    config = make_config(model__context_limit=32_000)
    client = _PreemptReplayClient()
    runtime = AgentRuntime(config, model_client=client)
    state = runtime.create_or_load_session()
    service = CommunicationService(runtime)
    holder: dict[str, Any] = {}

    thread = threading.Thread(target=lambda: holder.setdefault("result", runtime.run_turn_in_session(state, "Do the long main task.")), daemon=True)
    thread.start()
    # Synchronization only: context compilation/schema preparation can exceed two seconds under full-suite load.
    assert client.main_started.wait(timeout=10)

    answer = service.answer_status_question(state.session_id, "What is happening right now?")
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert answer == "The main agent is still working."
    assert holder["result"].assistant_text == "main finished"
    assert client.replay_verified is True
    assert len(client.requests) == 3
    assert client.requests[0] == client.requests[2]
    events = runtime.history.read_history(state.session_id)
    preempted = [event for event in events if event.event_type == "model_call_preempted"]
    replayed = [event for event in events if event.event_type == "model_call_replayed"]
    assert len(preempted) == 1
    assert len(replayed) == 1
    assert preempted[0].payload["request_sha256"] == replayed[0].payload["request_sha256"]
    assert preempted[0].payload["usage_evidence"] == {
        "backend_prompt_tokens": None,
        "backend_completion_tokens": None,
        "prompt_tokens_source": None,
        "completion_tokens_source": None,
    }
    assert replayed[0].payload["request"] == client.requests[0]
    inference = runtime.inference.list(session_id=state.session_id)
    assert any(item.status == "completed" and item.attempt_count == 2 for item in inference)
    assert any(event.event_type == "inference_request_requeued" for event in events)


def test_target_changing_communication_invalidates_replay_and_refreshes_history(make_config) -> None:
    config = make_config(model__context_limit=32_000)
    client = _InvalidationClient()
    runtime = AgentRuntime(config, model_client=client)
    state = runtime.create_or_load_session()
    service = CommunicationService(runtime)
    holder: dict[str, Any] = {}

    thread = threading.Thread(target=lambda: holder.setdefault("result", runtime.run_turn_in_session(state, "Original objective.")), daemon=True)
    thread.start()
    # This is a synchronization wait, not a latency SLO. Context compilation and
    # constrained-schema preparation can exceed two seconds under full-suite load.
    assert client.main_started.wait(timeout=10)

    request = service.submit(state.session_id, "redirect to the new objective")

    def apply_control(target_state):
        runtime.history.record_event(
            target_state,
            "message_added",
            {
                "message": {
                    "role": "user",
                    "content": "redirected objective",
                    "created_at": utc_now_iso(),
                    "name": None,
                    "metadata": {"source": "communication-test"},
                }
            },
        )
        runtime.history.mark_control_message_processed(target_state.session_id, request.correlation_id)
        return SimpleNamespace(assistant_text="redirect applied")

    original = runtime.run_pending_controls_in_session
    runtime.run_pending_controls_in_session = apply_control  # type: ignore[method-assign]
    try:
        processed = service.process_once(session_id=state.session_id)
    finally:
        runtime.run_pending_controls_in_session = original  # type: ignore[method-assign]

    thread.join(timeout=5)
    assert processed is not None and processed.status == "completed"
    assert not thread.is_alive()
    assert holder["result"].assistant_text == "continued after redirect"
    assert len(client.requests) == 2
    assert client.requests[0] != client.requests[1]
    assert "redirected objective" in str(client.requests[1]["prompt"])
    events = runtime.history.read_history(state.session_id)
    assert any(event.event_type == "model_call_replay_invalidated" for event in events)
    assert not any(event.event_type == "model_call_replayed" for event in events)
    inference = runtime.inference.list(session_id=state.session_id)
    assert any(item.status == "superseded" for item in inference)
    assert any(item.status == "completed" for item in inference)


def test_separate_assistant_model_answers_without_preempting_main(make_config) -> None:
    main_config = make_config(model__context_limit=32_000)
    assistant_config = make_config(model__context_limit=32_000)
    main_client = _HoldClient("main finished")
    assistant_client = _ImmediateClient("assistant status")
    main = AgentRuntime(main_config, model_client=main_client)
    assistant = AgentRuntime(assistant_config, model_client=assistant_client)
    state = main.create_or_load_session()
    service = CommunicationService(main, assistant_runtime=assistant)
    holder: dict[str, Any] = {}

    thread = threading.Thread(target=lambda: holder.setdefault("result", main.run_turn_in_session(state, "Long task.")), daemon=True)
    thread.start()
    # Context compilation and constrained-schema preparation can exceed two seconds
    # on a loaded Jetson; this synchronization wait is not a latency assertion.
    assert main_client.started.wait(timeout=10)

    answer = service.answer_status_question(state.session_id, "Status?")
    assert answer == "assistant status"
    assert thread.is_alive()
    active = main.preemption.active_call(state.session_id)
    assert active is not None
    assert main.preemption.pending_for_call(state.session_id, active.call_id) is None

    main_client.release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert holder["result"].assistant_text == "main finished"


def test_configured_communication_endpoint_builds_separate_runtime(make_config) -> None:
    config = make_config()
    config.communication.enabled = True
    config.communication.model_base_url = "http://127.0.0.1:14830"
    config.communication.enabled_tools = ["calculator"]
    main = AgentRuntime(config, model_client=_ImmediateClient("main"))

    service = CommunicationService.from_runtime(main)

    assert service.runtime is main
    assert service.assistant_runtime is not None
    assert service.assistant_runtime.config is not config
    assert service.assistant_runtime.config.model.base_url == "http://127.0.0.1:14830"
    assert service.assistant_runtime.config.tools.enabled == ["calculator"]
    assert service.assistant_runtime.config.tools.allow_side_effect_tools is False


def test_separate_assistant_semantically_escalates_to_stronger_model(
    make_config,
) -> None:
    main_client = _ImmediateClient("strong status")
    assistant_client = _EscalatingStatusClient("small-model draft")
    main = AgentRuntime(
        make_config(model__context_limit=32_000),
        model_client=main_client,
    )
    assistant = AgentRuntime(
        make_config(model__context_limit=32_000),
        model_client=assistant_client,
    )
    state = main.create_or_load_session()
    service = CommunicationService(main, assistant_runtime=assistant)

    answer = service.answer_status_question(state.session_id, "Explain the status.")

    assert answer == "strong status"
    assert len(assistant_client.requests) == 1
    assert len(main_client.requests) == 1
    assistant_events = [
        event
        for entry in assistant.history.list_session_entries(include_internal=True)
        for event in assistant.history.read_history(str(entry["session_id"]))
    ]
    requested = next(
        event
        for event in assistant_events
        if event.event_type == "communication_status_escalation_requested"
    )
    resolved = next(
        event
        for event in assistant_events
        if event.event_type == "communication_status_escalation_resolved"
    )
    assert resolved.payload["request_event_sequence"] == requested.sequence
    assert resolved.payload["request_event_hash"] == requested.hash
    assert resolved.payload["stronger_model_requested_further_escalation"] is False
    assert requested.payload["trigger"] == "semantic_request"


def test_separate_assistant_failure_falls_back_to_stronger_model(make_config) -> None:
    main = AgentRuntime(
        make_config(model__context_limit=32_000),
        model_client=_ImmediateClient("strong status"),
    )
    assistant = AgentRuntime(
        make_config(model__context_limit=32_000, model__max_retries=0),
        model_client=_FailingStatusClient("unused"),
    )
    state = main.create_or_load_session()
    service = CommunicationService(main, assistant_runtime=assistant)

    answer = service.answer_status_question(state.session_id, "Explain the status.")

    assert answer == "strong status"
    assistant_events = [
        event
        for entry in assistant.history.list_session_entries(include_internal=True)
        for event in assistant.history.read_history(str(entry["session_id"]))
    ]
    requested = next(
        event
        for event in assistant_events
        if event.event_type == "communication_status_escalation_requested"
    )
    assert requested.payload["trigger"] == "assistant_failure"
    assert any(
        event.event_type == "communication_status_unavailable"
        for event in assistant_events
    )
    assert any(
        event.event_type == "communication_status_escalation_resolved"
        for event in assistant_events
    )


def test_failed_stronger_status_escalation_is_durable(make_config) -> None:
    main = AgentRuntime(
        make_config(model__context_limit=32_000),
        model_client=_FailingStatusClient("unused"),
    )
    assistant = AgentRuntime(
        make_config(model__context_limit=32_000),
        model_client=_EscalatingStatusClient("small-model draft"),
    )
    state = main.create_or_load_session()
    service = CommunicationService(main, assistant_runtime=assistant)

    with pytest.raises(RuntimeError, match="strong status backend unavailable"):
        service.answer_status_question(state.session_id, "Explain the status.")

    assistant_events = [
        event
        for entry in assistant.history.list_session_entries(include_internal=True)
        for event in assistant.history.read_history(str(entry["session_id"]))
    ]
    requested = next(
        event
        for event in assistant_events
        if event.event_type == "communication_status_escalation_requested"
    )
    failed = next(
        event
        for event in assistant_events
        if event.event_type == "communication_status_escalation_failed"
    )
    assert failed.payload["request_event_sequence"] == requested.sequence
    assert failed.payload["request_event_hash"] == requested.hash
    assert failed.payload["error_type"] == "RuntimeError"


def test_benchmark_communication_probe_exercises_exact_replay(make_config, tmp_path) -> None:
    from swaag.benchmark.benchmark_runner import _run_turn_with_communication_probe
    from swaag.benchmark.task_definitions import BenchmarkVerificationContract, TaskScenario
    from swaag.benchmark.verifier import verify_benchmark_contract

    config = make_config(model__context_limit=32_000)
    client = _PreemptReplayClient()
    runtime = AgentRuntime(config, model_client=client)
    state = runtime.create_or_load_session()
    scenario = TaskScenario(
        prompt="Do the benchmark main task.",
        workspace=tmp_path,
        model_client=client,
        communication_probe_question="Benchmark status?",
        verification_contract=BenchmarkVerificationContract(
            task_type="multi_step",
            required_history_events=["model_call_preempted", "model_call_replayed", "turn_finished"],
            require_exact_preemption_replay=True,
        ),
    )
    turn = _run_turn_with_communication_probe(runtime, state, scenario)
    assert turn.assistant_text == "main finished"
    rebuilt = runtime.history.rebuild_from_history(state.session_id)
    events = runtime.history.read_history(state.session_id)
    report = verify_benchmark_contract(
        scenario.verification_contract,
        assistant_text=turn.assistant_text,
        state=rebuilt,
        events=events,
        workspace_before={},
        workspace_after={},
        workspace_root=str(tmp_path),
    )
    assert report.passed is True
    assert report.checks["exact_preemption_replay"] is True


def test_orchestrator_preempts_named_route_worker_on_shared_backend(make_config) -> None:
    main = AgentRuntime(
        make_config(
            model__base_url="http://127.0.0.1:19001",
            model__context_limit=32_000,
        ),
        model_client=_ImmediateClient("main"),
    )
    route_client = _PreemptReplayClient()
    route = AgentRuntime(
        make_config(
            model__base_url="http://127.0.0.1:19002",
            model__context_limit=32_000,
        ),
        model_client=route_client,
    )
    orchestrator = AgentRuntime(
        make_config(
            model__base_url="http://127.0.0.1:19002",
            model__context_limit=32_000,
            tools__enabled=["orchestration_control"],
            tools__allow_stateful_tools=True,
            tools__allow_side_effect_tools=True,
        ),
        model_client=_ImmediateClient("orchestrator reply"),
    )
    service = CommunicationService(
        main,
        orchestrator_runtime=orchestrator,
        worker_model_runtimes={"strong": route},
    )
    plan_id = service.orchestration_api.execute(
        "create", {"objective": "route preemption"}
    )["plan"]["plan_id"]
    service.orchestration_api.execute(
        "node.add",
        {
            "plan_id": plan_id,
            "objective": "long routed worker",
            "model_key": "strong",
        },
    )
    started = service.orchestration_api.execute(
        "start", {"plan_id": plan_id}
    )["started_worker_ids"]
    assert len(started) == 1
    assert route_client.main_started.wait(timeout=10)

    answer = service.orchestrator_message("Give me the current orchestration status.")

    assert answer["answer"] == "orchestrator reply"
    finished = service.worker_model_managers["strong"].wait(
        started[0], timeout_seconds=10
    )
    assert finished.status == "completed"
    assert route_client.replay_verified is True
    assert finished.model_key == "strong"
    events = route.history.read_history(finished.session_id)
    assert any(event.event_type == "model_call_preempted" for event in events)
    assert any(event.event_type == "model_call_replayed" for event in events)


class _UsagePreemptReplayClient(_PreemptReplayClient):
    def send_completion(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
        progress_callback=None,
        cancel_check=None,
    ) -> CompletionResult:
        copied = json.loads(stable_json_dumps(payload, indent=None))
        self.requests.append(copied)
        if payload.get("contract") == "communication_status":
            return self._result(payload, _status("The main agent is still working."))
        if payload.get("contract") == "orchestrator_interaction":
            return self._result(
                payload,
                _orchestrator_interaction("Yes, I can hear you."),
            )
        if self.first_main_request is None:
            self.first_main_request = copied
            self.main_started.set()
            if progress_callback is not None:
                progress_callback(
                    {
                        "completion_tokens": 17,
                        "backend_prompt_tokens": 321,
                        "backend_completion_tokens": 17,
                        "prompt_tokens_source": "tokens_evaluated",
                        "completion_tokens_source": "tokens_predicted",
                        "elapsed_seconds": 0.25,
                        "tokens_per_second": 68.0,
                        "first_token_seconds": 0.1,
                        "token_timeout_seconds": 30,
                    }
                )
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if cancel_check is not None and cancel_check():
                    raise ModelCallPreempted("usage-aware test preemption")
                time.sleep(0.005)
            raise AssertionError("main request was not preempted")
        assert copied == self.first_main_request
        self.replay_verified = True
        return self._result(payload, _action("main finished"))


def test_preemption_preserves_backend_reported_partial_usage(make_config) -> None:
    config = make_config(model__context_limit=32_000)
    client = _UsagePreemptReplayClient()
    runtime = AgentRuntime(config, model_client=client)
    state = runtime.create_or_load_session()
    service = CommunicationService(runtime)
    holder: dict[str, Any] = {}

    thread = threading.Thread(
        target=lambda: holder.setdefault(
            "result", runtime.run_turn_in_session(state, "Do the long main task.")
        ),
        daemon=True,
    )
    thread.start()
    assert client.main_started.wait(timeout=10)
    assert (
        service.answer_status_question(
            state.session_id, "What is happening right now?"
        )
        == "The main agent is still working."
    )
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert client.replay_verified is True
    events = runtime.history.read_history(state.session_id)
    preempted = [event for event in events if event.event_type == "model_call_preempted"]
    replayed = [event for event in events if event.event_type == "model_call_replayed"]
    assert len(preempted) == 1
    assert len(replayed) == 1
    usage = preempted[0].payload["usage_evidence"]
    assert usage == {
        "backend_prompt_tokens": 321,
        "backend_completion_tokens": 17,
        "prompt_tokens_source": "tokens_evaluated",
        "completion_tokens_source": "tokens_predicted",
    }
    assert replayed[0].payload["preempted_usage_evidence"] == usage

def test_benchmark_communication_probe_uses_task_deadline_for_replayed_turn(make_config, tmp_path) -> None:
    from swaag.benchmark.benchmark_runner import _run_turn_with_communication_probe
    from swaag.benchmark.task_definitions import BenchmarkVerificationContract, TaskScenario

    class _DelayedReplayClient(_PreemptReplayClient):
        def __init__(self) -> None:
            super().__init__()
            self.main_attempts = 0

        def send_completion(self, payload, **kwargs):
            contract = payload.get("contract")
            if contract == "agent_action":
                self.main_attempts += 1
                if self.main_attempts >= 2:
                    time.sleep(2.5)
            return super().send_completion(payload, **kwargs)

    config = make_config(model__context_limit=32_000)
    client = _DelayedReplayClient()
    runtime = AgentRuntime(config, model_client=client)
    state = runtime.create_or_load_session()
    scenario = TaskScenario(
        prompt="Do the benchmark main task.",
        workspace=tmp_path,
        model_client=client,
        communication_probe_question="Benchmark status?",
        communication_probe_wait_seconds=2.0,
        verification_contract=BenchmarkVerificationContract(task_type="multi_step"),
    )
    turn = _run_turn_with_communication_probe(
        runtime,
        state,
        scenario,
        resume_timeout_seconds=15.0,
    )
    assert turn.assistant_text == "main finished"
    assert client.main_attempts >= 2
    event_types = [event.event_type for event in runtime.history.read_history(state.session_id)]
    assert "model_call_preempted" in event_types
    assert "model_call_replayed" in event_types


def test_user_facing_orchestrator_exposes_core_tool_without_repeated_discovery(make_config) -> None:
    config = make_config(
        model__context_limit=32_000,
        tools__staged_discovery=True,
        communication__enabled=True,
        communication__enabled_tools=["orchestration_control"],
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
    )
    runtime = AgentRuntime(config, model_client=_ImmediateClient("unused"))
    service = CommunicationService.from_runtime(runtime)
    actual = service.orchestrator_runtime.config.tools
    assert actual.enabled == ["orchestration_control"]
    assert actual.staged_discovery is True
    assert actual.allow_stateful_tools is True
    assert actual.allow_side_effect_tools is True
    assert runtime.config.tools.staged_discovery is True


def test_orchestrator_lightweight_interaction_is_one_model_call_and_starts_no_worker(make_config) -> None:
    config = make_config(
        model__context_limit=32_000,
        tools__enabled=["orchestration_control"],
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
    )
    client = _ImmediateClient("Yes, I can hear you.")
    orchestrator = AgentRuntime(config, model_client=client)
    service = CommunicationService(orchestrator, orchestrator_runtime=orchestrator)

    before_workers = service.workers.list()
    answer = service.orchestrator_message("Hello, can you hear me?")
    after_workers = service.workers.list()

    assert answer["answer"] == "Yes, I can hear you."
    assert before_workers == []
    assert after_workers == []
    assert [request["contract"] for request in client.requests] == [
        "orchestrator_interaction"
    ]
    prompt = client.requests[0]["prompt"]
    assert "Do not mention backend URLs" in prompt
    assert "Never substitute runtime-status commentary" in prompt
    assert "user's exact current request and later corrections" in prompt
    assert "Do not replace them with a different objective" in prompt
    assert "invent an answer to an important unknown" in prompt
    assert "route to orchestration instead of guessing" in prompt
    assert "omit implementation noise and machine identifiers" in prompt
    assert "state material uncertainty or a blocker" in prompt
    state = orchestrator.create_or_load_user_session("SWAAG Orchestrator")
    visible = [
        message
        for message in state.messages
        if message.role in {"user", "assistant"}
        and not message.metadata.get("internal_action")
    ]
    assert [message.content for message in visible[-2:]] == [
        "Hello, can you hear me?",
        "Yes, I can hear you.",
    ]


def test_orchestrator_preserves_user_facing_answer_whitespace_verbatim(make_config) -> None:
    config = make_config(
        model__context_limit=32_000,
        tools__enabled=["orchestration_control"],
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
    )
    exact_answer = "\n  exact answer payload  \n"
    client = _ImmediateClient(exact_answer)
    orchestrator = AgentRuntime(config, model_client=client)
    service = CommunicationService(orchestrator, orchestrator_runtime=orchestrator)

    answer = service.orchestrator_message("Return the exact payload.")

    assert answer["answer"] == exact_answer
    state = orchestrator.create_or_load_user_session("SWAAG Orchestrator")
    visible = [
        message.content
        for message in state.messages
        if message.role == "assistant" and not message.metadata.get("internal_action")
    ]
    assert visible[-1] == exact_answer


def test_orchestrator_preserves_current_user_whitespace_verbatim(make_config) -> None:
    config = make_config(
        model__context_limit=32_000,
        tools__enabled=["orchestration_control"],
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
    )
    client = _ImmediateClient("preserved")
    orchestrator = AgentRuntime(config, model_client=client)
    service = CommunicationService(orchestrator, orchestrator_runtime=orchestrator)
    message = "\n  preserve this exact utterance  \n"

    answer = service.orchestrator_message(message)

    assert answer["answer"] == "preserved"
    prompt = str(client.requests[0]["prompt"])
    assert "Current user utterance, verbatim and authoritative:\n" + message in prompt
    state = orchestrator.create_or_load_user_session("SWAAG Orchestrator")
    visible = [item.content for item in state.messages if item.role == "user"]
    assert visible[-1] == message


class _RouteToPlannerClient(_ImmediateClient):
    def send_completion(self, payload: dict[str, Any], **kwargs) -> CompletionResult:
        self.requests.append(json.loads(stable_json_dumps(payload, indent=None)))
        if payload.get("contract") == "orchestrator_interaction":
            return self._result(
                payload,
                _orchestrator_interaction(
                    route="orchestrate",
                    reason="The user requested substantive delegated work.",
                ),
            )
        return self._result(payload, _action("planner handled the substantive request"))


class _BrokenFastRouterClient(_ImmediateClient):
    def send_completion(self, payload: dict[str, Any], **kwargs) -> CompletionResult:
        self.requests.append(json.loads(stable_json_dumps(payload, indent=None)))
        if payload.get("contract") == "orchestrator_interaction":
            return self._result(payload, _orchestrator_interaction(
                answer="I'll write a Python program and create the requested file.",
                route="respond",
                reason="The user wants work; promise it without tools.",
            ))
        return self._result(payload, _action("The planner handled the requested work."))


def test_false_fast_promise_routes_to_full_orchestrator(make_config) -> None:
    config = make_config(
        model__context_limit=32_000,
        tools__enabled=["orchestration_control"],
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__completion_evaluation_enabled=False,
    )
    client = _BrokenFastRouterClient("unused")
    runtime = AgentRuntime(config, model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    answer = service.orchestrator_message(
        "You shall do it now: let a worker write a Python program that creates a file."
    )
    assert answer["answer"] == "The planner handled the requested work."
    assert [x["contract"] for x in client.requests] == [
        "orchestrator_interaction", "agent_action",
    ]
    state = runtime.create_or_load_user_session("SWAAG Orchestrator")
    event_types = [x.event_type for x in runtime.history.read_history(state.session_id)]
    assert "orchestrator_direct_reply_rejected" in event_types
    assert "I'll write" not in answer["answer"]


def test_unexecuted_work_claim_guard_preserves_direct_noncommitments() -> None:
    from swaag.communication import CommunicationService
    yes = CommunicationService._fast_reply_claims_unperformed_work
    assert yes("I'll write a Python program that creates a file.")
    assert yes("I will write a Python program.")
    assert yes("I can confirm that I will write a program.")
    assert yes("I have started the worker.")
    assert yes("Starting the worker now.")
    assert not yes("No, I haven't started a worker.")
    assert not yes("There are no workers running.")
    assert not yes("Yes, I can hear you.")
    assert not yes("I will not delete your files.")


class _ExtraAnswerPlannerClient(_ImmediateClient):
    def send_completion(self, payload: dict[str, Any], **kwargs) -> CompletionResult:
        self.requests.append(json.loads(stable_json_dumps(payload, indent=None)))
        if payload.get("contract") == "orchestrator_interaction":
            return self._result(payload, _orchestrator_interaction(
                route="orchestrate",
                answer="I will write the file in the future.",
                reason="Full work must be delegated.",
            ))
        return self._result(payload, _action("Full planner accepted the real request."))


def test_fast_route_orchestrate_discards_invalid_extra_answer(make_config) -> None:
    config = make_config(
        model__context_limit=32_000,
        tools__enabled=["orchestration_control"],
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__completion_evaluation_enabled=False,
    )
    client = _ExtraAnswerPlannerClient("unused")
    runtime = AgentRuntime(config, model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    answer = service.orchestrator_message("Start the requested worker now.")
    assert answer["answer"] == "Full planner accepted the real request."
    assert [item["contract"] for item in client.requests] == [
        "orchestrator_interaction", "agent_action",
    ]


def test_orchestrator_substantive_command_enters_full_planner_after_gate(make_config) -> None:
    config = make_config(
        model__context_limit=32_000,
        tools__enabled=["orchestration_control"],
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
        runtime__completion_evaluation_enabled=False,
    )
    client = _RouteToPlannerClient("unused")
    orchestrator = AgentRuntime(config, model_client=client)
    service = CommunicationService(orchestrator, orchestrator_runtime=orchestrator)

    answer = service.orchestrator_message(
        "Start a new worker project and build the requested large program."
    )

    assert answer["answer"] == "planner handled the substantive request"
    assert [request["contract"] for request in client.requests] == [
        "orchestrator_interaction",
        "agent_action",
    ]

def test_orchestrator_fast_interaction_receives_exact_prior_conversation(make_config) -> None:
    config = make_config(
        model__context_limit=32_000,
        tools__enabled=["orchestration_control"],
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
    )
    client = _ImmediateClient("Yes.")
    orchestrator = AgentRuntime(config, model_client=client)
    service = CommunicationService(orchestrator, orchestrator_runtime=orchestrator)

    first = service.orchestrator_message("Remember the phrase blue lantern.")
    second = service.orchestrator_message("What phrase did I just say?")

    assert first["answer"] == "Yes."
    assert second["answer"] == "Yes."
    assert [request["contract"] for request in client.requests] == [
        "orchestrator_interaction",
        "orchestrator_interaction",
    ]
    assert "Remember the phrase blue lantern." in client.requests[1]["prompt"]
    assert "What phrase did I just say?" in client.requests[1]["prompt"]

def test_orchestrator_fast_status_call_receives_current_worker_snapshot(make_config) -> None:
    config = make_config(
        model__context_limit=32_000,
        tools__enabled=["orchestration_control"],
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
    )
    client = _ImmediateClient("One worker is currently prepared.")
    orchestrator = AgentRuntime(config, model_client=client)
    service = CommunicationService(orchestrator, orchestrator_runtime=orchestrator)
    created = service.task_api.execute(
        "create",
        {"objective": "inspect the blue lantern file", "start": False},
    )

    answer = service.orchestrator_message("What is currently running?")

    assert answer["answer"] == "One worker is currently prepared."
    assert [request["contract"] for request in client.requests] == [
        "orchestrator_interaction"
    ]
    prompt = client.requests[0]["prompt"]
    assert created["worker"]["worker_id"] in prompt
    assert "inspect the blue lantern file" in prompt
    assert '"status":"created"' in prompt.replace(" ", "").replace("\n", "")


def test_benchmark_communication_probe_allows_context_preparation_past_probe_window(make_config, tmp_path) -> None:
    from swaag.benchmark.benchmark_runner import _run_turn_with_communication_probe
    from swaag.benchmark.task_definitions import BenchmarkVerificationContract, TaskScenario

    class _SlowPreparationClient(_PreemptReplayClient):
        def __init__(self) -> None:
            super().__init__()
            self.delayed = False

        def context_limit_resolution(self):
            if not self.delayed:
                self.delayed = True
                time.sleep(0.25)
            return 32_000, "test:slow-preparation"

    config = make_config(model__context_limit=32_000)
    client = _SlowPreparationClient()
    runtime = AgentRuntime(config, model_client=client)
    state = runtime.create_or_load_session()
    scenario = TaskScenario(
        prompt="Do the benchmark main task after context preparation.",
        workspace=tmp_path,
        model_client=client,
        communication_probe_question="Benchmark status?",
        communication_probe_wait_seconds=0.05,
        verification_contract=BenchmarkVerificationContract(task_type="multi_step"),
    )

    turn = _run_turn_with_communication_probe(
        runtime,
        state,
        scenario,
        resume_timeout_seconds=8.0,
    )

    assert turn.assistant_text == "main finished"
    event_types = [
        event.event_type for event in runtime.history.read_history(state.session_id)
    ]
    assert "model_call_preempted" in event_types
    assert "model_call_replayed" in event_types

class _ConcurrentOrchestratorClient(_ImmediateClient):
    def __init__(self) -> None:
        super().__init__("unused")
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self._orchestrator_calls = 0
        self._orchestrator_lock = threading.Lock()

    def send_completion(self, payload: dict[str, Any], **kwargs) -> CompletionResult:
        if payload.get("contract") != "orchestrator_interaction":
            return super().send_completion(payload, **kwargs)
        with self._orchestrator_lock:
            self._orchestrator_calls += 1
            call_index = self._orchestrator_calls
        self.requests.append(json.loads(stable_json_dumps(payload, indent=None)))
        if call_index == 1:
            self.first_started.set()
            assert self.release_first.wait(timeout=5)
            answer = "first reply"
        else:
            answer = "second reply"
        return self._result(payload, _orchestrator_interaction(answer))


def test_concurrent_orchestrator_messages_preserve_order_and_history(make_config) -> None:
    config = make_config(
        model__context_limit=32_000,
        tools__enabled=["orchestration_control"],
        tools__allow_stateful_tools=True,
        tools__allow_side_effect_tools=True,
    )
    client = _ConcurrentOrchestratorClient()
    runtime = AgentRuntime(config, model_client=client)
    service = CommunicationService(runtime, orchestrator_runtime=runtime)
    results: dict[str, dict[str, str]] = {}
    errors: dict[str, Exception] = {}

    def call(key: str, message: str) -> None:
        try:
            results[key] = service.orchestrator_message(message)
        except Exception as exc:
            errors[key] = exc

    first = threading.Thread(target=call, args=("first", "first user"), daemon=True)
    second = threading.Thread(target=call, args=("second", "second user"), daemon=True)
    first.start()
    assert client.first_started.wait(timeout=5)
    second.start()
    second.join(timeout=0.2)
    assert second.is_alive(), "second orchestrator message overtook the active turn"
    client.release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not errors
    assert not first.is_alive() and not second.is_alive()
    assert results["first"]["answer"] == "first reply"
    assert results["second"]["answer"] == "second reply"
    state = runtime.create_or_load_user_session("SWAAG Orchestrator")
    visible = [
        message.content
        for message in state.messages
        if message.role in {"user", "assistant"}
        and not message.metadata.get("internal_action")
    ]
    assert visible[-4:] == [
        "first user",
        "first reply",
        "second user",
        "second reply",
    ]

def test_run_cancellation_preserves_reason_whitespace_verbatim(make_config) -> None:
    from swaag.preemption import ModelPreemptionCoordinator

    store = ModelPreemptionCoordinator(make_config().sessions.root)
    reason = "\n  preserve cancellation reason exactly  \n"
    item = store.request_run_cancellation("session-x", "run-x", reason=reason)
    assert item.reason == reason


class _AutonomousIdeaClient(_BaseClient):
    def __init__(self, *, create: bool) -> None:
        super().__init__()
        self.create = create

    def send_completion(self, payload: dict[str, Any], **kwargs) -> CompletionResult:
        self.requests.append(json.loads(stable_json_dumps(payload, indent=None)))
        assert payload.get("contract") == "autonomous_work_idea"
        body = ({
            "create": True,
            "objective": "Improve repository verification",
            "worker_objective": "Inspect prior verification failures and propose one evidence-backed improvement",
            "reason": "The supplied history repeatedly prioritizes verification quality.",
        } if self.create else {
            "create": False, "objective": "", "worker_objective": "",
            "reason": "No sufficiently grounded useful task is available.",
        })
        return self._result(payload, json.dumps(body))


def test_autonomous_idea_semantic_call_is_history_grounded_and_can_decline(make_config) -> None:
    config = make_config(model__context_limit=32_000)
    for create in (True, False):
        client = _AutonomousIdeaClient(create=create)
        runtime = AgentRuntime(config, model_client=client)
        result = runtime.generate_autonomous_work_idea(
            conversation_messages=[Message(role="user", content="Verification quality matters most.", created_at=utc_now_iso())],
            runtime_snapshot={"recent_plans": [], "open_questions": {"questions": []}},
        )
        assert result["create"] is create
        prompt = client.requests[0]["prompt"]
        assert "explicitly enabled autonomous keep-working mode" in prompt
        assert "Verification quality matters most." in prompt
        assert "Do not invent a task merely to stay busy" in prompt
        assert "Never pretend the user explicitly requested" in prompt
        assert result["objective"] == ("Improve repository verification" if create else "")
