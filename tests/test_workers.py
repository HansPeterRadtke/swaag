from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from swaag.model import CompletionRequestPolicy
from swaag.preemption import ModelCallPreempted
from swaag.types import CompletionResult, ContractSpec
from swaag.workers import WorkerManager


def _action(message: str, *, blocking_question: str = "") -> str:
    questions = []
    if blocking_question:
        questions.append(
            {
                "question": blocking_question,
                "criticality": "blocking",
                "reason": "The missing answer changes the result.",
                "assumption_if_unanswered": "",
            }
        )
    return json.dumps(
        {
            "assistant_message": message,
            "tool_calls": [],
            "continue_loop": False,
            "silent_completion": False,
            "questions": questions,
            "status": {
                "situation": "Working on the objective.",
                "action": "Return the current result.",
                "reason": "The available evidence determines the next response.",
                "importance": "normal",
            },
        }
    )


class _WorkerClient:
    is_deterministic_test_client = True

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
            effective_timeout_seconds=5,
            progress_poll_seconds=0.01,
        )

    def resolve_contract(self, contract: ContractSpec, **kwargs):
        return contract, self.select_request_policy(contract=contract, **kwargs)

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
            "contract": contract.name,
            "json_schema": contract.json_schema,
        }

    @staticmethod
    def result(payload: dict[str, Any], text: str) -> CompletionResult:
        return CompletionResult(
            text=text,
            raw_request=payload,
            raw_response={"content": text},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


class _ObjectiveClient(_WorkerClient):
    def send_completion(self, payload: dict[str, Any], **_kwargs) -> CompletionResult:
        prompt = str(payload["prompt"])
        answer = "alpha complete" if "alpha objective" in prompt else "beta complete"
        return self.result(payload, _action(answer))


class _CancellableClient(_WorkerClient):
    def __init__(self) -> None:
        self.started = threading.Event()

    def send_completion(self, payload: dict[str, Any], *, cancel_check=None, **_kwargs):
        self.started.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if cancel_check is not None and cancel_check():
                raise ModelCallPreempted("cancel observed")
            time.sleep(0.005)
        raise AssertionError("worker cancellation was not observed")


class _RedirectClient(_WorkerClient):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.calls = 0

    def send_completion(self, payload: dict[str, Any], *, cancel_check=None, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if cancel_check is not None and cancel_check():
                    raise ModelCallPreempted("redirect observed")
                time.sleep(0.005)
            raise AssertionError("worker redirect was not observed")
        assert "new exact direction" in str(payload["prompt"])
        return self.result(payload, _action("redirect complete"))


class _CompletionRedirectClient(_WorkerClient):
    def __init__(self) -> None:
        self.completion_started = threading.Event()
        self.action_calls = 0
        self.completion_calls = 0

    def send_completion(self, payload: dict[str, Any], *, cancel_check=None, **_kwargs):
        contract = str(payload["contract"])
        if contract == "agent_action":
            self.action_calls += 1
            prompt = str(payload["prompt"])
            if self.action_calls == 1:
                return self.result(payload, _action("stale candidate"))
            assert "new exact direction" in prompt
            return self.result(payload, _action("revised complete"))
        assert contract == "completion_evaluation"
        self.completion_calls += 1
        if self.completion_calls == 1:
            self.completion_started.set()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if cancel_check is not None and cancel_check():
                    raise ModelCallPreempted("completion redirect observed")
                time.sleep(0.005)
            raise AssertionError("completion redirect was not observed")
        return self.result(
            payload,
            json.dumps(
                {
                    "complete": True,
                    "reason": "The revised direction is reflected in the candidate.",
                    "remaining_work": [],
                }
            ),
        )


class _PresentationRedirectClient(_WorkerClient):
    def __init__(self) -> None:
        self.calls = 0

    def send_completion(self, payload: dict[str, Any], **_kwargs):
        self.calls += 1
        prompt = str(payload["prompt"])
        if self.calls == 1:
            return self.result(payload, _action("stale candidate"))
        assert "new exact direction" in prompt
        return self.result(payload, _action("revised complete"))


class _InputClient(_WorkerClient):
    def __init__(self) -> None:
        self.calls = 0

    def send_completion(self, payload: dict[str, Any], **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return self.result(payload, _action("Which target?", blocking_question="Which target?"))
        assert "Use target green" in str(payload["prompt"])
        return self.result(payload, _action("green complete"))


class _OptionalInputClient(_WorkerClient):
    def __init__(self) -> None:
        self.calls = 0

    def send_completion(self, payload: dict[str, Any], **_kwargs):
        self.calls += 1
        prompt = str(payload["prompt"])
        if self.calls == 1:
            return self.result(
                payload,
                json.dumps(
                    {
                        "assistant_message": "I can proceed with blue unless you prefer another target.",
                        "tool_calls": [
                            {
                                "tool_name": "calculator",
                                "arguments": {"expression": "1 + 1"},
                            }
                        ],
                        "continue_loop": True,
                        "silent_completion": False,
                        "questions": [
                            {
                                "question": "Do you prefer a target other than blue?",
                                "criticality": "optional",
                                "reason": "Blue is a safe provisional choice.",
                                "assumption_if_unanswered": "Use blue.",
                            }
                        ],
                        "status": {
                            "situation": "A safe provisional target is available.",
                            "action": "Continue useful work with blue.",
                            "reason": "The optional preference does not block progress.",
                            "importance": "normal",
                        },
                    }
                ),
            )
        if "Use target green" in prompt:
            return self.result(payload, _action("green revision complete"))
        assert '"result": 2' in prompt
        return self.result(payload, _action("blue provisional complete"))


class _ContinuousClient(_WorkerClient):
    def __init__(self) -> None:
        self.calls = 0
        self.second_started = threading.Event()
        self.second_prompt = ""

    def send_completion(self, payload: dict[str, Any], *, cancel_check=None, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return self.result(payload, _action("first provisional result"))
        self.second_prompt = str(payload["prompt"])
        self.second_started.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if cancel_check is not None and cancel_check():
                raise ModelCallPreempted("continuous cancellation observed")
            time.sleep(0.005)
        raise AssertionError("continuous worker cancellation was not observed")


def test_multiple_workers_have_independent_durable_sessions(make_config) -> None:
    from swaag.runtime import AgentRuntime

    runtime = AgentRuntime(make_config(model__context_limit=32_000), model_client=_ObjectiveClient())
    manager = WorkerManager(runtime, max_workers=2)
    alpha = manager.create("alpha objective", name="alpha-worker")
    beta = manager.create("beta objective", name="beta-worker")

    manager.start(alpha.worker_id)
    manager.start(beta.worker_id)
    alpha_done = manager.wait(alpha.worker_id, timeout_seconds=10)
    beta_done = manager.wait(beta.worker_id, timeout_seconds=10)
    manager.shutdown()

    assert alpha_done.status == beta_done.status == "completed"
    assert alpha_done.session_id != beta_done.session_id
    assert alpha_done.result == "alpha complete"
    assert beta_done.result == "beta complete"
    assert manager.events(alpha.worker_id)[0].event_type == "worker_created"
    completed_event = next(
        event for event in manager.events(alpha.worker_id) if event.event_type == "worker_completed"
    )
    assert completed_event.payload["result"] == "alpha complete"
    assert completed_event.payload["run_count"] == 1


def test_worker_inspection_returns_complete_inference_lifecycle(make_config) -> None:
    from swaag.runtime import AgentRuntime

    runtime = AgentRuntime(make_config(), model_client=object())
    manager = WorkerManager(runtime)
    worker = manager.create("inspect every inference request")
    for index in range(12):
        runtime.inference.enqueue(
            session_id=worker.session_id,
            run_id=f"run-{index}",
            call_id=f"call-{index}",
            call_kind="agent_action",
            priority=0,
            source="test",
        )

    inspection = manager.inspect(worker.worker_id)
    manager.shutdown()

    assert [item["call_id"] for item in inspection["inference_requests"]] == [
        f"call-{index}" for index in range(12)
    ]


def test_worker_response_presentations_are_opt_in_and_durable(
    make_config,
    monkeypatch,
) -> None:
    from swaag.runtime import AgentRuntime

    runtime = AgentRuntime(
        make_config(model__context_limit=32_000), model_client=_ObjectiveClient()
    )
    calls: list[dict[str, Any]] = []

    def generate(_state, **kwargs):
        calls.append(kwargs)
        return {
            "raw": kwargs["assistant_message"],
            "visual": "Alpha is complete.",
            "audio": "Alpha is complete.",
            "requested_modes": ["audio", "visual"],
            "completed_modes": ["visual", "audio"],
        }

    monkeypatch.setattr(runtime, "generate_response_presentations", generate)
    manager = WorkerManager(runtime)
    worker = manager.create(
        "alpha objective",
        presentation_modes=["visual", "audio", "visual"],
    )

    manager.start(worker.worker_id)
    finished = manager.wait(worker.worker_id, timeout_seconds=10)
    presentations = manager.presentations(worker.worker_id)
    inspection = manager.inspect(worker.worker_id)
    manager.shutdown()

    assert worker.presentation_modes == ["audio", "visual"]
    assert finished.result == "alpha complete"
    assert calls[0]["modes"] == ["audio", "visual"]
    assert presentations is not None
    assert presentations["audio"] == "Alpha is complete."
    assert inspection["presentations"] == presentations
    assert next(
        event
        for event in manager.events(worker.worker_id)
        if event.event_type == "worker_completed"
    ).payload["presentations"] == presentations


def test_worker_cancellation_is_durable_and_stops_active_inference(make_config) -> None:
    from swaag.runtime import AgentRuntime
    from swaag.scheduler import WakeupStore

    client = _CancellableClient()
    runtime = AgentRuntime(make_config(model__context_limit=32_000), model_client=client)
    manager = WorkerManager(runtime)
    worker = manager.create("long objective")
    manager.start(worker.worker_id)
    assert client.started.wait(timeout=10)
    wakeup = WakeupStore(runtime.config.sessions.root).schedule(
        session_id=worker.session_id,
        reason="must not resume a canceled worker",
        duration="1 hour",
    )

    inspection = manager.inspect(worker.worker_id)
    diagnostics = inspection["execution_diagnostics"]
    assert diagnostics["last_transition"]["to_status"] == "working"
    assert diagnostics["active_operation"]["active_kind"] == "model"
    assert diagnostics["active_operation"]["active_id"].startswith("model_call_")
    assert diagnostics["active_operation"]["phase"] == "inference"
    assert diagnostics["active_operation"]["substate"] in {
        "dispatching",
        "awaiting_result",
        "streaming",
    }
    assert diagnostics["active_operation"]["operation_kind"] == "action"
    assert diagnostics["active_operation"]["activity_sequence"] > 0
    assert diagnostics["active_operation"]["phase_started_at"]
    assert diagnostics["active_operation"]["substate_started_at"]
    assert diagnostics["active_operation"]["pid_alive"] is True
    assert diagnostics["active_operation"]["heartbeat_age_seconds"] >= 0
    assert diagnostics["local_supervisor"]["manager_process_alive"] is True
    assert diagnostics["local_supervisor"]["run_state"] == "running"

    requested = manager.cancel(worker.worker_id, reason="stop now")
    finished = manager.wait(worker.worker_id, timeout_seconds=10)
    manager.shutdown()

    assert requested.status == "cancellation_requested"
    assert finished.status == "canceled"
    assert runtime.history.read_active_run(worker.session_id) is None
    assert WakeupStore(runtime.config.sessions.root).list(
        session_id=worker.session_id, include_cancelled=True
    )[0].status == "cancelled"
    history = runtime.history.read_history(worker.session_id)
    assert any(
        event.event_type == "model_call_preempted"
        and event.payload.get("reason") == "run_cancellation_requested"
        for event in history
    )
    assert any(
        event
        for event in manager.events(worker.worker_id)
        if event.event_type == "worker_cancellation_requested"
    )
    wakeup_cancellation = next(
        event
        for event in manager.events(worker.worker_id)
        if event.event_type == "worker_wakeups_cancelled"
    )
    assert wakeup_cancellation.payload["wakeup_ids"] == [wakeup.wakeup_id]


def test_worker_message_preempts_stale_request_and_rebuilds_from_control(make_config) -> None:
    from swaag.runtime import AgentRuntime

    client = _RedirectClient()
    runtime = AgentRuntime(make_config(model__context_limit=32_000), model_client=client)
    manager = WorkerManager(runtime)
    worker = manager.create("original direction")
    manager.start(worker.worker_id)
    assert client.started.wait(timeout=10)

    manager.message(worker.worker_id, "new exact direction")
    finished = manager.wait(worker.worker_id, timeout_seconds=10)
    manager.shutdown()

    assert finished.status == "completed"
    assert finished.result == "redirect complete"
    assert client.calls == 2
    history = runtime.history.read_history(worker.session_id)
    assert any(event.event_type == "model_call_replay_invalidated" for event in history)


def test_worker_message_preempts_provisional_completion_evaluation(make_config) -> None:
    from swaag.runtime import AgentRuntime

    client = _CompletionRedirectClient()
    runtime = AgentRuntime(
        make_config(
            model__context_limit=32_000,
            runtime__completion_evaluation_enabled=True,
        ),
        model_client=client,
    )
    manager = WorkerManager(runtime)
    worker = manager.create("original direction")
    manager.start(worker.worker_id)
    assert client.completion_started.wait(timeout=10)

    manager.message(worker.worker_id, "new exact direction")
    finished = manager.wait(worker.worker_id, timeout_seconds=10)
    manager.shutdown()

    assert finished.status == "completed"
    assert finished.result == "revised complete"
    assert finished.error is None
    assert client.action_calls == 2
    assert client.completion_calls == 2
    assert runtime.history.list_pending_control_messages(worker.session_id) == []
    history = runtime.history.read_history(worker.session_id)
    assert any(
        event.event_type == "completion_rejected"
        and "superseded by a user control" in str(event.payload.get("reason", ""))
        for event in history
    )


def test_worker_message_during_presentation_rebuilds_terminal_result(
    make_config,
    monkeypatch,
) -> None:
    from swaag.runtime import AgentRuntime

    client = _PresentationRedirectClient()
    runtime = AgentRuntime(
        make_config(model__context_limit=32_000), model_client=client
    )
    presentation_started = threading.Event()
    presentation_release = threading.Event()
    presentation_inputs: list[str] = []

    def generate(_state, **kwargs):
        assistant_message = str(kwargs["assistant_message"])
        presentation_inputs.append(assistant_message)
        if len(presentation_inputs) == 1:
            presentation_started.set()
            assert presentation_release.wait(timeout=5)
        return {
            "raw": assistant_message,
            "audio": assistant_message,
            "requested_modes": ["audio"],
            "completed_modes": ["audio"],
        }

    monkeypatch.setattr(runtime, "generate_response_presentations", generate)
    manager = WorkerManager(runtime)
    worker = manager.create("original direction", presentation_modes=["audio"])
    manager.start(worker.worker_id)
    assert presentation_started.wait(timeout=10)

    manager.message(worker.worker_id, "new exact direction")
    presentation_release.set()
    finished = manager.wait(worker.worker_id, timeout_seconds=10)
    presentations = manager.presentations(worker.worker_id)
    events = manager.events(worker.worker_id)
    manager.shutdown()

    assert finished.status == "completed"
    assert finished.result == "revised complete"
    assert finished.run_count == 2
    assert presentation_inputs == ["stale candidate", "revised complete"]
    assert presentations is not None
    assert presentations["audio"] == "revised complete"
    continuation = next(
        event
        for event in events
        if event.event_type == "worker_control_continuation_started"
    )
    assert continuation.payload["phase"] == "response_presentation"
    assert continuation.payload["provisional"] is True


def test_input_required_worker_resumes_without_duplicate_original_request(make_config) -> None:
    from swaag.runtime import AgentRuntime

    client = _InputClient()
    runtime = AgentRuntime(make_config(model__context_limit=32_000), model_client=client)
    manager = WorkerManager(runtime)
    worker = manager.create("choose a target")
    manager.start(worker.worker_id)
    waiting = manager.wait(worker.worker_id, timeout_seconds=10)
    assert waiting.status == "input_required"

    manager.message(worker.worker_id, "Use target green")
    finished = manager.wait(worker.worker_id, timeout_seconds=10)
    state = runtime.history.rebuild_from_history(worker.session_id, write_projections=False)
    manager.shutdown()

    assert finished.status == "completed"
    assert finished.result == "green complete"
    assert [message.content for message in state.messages if message.role == "user"] == [
        "choose a target"
    ]


def test_optional_question_continues_work_and_later_answer_redirects(make_config) -> None:
    from swaag.runtime import AgentRuntime

    client = _OptionalInputClient()
    runtime = AgentRuntime(make_config(model__context_limit=32_000), model_client=client)
    manager = WorkerManager(runtime)
    worker = manager.create("choose a target and complete the calculation")
    manager.start(worker.worker_id)
    provisional = manager.wait(worker.worker_id, timeout_seconds=10)

    assert provisional.status == "completed"
    assert provisional.result == "blue provisional complete"
    question = next(
        event
        for event in runtime.history.read_history(worker.session_id)
        if event.event_type == "agent_question"
    )
    assert question.payload["criticality"] == "optional"
    assert question.payload["assumption_if_unanswered"] == "Use blue."
    assert any(
        event.event_type == "tool_result"
        and event.payload.get("tool_name") == "calculator"
        for event in runtime.history.read_history(worker.session_id)
    )

    manager.message(worker.worker_id, "Use target green")
    revised = manager.wait(worker.worker_id, timeout_seconds=10)
    manager.shutdown()

    assert revised.status == "completed"
    assert revised.result == "green revision complete"
    assert revised.run_count == 2


def test_continuous_worker_keeps_provisional_results_and_runs_until_canceled(
    make_config,
) -> None:
    from swaag.runtime import AgentRuntime

    client = _ContinuousClient()
    runtime = AgentRuntime(
        make_config(model__context_limit=32_000), model_client=client
    )
    manager = WorkerManager(runtime)
    worker = manager.create(
        "keep improving the evidence",
        completion_mode="continuous",
    )
    manager.start(worker.worker_id)
    assert client.second_started.wait(timeout=10)

    active = manager.store.get(worker.worker_id)
    events = manager.events(worker.worker_id)
    manager.cancel(worker.worker_id, reason="enough improvement")
    finished = manager.wait(worker.worker_id, timeout_seconds=10)
    manager.shutdown()

    assert active.status == "working"
    assert active.completion_mode == "continuous"
    assert active.result == "first provisional result"
    assert active.run_count == 2
    assert "explicit continuous completion mode" in client.second_prompt
    assert any(event.event_type == "worker_iteration_completed" for event in events)
    assert any(event.event_type == "worker_continuation_started" for event in events)
    assert finished.status == "canceled"
    assert finished.result == "first provisional result"


def test_continuous_worker_mode_is_explicit_and_rejects_terminal_schema(
    make_config,
) -> None:
    from swaag.runtime import AgentRuntime

    manager = WorkerManager(AgentRuntime(make_config(), model_client=object()))
    with pytest.raises(ValueError, match="completion_mode"):
        manager.create("objective", completion_mode="eventually")
    with pytest.raises(ValueError, match="terminal output_schema"):
        manager.create(
            "objective",
            completion_mode="continuous",
            output_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        )
    with pytest.raises(ValueError, match="terminal response presentations"):
        manager.create(
            "objective",
            completion_mode="continuous",
            presentation_modes=["audio"],
        )
    with pytest.raises(ValueError, match="visual and/or audio"):
        manager.create("objective", presentation_modes=["hologram"])
    manager.shutdown()


def test_worker_archive_preserves_exact_history_and_prevents_restart(make_config) -> None:
    from swaag.runtime import AgentRuntime

    runtime = AgentRuntime(make_config(model__context_limit=32_000), model_client=_ObjectiveClient())
    manager = WorkerManager(runtime)
    worker = manager.create("alpha objective")
    manager.start(worker.worker_id)
    manager.wait(worker.worker_id, timeout_seconds=10)

    archived = manager.archive(worker.worker_id)

    assert archived.status == "completed"
    assert archived.archived_at is not None
    assert runtime.history.read_history(worker.session_id)
    with pytest.raises(ValueError, match="archived"):
        manager.resume(worker.worker_id)
    manager.shutdown()


def test_worker_events_link_and_rehydrate_canonical_session_history(make_config) -> None:
    from swaag.runtime import AgentRuntime

    config = make_config(model__context_limit=32_000)
    runtime = AgentRuntime(config, model_client=object())
    manager = WorkerManager(runtime)
    worker = manager.create("stream canonical activity")
    state = runtime.history.rebuild_from_history(
        worker.session_id, write_projections=False
    )
    source = runtime.history.record_event(
        state,
        "tool_called",
        {
            "call_id": "call_exact",
            "tool_name": "reader",
            "tool_input": {"path": "evidence.txt"},
        },
    )
    runtime.history.record_event(
        state,
        "tool_result",
        {
            "call_id": "call_exact",
            "tool_name": "reader",
            "raw_input": {"path": "evidence.txt"},
            "validated_input": {"path": "evidence.txt"},
            "output": {"text": "durable exact result"},
            "source_event_references": [],
        },
    )

    projected_once = manager.events(worker.worker_id)
    projected_twice = manager.events(worker.worker_id)
    durable = manager.store.events(worker.worker_id)
    manager.shutdown()

    linked = [
        event for event in projected_once if event.event_type == "worker_history_event"
    ]
    assert len(linked) == 2
    assert len(projected_twice) == len(projected_once)
    assert linked[0].payload["history_event_hash"] == source.hash
    assert linked[0].payload["canonical_event"]["payload"]["tool_input"] == {
        "path": "evidence.txt"
    }
    assert all("canonical_event" not in event.payload for event in durable)

    restarted = WorkerManager(AgentRuntime(config, model_client=object()))
    replayed = restarted.events(worker.worker_id)
    restarted.shutdown()
    replayed_link = next(
        event for event in replayed if event.event_type == "worker_history_event"
    )
    assert replayed_link.payload["canonical_event"]["hash"] == source.hash
