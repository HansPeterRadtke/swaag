from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from typing import Any

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
        prompt = str(payload.get("prompt", ""))
        if "communication assistant for another SWAAG agent" in prompt:
            return self._result(payload, _action("The main agent is still working."))
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
    assert client.main_started.wait(timeout=2)

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
    assert replayed[0].payload["request"] == client.requests[0]


def test_target_changing_communication_invalidates_replay_and_refreshes_history(make_config) -> None:
    config = make_config(model__context_limit=32_000)
    client = _InvalidationClient()
    runtime = AgentRuntime(config, model_client=client)
    state = runtime.create_or_load_session()
    service = CommunicationService(runtime)
    holder: dict[str, Any] = {}

    thread = threading.Thread(target=lambda: holder.setdefault("result", runtime.run_turn_in_session(state, "Original objective.")), daemon=True)
    thread.start()
    assert client.main_started.wait(timeout=2)

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
