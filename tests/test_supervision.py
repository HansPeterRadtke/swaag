from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from swaag.model import LlamaCppClient
from swaag.preemption import ModelCallPreempted
from swaag.supervision import BackendActivityMonitor, RuntimeSupervisor, llama_slot_activity


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true")
        time.sleep(.005)


def slots(processing=False, processed=0):
    return [{"id": 0, "id_task": 1, "is_processing": processing,
             "n_prompt_tokens": 100, "n_prompt_tokens_processed": processed,
             "next_token": [{"n_decoded": 0}]}]


def test_busy_flag_is_not_fabricated_progress():
    state = {"processed": 0}
    monitor = BackendActivityMonitor(lambda: llama_slot_activity(slots(True, state["processed"])), interval=.01)
    with monitor.observing():
        assert monitor.wait_for_sample(1)
        assert monitor.snapshot()["state"] == "processing"
        assert monitor.snapshot()["progress_observed_at"] is None
        state["processed"] = 4
        wait_until(lambda: monitor.snapshot()["progress_observed_at"] is not None)
        assert monitor.snapshot()["slots"][0]["phase"] == "prefill"


def test_supervisor_stays_responsive_while_a_backend_probe_is_blocked():
    release = threading.Event()
    blocked = BackendActivityMonitor(lambda: (release.wait(2), llama_slot_activity(slots()))[1], interval=.01)
    working = BackendActivityMonitor(lambda: llama_slot_activity(slots(True, 10)), interval=.01)
    def runtime(url, monitor):
        return SimpleNamespace(config=SimpleNamespace(model=SimpleNamespace(base_url=url)),
                               client=SimpleNamespace(activity_monitor=lambda: monitor))
    orchestrator, worker = runtime("http://one", blocked), runtime("http://two", working)
    supervisor = RuntimeSupervisor({"orchestrator": orchestrator, "worker:two": worker}, interval=.01)
    supervisor.start()
    try:
        supervisor.observe_runtime(orchestrator, "session-a", {"phase": "inference", "heartbeat_at": "now"})
        assert working.wait_for_sample(1)
        initial = supervisor.snapshot()["supervisor"]["heartbeat_at"]
        wait_until(lambda: supervisor.snapshot()["supervisor"]["heartbeat_at"] != initial)
        started = time.monotonic()
        result = supervisor.snapshot()
        assert time.monotonic() - started < .1
        assert result["backends"]["http://two"]["state"] == "processing"
        assert result["active_sessions"][0]["roles"] == ["orchestrator"]
        assert result["active_sessions"][0]["heartbeat_is_computation_evidence"] is False
        assert result["automatic_semantic_intervention"] is False
        supervisor.observe_runtime(orchestrator, "session-a", {"phase": "completed"})
        assert supervisor.snapshot()["active_sessions"] == []
    finally:
        release.set()
        supervisor.close()


@pytest.fixture
def local_backend():
    state = {"processing": False, "processed": 0, "delay": 0, "headers_first": False, "http_status": 200}
    entered = threading.Event()
    release = threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_GET(self):
            body = json.dumps(slots(state["processing"], state["processed"])).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            state["processing"] = True
            entered.set()
            try:
                if state["headers_first"]:
                    self.send_response(state["http_status"])
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.flush()
                started = time.monotonic()
                while not release.wait(.02):
                    state["processed"] += 1
                    if state["delay"] and time.monotonic() - started >= state["delay"]:
                        break
                if not state["headers_first"]:
                    self.send_response(state["http_status"])
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                # Multi-byte UTF-8 split across writes must remain exact.
                body = ('data: ' + json.dumps({"content": "Grüße", "stop": True, "tokens_predicted": 1}, ensure_ascii=False) + '\n\n').encode()
                for byte in body:
                    self.wfile.write(bytes([byte]))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                state["processing"] = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", state, entered, release
    release.set()
    server.shutdown()
    server.server_close()
    thread.join(1)


@pytest.mark.parametrize("headers_first", [False, True])
def test_active_prefill_survives_estimated_read_deadline(make_config, local_backend, headers_first):
    url, state, _, _ = local_backend
    state.update(delay=1.3, headers_first=headers_first)
    client = LlamaCppClient(make_config(model__base_url=url, model__timeout_seconds=1))
    progress = []
    result = client.send_completion({"prompt": "required input", "n_predict": 32}, progress_callback=progress.append)
    assert result.text == "Grüße"
    assert result.elapsed_seconds >= 1.3
    assert result.raw_response["timeout_policy"] == "observed_local_activity_no_speculative_deadline"
    assert any(item.get("backend_activity", {}).get("state") == "processing" for item in progress)


@pytest.mark.parametrize("headers_first", [False, True])
def test_explicit_cancellation_interrupts_headers_or_silent_body(make_config, local_backend, headers_first):
    url, state, entered, release = local_backend
    state["headers_first"] = headers_first
    client = LlamaCppClient(make_config(model__base_url=url, model__timeout_seconds=1))
    cancel = threading.Event()
    outcomes = []
    def run():
        try:
            client.send_completion({"prompt": "x", "n_predict": 32}, cancel_check=cancel.is_set, cancel_poll_seconds=.01)
        except Exception as exc:
            outcomes.append(exc)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert entered.wait(1)
        cancel.set()
        thread.join(1)
        assert not thread.is_alive(), "explicit cancellation was blocked on the model response"
        assert len(outcomes) == 1 and isinstance(outcomes[0], ModelCallPreempted)
    finally:
        release.set()
        thread.join(2)


def test_generation_evidence_takes_precedence_over_partial_prompt_counters():
    rows = slots(True, 50)
    rows[0]['next_token'][0]['n_decoded'] = 2
    assert llama_slot_activity(rows)['slots'][0]['phase'] == 'generation'
    rows[0]['next_token'][0]['n_decoded'] = 0
    rows[0]['n_prompt_tokens_cache'] = 50
    assert llama_slot_activity(rows)['slots'][0]['phase'] == 'generation'


def test_cancel_interrupts_silent_http_error_body(make_config, local_backend):
    url, state, entered, release = local_backend
    state.update(headers_first=True, http_status=500)
    client = LlamaCppClient(make_config(model__base_url=url))
    cancel = threading.Event()
    errors = []
    def call():
        try:
            client.send_completion({'prompt': 'hello', 'n_predict': 1}, cancel_check=cancel.is_set)
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=call)
    thread.start()
    assert entered.wait(2)
    time.sleep(.1)
    cancel.set()
    thread.join(1)
    release.set()
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], ModelCallPreempted)
