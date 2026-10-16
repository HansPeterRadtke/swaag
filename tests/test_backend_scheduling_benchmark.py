from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

from swaag.benchmark.backend_scheduling import (
    _parse_vllm_scheduler_metrics,
    run_backend_scheduling_benchmark,
)
from swaag.benchmark.benchmark_runner import _build_parser


def test_parse_vllm_scheduler_metrics() -> None:
    text = '''
# HELP ignored x
vllm:num_requests_running{engine="0",model_name="x"} 2.0
vllm:num_requests_waiting{engine="0",model_name="x"} 1.0
'''
    assert _parse_vllm_scheduler_metrics(text) == {
        "num_requests_running": 2.0,
        "num_requests_waiting": 1.0,
    }


def test_backend_scheduling_records_absolute_order_and_metrics(tmp_path: Path) -> None:
    state = {"running": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_GET(self):  # noqa: N802
            if self.path != "/metrics":
                self.send_error(404)
                return
            with lock:
                running = state["running"]
            body = (
                'vllm:num_requests_running{engine="0",model_name="fake"} '
                f'{running}.0\n'
                'vllm:num_requests_waiting{engine="0",model_name="fake"} 0.0\n'
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            prompt = payload["messages"][0]["content"]
            is_long = prompt.startswith("alpha ")
            with lock:
                state["running"] += 1
            try:
                time.sleep(0.16 if is_long else 0.02)
                chunks = ["long-a", "long-b"] if is_long else ["short"]
                lines = []
                if is_long:
                    item = {"choices": [{"delta": {"reasoning_content": "thinking"}, "finish_reason": None}]}
                    lines.append("data: " + json.dumps(item) + "\n\n")
                for content in chunks:
                    item = {"choices": [{"delta": {"content": content}, "finish_reason": None}]}
                    lines.append("data: " + json.dumps(item) + "\n\n")
                lines.append("data: [DONE]\n\n")
                body = "".join(lines).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            finally:
                with lock:
                    state["running"] -= 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        report = run_backend_scheduling_benchmark(
            output_dir=tmp_path / "run",
            endpoint=base + "/v1/chat/completions",
            model="fake",
            metrics_url=base + "/metrics",
            long_prompt_repetitions=10,
            long_max_tokens=8,
            short_max_tokens=4,
            short_submit_delay_seconds=0.03,
            scheduler_sample_delay_seconds=0.01,
            timeout_seconds=2,
            clean=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert report["passed"] is True
    assert report["transport_succeeded"] is True
    assert report["short_finished_before_long"] is True
    assert report["short_submitted_before_long_finished"] is True
    assert report["short_first_token_before_long_finished"] is True
    assert report["interactive_overlap_observed"] is True
    assert report["short_serialized_behind_long"] is False
    assert report["overlap_seconds"] > 0
    assert report["long"]["ttft_seconds"] is not None
    assert report["long"]["visible_ttft_seconds"] is not None
    assert report["long"]["reasoning_chars"] == len("thinking")
    assert report["short"]["submitted_offset_seconds"] > report["long"]["submitted_offset_seconds"]
    assert report["short"]["finished_offset_seconds"] < report["long"]["finished_offset_seconds"]
    assert report["scheduler_during_overlap"]["metrics"]["num_requests_running"] == 2.0
    assert report["scheduler_samples"]
    assert report["scheduler_peak_running"] == 2.0
    assert report["scheduler_peak_waiting"] == 0.0
    assert report["scheduler_concurrency_observed"] is True
    assert (tmp_path / "run" / "backend_scheduling_results.json").exists()



def test_backend_scheduling_identifies_serialized_short_request(tmp_path: Path) -> None:
    request_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            prompt = payload["messages"][0]["content"]
            is_long = prompt.startswith("alpha ")
            with request_lock:
                time.sleep(0.12 if is_long else 0.01)
                content = "long" if is_long else "short"
                body = (
                    "data: "
                    + json.dumps({"choices": [{"delta": {"content": content}, "finish_reason": None}]})
                    + "\n\ndata: [DONE]\n\n"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        report = run_backend_scheduling_benchmark(
            output_dir=tmp_path / "serialized",
            endpoint=base + "/v1/chat/completions",
            model="fake",
            long_prompt_repetitions=10,
            long_max_tokens=8,
            short_max_tokens=4,
            short_submit_delay_seconds=0.02,
            scheduler_sample_delay_seconds=0.01,
            timeout_seconds=2,
            clean=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert report["transport_succeeded"] is True
    assert report["short_submitted_before_long_finished"] is True
    assert report["short_first_token_before_long_finished"] is False
    assert report["interactive_overlap_observed"] is False
    assert report["short_serialized_behind_long"] is True
    assert report["short_finished_before_long"] is False
    assert report["scheduler_peak_running"] is None
    assert report["scheduler_peak_waiting"] is None
    assert report["scheduler_concurrency_observed"] is None

def test_backend_scheduling_cli_arguments() -> None:
    args = _build_parser().parse_args(
        [
            "backend-scheduling",
            "--endpoint",
            "http://127.0.0.1:9000/v1/chat/completions",
            "--model",
            "example",
            "--metrics-url",
            "http://127.0.0.1:9000/metrics",
            "--long-max-tokens",
            "128",
        ]
    )
    assert args.command == "backend-scheduling"
    assert args.model == "example"
    assert args.long_max_tokens == 128
    assert args.metrics_url.endswith("/metrics")
