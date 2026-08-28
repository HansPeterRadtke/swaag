from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
from threading import Thread


def test_otlp_host_exporter_sends_swaag_spans_and_metrics(tmp_path: Path) -> None:
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - HTTP handler API
            size = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(size)
            requests.append(
                {
                    "path": self.path,
                    "content_type": self.headers.get("Content-Type", ""),
                    "body_size": len(body),
                }
            )
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    script = tmp_path / "emit_telemetry.py"
    script.write_text(
        "\n".join(
            [
                "from swaag.telemetry import OperationalTelemetry",
                "from swaag.telemetry_export import configure_otlp_export_from_environment",
                "runtime = configure_otlp_export_from_environment()",
                "assert runtime is not None",
                "telemetry = OperationalTelemetry()",
                "with telemetry.agent_invocation(session_id='session_probe', run_id='run_probe', model_name='model_probe'):",
                "    pass",
                "runtime.shutdown()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = {
        "PATH": str(Path(sys.executable).parent),
        "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        "OTEL_EXPORTER_OTLP_ENDPOINT": endpoint,
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
        "OTEL_SERVICE_NAME": "swaag-export-test",
        "OTEL_METRIC_EXPORT_INTERVAL": "600000",
    }
    try:
        completed = subprocess.run(
            [sys.executable, str(script)],
            text=True,
            capture_output=True,
            timeout=30,
            env=env,
            check=False,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert completed.returncode == 0, completed.stderr
    paths = {str(item["path"]) for item in requests}
    assert paths == {"/v1/traces", "/v1/metrics"}
    assert all(int(item["body_size"]) > 0 for item in requests)
    assert all(
        item["content_type"] == "application/x-protobuf" for item in requests
    )
    (tmp_path / "requests.json").write_text(
        json.dumps(requests, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_otlp_host_exporter_rejects_invalid_metric_interval(
    monkeypatch,
) -> None:
    from swaag.telemetry_export import configure_otlp_export_from_environment

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setenv("OTEL_METRIC_EXPORT_INTERVAL", "not-a-number")

    try:
        configure_otlp_export_from_environment()
    except ValueError as exc:
        assert "OTEL_METRIC_EXPORT_INTERVAL" in str(exc)
    else:
        raise AssertionError("invalid export interval was accepted")
