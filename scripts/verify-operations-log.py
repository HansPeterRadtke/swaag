#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from swaag.config import load_config
from swaag.operations_log import OperationsLog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.parent / "operations.jsonl"
    for path in output.parent.glob("operations.jsonl*"):
        path.unlink()
    config = load_config(
        env={
            "SWAAG__SESSIONS__ROOT": str(output.parent / "sessions"),
            "SWAAG__LOGGING__FILE_PATH": str(log_path),
            "SWAAG__LOGGING__QUEUE_CAPACITY": "2",
            # Startup config/source/metadata is intentionally large; choose a
            # realistic bound above one complete startup record, then force
            # rotation with ordinary records.
            "SWAAG__LOGGING__MAX_BYTES": "200000",
            "SWAAG__LOGGING__BACKUP_COUNT": "2",
        }
    )
    verifier_secret = "live-ops-configured-secret-DO-NOT-PERSIST"
    config.a2a_authorization.bearer_token = verifier_secret
    runtime = OperationsLog(config, component="live-ops-verifier")
    runtime.event(
        "secret-redaction-probe",
        authorization="Bearer live-authorization-secret",
        access_token="live-access-token-secret",
        service_api_key="live-api-key-secret",
        detail=(
            "Authorization: Bearer embedded-live-secret "
            "api_key=embedded-live-api " + verifier_secret
        ),
    )
    original = runtime._write_line
    fallback_path = output.parent / "overflow-fallback.jsonl"

    def slow(line: str) -> None:
        time.sleep(0.003)
        original(line)

    runtime._write_line = slow  # intentional live stress of bounded queue
    # Capture the documented stderr fallback without flooding the control channel.
    old_stderr = __import__("os").dup(2)
    fallback_fd = __import__("os").open(
        fallback_path,
        __import__("os").O_WRONLY | __import__("os").O_CREAT | __import__("os").O_TRUNC,
        0o600,
    )
    __import__("os").dup2(fallback_fd, 2)
    try:
        for index in range(500):
            runtime.event("stress", index=index, payload="x" * 500)
        runtime._queue.join()
    finally:
        __import__("os").dup2(old_stderr, 2)
        __import__("os").close(old_stderr)
        __import__("os").close(fallback_fd)
    runtime._write_line = original
    # Guaranteed writes after the overflow phase make the rotation/retention path
    # deterministic while still using the public event API.
    for index in range(30):
        runtime.event("rotation", index=index, payload="r" * 5000)
        runtime._queue.join()
    runtime.shutdown(reason="live_verifier_done")
    files = [p for p in [log_path, Path(str(log_path)+".1"), Path(str(log_path)+".2")] if p.exists()]
    rows = []
    for path in files:
        for line in path.read_text(errors="replace").splitlines():
            try: rows.append(json.loads(line))
            except Exception: pass
    names = [row.get("event") for row in rows]
    persisted_text = "\n".join(
        path.read_text(errors="replace") for path in files if path.exists()
    )
    redaction_secrets = (
        verifier_secret,
        "live-authorization-secret",
        "live-access-token-secret",
        "live-api-key-secret",
        "embedded-live-secret",
        "embedded-live-api",
    )
    checks = [
        {
            "name": "secrets_redacted",
            "passed": all(secret not in persisted_text for secret in redaction_secrets)
            and "[REDACTED]:sha256=" in persisted_text,
            "evidence": {
                "secret_count": len(redaction_secrets),
                "redaction_marker_present": "[REDACTED]:sha256=" in persisted_text,
            },
        },
        {"name": "queue_overflow_observed", "passed": runtime.dropped_count > 0 and fallback_path.exists() and "queue_overflow" in fallback_path.read_text(errors="replace"), "evidence": {"dropped": runtime.dropped_count, "fallback_bytes": fallback_path.stat().st_size if fallback_path.exists() else 0}},
        {"name": "rotation_observed", "passed": len(files) >= 2, "evidence": {"files": [str(p) for p in files]}},
        {"name": "retention_bounded", "passed": not Path(str(log_path)+".3").exists(), "evidence": {}},
        {"name": "shutdown_flushed", "passed": "shutdown" in names, "evidence": {"events": names[-20:]}},
        {"name": "files_bounded", "passed": all(p.stat().st_size <= 205000 for p in files), "evidence": {str(p): p.stat().st_size for p in files}},
    ]
    report = {"benchmark": "operations-log-live-verifier", "checks": checks, "passed": all(x["passed"] for x in checks)}
    output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": len(checks), "output": str(output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
