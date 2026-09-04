from __future__ import annotations

import json

from swaag.history import HistoryStore
from swaag.redaction import redact_for_persistence, redact_text


def test_recursive_redaction_preserves_nonsecret_token_metrics():
    value = {
        "authorization": "Bearer top-secret",
        "nested": {
            "client_secret": "client-xyz",
            "prompt_tokens": 123,
            "token_count": 456,
            "message": "request failed Authorization: Bearer abc.def.ghi",
        },
        "url": "https://user:passw0rd@example.test/path",
    }
    redacted = redact_for_persistence(value)
    encoded = json.dumps(redacted)
    assert "top-secret" not in encoded
    assert "client-xyz" not in encoded
    assert "abc.def.ghi" not in encoded
    assert "passw0rd" not in encoded
    assert redacted["nested"]["prompt_tokens"] == 123
    assert redacted["nested"]["token_count"] == 456
    assert "[REDACTED]" in encoded


def test_known_configured_secret_is_scrubbed_from_unstructured_text():
    text = "remote failure included opaque-value-123 in provider diagnostics"
    redacted = redact_text(text, secret_values=["opaque-value-123"])
    assert "opaque-value-123" not in redacted
    assert "[REDACTED]" in redacted


def test_history_store_redacts_before_hashing_and_persistence(tmp_path):
    history = HistoryStore(tmp_path / "sessions", secret_values=["known-secret-value"])
    state = history.create(config_fingerprint="cfg", model_base_url="http://model")
    event = history.record_event(
        state,
        "error",
        {
            "operation": "remote_call",
            "error_type": "RemoteError",
            "error": "Bearer raw-bearer and known-secret-value",
            "headers": {
                "Authorization": "Bearer header-secret",
                "x-safe": "visible",
            },
            "access_token": "access-secret",
            "completion_tokens": 19,
        },
        metadata={"password": "metadata-secret", "safe": "yes"},
    )
    assert event.payload["completion_tokens"] == 19
    assert event.payload["headers"]["x-safe"] == "visible"
    serialized = history.history_path(state.session_id).read_text(encoding="utf-8")
    for secret in (
        "raw-bearer",
        "known-secret-value",
        "header-secret",
        "access-secret",
        "metadata-secret",
    ):
        assert secret not in serialized
    assert "[REDACTED]" in serialized
    rebuilt = history.read_history(state.session_id)[-1]
    assert rebuilt.hash == event.hash
    assert rebuilt.payload == event.payload


def test_prompt_instruction_store_redacts_configured_secret(make_config, monkeypatch):
    from swaag.prompt_instruction_store import PromptInstructionStore
    config = make_config()
    config.mcp.authorization.introspection_client_secret = "store-secret"
    config.a2a_authorization.bearer_token = "a2a-secret"
    store = PromptInstructionStore(config.sessions.root, config)
    store.add(
        title="Sensitive learned rule",
        content="provider diagnostic leaked store-secret",
        scopes=["communication_status"],
        origin_session_id="session-x",
    )
    raw = store.path.read_bytes()
    assert b"store-secret" not in raw
    rebuilt = store.list()
    assert len(rebuilt) == 1
    assert "store-secret" not in rebuilt[0].content
    assert "[REDACTED]" in rebuilt[0].content


def test_result_collector_redacts_reports(tmp_path):
    from swaag.benchmark.result_collector import BenchmarkTaskResult, ResultCollector
    collector = ResultCollector(secret_values=("report-secret",))
    collector.add(BenchmarkTaskResult(
        task_id="t", task_type="quality", difficulty="easy", tags=[], description="report-secret",
        expected_outcome="success", success=False, false_positive=False, session_id="s",
        assistant_text="Bearer report-bearer", deterministic_verification_passed=False,
        verification_summary={"checks": {}, "evidence": {}, "reason": "report-secret"},
        quality_summary={"passed": False, "checks": {}, "evidence": {}, "oracle": {}},
        metrics={}, failure_category="x", failure_reason="report-secret", failure_subsystem="model",
    ))
    collector.write(tmp_path, run_metadata={"output_dir": str(tmp_path)})
    combined = (tmp_path / "benchmark_results.json").read_text() + (tmp_path / "benchmark_report.md").read_text()
    assert "report-secret" not in combined
    assert "report-bearer" not in combined
    assert "[REDACTED]" in combined


def test_configured_secret_values_include_a2a_bearer(make_config) -> None:
    from swaag.redaction import configured_secret_values, redact_text

    config = make_config()
    config.a2a_authorization.bearer_token = "a2a-secret-value"
    secrets = configured_secret_values(config)
    assert "a2a-secret-value" in secrets
    assert "a2a-secret-value" not in redact_text(
        "Authorization: Bearer a2a-secret-value", secret_values=secrets
    )
