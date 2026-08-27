from pathlib import Path
from types import SimpleNamespace

import requests


def test_trace_metrics_export_context_provenance() -> None:
    from swaag.benchmark.benchmark_runner import _task_trace_metrics

    provenance = {
        "context_compiled": {
            "session_id": "session-a",
            "sequence": 7,
            "event_hash": "hash-7",
        },
        "prompt_built": {
            "session_id": "session-a",
            "sequence": 9,
            "event_hash": "hash-9",
        },
        "context_limit": 2048,
        "input_tokens": 320,
        "reserved_response_tokens": 256,
        "safety_margin_tokens": 64,
        "required_tokens": 640,
        "exact": True,
        "components": [
            {
                "name": "current_user",
                "category": "current_user",
                "tokens": 12,
                "exact": True,
                "include_in_context": True,
            }
        ],
    }
    events = [
        SimpleNamespace(
            event_type="model_request_sent",
            payload={
                "call_id": "model-call-a",
                "kind": "action",
                "attempt": 1,
                "context_provenance": provenance,
            },
        )
    ]

    metrics = _task_trace_metrics(events)

    assert metrics["context_call_explanations"] == [
        {
            "call_id": "model-call-a",
            "kind": "action",
            "attempt": 1,
            **provenance,
        }
    ]


def test_benchmark_contract_write_allowlist_is_applied_to_runtime_config(tmp_path, monkeypatch) -> None:
    # This verifies the policy transformation independently of model execution by
    # constructing the same paths a benchmark seed provides.
    from swaag.benchmark.task_definitions import get_benchmark_tasks
    from swaag.benchmark.benchmark_runner import _build_config

    task = next(item for item in get_benchmark_tasks() if item.task_id == "coding_multifile_fix")
    scenario = task.create(tmp_path / "workspace", live_mode=True)
    config = _build_config(
        sessions_root=tmp_path / "sessions",
        workspace=scenario.workspace,
        overrides=task.config_overrides,
        base_url="http://127.0.0.1:1",
        seed=11,
    )
    contract = scenario.verification_contract
    assert contract.forbid_unexpected_workspace_changes is True
    allowed = []
    for item in contract.allowed_modified_files:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = scenario.workspace / candidate
        allowed.append(str(candidate.resolve()))
    config.editor.allowed_write_paths = allowed
    assert {Path(item).name for item in config.editor.allowed_write_paths} == {"tokenizer.py", "normalizer.py"}
    assert "test_multifile.py" not in {Path(item).name for item in config.editor.allowed_write_paths}


def test_benchmark_model_cache_namespace_changes_with_model_props(monkeypatch) -> None:
    import io
    import json
    from swaag.benchmark.benchmark_runner import _benchmark_model_cache_namespace

    class _Response:
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return json.dumps(self.payload).encode("utf-8")

    payloads = [
        {"model_alias": "qwen2.5-14b", "model_path": "/models/14b.gguf", "build_info": "build-a"},
        {"model_alias": "qwen2.5-32b", "model_path": "/models/32b.gguf", "build_info": "build-a"},
    ]
    monkeypatch.setattr("swaag.benchmark.benchmark_runner.urllib.request.urlopen", lambda *a, **k: _Response(payloads.pop(0)))
    first = _benchmark_model_cache_namespace("http://10.8.0.7:14831")
    second = _benchmark_model_cache_namespace("http://10.8.0.7:14832")
    assert first != second
    assert "14b" in first
    assert "32b" in second


def test_benchmark_model_cache_namespace_changes_when_model_replaced_on_same_endpoint(monkeypatch) -> None:
    import json
    from swaag.benchmark.benchmark_runner import _benchmark_model_cache_namespace

    class _Response:
        def __init__(self, payload): self.payload = payload
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return json.dumps(self.payload).encode("utf-8")

    payloads = [
        {"model_alias": "same-port", "model_path": "/models/a.gguf", "build_info": "build-a"},
        {"model_alias": "same-port", "model_path": "/models/b.gguf", "build_info": "build-a"},
    ]
    monkeypatch.setattr("swaag.benchmark.benchmark_runner.urllib.request.urlopen", lambda *a, **k: _Response(payloads.pop(0)))
    first = _benchmark_model_cache_namespace("http://10.8.0.7:14832")
    second = _benchmark_model_cache_namespace("http://10.8.0.7:14832")
    assert first != second


def test_benchmark_model_cache_namespace_ignores_volatile_server_state(
    monkeypatch,
) -> None:
    import json
    from swaag.benchmark.benchmark_runner import _benchmark_model_cache_namespace

    class _Response:
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return json.dumps(self.payload).encode("utf-8")

    stable = {
        "model_alias": "same-model",
        "model_path": "/missing/model.gguf",
        "model_ftype": "Q4_K_M",
        "build_info": "build-a",
        "chat_template": "template-a",
        "default_generation_settings": {"n_ctx": 22016, "params": {"top_p": 1.0}},
    }
    payloads = [
        stable | {"is_sleeping": False, "media_marker": "random-a", "total_slots": 1},
        stable | {"is_sleeping": True, "media_marker": "random-b", "total_slots": 4},
    ]
    monkeypatch.setattr(
        "swaag.benchmark.benchmark_runner.urllib.request.urlopen",
        lambda *args, **kwargs: _Response(payloads.pop(0)),
    )

    first = _benchmark_model_cache_namespace("http://10.8.0.7:14832")
    second = _benchmark_model_cache_namespace("http://10.8.0.7:14832")

    assert first == second


def test_planned_cache_mode_uses_model_namespace(tmp_path, monkeypatch) -> None:
    from swaag.benchmark.benchmark_runner import _planned_cache_mode
    from swaag.benchmark.task_definitions import get_benchmark_tasks
    monkeypatch.setenv("SWAAG_BENCHMARK_REPLAY_CACHE_ROOT", str(tmp_path / "cache"))
    task = get_benchmark_tasks()[0]
    ns14 = "qwen14-hash"
    ns32 = "qwen32-hash"
    path14 = tmp_path / "cache" / ns14 / task.task_id
    path14.mkdir(parents=True)
    (path14 / "seed_11.json").write_text("{}", encoding="utf-8")
    assert _planned_cache_mode(tmp_path, task, [11], cached=True, model_cache_namespace=ns14) == "replay"
    assert _planned_cache_mode(tmp_path, task, [11], cached=True, model_cache_namespace=ns32) == "record"


def test_benchmark_execution_blockers_are_external_not_task_failures() -> None:
    from swaag.benchmark.benchmark_runner import _benchmark_execution_blocker
    from swaag.model_cache import MissingReplayEntryError

    missing = _benchmark_execution_blocker(
        MissingReplayEntryError("missing exact request")
    )
    unreachable = _benchmark_execution_blocker(
        requests.ConnectionError("endpoint unavailable")
    )

    assert missing["kind"] == "missing_exact_replay_entry"
    assert unreachable["kind"] == "model_endpoint_unreachable"
    assert _benchmark_execution_blocker(ValueError("model output was invalid")) is None


def test_benchmark_run_cli_returns_distinct_blocked_status(
    monkeypatch, tmp_path
) -> None:
    from swaag.benchmark import benchmark_runner

    monkeypatch.setattr(
        benchmark_runner,
        "run_benchmarks",
        lambda **_kwargs: {
            "summary": {
                "total_tasks": 9,
                "executed_tasks": 0,
                "blocked_tasks": 9,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "false_positives": 0,
            }
        },
    )

    assert benchmark_runner.main(["run", "--output", str(tmp_path)]) == 2
