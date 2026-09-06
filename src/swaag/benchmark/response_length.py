from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from swaag.config import AgentConfig, load_config
from swaag.grammar import _contract
from swaag.runtime import AgentRuntime
from swaag.types import PromptComponent
from swaag.utils import stable_json_dumps


CASE_TEXT = (
    "Explain why append-only event history is useful in a durable agent harness. "
    "Cover recovery after restart, provenance, and the distinction between authoritative "
    "events and disposable prompt projections."
)

LENGTH_CASES: dict[str, dict[str, Any]] = {
    "exact_words_45": {
        "instruction": "Answer in exactly 45 words.",
        "target_words": 45,
        "minimum_words": 45,
        "maximum_words": 45,
        "instruction_kind": "exact_words",
    },
    "short": {
        "instruction": "Give a short answer.",
        "target_words": None,
        "minimum_words": 20,
        "maximum_words": 70,
        "instruction_kind": "qualitative",
    },
    "medium": {
        "instruction": "Give a medium-length answer with enough detail to be useful.",
        "target_words": None,
        "minimum_words": 70,
        "maximum_words": 180,
        "instruction_kind": "qualitative",
    },
    "detailed": {
        "instruction": "Give a detailed answer.",
        "target_words": None,
        "minimum_words": 180,
        "maximum_words": 420,
        "instruction_kind": "qualitative",
    },
}

_RESPONSE_LENGTH_SYSTEM = """You are measuring response-length instruction following. Answer the supplied task accurately and obey the requested response-size instruction. Do not discuss the benchmark or the instruction itself. Return only JSON matching the supplied schema."""


def response_length_contract():
    return _contract(
        "response_length_measurement",
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )


def select_cases(names: Iterable[str] = ()) -> list[str]:
    requested = list(names)
    unknown = sorted(set(requested) - set(LENGTH_CASES))
    if unknown:
        raise ValueError("Unknown response-length case: " + ", ".join(unknown))
    return requested or list(LENGTH_CASES)


def _case_config(base: AgentConfig, *, sessions_root: Path) -> AgentConfig:
    config = copy.deepcopy(base)
    config.sessions.root = sessions_root
    config.model.cache_enabled = False
    config.tools.enabled = []
    config.tools.allow_stateful_tools = False
    config.tools.allow_side_effect_tools = False
    config.runtime.completion_evaluation_enabled = False
    return config


def _model_identity(runtime: AgentRuntime) -> Any:
    provider = getattr(runtime.client, "cache_identity", None)
    return provider() if callable(provider) else type(runtime.client).__name__


def _word_count(text: str) -> int:
    return len(text.split())


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_response_length_benchmark(
    *,
    output_dir: Path,
    config: AgentConfig | None = None,
    case_names: Iterable[str] = (),
    clean: bool = False,
    runtime_factory: Callable[[AgentConfig], AgentRuntime] = AgentRuntime,
    model_identity: Any | None = None,
) -> dict[str, Any]:
    base = config or load_config()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "response_length_results.json"
    selected = select_cases(case_names)
    results: list[dict[str, Any]] = []
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("planned_cases") != selected:
            raise ValueError("Response-length checkpoint case list does not match this run")
        results = [dict(item) for item in previous.get("results", [])]
        checkpoint_identity = previous.get("model_identity")
        if model_identity is not None and model_identity != checkpoint_identity:
            raise ValueError("Response-length checkpoint model identity does not match this run")
        model_identity = checkpoint_identity if model_identity is None else model_identity

    for index, case_name in enumerate(selected, start=1):
        if any(item.get("case") == case_name for item in results):
            continue
        case = LENGTH_CASES[case_name]
        runtime = runtime_factory(
            _case_config(base, sessions_root=output_dir / "runs" / f"{index:02d}-{case_name}" / "sessions")
        )
        current_identity = _model_identity(runtime)
        if model_identity is None:
            model_identity = current_identity
        elif current_identity != model_identity:
            raise ValueError("Response-length runtime model identity changed")
        state = runtime.create_or_load_session()
        assembly = runtime.prompts.build_semantic_operation_prompt(
            kind="response_length_measurement",
            system_instruction=_RESPONSE_LENGTH_SYSTEM,
            components=[
                PromptComponent(
                    name="response_length_task",
                    category="turn_context",
                    text=(
                        "Task:\n"
                        + CASE_TEXT
                        + "\n\nResponse-size instruction:\n"
                        + str(case["instruction"])
                    ),
                )
            ],
        )
        started = time.monotonic()
        payload = runtime._execute_compiled_presentation_call(
            state,
            assembly,
            response_length_contract(),
            validator=lambda value: value if isinstance(value.get("answer"), str) and value["answer"].strip() else (_ for _ in ()).throw(ValueError("answer must be a non-empty string")),
        )
        answer = str(payload["answer"]).strip()
        words = _word_count(answer)
        minimum = int(case["minimum_words"])
        maximum = int(case["maximum_words"])
        passed = minimum <= words <= maximum
        results.append(
            {
                "case": case_name,
                "instruction_kind": case["instruction_kind"],
                "instruction": case["instruction"],
                "answer": answer,
                "word_count": words,
                "target_words": case["target_words"],
                "minimum_words": minimum,
                "maximum_words": maximum,
                "absolute_target_error_words": (
                    None
                    if case["target_words"] is None
                    else abs(words - int(case["target_words"]))
                ),
                "passed": passed,
                "elapsed_seconds": time.monotonic() - started,
                "session_id": state.session_id,
            }
        )
        _write_report(
            report_path,
            {
                "benchmark": "response_length_instruction_following",
                "status": "running",
                "model_identity": model_identity,
                "planned_cases": selected,
                "results": results,
                "passed": sum(bool(item["passed"]) for item in results),
                "total": len(results),
                "complete": len(results) == len(selected),
            },
        )
    report = {
        "benchmark": "response_length_instruction_following",
        "status": "completed" if len(results) == len(selected) else "running",
        "model_identity": model_identity,
        "planned_cases": selected,
        "results": results,
        "passed": sum(bool(item["passed"]) for item in results),
        "total": len(results),
        "complete": len(results) == len(selected),
    }
    _write_report(report_path, report)
    return report
