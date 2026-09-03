from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

from swaag.config import AgentConfig, load_config
from swaag.grammar import (
    agent_action_contract,
    agent_capability_selection_contract,
    agent_terminal_response_contract,
    agent_tool_call_contract,
    communication_status_contract,
    completion_evaluation_contract,
    note_selection_contract,
    prompt_instruction_selection_contract,
    summary_contract,
    yes_no_contract,
)
from swaag.model import LlamaCppClient
from swaag.redaction import configured_secret_values, redact_for_persistence
from swaag.tools.base import ToolValidationError, _validate_schema_value
from swaag.types import ContractSpec
from swaag.utils import sha256_text, stable_json_dumps


BENCHMARK_VERSION = 1
DEFAULT_SEEDS: tuple[int, ...] = (17, 42, 91)


@dataclass(slots=True, frozen=True)
class ConstraintDecodingCase:
    case_id: str
    prompt: str
    contract: ContractSpec
    semantic_check: Callable[[dict[str, Any]], bool]


@dataclass(slots=True, frozen=True)
class ConstraintDecodingResult:
    case_id: str
    seed: int
    repetition: int
    contract_name: str
    schema_sha256: str
    request_schema_sha256: str | None
    constraint_present_in_request: bool
    json_parsed: bool
    schema_valid: bool
    structurally_valid: bool
    semantic_passed: bool
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    elapsed_seconds: float | None
    tokens_per_second: float | None
    text_sha256: str | None
    error_type: str | None = None
    error_reason: str | None = None


def _tool_schema(properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def build_constraint_decoding_cases() -> list[ConstraintDecodingCase]:
    tool_specs = [
        (
            "calculator",
            "Evaluate an arithmetic expression.",
            _tool_schema({"expression": {"type": "string"}}),
        ),
        (
            "read_file",
            "Read one file.",
            _tool_schema({"path": {"type": "string"}}),
        ),
    ]
    return [
        ConstraintDecodingCase(
            "yes_no_enum",
            "Return yes in the constrained answer field.",
            yes_no_contract(),
            lambda p: p.get("answer") == "yes",
        ),
        ConstraintDecodingCase(
            "summary_object",
            "Summarize the word alpha as exactly alpha and preserve zero recent messages.",
            summary_contract(),
            lambda p: p.get("summary") == "alpha" and p.get("preserve_recent_messages") == 0,
        ),
        ConstraintDecodingCase(
            "communication_status_enums_arrays",
            (
                "Report a major blocked situation. Set importance to major, escalation false, "
                "evidence_sequences to [7], and keep all required string fields non-empty."
            ),
            communication_status_contract(),
            lambda p: p.get("importance") == "major"
            and p.get("escalate_to_stronger_model") is False
            and p.get("evidence_sequences") == [7],
        ),
        ConstraintDecodingCase(
            "completion_without_evidence_field",
            "The task is incomplete. Return complete false with remaining work containing verify.",
            completion_evaluation_contract(),
            lambda p: p.get("complete") is False
            and "evidence_requests" not in p
            and any("verify" in str(v).lower() for v in p.get("remaining_work", [])),
        ),
        ConstraintDecodingCase(
            "completion_with_state_dependent_evidence_union",
            (
                "The task is incomplete. Request tool_result/source-7 as evidence with purpose verify."
            ),
            completion_evaluation_contract(
                [("tool_result", "source-7"), ("attachment", "attachment-2")]
            ),
            lambda p: p.get("complete") is False
            and any(
                row.get("source_kind") == "tool_result" and row.get("source_id") == "source-7"
                for row in p.get("evidence_requests", [])
                if isinstance(row, dict)
            ),
        ),
        ConstraintDecodingCase(
            "capability_selection_enum",
            "Select the search capability.",
            agent_capability_selection_contract(["search", "reading", "shell"]),
            lambda p: p.get("capability") == "search",
        ),
        ConstraintDecodingCase(
            "tool_call_state_dependent_union",
            "Call calculator with expression 2+3 and no other tool.",
            agent_tool_call_contract(tool_specs),
            lambda p: len(p.get("tool_calls", [])) == 1
            and p["tool_calls"][0].get("tool_name") == "calculator"
            and p["tool_calls"][0].get("arguments", {}).get("expression") == "2+3",
        ),
        ConstraintDecodingCase(
            "tool_call_zero_tool_state",
            "No tools are available. Return an empty tool_calls array.",
            agent_tool_call_contract([]),
            lambda p: p.get("tool_calls") == [],
        ),
        ConstraintDecodingCase(
            "terminal_response_fixed_false_enum",
            "Respond with READY and do not use silent completion.",
            agent_terminal_response_contract(allow_silent_completion=False),
            lambda p: p.get("assistant_message") == "READY" and p.get("silent_completion") is False,
        ),
        ConstraintDecodingCase(
            "agent_action_zero_tool_state",
            (
                "No tools are available. Respond READY, use no tool calls, continue_loop false, "
                "silent_completion false, importance normal, and no questions."
            ),
            agent_action_contract([], allow_silent_completion=False),
            lambda p: p.get("tool_calls") == []
            and p.get("continue_loop") is False
            and p.get("silent_completion") is False,
        ),
        ConstraintDecodingCase(
            "note_selection_dynamic_enum",
            "Select only note-b.",
            note_selection_contract(["note-a", "note-b", "note-c"]),
            lambda p: p.get("selected_note_ids") == ["note-b"],
        ),
        ConstraintDecodingCase(
            "instruction_selection_dynamic_anyof",
            "Select only the user instruction inst-b.",
            prompt_instruction_selection_contract(
                [("user", "inst-a"), ("user", "inst-b"), ("session", "inst-c")]
            ),
            lambda p: p.get("selected_instructions")
            == [{"instruction_store": "user", "instruction_id": "inst-b"}],
        ),
    ]


def _model_identity(client: Any) -> Any:
    identity = getattr(client, "cache_identity", None)
    value = identity() if callable(identity) else type(client).__name__
    if not isinstance(value, dict):
        return value
    stable_keys = (
        "base_url",
        "completion_endpoint",
        "configured_model_identity",
        "model_alias",
        "model_file",
        "profile_name",
        "server_build_info",
        "local_server_process_sha256",
    )
    return {key: value.get(key) for key in stable_keys}


def _request_schema(payload: dict[str, Any]) -> dict[str, Any] | None:
    direct = payload.get("json_schema")
    if isinstance(direct, dict):
        return direct
    response_format = payload.get("response_format")
    if not isinstance(response_format, dict):
        return None
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return None
    schema = json_schema.get("schema")
    return schema if isinstance(schema, dict) else None


def _atomic_checkpoint(path: Path, payload: dict[str, Any], *, secret_values: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    sanitized = redact_for_persistence(payload, secret_values=secret_values)
    raw = stable_json_dumps(sanitized, indent=2) + "\n"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, raw.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)


def run_constraint_decoding_benchmark(
    *,
    config: AgentConfig | None = None,
    output_path: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    repetitions_per_seed: int = 10,
    case_ids: Iterable[str] = (),
    client: Any | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    if repetitions_per_seed <= 0:
        raise ValueError("repetitions_per_seed must be positive")
    config = deepcopy(config or load_config())
    config.model.structured_output_mode = "server_schema"
    all_cases = build_constraint_decoding_cases()
    requested = list(case_ids)
    by_id = {case.case_id: case for case in all_cases}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown constraint-decoding cases: {', '.join(unknown)}")
    cases = [by_id[case_id] for case_id in requested] if requested else all_cases
    exact_seeds = [int(seed) for seed in seeds]
    if not exact_seeds:
        raise ValueError("constraint-decoding benchmark requires at least one seed")
    client = client or LlamaCppClient(config)
    identity = _model_identity(client)
    secret_values = configured_secret_values(config)
    schema_hashes = {
        case.case_id: sha256_text(stable_json_dumps(case.contract.json_schema, indent=None))
        for case in cases
    }
    signature = {
        "version": BENCHMARK_VERSION,
        "case_ids": [case.case_id for case in cases],
        "schema_hashes": schema_hashes,
        "seeds": exact_seeds,
        "repetitions_per_seed": int(repetitions_per_seed),
        "model_identity": identity,
        "structured_output_mode": config.model.structured_output_mode,
    }
    results: list[ConstraintDecodingResult] = []
    if resume and output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("signature") != signature:
            raise ValueError("Constraint-decoding checkpoint does not match the current benchmark")
        rows = previous.get("results", [])
        if not isinstance(rows, list):
            raise ValueError("Constraint-decoding checkpoint results must be an array")
        results = [ConstraintDecodingResult(**row) for row in rows]
    completed = {(r.case_id, r.seed, r.repetition) for r in results}

    def report_payload() -> dict[str, Any]:
        planned = len(cases) * len(exact_seeds) * repetitions_per_seed
        structural = sum(r.structurally_valid for r in results)
        semantic = sum(r.semantic_passed for r in results)
        constraint_sent = sum(r.constraint_present_in_request for r in results)
        transport_failures = sum(r.error_type is not None for r in results)
        parse_failures = sum(not r.json_parsed and r.error_type is None for r in results)
        schema_failures = sum(r.json_parsed and not r.schema_valid for r in results)
        return {
            "benchmark": "constraint_decoding",
            "signature": signature,
            "planned_calls": planned,
            "completed_calls": len(results),
            "complete": len(results) == planned,
            "structurally_valid": structural,
            "structural_valid_percent": round(100.0 * structural / max(1, len(results)), 3),
            "semantic_passed": semantic,
            "semantic_pass_percent": round(100.0 * semantic / max(1, len(results)), 3),
            "constraint_present_in_request": constraint_sent,
            "transport_or_grammar_failures": transport_failures,
            "json_parse_failures": parse_failures,
            "schema_validation_failures": schema_failures,
            "results": [asdict(result) for result in results],
            "measurement_scope": (
                "Structural validity measures generation-time constrained-output robustness. "
                "Semantic correctness is reported separately and must not be inferred from schema validity."
            ),
        }

    for case in cases:
        schema = case.contract.json_schema or {}
        expected_schema_hash = schema_hashes[case.case_id]
        for seed in exact_seeds:
            for repetition in range(1, repetitions_per_seed + 1):
                key = (case.case_id, seed, repetition)
                if key in completed:
                    continue
                config.model.seed = seed
                # LlamaCppClient keeps the config object by reference. Test clients may expose it too.
                client_config = getattr(client, "config", None)
                if client_config is not None and getattr(client_config, "model", None) is not None:
                    client_config.model.seed = seed
                parsed = False
                schema_valid = False
                semantic_passed = False
                request_schema_hash: str | None = None
                constraint_present = False
                completion = None
                error_type: str | None = None
                error_reason: str | None = None
                try:
                    completion = client.complete(
                        case.prompt,
                        max_tokens=256,
                        contract=case.contract,
                        temperature=0.0,
                        kind="benchmark_quality_judge",
                        live_mode=True,
                    )
                    request_schema = _request_schema(dict(completion.raw_request))
                    if request_schema is not None:
                        request_schema_hash = sha256_text(
                            stable_json_dumps(request_schema, indent=None)
                        )
                        constraint_present = request_schema_hash == expected_schema_hash
                    payload = json.loads(completion.text)
                    parsed = isinstance(payload, dict)
                    if parsed:
                        _validate_schema_value(payload, schema, path=case.contract.name)
                        schema_valid = True
                        semantic_passed = bool(case.semantic_check(payload))
                except (json.JSONDecodeError, ToolValidationError) as exc:
                    error_reason = str(exc)
                except Exception as exc:
                    error_type = type(exc).__name__
                    error_reason = str(exc)
                structurally_valid = bool(parsed and schema_valid and constraint_present and error_type is None)
                results.append(
                    ConstraintDecodingResult(
                        case_id=case.case_id,
                        seed=seed,
                        repetition=repetition,
                        contract_name=case.contract.name,
                        schema_sha256=expected_schema_hash,
                        request_schema_sha256=request_schema_hash,
                        constraint_present_in_request=constraint_present,
                        json_parsed=parsed,
                        schema_valid=schema_valid,
                        structurally_valid=structurally_valid,
                        semantic_passed=semantic_passed,
                        finish_reason=getattr(completion, "finish_reason", None),
                        prompt_tokens=getattr(completion, "prompt_tokens", None),
                        completion_tokens=getattr(completion, "completion_tokens", None),
                        elapsed_seconds=getattr(completion, "elapsed_seconds", None),
                        tokens_per_second=getattr(completion, "tokens_per_second", None),
                        text_sha256=(
                            sha256_text(str(completion.text)) if completion is not None else None
                        ),
                        error_type=error_type,
                        error_reason=error_reason,
                    )
                )
                completed.add(key)
                _atomic_checkpoint(output_path, report_payload(), secret_values=secret_values)
    final = report_payload()
    _atomic_checkpoint(output_path, final, secret_values=secret_values)
    return final
