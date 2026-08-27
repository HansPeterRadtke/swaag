from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Iterable

from swaag.benchmark.context_order import (
    _stable_model_identity,
    _verification_output_reserve,
)
from swaag.config import AgentConfig, load_config
from swaag.model import LlamaCppClient
from swaag.types import ContractSpec
from swaag.utils import sha256_text, stable_json_dumps


BENCHMARK_VERSION = "context_semantic_layout_v2_dynamic_output"
SYSTEM_FIELD = "system_instruction"
USER_FIELDS = (
    "task_instruction",
    "current_request",
    "conversation_history",
    "tool_definitions",
    "tool_results",
    "retrieved_evidence",
)
ALL_FIELDS = (SYSTEM_FIELD, *USER_FIELDS)
DEFAULT_UTILIZATIONS = (0.50,)


@dataclass(slots=True, frozen=True)
class ContextLayoutCase:
    variant: str
    requested_utilization: float
    user_section_order: tuple[str, ...]
    system_instruction: str
    user_prompt: str
    expected: dict[str, str]


@dataclass(slots=True)
class ContextLayoutResult:
    variant: str
    requested_utilization: float
    user_section_order: list[str]
    answer: dict[str, str]
    field_checks: dict[str, bool]
    passed: bool
    preflight_prompt_tokens: int
    prompt_tokens: int | None
    completion_tokens: int | None
    context_limit: int
    actual_input_utilization: float
    section_token_fractions: dict[str, float]
    elapsed_seconds: float | None
    first_token_seconds: float | None
    finish_reason: str | None
    reserved_output_tokens: int
    serialized_prompt_sha256: str
    prompt_protocol_sha256: str


def answer_contract() -> ContractSpec:
    properties = {name: {"type": "string"} for name in ALL_FIELDS}
    return ContractSpec(
        name="context_semantic_layout",
        mode="json_schema",
        json_schema={
            "type": "object",
            "properties": properties,
            "required": list(ALL_FIELDS),
            "additionalProperties": False,
        },
    )


def _codes(seed: int) -> dict[str, str]:
    return {
        name: f"SWAAG-LAYOUT-{seed:04d}-{index:02d}"
        for index, name in enumerate(ALL_FIELDS, start=1)
    }


def _filler(index: int) -> str:
    return (
        f"Ordinary context record {index:05d}: this calibration, maintenance, and logistics note "
        "is irrelevant to every requested retrieval field and contains no answer code.\n"
    )


def _section(name: str, code: str, filler: str) -> str:
    labels = {
        "task_instruction": "TASK INSTRUCTION",
        "current_request": "CURRENT USER REQUEST",
        "conversation_history": "PRIOR CONVERSATION HISTORY",
        "tool_definitions": "AVAILABLE TOOL DEFINITIONS",
        "tool_results": "AUTHORITATIVE TOOL RESULTS",
        "retrieved_evidence": "RETRIEVED EXTERNAL EVIDENCE",
    }
    return (
        f"\n## {labels[name]}\n"
        f"The exact code for {name} is {code}. Preserve it verbatim.\n"
        f"{filler}"
    )


def _render_case(
    *,
    order: tuple[str, ...],
    requested_utilization: float,
    seed: int,
    filler_blocks: int,
) -> ContextLayoutCase:
    codes = _codes(seed)
    fillers = {name: [] for name in USER_FIELDS}
    for index in range(filler_blocks):
        fillers[USER_FIELDS[index % len(USER_FIELDS)]].append(_filler(index))
    sections = {
        name: _section(name, codes[name], "".join(fillers[name]))
        for name in USER_FIELDS
    }
    query = (
        "\n## RESPONSE REQUEST\n"
        "Return the exact code from every named semantic section in the matching JSON field. "
        "Use only the supplied records; do not infer or substitute a code.\n"
    )
    system_instruction = (
        "Follow the user's retrieval request and the required JSON schema. "
        f"The exact code for {SYSTEM_FIELD} is {codes[SYSTEM_FIELD]}; preserve it verbatim."
    )
    return ContextLayoutCase(
        variant="-".join(order),
        requested_utilization=requested_utilization,
        user_section_order=order,
        system_instruction=system_instruction,
        user_prompt="".join(sections[name] for name in order) + query,
        expected=codes,
    )


def build_cases(
    *,
    context_limit: int,
    utilizations: Iterable[float] = DEFAULT_UTILIZATIONS,
    seed: int = 29,
    token_counter: Callable[[list[dict[str, str]]], int] | None = None,
) -> list[ContextLayoutCase]:
    cases: list[ContextLayoutCase] = []
    for raw_utilization in utilizations:
        utilization = float(raw_utilization)
        if not 0.05 <= utilization <= 0.90:
            raise ValueError("requested utilization must be between 0.05 and 0.90")

        def count(blocks: int) -> int:
            candidate = _render_case(
                order=USER_FIELDS,
                requested_utilization=utilization,
                seed=seed,
                filler_blocks=blocks,
            )
            messages = [
                {"role": "system", "content": candidate.system_instruction},
                {"role": "user", "content": candidate.user_prompt},
            ]
            if token_counter is not None:
                return int(token_counter(messages))
            return max(1, sum(len(message["content"]) for message in messages) // 3)

        target = max(1, int(round(context_limit * utilization)))
        low, high = 0, 1
        while count(high) < target:
            low, high = high, high * 2
        while high - low > 1:
            middle = (low + high) // 2
            if count(middle) < target:
                low = middle
            else:
                high = middle
        filler_blocks = min((low, high), key=lambda value: abs(count(value) - target))
        for rotation in range(len(USER_FIELDS)):
            order = USER_FIELDS[rotation:] + USER_FIELDS[:rotation]
            cases.append(
                _render_case(
                    order=order,
                    requested_utilization=utilization,
                    seed=seed,
                    filler_blocks=filler_blocks,
                )
            )
    return cases


def run_context_layout_benchmark(
    *,
    config: AgentConfig | None = None,
    utilizations: Iterable[float] = DEFAULT_UTILIZATIONS,
    seed: int = 29,
    output_path: Path | None = None,
    resume: bool = True,
    client_factory: Callable[[AgentConfig], object] = LlamaCppClient,
) -> dict:
    config = config or load_config()
    client = client_factory(config)
    identity = client.cache_identity()
    context_limit, context_limit_source = client.context_limit_resolution()
    utilization_values = tuple(float(value) for value in utilizations)

    def exact_count(messages: list[dict[str, str]]) -> int:
        rendering = client.render_chat_prompt(messages)
        return int(client.tokenize(rendering["prompt"]))

    cases = build_cases(
        context_limit=context_limit,
        utilizations=utilization_values,
        seed=seed,
        token_counter=exact_count,
    )
    results: list[ContextLayoutResult] = []
    expected_header = {
        "benchmark": BENCHMARK_VERSION,
        "context_limit": context_limit,
        "context_limit_source": context_limit_source,
        "seed": seed,
        "requested_utilizations": list(utilization_values),
        "planned": len(cases),
    }
    if resume and output_path is not None and output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        mismatches = {
            key: {"checkpoint": previous.get(key), "current": value}
            for key, value in expected_header.items()
            if previous.get(key) != value
        }
        if _stable_model_identity(previous.get("model_identity")) != _stable_model_identity(identity):
            mismatches["model_identity"] = {
                "checkpoint": previous.get("model_identity"),
                "current": identity,
            }
        if mismatches:
            raise ValueError(
                "Context-layout checkpoint does not match the current benchmark: "
                + stable_json_dumps(mismatches, indent=None)
            )
        raw_results = previous.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("Context-layout checkpoint results must be an array")
        results = [ContextLayoutResult(**row) for row in raw_results]

    completed = {
        (result.variant, result.requested_utilization) for result in results
    }
    if len(completed) != len(results):
        raise ValueError("Context-layout checkpoint contains duplicate cases")

    def report(complete: bool) -> dict:
        rows = [asdict(result) for result in results]
        by_field = {
            field: {
                "passed": sum(1 for row in rows if row["field_checks"].get(field)),
                "completed": len(rows),
                "planned": len(cases),
            }
            for field in ALL_FIELDS
        }
        by_user_position = {
            str(position): {
                "passed": sum(
                    1
                    for row in rows
                    if row["field_checks"].get(row["user_section_order"][position])
                ),
                "completed": len(rows),
                "planned": len(cases),
            }
            for position in range(len(USER_FIELDS))
        }
        return {
            **expected_header,
            "model_identity": identity,
            "results": rows,
            "by_field": by_field,
            "by_user_position": by_user_position,
            "passed": sum(1 for row in rows if row["passed"]),
            "completed": len(rows),
            "total": len(rows),
            "complete": complete,
        }

    def checkpoint(payload: dict) -> None:
        if output_path is None:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        temporary.write_text(stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output_path)

    contract = answer_contract()
    for case in cases:
        key = (case.variant, case.requested_utilization)
        if key in completed:
            continue
        messages = [
            {"role": "system", "content": case.system_instruction},
            {"role": "user", "content": case.user_prompt},
        ]
        rendering = client.render_chat_prompt(messages)
        serialized_prompt = rendering["prompt"]
        protocol_hash = rendering["prompt_protocol_sha256"]
        client.verify_prompt_protocol(protocol_hash)
        preflight_tokens = int(client.tokenize(serialized_prompt))
        output_reserve = _verification_output_reserve(
            config,
            context_limit=context_limit,
            input_tokens=preflight_tokens,
        )
        completion = client.complete(
            serialized_prompt,
            max_tokens=output_reserve,
            contract=contract,
            temperature=0.0,
            kind="verification",
            live_mode=True,
            messages=messages,
        )
        try:
            payload = json.loads(completion.text)
        except json.JSONDecodeError:
            payload = {}
        answer = {
            field: str(payload.get(field, "")).strip()
            for field in ALL_FIELDS
        }
        checks = {
            field: answer[field] == case.expected[field]
            for field in ALL_FIELDS
        }
        token_fractions = {
            field: client.tokenize(serialized_prompt[: serialized_prompt.index(code)])
            / max(1, preflight_tokens)
            for field, code in case.expected.items()
        }
        actual_prompt_tokens = (
            completion.prompt_tokens
            if isinstance(completion.prompt_tokens, int)
            else preflight_tokens
        )
        results.append(
            ContextLayoutResult(
                variant=case.variant,
                requested_utilization=case.requested_utilization,
                user_section_order=list(case.user_section_order),
                answer=answer,
                field_checks=checks,
                passed=all(checks.values())
                and completion.finish_reason not in {"length", "context_overflow"},
                preflight_prompt_tokens=preflight_tokens,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                context_limit=context_limit,
                actual_input_utilization=actual_prompt_tokens / context_limit,
                section_token_fractions=token_fractions,
                elapsed_seconds=completion.elapsed_seconds,
                first_token_seconds=completion.first_token_seconds,
                finish_reason=completion.finish_reason,
                reserved_output_tokens=output_reserve,
                serialized_prompt_sha256=sha256_text(serialized_prompt),
                prompt_protocol_sha256=protocol_hash,
            )
        )
        checkpoint(report(len(results) == len(cases)))
    final = report(True)
    checkpoint(final)
    return final
