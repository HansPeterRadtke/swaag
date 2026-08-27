from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Iterable

from swaag.budgeting import compute_call_budget
from swaag.config import AgentConfig, load_config
from swaag.model import LlamaCppClient
from swaag.types import ContractSpec
from swaag.utils import sha256_text, stable_json_dumps

POSITIONS = ("early", "middle", "late")
DEFAULT_UTILIZATIONS = (0.10, 0.25, 0.50, 0.75, 0.90)
BENCHMARK_VERSION = "context_order_retrieval_v3_dynamic_output"
SYSTEM_INSTRUCTION = (
    "Follow the user's retrieval request using only the supplied record. "
    "Return the answer through the required JSON schema."
)


@dataclass(slots=True, frozen=True)
class ContextOrderCase:
    position: str
    requested_utilization: float
    expected_code: str
    prompt: str
    marker_char_fraction: float


@dataclass(slots=True)
class ContextOrderResult:
    position: str
    requested_utilization: float
    expected_code: str
    answer: str
    passed: bool
    preflight_prompt_tokens: int
    prompt_tokens: int | None
    completion_tokens: int | None
    context_limit: int
    actual_input_utilization: float | None
    marker_char_fraction: float
    marker_token_fraction: float
    elapsed_seconds: float | None
    first_token_seconds: float | None
    finish_reason: str | None
    reserved_output_tokens: int
    serialized_prompt_sha256: str | None = None
    prompt_protocol_sha256: str | None = None


def answer_contract() -> ContractSpec:
    return ContractSpec(
        name="context_order_retrieval",
        mode="json_schema",
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )


def chat_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt},
    ]


def _filler_block(index: int) -> str:
    return (
        f"Reference note {index:05d}: The observatory stores ordinary calibration records for instruments, "
        "weather, maintenance, and logistics. This note is intentionally irrelevant to the retrieval question. "
        "Do not infer an answer from this note.\n"
    )


def _render_case(*, position: str, requested_utilization: float, filler: str, seed: int) -> ContextOrderCase:
    if position not in POSITIONS:
        raise ValueError(f"position must be one of {POSITIONS}")
    expected = f"SWAAG-{seed:04d}-ORBIT"
    marker = (
        "CRITICAL RETRIEVAL RECORD. The exact retrieval code for this experiment is "
        f"{expected}. When asked for the retrieval code, return this exact code and no substitute.\n"
    )
    query = (
        "\nRETRIEVAL QUESTION: What is the exact retrieval code stated in the critical retrieval record? "
        "Return it in the constrained answer field.\n"
    )
    if position == "early":
        insert_at = max(0, int(len(filler) * 0.05))
    elif position == "middle":
        insert_at = len(filler) // 2
    else:
        insert_at = int(len(filler) * 0.95)
    prompt = filler[:insert_at] + marker + filler[insert_at:] + query
    marker_start = prompt.index(marker)
    return ContextOrderCase(
        position=position, requested_utilization=requested_utilization, expected_code=expected, prompt=prompt,
        marker_char_fraction=marker_start / max(1, len(prompt)),
    )


def _filler_for_target(
    *,
    requested_utilization: float,
    context_limit: int,
    seed: int,
    token_counter: Callable[[str], int] | None,
) -> str:
    target_chars = max(2400, int(context_limit * requested_utilization * 3.2))
    initial_blocks = max(1, target_chars // max(1, len(_filler_block(0))))
    if token_counter is None:
        count = initial_blocks
        filler = "".join(_filler_block(index) for index in range(count))
        while len(filler) < target_chars:
            filler += _filler_block(count)
            count += 1
        return filler

    target_tokens = max(1, int(round(context_limit * requested_utilization)))

    def measured(block_count: int) -> tuple[int, str]:
        filler = "".join(_filler_block(index) for index in range(block_count))
        case = _render_case(
            position="middle",
            requested_utilization=requested_utilization,
            filler=filler,
            seed=seed,
        )
        return int(token_counter(case.prompt)), filler

    low = 0
    high = initial_blocks
    high_tokens, _ = measured(high)
    while high_tokens < target_tokens:
        low = high
        high *= 2
        high_tokens, _ = measured(high)
    while high - low > 1:
        middle = (low + high) // 2
        middle_tokens, _ = measured(middle)
        if middle_tokens < target_tokens:
            low = middle
        else:
            high = middle
    candidates = [measured(low), measured(high)]
    _tokens, filler = min(candidates, key=lambda item: abs(item[0] - target_tokens))
    return filler


def build_case(
    *,
    position: str,
    requested_utilization: float,
    context_limit: int,
    seed: int = 17,
    token_counter: Callable[[str], int] | None = None,
) -> ContextOrderCase:
    if not 0.02 <= requested_utilization <= 0.95:
        raise ValueError("requested_utilization must be between 0.02 and 0.95")
    filler = _filler_for_target(
        requested_utilization=requested_utilization,
        context_limit=context_limit,
        seed=seed,
        token_counter=token_counter,
    )
    return _render_case(
        position=position,
        requested_utilization=requested_utilization,
        filler=filler,
        seed=seed,
    )


def build_matrix(
    *,
    context_limit: int,
    utilizations: Iterable[float] = DEFAULT_UTILIZATIONS,
    seed: int = 17,
    token_counter: Callable[[str], int] | None = None,
) -> list[ContextOrderCase]:
    cases: list[ContextOrderCase] = []
    for raw_utilization in utilizations:
        utilization = float(raw_utilization)
        if not 0.02 <= utilization <= 0.95:
            raise ValueError("requested_utilization must be between 0.02 and 0.95")
        filler = _filler_for_target(
            requested_utilization=utilization,
            context_limit=context_limit,
            seed=seed,
            token_counter=token_counter,
        )
        cases.extend(
            _render_case(
                position=position,
                requested_utilization=utilization,
                filler=filler,
                seed=seed,
            )
            for position in POSITIONS
        )
    return cases


def _stable_model_identity(identity: object) -> object:
    if not isinstance(identity, dict):
        return identity
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
    if not any(key in identity for key in stable_keys):
        return identity
    return {key: identity.get(key) for key in stable_keys}


def _verification_output_reserve(
    config: AgentConfig,
    *,
    context_limit: int,
    input_tokens: int,
) -> int:
    minimum = max(1, int(config.context.reserved_response_tokens))
    safety = max(0, int(config.context.safety_margin_tokens))
    available = max(0, int(context_limit) - int(input_tokens) - safety)
    if available < minimum:
        raise ValueError(
            "Benchmark case does not fit its minimum output requirement: "
            f"input={input_tokens} minimum_output={minimum} safety={safety} "
            f"limit={context_limit}"
        )
    desired = compute_call_budget(
        config,
        call_kind="benchmark_quality_judge",
        context_limit=context_limit,
    ).output_tokens
    return min(max(minimum, int(desired)), available)


def run_context_order_benchmark(
    *, config: AgentConfig | None = None, utilizations: Iterable[float] = DEFAULT_UTILIZATIONS, seed: int = 17,
    output_path: Path | None = None, resume: bool = True,
) -> dict:
    config = config or load_config()
    client = LlamaCppClient(config)
    identity = client.cache_identity()
    context_limit, context_limit_source = client.context_limit_resolution()
    utilization_values = tuple(float(value) for value in utilizations)
    cases = build_matrix(
        context_limit=context_limit,
        utilizations=utilization_values,
        seed=seed,
        token_counter=lambda prompt: client.tokenize(
            client.render_chat_prompt(chat_messages(prompt))["prompt"]
        ),
    )
    results: list[ContextOrderResult] = []
    model_identity_history: list[object] = [identity]
    contract = answer_contract()

    if resume and output_path is not None and output_path.exists():
        try:
            previous = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot resume invalid context-order checkpoint {output_path}: {exc}"
            ) from exc
        expected_header = {
            "benchmark": BENCHMARK_VERSION,
            "context_limit": context_limit,
            "context_limit_source": context_limit_source,
            "seed": seed,
            "requested_utilizations": list(utilization_values),
            "planned": len(cases),
        }
        previous_identity = previous.get("model_identity")
        if _stable_model_identity(previous_identity) != _stable_model_identity(identity):
            expected_header["model_identity"] = identity
        mismatches = {
            key: {"checkpoint": previous.get(key), "current": value}
            for key, value in expected_header.items()
            if previous.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Context-order checkpoint does not match the current benchmark: "
                + stable_json_dumps(mismatches, indent=None)
            )
        previous_history = previous.get("model_identity_history", [previous_identity])
        if not isinstance(previous_history, list):
            raise ValueError("Context-order checkpoint model_identity_history must be an array")
        model_identity_history = list(previous_history)
        if identity not in model_identity_history:
            model_identity_history.append(identity)
        raw_results = previous.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("Context-order checkpoint results must be an array")
        try:
            results = [ContextOrderResult(**row) for row in raw_results]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Context-order checkpoint has invalid result rows: {exc}"
            ) from exc

    completed_keys = {
        (result.position, result.requested_utilization, result.expected_code)
        for result in results
    }
    if len(completed_keys) != len(results):
        raise ValueError("Context-order checkpoint contains duplicate completed cases")

    def build_report(*, complete: bool) -> dict:
        rows = [asdict(result) for result in results]
        by_position = {
            position: {
                "passed": sum(1 for row in rows if row["position"] == position and row["passed"]),
                "completed": sum(1 for row in rows if row["position"] == position),
                "planned": sum(1 for case in cases if case.position == position),
            }
            for position in POSITIONS
        }
        return {
            "benchmark": BENCHMARK_VERSION,
            "model_identity": identity,
            "model_identity_history": model_identity_history,
            "context_limit": context_limit,
            "context_limit_source": context_limit_source,
            "seed": seed,
            "requested_utilizations": list(utilization_values),
            "results": rows,
            "by_position": by_position,
            "passed": sum(1 for row in rows if row["passed"]),
            "completed": len(rows),
            "planned": len(cases),
            "complete": complete,
            # Retain the original final-report field for existing consumers.
            "total": len(rows),
        }

    def checkpoint(report: dict) -> None:
        if output_path is None:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(output_path.name + ".tmp")
        temporary_path.write_text(
            stable_json_dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)

    for case in cases:
        case_key = (case.position, case.requested_utilization, case.expected_code)
        if case_key in completed_keys:
            continue
        messages = chat_messages(case.prompt)
        rendering = client.render_chat_prompt(messages)
        serialized_prompt = rendering["prompt"]
        prompt_protocol_sha256 = rendering["prompt_protocol_sha256"]
        client.verify_prompt_protocol(prompt_protocol_sha256)
        preflight_prompt_tokens = client.tokenize(serialized_prompt)
        output_reserve = _verification_output_reserve(
            config,
            context_limit=context_limit,
            input_tokens=preflight_prompt_tokens,
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
        answer = str(payload.get("answer", "")).strip()
        prompt_tokens = completion.prompt_tokens
        actual_prompt_tokens = prompt_tokens if isinstance(prompt_tokens, int) else preflight_prompt_tokens
        actual = actual_prompt_tokens / context_limit
        marker_start = serialized_prompt.index("CRITICAL RETRIEVAL RECORD.")
        marker_token_fraction = client.tokenize(serialized_prompt[:marker_start]) / max(
            1, preflight_prompt_tokens
        )
        results.append(ContextOrderResult(
            position=case.position, requested_utilization=case.requested_utilization, expected_code=case.expected_code,
            answer=answer,
            passed=answer == case.expected_code and completion.finish_reason not in {"length", "context_overflow"},
            preflight_prompt_tokens=preflight_prompt_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion.completion_tokens, context_limit=context_limit,
            actual_input_utilization=actual, marker_char_fraction=case.marker_char_fraction,
            marker_token_fraction=marker_token_fraction,
            elapsed_seconds=completion.elapsed_seconds, first_token_seconds=completion.first_token_seconds,
            finish_reason=completion.finish_reason,
            reserved_output_tokens=output_reserve,
            serialized_prompt_sha256=sha256_text(serialized_prompt),
            prompt_protocol_sha256=prompt_protocol_sha256,
        ))
        checkpoint(build_report(complete=len(results) == len(cases)))
    report = build_report(complete=True)
    if not cases:
        checkpoint(report)
    return report
