from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

from swaag.config import AgentConfig, load_config
from swaag.model import LlamaCppClient
from swaag.types import ContractSpec
from swaag.utils import stable_json_dumps

POSITIONS = ("early", "middle", "late")
DEFAULT_UTILIZATIONS = (0.10, 0.25, 0.50)


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
    prompt_tokens: int | None
    completion_tokens: int | None
    context_limit: int
    actual_input_utilization: float | None
    marker_char_fraction: float
    elapsed_seconds: float | None
    first_token_seconds: float | None


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


def _filler_block(index: int) -> str:
    return (
        f"Reference note {index:05d}: The observatory stores ordinary calibration records for instruments, "
        "weather, maintenance, and logistics. This note is intentionally irrelevant to the retrieval question. "
        "Do not infer an answer from this note.\n"
    )


def build_case(*, position: str, requested_utilization: float, context_limit: int, seed: int = 17) -> ContextOrderCase:
    if position not in POSITIONS:
        raise ValueError(f"position must be one of {POSITIONS}")
    if not 0.02 <= requested_utilization <= 0.95:
        raise ValueError("requested_utilization must be between 0.02 and 0.95")
    # Conservative character target; live results record actual tokenizer counts.
    target_chars = max(2400, int(context_limit * requested_utilization * 3.2))
    expected = f"SWAAG-{seed:04d}-ORBIT"
    marker = (
        "CRITICAL RETRIEVAL RECORD. The exact retrieval code for this experiment is "
        f"{expected}. When asked for the retrieval code, return this exact code and no substitute.\n"
    )
    query = (
        "\nRETRIEVAL QUESTION: What is the exact retrieval code stated in the critical retrieval record? "
        "Return it in the constrained answer field.\n"
    )
    blocks: list[str] = []
    size = 0
    i = 0
    while size < target_chars:
        block = _filler_block(i)
        blocks.append(block)
        size += len(block)
        i += 1
    filler = "".join(blocks)
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


def build_matrix(*, context_limit: int, utilizations: Iterable[float] = DEFAULT_UTILIZATIONS, seed: int = 17) -> list[ContextOrderCase]:
    return [
        build_case(position=position, requested_utilization=float(util), context_limit=context_limit, seed=seed)
        for util in utilizations
        for position in POSITIONS
    ]


def run_context_order_benchmark(
    *, config: AgentConfig | None = None, utilizations: Iterable[float] = DEFAULT_UTILIZATIONS, seed: int = 17,
    output_path: Path | None = None,
) -> dict:
    config = config or load_config()
    client = LlamaCppClient(config)
    identity = client.cache_identity()
    cases = build_matrix(context_limit=config.model.context_limit, utilizations=utilizations, seed=seed)
    results: list[ContextOrderResult] = []
    contract = answer_contract()
    for case in cases:
        completion = client.complete(
            case.prompt, max_tokens=96, contract=contract, temperature=0.0, kind="verification", live_mode=True
        )
        try:
            payload = json.loads(completion.text)
        except json.JSONDecodeError:
            payload = {}
        answer = str(payload.get("answer", "")).strip()
        prompt_tokens = completion.prompt_tokens
        actual = (prompt_tokens / config.model.context_limit) if isinstance(prompt_tokens, int) else None
        results.append(ContextOrderResult(
            position=case.position, requested_utilization=case.requested_utilization, expected_code=case.expected_code,
            answer=answer, passed=answer == case.expected_code, prompt_tokens=prompt_tokens,
            completion_tokens=completion.completion_tokens, context_limit=config.model.context_limit,
            actual_input_utilization=actual, marker_char_fraction=case.marker_char_fraction,
            elapsed_seconds=completion.elapsed_seconds, first_token_seconds=completion.first_token_seconds,
        ))
    rows = [asdict(result) for result in results]
    by_position = {
        position: {
            "passed": sum(1 for row in rows if row["position"] == position and row["passed"]),
            "total": sum(1 for row in rows if row["position"] == position),
        }
        for position in POSITIONS
    }
    report = {
        "benchmark": "context_order_retrieval",
        "model_identity": identity,
        "context_limit": config.model.context_limit,
        "seed": seed,
        "requested_utilizations": [float(value) for value in utilizations],
        "results": rows,
        "by_position": by_position,
        "passed": sum(1 for row in rows if row["passed"]),
        "total": len(rows),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(stable_json_dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
