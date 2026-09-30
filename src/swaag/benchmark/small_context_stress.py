from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from swaag.action import AgentAction, AgentStatus
from swaag.config import AgentConfig, load_config
from swaag.context_compiler import ContextCompiler
from swaag.grammar import agent_action_contract, yes_no_contract
from swaag.model import LlamaCppClient
from swaag.runtime import AgentRuntime, BudgetExceededError
from swaag.tokens import ExactTokenCounter
from swaag.types import Message, PromptAssembly, PromptComponent
from swaag.utils import stable_json_dumps, utc_now_iso


CASE_IDS = (
    "server_context_identity",
    "near_full_boundary",
    "exact_full_boundary",
    "one_token_over_boundary",
    "million_token_user_request_rejected",
    "irreducible_output_schema_rejected",
    "oversized_history_recovery",
    "oversized_tool_result_recovery",
    "oversized_attachment_recovery",
)


class _NoInferenceClient:
    def __init__(self, delegate: LlamaCppClient):
        self.delegate = delegate
        self.send_calls = 0

    def send_completion(self, *args, **kwargs):
        self.send_calls += 1
        raise AssertionError("small-context mechanical case unexpectedly reached generation")

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)


def _config(base_url: str, root: Path, *, context_limit: int) -> AgentConfig:
    config = load_config()
    config.sessions.root = root / "sessions"
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config.tools.read_roots = [workspace]
    config.tools.enabled = []
    config.tools.staged_discovery = False
    config.model.base_url = base_url.rstrip("/")
    config.model.context_limit = int(context_limit)
    config.model.cache_enabled = False
    config.runtime.completion_evaluation_enabled = False
    config.context.semantic_reduction_max_input_tokens = min(
        max(512, int(context_limit * 0.6)),
        int(config.context.semantic_reduction_max_input_tokens or context_limit),
    )
    return config


def _boundary_compilation(
    config: AgentConfig,
    client: LlamaCppClient,
    *,
    repetitions: int,
    context_limit: int,
):
    filler = " x" * max(0, int(repetitions))
    assembly = PromptAssembly(
        kind="verification",
        prompt_text=filler,
        components=[
            PromptComponent(
                name="user_request",
                category="current_user",
                text=filler,
            )
        ],
        prompt_mode="lean",
    )
    compiler = ContextCompiler(config)
    counter = ExactTokenCounter(client.tokenize)
    return compiler.compile(
        assembly,
        yes_no_contract(),
        counter,
        minimum_output_tokens=1,
        desired_output_tokens=1,
        context_limit=context_limit,
        context_limit_source="server_props:n_ctx",
    )


def _find_boundary(
    config: AgentConfig,
    client: LlamaCppClient,
    *,
    target_required_tokens: int,
    context_limit: int,
) -> tuple[int, Any]:
    low = 0
    high = max(32, context_limit * 2)
    while _boundary_compilation(
        config, client, repetitions=high, context_limit=context_limit
    ).report.required_tokens < target_required_tokens:
        high *= 2
        if high > context_limit * 32:
            raise RuntimeError("could not bracket small-context boundary")
    while low <= high:
        middle = (low + high) // 2
        compilation = _boundary_compilation(
            config, client, repetitions=middle, context_limit=context_limit
        )
        value = compilation.report.required_tokens
        if value == target_required_tokens:
            return middle, compilation
        if value < target_required_tokens:
            low = middle + 1
        else:
            high = middle - 1
    # Search a tight neighborhood in case the tokenizer is locally non-linear.
    for repetitions in range(max(0, high - 64), low + 65):
        compilation = _boundary_compilation(
            config, client, repetitions=repetitions, context_limit=context_limit
        )
        if compilation.report.required_tokens == target_required_tokens:
            return repetitions, compilation
    raise RuntimeError(
        f"live tokenizer could not synthesize exact required-token boundary {target_required_tokens}"
    )


def _mechanical_boundary_case(
    config: AgentConfig,
    client: LlamaCppClient,
    *,
    target: int,
    expected_fits: bool,
) -> dict[str, Any]:
    repetitions, compilation = _find_boundary(
        config,
        client,
        target_required_tokens=target,
        context_limit=config.model.context_limit,
    )
    report = compilation.report
    if report.required_tokens != target or report.fits is not expected_fits:
        raise AssertionError(
            f"boundary mismatch target={target} required={report.required_tokens} fits={report.fits}"
        )
    return {
        "repetitions": repetitions,
        "accounting": compilation.accounting(),
    }


def _million_token_user_case(config: AgentConfig, client: LlamaCppClient) -> dict[str, Any]:
    # Calibrate one-token-ish live text, then materialize a real giant user string.
    sample_repetitions = 4096
    sample = " x" * sample_repetitions
    sample_tokens = client.tokenize(sample)
    if sample_tokens <= 0:
        raise AssertionError("live tokenizer returned no tokens for calibration text")
    desired = 1_000_000
    repetitions = max(1, int(desired * sample_repetitions / sample_tokens))
    text = " x" * repetitions
    exact_tokens = client.tokenize(text)
    # One correction is enough for the repeated-token calibration and keeps the
    # actual stress request close to one million exact live tokens.
    if exact_tokens < 900_000 or exact_tokens > 1_100_000:
        repetitions = max(1, int(repetitions * desired / max(1, exact_tokens)))
        text = " x" * repetitions
        exact_tokens = client.tokenize(text)
    if not 900_000 <= exact_tokens <= 1_100_000:
        raise AssertionError(f"giant user request token count not close to one million: {exact_tokens}")

    delegate = LlamaCppClient(config)
    no_inference = _NoInferenceClient(delegate)
    runtime = AgentRuntime(config, model_client=no_inference)
    state = runtime.create_or_load_session()
    try:
        runtime._prepare_action_call(
            state,
            original_request=text,
            pending_messages=[],
            tool_specs=[],
            capability_index=[],
            contract=agent_action_contract([]),
            validation_feedback="",
            minimum_output_tokens=64,
        )
    except BudgetExceededError as exc:
        if exc.report is None or exc.report.required_tokens <= exc.report.context_limit:
            raise AssertionError("million-token request failed without measured overflow") from exc
        if no_inference.send_calls != 0:
            raise AssertionError("million-token irreducible request reached generation")
        return {
            "exact_user_tokens": exact_tokens,
            "chars": len(text),
            "send_calls": no_inference.send_calls,
            "final_budget": asdict(exc.report),
        }
    raise AssertionError("million-token current user request unexpectedly fit the tiny context")


def _irreducible_schema_case(config: AgentConfig, client: LlamaCppClient) -> dict[str, Any]:
    # A giant structured-output contract can make a call impossible even with no
    # semantic input. Mechanical admission must reject it without generation.
    properties = {f"field_{i:04d}": {"type": "string"} for i in range(600)}
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    from swaag.types import ContractSpec

    contract = ContractSpec(name="tiny_context_irreducible_schema", mode="json_schema", json_schema=schema)
    assembly = PromptAssembly(
        kind="verification",
        prompt_text="tiny request",
        components=[PromptComponent(name="user_request", category="current_user", text="tiny request")],
        prompt_mode="lean",
    )
    compilation = ContextCompiler(config).compile(
        assembly,
        contract,
        ExactTokenCounter(client.tokenize),
        minimum_output_tokens=1,
        desired_output_tokens=1,
        context_limit=config.model.context_limit,
        context_limit_source="server_props:n_ctx",
    )
    if compilation.report.fits:
        raise AssertionError("irreducible output schema unexpectedly fit tiny context")
    return {"accounting": compilation.accounting(), "schema_fields": len(properties)}


def _history_case(config: AgentConfig) -> dict[str, Any]:
    marker = "SMALLCTX-HISTORY-AUTH-731"
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()
    runtime._record_message(
        state,
        Message(
            role="user",
            content=(
                f"Authoritative early fact {marker}. Keep this exact marker. "
                + "authoritative evidence " * 350
            ),
            created_at=utc_now_iso(),
        ),
    )
    for index in range(8):
        runtime._record_message(
            state,
            Message(
                role="assistant" if index % 2 else "user",
                content=(f"Routine unrelated progress block {index}. " * 220),
                created_at=utc_now_iso(),
            ),
        )
    current_request = f"Continue while preserving exact marker {marker}."
    runtime._record_message(
        state,
        Message(role="user", content=current_request, created_at=utc_now_iso()),
    )
    prepared = runtime._prepare_action_call(
        state,
        original_request=current_request,
        pending_messages=[],
        tool_specs=[],
        capability_index=[],
        contract=agent_action_contract([]),
        validation_feedback="",
        minimum_output_tokens=64,
    )
    if not prepared.report.fits:
        raise AssertionError("oversized history was not recovered to a fitting action prompt")
    events = runtime.history.read_history(state.session_id)
    compressed = [event for event in events if event.event_type in {"history_compressed", "history_reprojected"}]
    if not compressed:
        raise AssertionError("oversized history did not trigger semantic history reduction")
    authoritative = runtime.history.read_authoritative_messages(state.session_id)
    if not any(marker in message.content for message in authoritative):
        raise AssertionError("authoritative history marker was not recoverable")
    return {
        "final_required_tokens": prepared.report.required_tokens,
        "context_limit": prepared.report.context_limit,
        "reduction_events": [event.event_type for event in compressed],
        "marker_recoverable": True,
    }


def _tool_result_case(config: AgentConfig) -> dict[str, Any]:
    marker = "SMALLCTX-TOOL-RESULT-947"
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()
    source_text = ("irrelevant tool bulk " * 2500) + marker
    current_request = f"Use the exact tool evidence and preserve marker {marker}."
    runtime._record_message(
        state,
        Message(role="user", content=current_request, created_at=utc_now_iso()),
    )
    source = runtime.history.record_event(
        state,
        "tool_result",
        {
            "tool_name": "generic_reader",
            "raw_input": {},
            "validated_input": {},
            "output": {"text": source_text},
        },
    )
    runtime._record_message(
        state,
        Message(
            role="tool",
            name="generic_reader",
            content=source_text,
            created_at=utc_now_iso(),
            metadata={
                "output": {"text": source_text},
                "source_event_sequence": source.sequence,
                "source_event_hash": source.hash,
                "source_event_type": source.event_type,
                "source_event_session_id": state.session_id,
                "source_event_references": [],
            },
        ),
    )
    prepared = runtime._prepare_action_call(
        state,
        original_request=current_request,
        pending_messages=[],
        tool_specs=[],
        capability_index=[],
        contract=agent_action_contract([]),
        validation_feedback="",
        minimum_output_tokens=64,
    )
    events = runtime.history.read_history(state.session_id)
    projected = [event for event in events if event.event_type == "tool_result_projected"]
    if not projected:
        raise AssertionError("oversized tool result did not create a semantic projection")
    raw = next(
        (
            event
            for event in runtime.history.read_history(state.session_id)
            if event.sequence == source.sequence
        ),
        None,
    )
    if raw is None or marker not in stable_json_dumps(raw.payload, indent=None):
        raise AssertionError("raw tool-result marker lost from authoritative history")
    if not prepared.report.fits:
        raise AssertionError("projected tool result did not fit tiny context")
    return {
        "source_sequence": source.sequence,
        "source_hash": source.hash,
        "projection_sequences": [event.sequence for event in projected],
        "final_required_tokens": prepared.report.required_tokens,
        "marker_recoverable": True,
    }


def _candidate(message: str) -> AgentAction:
    return AgentAction(
        assistant_message=message,
        tool_calls=[],
        continue_loop=False,
        silent_completion=False,
        status=AgentStatus("Candidate ready.", "Verify exact attachment.", "Candidate claims completion.", "normal"),
        questions=[],
    )


def _attachment_case(config: AgentConfig) -> dict[str, Any]:
    marker = "SMALLCTX-ATTACHMENT-389"
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()
    source_text = ("large attachment bulk " * 6000) + marker
    raw_tokens = runtime._counter(state).count_text(source_text).tokens
    if raw_tokens <= config.model.context_limit:
        raise AssertionError("attachment stress source was not actually oversized")
    data = source_text.encode("utf-8")
    reference = runtime.add_attachment(
        data,
        original_name="tiny-context-huge-evidence.txt",
        media_type="text/plain",
        session_id=state.session_id,
    )
    state = runtime.create_or_load_session(state.session_id)
    objective = f"Complete only if exact attachment evidence proves marker {marker}."
    runtime._record_message(
        state,
        Message(role="user", content=objective, created_at=utc_now_iso()),
    )
    # The independent evaluator chooses whether it needs exact re-expansion. This
    # case is deliberately phrased so the attachment is the only completion proof.
    runtime.config.runtime.completion_evaluation_enabled = True
    result = runtime._evaluate_completion(
        state,
        original_request=objective,
        selected_action=_candidate("The attachment proves the required marker; task complete."),
        tool_results=[],
    )
    rows = result.get("reexpanded_evidence_sources", [])
    row = next((item for item in rows if item.get("source_id") == reference.attachment_id), None)
    if row is None or row.get("integrity_verified") is not True:
        raise AssertionError("huge attachment was not integrity-checked/re-expanded")
    if not (
        row.get("projected") is True
        or row.get("exact_search_excerpted") is True
    ):
        raise AssertionError(
            "oversized attachment did not use a bounded model-selected evidence view"
        )
    return {
        "attachment_id": reference.attachment_id,
        "sha256": reference.sha256,
        "bytes": reference.size_bytes,
        "raw_tokens": raw_tokens,
        "complete": result.get("complete"),
        "source": row,
    }


def run_small_context_stress_benchmark(
    *,
    output_dir: Path,
    base_url: str,
    expected_context_limit: int = 2048,
    case_ids: list[str] | None = None,
    clean: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = list(case_ids or CASE_IDS)
    unknown = sorted(set(requested) - set(CASE_IDS))
    if unknown:
        raise ValueError("Unknown small-context stress case: " + ", ".join(unknown))
    base_config = _config(base_url, output_dir / "identity", context_limit=expected_context_limit)
    identity_client = LlamaCppClient(base_config)
    actual_limit, limit_source = identity_client.context_limit_resolution()
    if actual_limit != int(expected_context_limit):
        raise RuntimeError(
            f"small-context endpoint advertises n_ctx={actual_limit}, expected exactly {expected_context_limit}"
        )
    identity = identity_client.cache_identity()
    report_path = output_dir / "small_context_stress_results.json"
    results: list[dict[str, Any]] = []

    for index, case_id in enumerate(requested, 1):
        case_root = output_dir / "runs" / f"{index:02d}-{case_id}"
        config = _config(base_url, case_root, context_limit=actual_limit)
        client = LlamaCppClient(config)
        started = time.monotonic()
        error = ""
        evidence: dict[str, Any] = {}
        try:
            if case_id == "server_context_identity":
                evidence = {"context_limit": actual_limit, "context_limit_source": limit_source, "identity": identity}
            elif case_id == "near_full_boundary":
                evidence = _mechanical_boundary_case(config, client, target=actual_limit - 1, expected_fits=True)
            elif case_id == "exact_full_boundary":
                evidence = _mechanical_boundary_case(config, client, target=actual_limit, expected_fits=True)
            elif case_id == "one_token_over_boundary":
                evidence = _mechanical_boundary_case(config, client, target=actual_limit + 1, expected_fits=False)
            elif case_id == "million_token_user_request_rejected":
                evidence = _million_token_user_case(config, client)
            elif case_id == "irreducible_output_schema_rejected":
                evidence = _irreducible_schema_case(config, client)
            elif case_id == "oversized_history_recovery":
                evidence = _history_case(config)
            elif case_id == "oversized_tool_result_recovery":
                evidence = _tool_result_case(config)
            elif case_id == "oversized_attachment_recovery":
                evidence = _attachment_case(config)
            else:
                raise AssertionError(case_id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "case_id": case_id,
                "passed": not error,
                "elapsed_seconds": time.monotonic() - started,
                "error": error,
                "evidence": evidence,
            }
        )
        report = {
            "benchmark": "small-context-stress",
            "generated_at": utc_now_iso(),
            "base_url": base_url.rstrip("/"),
            "expected_context_limit": int(expected_context_limit),
            "actual_context_limit": actual_limit,
            "context_limit_source": limit_source,
            "model_identity": identity,
            "planned_cases": requested,
            "complete": len(results) == len(requested),
            "passed": sum(bool(row["passed"]) for row in results),
            "total": len(results),
            "results": results,
        }
        report_path.write_text(stable_json_dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
