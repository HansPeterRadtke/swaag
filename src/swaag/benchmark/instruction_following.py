from __future__ import annotations

import copy
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from swaag.config import AgentConfig, load_config
from swaag.prompt_instruction_store import PromptInstructionStore
from swaag.runtime import AgentRuntime
from swaag.types import PromptInstruction, SessionState
from swaag.utils import stable_json_dumps


@dataclass(slots=True, frozen=True)
class InstructionFollowingCase:
    case_id: str
    split: str
    prompt: str
    setup: str


CASES = (
    InstructionFollowingCase(
        case_id="simultaneous_format_constraints",
        split="baseline",
        setup="format_policy",
        prompt="Provide the standing status-format exercise now.",
    ),
    InstructionFollowingCase(
        case_id="conditional_tool_constraint",
        split="baseline",
        setup="calculation_policy",
        prompt="Multiply the operational counts 17 and 19 and report as instructed.",
    ),
    InstructionFollowingCase(
        case_id="user_report_relevance_constraints",
        split="held_out",
        setup="report_policy",
        prompt=(
            "Prepare the user report. Tests passed, deployment is blocked by authentication, "
            "commit 0123456789abcdef exists, the workspace is /srv/internal/project, and the "
            "operator ran sudo systemctl restart example.service."
        ),
    ),
    InstructionFollowingCase(
        case_id="dense_standing_policy",
        split="held_out",
        setup="dense_policy",
        prompt="Return the standing handoff object now.",
    ),
    InstructionFollowingCase(
        case_id="revised_instruction_applies",
        split="held_out",
        setup="revised_policy",
        prompt="Return the currently operative route marker and nothing else.",
    ),
)


def select_cases(case_ids: Iterable[str] = ()) -> list[InstructionFollowingCase]:
    requested = list(case_ids)
    by_id = {case.case_id: case for case in CASES}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(
            "Unknown instruction-following case: " + ", ".join(unknown)
        )
    return [by_id[case_id] for case_id in requested] if requested else list(CASES)


def _case_config(base: AgentConfig, *, sessions_root: Path) -> AgentConfig:
    config = copy.deepcopy(base)
    config.sessions.root = sessions_root
    config.model.cache_enabled = False
    config.tools.enabled = ["calculator"]
    config.tools.allow_stateful_tools = False
    config.tools.allow_side_effect_tools = False
    config.tools.staged_discovery = False
    config.runtime.completion_evaluation_enabled = False
    config.runtime.tool_call_budget = max(4, config.runtime.tool_call_budget)
    config.runtime.max_total_actions = max(8, config.runtime.max_total_actions)
    return config


def _model_identity(runtime: AgentRuntime) -> Any:
    provider = getattr(runtime.client, "cache_identity", None)
    return provider() if callable(provider) else type(runtime.client).__name__


def _write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        stable_json_dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _add_user_instruction(
    store: PromptInstructionStore,
    state: SessionState,
    *,
    title: str,
    content: str,
) -> PromptInstruction:
    mutation = store.add(
        title=title,
        content=content,
        scopes=["action"],
        origin_session_id=state.session_id,
    )
    assert mutation.instruction is not None
    return mutation.instruction


def _seed_case(
    case: InstructionFollowingCase,
    store: PromptInstructionStore,
    state: SessionState,
) -> list[str]:
    if case.setup == "format_policy":
        instruction = _add_user_instruction(
            store,
            state,
            title="Status exercise format",
            content=(
                "For the standing status-format exercise, the complete final user answer must "
                "contain exactly these four non-empty lines and no other text: first "
                "`OPERATIONS UPDATE`, then `status: ready`, then `owner: field-team`, then "
                "`next: verify`. Do not add Markdown fences, bullets, prefixes, or suffixes."
            ),
        )
        return [instruction.instruction_id]
    if case.setup == "calculation_policy":
        instruction = _add_user_instruction(
            store,
            state,
            title="Operational count reporting",
            content=(
                "When asked to multiply operational counts, call the calculator capability "
                "before answering. If the exact result exceeds 300, the complete final answer "
                "must be `review-required`; otherwise it must be `clear`. Do not expose the "
                "numeric result or add explanation."
            ),
        )
        return [instruction.instruction_id]
    if case.setup == "report_policy":
        instruction = _add_user_instruction(
            store,
            state,
            title="Meaningful user reports",
            content=(
                "For the user-report exercise, write exactly two sentences. State that tests "
                "passed and that deployment remains blocked by authentication. Omit commit "
                "hashes, filesystem paths, command transcripts, service names, and incidental "
                "operator mechanics. Do not use a list or heading."
            ),
        )
        return [instruction.instruction_id]
    if case.setup == "dense_policy":
        instruction = _add_user_instruction(
            store,
            state,
            title="Structured handoff policy",
            content=(
                "For the standing handoff-object exercise, return one single-line JSON object "
                "without Markdown or surrounding prose. It must have exactly five keys in this "
                "order: `alpha`, `beta`, `gamma`, `ready`, `count`. Use lower-case key names. "
                "Set alpha to the exact string `A17`; set beta to the exact string `B29`; set "
                "gamma to an array containing exactly `north` followed by `east`; set ready to "
                "the JSON boolean true; and set count to the JSON number 3. Do not stringify "
                "the boolean or number. Do not add nulls, comments, metadata, explanations, "
                "timestamps, confidence, or extra keys. Preserve the array order. The entire "
                "response must parse as JSON as supplied, without extracting a fenced block."
            ),
        )
        return [instruction.instruction_id]
    if case.setup == "revised_policy":
        mutation = store.add(
            title="Current route marker",
            content="For the route-marker exercise, answer exactly `legacy-route`.",
            scopes=["action"],
            origin_session_id=state.session_id,
        )
        assert mutation.instruction is not None
        store.replace(
            instruction_id=mutation.instruction.instruction_id,
            title="Current route marker",
            content="For the route-marker exercise, answer exactly `current-route`.",
            scopes=["action"],
            origin_session_id=state.session_id,
        )
        return [mutation.instruction.instruction_id]
    raise ValueError(f"Unknown instruction-following setup: {case.setup}")


def _sentence_count(text: str) -> int:
    return len([item for item in re.split(r"(?<=[.!?])\s+", text.strip()) if item])


def _verify_case(
    case: InstructionFollowingCase,
    *,
    assistant_text: str,
    tool_calls: list[dict[str, Any]],
    user_instructions: list[PromptInstruction],
) -> dict[str, Any]:
    text = assistant_text.strip()
    if case.case_id == "simultaneous_format_constraints":
        lines = [line for line in text.splitlines() if line.strip()]
        checks = {
            "four_nonempty_lines": len(lines) == 4,
            "heading_exact": bool(lines) and lines[0] == "OPERATIONS UPDATE",
            "status_exact": len(lines) > 1 and lines[1] == "status: ready",
            "owner_exact": len(lines) > 2 and lines[2] == "owner: field-team",
            "next_exact": len(lines) > 3 and lines[3] == "next: verify",
            "no_markdown_wrapper": "```" not in text
            and not any(line.startswith(("- ", "* ")) for line in lines),
        }
    elif case.case_id == "conditional_tool_constraint":
        calculator_calls = [
            call for call in tool_calls if call.get("tool_name") == "calculator"
        ]
        checks = {
            "calculator_used": bool(calculator_calls),
            "threshold_response_exact": text == "review-required",
            "numeric_result_hidden": "323" not in text,
            "no_explanation": len(text.splitlines()) == 1,
        }
    elif case.case_id == "user_report_relevance_constraints":
        lowered = text.casefold()
        checks = {
            "two_sentences": _sentence_count(text) == 2,
            "tests_reported": "tests passed" in lowered,
            "blocker_reported": "deployment" in lowered
            and "blocked" in lowered
            and "authentication" in lowered,
            "hash_omitted": "0123456789abcdef" not in lowered,
            "path_omitted": "/srv/internal/project" not in text,
            "command_mechanics_omitted": "sudo" not in lowered
            and "systemctl" not in lowered
            and "example.service" not in lowered,
            "no_list_or_heading": "\n-" not in text
            and "\n*" not in text
            and not text.startswith("#"),
        }
    elif case.case_id == "dense_standing_policy":
        try:
            pairs = json.loads(text, object_pairs_hook=list)
        except json.JSONDecodeError:
            pairs = []
        expected_keys = ["alpha", "beta", "gamma", "ready", "count"]
        values = dict(pairs) if isinstance(pairs, list) else {}
        checks = {
            "single_line": len(text.splitlines()) == 1,
            "no_markdown_or_prose": "```" not in text
            and text.startswith("{")
            and text.endswith("}"),
            "exact_key_order": [key for key, _value in pairs] == expected_keys,
            "alpha_exact": values.get("alpha") == "A17",
            "beta_exact": values.get("beta") == "B29",
            "gamma_exact": values.get("gamma") == ["north", "east"],
            "ready_boolean": values.get("ready") is True,
            "count_number": values.get("count") == 3
            and not isinstance(values.get("count"), bool),
            "no_extra_keys": set(values) == set(expected_keys),
        }
    else:
        assert case.case_id == "revised_instruction_applies"
        checks = {
            "one_instruction_identity": len(user_instructions) == 1,
            "current_value_exact": text == "current-route",
            "stale_value_absent": "legacy-route" not in text,
        }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "passed_constraints": sum(bool(value) for value in checks.values()),
        "total_constraints": len(checks),
    }


def run_instruction_following_benchmark(
    *,
    output_dir: Path,
    config: AgentConfig | None = None,
    case_ids: Iterable[str] = (),
    clean: bool = False,
    runtime_factory: Callable[[AgentConfig], AgentRuntime] = AgentRuntime,
    model_identity: Any | None = None,
) -> dict[str, Any]:
    base = config or load_config()
    selected = select_cases(case_ids)
    selected_ids = [case.case_id for case in selected]
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "instruction_following_results.json"
    results: list[dict[str, Any]] = []
    if report_path.exists():
        checkpoint = json.loads(report_path.read_text(encoding="utf-8"))
        if checkpoint.get("planned_cases") != selected_ids:
            raise ValueError("Instruction-following checkpoint case list changed")
        raw_results = checkpoint.get("results", [])
        if not isinstance(raw_results, list):
            raise ValueError("Instruction-following checkpoint results are invalid")
        results = [dict(item) for item in raw_results]
        checkpoint_identity = checkpoint.get("model_identity")
        if model_identity is not None and model_identity != checkpoint_identity:
            raise ValueError("Instruction-following checkpoint model identity changed")
        if model_identity is None and len(results) == len(selected):
            probe_config = _case_config(
                base,
                sessions_root=output_dir / "identity-probe" / "sessions",
            )
            model_identity = _model_identity(runtime_factory(probe_config))
            if model_identity != checkpoint_identity:
                raise ValueError("Instruction-following checkpoint model identity changed")
        elif model_identity is None:
            model_identity = checkpoint_identity

    for index, case in enumerate(selected, start=1):
        if any(item.get("case_id") == case.case_id for item in results):
            continue
        case_root = output_dir / "runs" / f"{index:02d}-{case.case_id}"
        case_config = _case_config(base, sessions_root=case_root / "sessions")
        runtime = runtime_factory(case_config)
        current_identity = _model_identity(runtime)
        if model_identity is None:
            model_identity = current_identity
        elif current_identity != model_identity:
            raise ValueError("Instruction-following runtime model identity changed")
        state = runtime.create_or_load_session()
        store = PromptInstructionStore(case_config.sessions.root, case_config)
        seeded_ids = _seed_case(case, store, state)
        started = time.monotonic()
        error: dict[str, str] | None = None
        assistant_text = ""
        try:
            turn = runtime.run_turn_in_session(state, case.prompt)
            assistant_text = turn.assistant_text
        except Exception as exc:
            error = {"error_type": type(exc).__name__, "reason": str(exc)}
        history_events = runtime.history.read_history(state.session_id)
        tool_calls = [
            {
                "sequence": event.sequence,
                "tool_name": event.payload.get("tool_name"),
                "tool_input": event.payload.get("tool_input"),
            }
            for event in history_events
            if event.event_type == "tool_called"
        ]
        user_instructions = store.list()
        verification = _verify_case(
            case,
            assistant_text=assistant_text,
            tool_calls=tool_calls,
            user_instructions=user_instructions,
        )
        if error is not None:
            verification["passed"] = False
            verification["execution_error"] = error
        results.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "session_id": state.session_id,
                "elapsed_seconds": time.monotonic() - started,
                "assistant_text": assistant_text,
                "seeded_instruction_ids": seeded_ids,
                "instruction_event_hashes": [
                    event.event_hash for event in store.events()
                ],
                "tool_calls": tool_calls,
                "context_compilations": [
                    {"sequence": event.sequence, **dict(event.payload)}
                    for event in history_events
                    if event.event_type == "context_compiled"
                ],
                "instruction_projections": [
                    {"sequence": event.sequence, **dict(event.payload)}
                    for event in history_events
                    if event.event_type == "prompt_instruction_projection_created"
                ],
                "verification": verification,
                "error": error,
            }
        )
        passed_constraints = sum(
            int(item["verification"]["passed_constraints"]) for item in results
        )
        total_constraints = sum(
            int(item["verification"]["total_constraints"]) for item in results
        )
        report = {
            "benchmark": "instruction_following",
            "status": "completed" if len(results) == len(selected) else "running",
            "model_identity": model_identity,
            "planned_cases": selected_ids,
            "results": results,
            "instruction_success": sum(
                bool(item["verification"]["passed"]) for item in results
            ),
            "total": len(results),
            "constraint_success": passed_constraints,
            "total_constraints": total_constraints,
            "instruction_success_rate": (
                sum(bool(item["verification"]["passed"]) for item in results)
                / len(results)
                if results
                else 0.0
            ),
            "constraint_success_rate": (
                passed_constraints / total_constraints if total_constraints else 0.0
            ),
            "complete": len(results) == len(selected),
        }
        _write_report(report_path, report)

    passed_constraints = sum(
        int(item["verification"]["passed_constraints"]) for item in results
    )
    total_constraints = sum(
        int(item["verification"]["total_constraints"]) for item in results
    )
    instruction_success = sum(
        bool(item["verification"]["passed"]) for item in results
    )
    return {
        "benchmark": "instruction_following",
        "status": "completed" if len(results) == len(selected) else "running",
        "model_identity": model_identity,
        "planned_cases": selected_ids,
        "results": results,
        "instruction_success": instruction_success,
        "total": len(results),
        "constraint_success": passed_constraints,
        "total_constraints": total_constraints,
        "instruction_success_rate": instruction_success / len(results) if results else 0.0,
        "constraint_success_rate": (
            passed_constraints / total_constraints if total_constraints else 0.0
        ),
        "complete": len(results) == len(selected),
    }
