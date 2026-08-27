from __future__ import annotations

import copy
import json
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
class PromptInstructionBehaviorCase:
    case_id: str
    split: str
    prompt: str
    setup: str


CASES = (
    PromptInstructionBehaviorCase(
        case_id="cross_session_scope",
        split="baseline",
        setup="empty",
        prompt=(
            "Use the prompt_instructions capability now. First inspect the local-user "
            "instruction store. Ensure it contains exactly one cross-session instruction "
            "for communication_status calls only, titled 'Private operational identifiers', "
            "saying not to expose internal identifiers unless I explicitly request them. "
            "Do not create a session-only copy and do not append an equivalent duplicate."
        ),
    ),
    PromptInstructionBehaviorCase(
        case_id="session_only_scope",
        split="baseline",
        setup="empty",
        prompt=(
            "Use the prompt_instructions capability now. Add this task-only rule to the "
            "session store, scoped only to action calls: verify explicit units before using "
            "a calculator. Do not place it in the cross-session user store."
        ),
    ),
    PromptInstructionBehaviorCase(
        case_id="revise_stale_rule",
        split="held_out",
        setup="stale_user_rule",
        prompt=(
            "The existing local-user reporting instruction is wrong. Inspect the user store "
            "and revise that same entry instead of appending another one. It must now say to "
            "omit internal identifiers unless I explicitly request them, and it must remain "
            "scoped only to communication_status."
        ),
    ),
    PromptInstructionBehaviorCase(
        case_id="repair_duplicate_bloat",
        split="held_out",
        setup="duplicate_user_rules",
        prompt=(
            "Audit the local-user prompt instructions. Two entries repeat the same reporting "
            "rule. Use the capability to remove the duplicate bloat while preserving exactly "
            "one rule that says to report meaningful evidence only. Do not add a third entry."
        ),
    ),
    PromptInstructionBehaviorCase(
        case_id="erase_obsolete_rule",
        split="held_out",
        setup="obsolete_user_rule",
        prompt=(
            "The local-user instruction titled 'Obsolete workaround' is no longer valid. "
            "Inspect the user instruction store and erase that entry now. Do not replace it "
            "and do not create another instruction."
        ),
    ),
)


def select_cases(case_ids: Iterable[str] = ()) -> list[PromptInstructionBehaviorCase]:
    requested = list(case_ids)
    by_id = {case.case_id: case for case in CASES}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(
            "Unknown prompt-instruction behavior case: " + ", ".join(unknown)
        )
    return [by_id[case_id] for case_id in requested] if requested else list(CASES)


def _case_config(base: AgentConfig, *, sessions_root: Path) -> AgentConfig:
    config = copy.deepcopy(base)
    config.sessions.root = sessions_root
    config.model.cache_enabled = False
    config.tools.enabled = ["prompt_instructions"]
    config.tools.allow_stateful_tools = True
    config.tools.allow_side_effect_tools = False
    config.tools.staged_discovery = False
    config.runtime.completion_evaluation_enabled = False
    config.runtime.tool_call_budget = max(12, config.runtime.tool_call_budget)
    config.runtime.max_total_actions = max(16, config.runtime.max_total_actions)
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


def _seed_case(
    case: PromptInstructionBehaviorCase,
    store: PromptInstructionStore,
    state: SessionState,
) -> list[str]:
    if case.setup == "empty":
        return []
    if case.setup == "stale_user_rule":
        mutation = store.add(
            title="Private operational identifiers",
            content="Always expose every internal identifier in status reports.",
            scopes=["communication_status"],
            origin_session_id=state.session_id,
        )
        assert mutation.instruction is not None
        return [mutation.instruction.instruction_id]
    if case.setup == "duplicate_user_rules":
        seeded: list[str] = []
        for title in ("Meaningful reporting", "Repeated meaningful reporting"):
            mutation = store.add(
                title=title,
                content="Report meaningful evidence only.",
                scopes=["communication_status"],
                origin_session_id=state.session_id,
            )
            assert mutation.instruction is not None
            seeded.append(mutation.instruction.instruction_id)
        return seeded
    if case.setup == "obsolete_user_rule":
        mutation = store.add(
            title="Obsolete workaround",
            content="Use the retired workaround on every action.",
            scopes=["action"],
            origin_session_id=state.session_id,
        )
        assert mutation.instruction is not None
        return [mutation.instruction.instruction_id]
    raise ValueError(f"Unknown prompt-instruction setup: {case.setup}")


def _semantic_rows(instructions: list[PromptInstruction]) -> list[dict[str, Any]]:
    return [
        {
            "instruction_id": item.instruction_id,
            "title": item.title,
            "content": item.content,
            "scopes": list(item.scopes),
        }
        for item in instructions
    ]


def _verify_case(
    case: PromptInstructionBehaviorCase,
    *,
    seeded_ids: list[str],
    user_instructions: list[PromptInstruction],
    session_instructions: list[PromptInstruction],
    store_actions: list[str],
) -> dict[str, Any]:
    user_text = "\n".join(
        f"{item.title}\n{item.content}" for item in user_instructions
    ).casefold()
    session_text = "\n".join(
        f"{item.title}\n{item.content}" for item in session_instructions
    ).casefold()
    if case.case_id == "cross_session_scope":
        checks = {
            "one_user_instruction": len(user_instructions) == 1,
            "no_session_copy": not session_instructions,
            "correct_scope": bool(user_instructions)
            and user_instructions[0].scopes == ["communication_status"],
            "meaning_preserved": "internal identifier" in user_text
            and "explicit" in user_text,
            "no_duplicate_mutation": store_actions == ["add"],
        }
    elif case.case_id == "session_only_scope":
        checks = {
            "no_user_instruction": not user_instructions,
            "one_session_instruction": len(session_instructions) == 1,
            "correct_scope": bool(session_instructions)
            and session_instructions[0].scopes == ["action"],
            "meaning_preserved": "unit" in session_text
            and "calculator" in session_text,
        }
    elif case.case_id == "revise_stale_rule":
        checks = {
            "same_single_identity": len(user_instructions) == 1
            and bool(seeded_ids)
            and user_instructions[0].instruction_id == seeded_ids[0],
            "used_revision": store_actions == ["add", "replace"],
            "correct_scope": bool(user_instructions)
            and user_instructions[0].scopes == ["communication_status"],
            "stale_rule_removed": "always expose" not in user_text,
            "correction_preserved": "internal identifier" in user_text
            and "explicit" in user_text,
        }
    elif case.case_id == "repair_duplicate_bloat":
        checks = {
            "one_instruction_remains": len(user_instructions) == 1,
            "no_new_entry": store_actions.count("add") == 2,
            "removed_duplicate": "remove" in store_actions,
            "meaning_preserved": "meaningful evidence" in user_text,
        }
    else:
        assert case.case_id == "erase_obsolete_rule"
        checks = {
            "store_is_empty": not user_instructions,
            "removed_without_replacement": store_actions == ["add", "remove"],
            "session_store_unchanged": not session_instructions,
        }
    return {"passed": all(checks.values()), "checks": checks}


def run_prompt_instruction_behavior_benchmark(
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
    report_path = output_dir / "prompt_instruction_behavior_results.json"
    results: list[dict[str, Any]] = []
    if report_path.exists():
        checkpoint = json.loads(report_path.read_text(encoding="utf-8"))
        if checkpoint.get("planned_cases") != selected_ids:
            raise ValueError(
                "Prompt-instruction checkpoint case list does not match this run"
            )
        raw_results = checkpoint.get("results", [])
        if not isinstance(raw_results, list):
            raise ValueError("Prompt-instruction checkpoint results are invalid")
        results = [dict(item) for item in raw_results]
        checkpoint_identity = checkpoint.get("model_identity")
        if model_identity is not None and model_identity != checkpoint_identity:
            raise ValueError(
                "Prompt-instruction checkpoint model identity does not match this run"
            )
        if model_identity is None and len(results) == len(selected):
            probe_config = _case_config(
                base,
                sessions_root=output_dir / "identity-probe" / "sessions",
            )
            model_identity = _model_identity(runtime_factory(probe_config))
            if model_identity != checkpoint_identity:
                raise ValueError(
                    "Prompt-instruction checkpoint model identity changed"
                )
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
            raise ValueError("Prompt-instruction runtime model identity changed")
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
            error = {
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }
        rebuilt = runtime.history.rebuild_from_history(
            state.session_id,
            prefer_checkpoint=False,
        )
        user_instructions = store.list()
        store_events = store.events()
        verification = _verify_case(
            case,
            seeded_ids=seeded_ids,
            user_instructions=user_instructions,
            session_instructions=rebuilt.prompt_instructions,
            store_actions=[event.action for event in store_events],
        )
        if error is not None:
            verification = {
                "passed": False,
                "checks": verification["checks"],
                "execution_error": error,
            }
        history_events = runtime.history.read_history(state.session_id)
        results.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "session_id": state.session_id,
                "elapsed_seconds": time.monotonic() - started,
                "assistant_text": assistant_text,
                "seeded_instruction_ids": seeded_ids,
                "user_instructions": _semantic_rows(user_instructions),
                "session_instructions": _semantic_rows(
                    rebuilt.prompt_instructions
                ),
                "store_events": [
                    {
                        "sequence": event.sequence,
                        "event_id": event.event_id,
                        "action": event.action,
                        "instruction_id": event.instruction_id,
                        "event_hash": event.event_hash,
                    }
                    for event in store_events
                ],
                "tool_calls": [
                    {
                        "sequence": event.sequence,
                        "tool_name": event.payload.get("tool_name"),
                        "tool_input": event.payload.get("tool_input"),
                    }
                    for event in history_events
                    if event.event_type == "tool_called"
                ],
                "context_compilations": [
                    {"sequence": event.sequence, **dict(event.payload)}
                    for event in history_events
                    if event.event_type == "context_compiled"
                ],
                "verification": verification,
                "error": error,
            }
        )
        report = {
            "benchmark": "prompt_instruction_behavior",
            "status": "running",
            "model_identity": model_identity,
            "planned_cases": selected_ids,
            "results": results,
            "passed": sum(
                bool(item["verification"]["passed"]) for item in results
            ),
            "total": len(results),
            "complete": len(results) == len(selected),
        }
        if report["complete"]:
            report["status"] = "completed"
        _write_report(report_path, report)

    return {
        "benchmark": "prompt_instruction_behavior",
        "status": "completed" if len(results) == len(selected) else "running",
        "model_identity": model_identity,
        "planned_cases": selected_ids,
        "results": results,
        "passed": sum(bool(item["verification"]["passed"]) for item in results),
        "total": len(results),
        "complete": len(results) == len(selected),
    }
