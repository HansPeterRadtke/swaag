from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from swaag.config import AgentConfig, load_config
from swaag.notes import make_note
from swaag.runtime import AgentRuntime
from swaag.types import Note, SessionState
from swaag.utils import sha256_text, stable_json_dumps


@dataclass(slots=True, frozen=True)
class NoteBehaviorCase:
    case_id: str
    split: str
    prompt: str
    setup: str


CASES = (
    NoteBehaviorCase(
        case_id="categorized_lifecycle",
        split="baseline",
        setup="empty",
        prompt=(
            "Use the notes capability to inspect the current durable notes, then add two "
            "different working notes. One is for software implementation and testing: a "
            "claimed defect must be reproduced before editing and every change must be "
            "tested. The other is for research and source verification: prefer primary "
            "sources and verify version-specific claims. Give each note concise free-form "
            "semantic categories so later action calls can select it by meaning. Do not "
            "combine the two concerns into one note."
        ),
    ),
    NoteBehaviorCase(
        case_id="revise_and_erase",
        split="held_out",
        setup="stale_and_obsolete",
        prompt=(
            "Inspect the durable notes. Replace the existing 'Version evidence' note in "
            "place so it says that installed behavior must be reproduced because the "
            "documentation may be stale, and categorize it for version research. Erase the "
            "note titled 'Retired workaround' completely. Preserve the first note's identity "
            "instead of adding a replacement copy."
        ),
    ),
    NoteBehaviorCase(
        case_id="semantic_application_isolation",
        split="held_out",
        setup="implementation_and_reporting",
        prompt=(
            "Which exact durable marker governs the current parser implementation? Answer "
            "in one sentence. Do not list, add, replace, remove, or compact notes."
        ),
    ),
)


def select_cases(case_ids: Iterable[str] = ()) -> list[NoteBehaviorCase]:
    requested = list(case_ids)
    by_id = {case.case_id: case for case in CASES}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError("Unknown note behavior case: " + ", ".join(unknown))
    return [by_id[case_id] for case_id in requested] if requested else list(CASES)


def _case_config(
    base: AgentConfig,
    *,
    sessions_root: Path,
    workspace: Path,
) -> AgentConfig:
    config = copy.deepcopy(base)
    config.sessions.root = sessions_root
    config.tools.read_roots = [workspace]
    config.model.cache_enabled = False
    config.tools.enabled = ["notes"]
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


def _record_seed_note(
    runtime: AgentRuntime,
    state: SessionState,
    *,
    title: str,
    content: str,
    categories: list[str],
) -> str:
    note = make_note(
        runtime.config,
        title=title,
        content=content,
        categories=categories,
    )
    runtime.history.record_event(state, "note_added", {"note": asdict(note)})
    return note.note_id


def _seed_case(
    case: NoteBehaviorCase,
    runtime: AgentRuntime,
    state: SessionState,
) -> tuple[SessionState, list[str]]:
    if case.setup == "empty":
        return state, []
    if case.setup == "stale_and_obsolete":
        seeded = [
            _record_seed_note(
                runtime,
                state,
                title="Version evidence",
                content="Trust the documentation without checking installed behavior.",
                categories=["documentation"],
            ),
            _record_seed_note(
                runtime,
                state,
                title="Retired workaround",
                content="Always apply the obsolete parser workaround.",
                categories=["software implementation"],
            ),
        ]
    elif case.setup == "implementation_and_reporting":
        seeded = [
            _record_seed_note(
                runtime,
                state,
                title="Parser implementation marker",
                content="The exact current implementation marker is PARSER-ALPHA-731.",
                categories=["software implementation", "parser work"],
            ),
            _record_seed_note(
                runtime,
                state,
                title="Later audio report marker",
                content="The exact later presentation marker is REPORT-BETA-902.",
                categories=["audio reporting"],
            ),
        ]
    else:
        raise ValueError(f"Unknown note behavior setup: {case.setup}")
    return (
        runtime.history.rebuild_from_history(
            state.session_id,
            prefer_checkpoint=False,
        ),
        seeded,
    )


def _note_rows(notes: list[Note]) -> list[dict[str, Any]]:
    return [asdict(note) for note in notes]


def _verify_case(
    case: NoteBehaviorCase,
    *,
    seeded_ids: list[str],
    notes: list[Note],
    tool_actions: list[str],
    note_mutations: list[str],
    selection_payloads: list[dict[str, Any]],
    assistant_text: str,
) -> dict[str, Any]:
    if case.case_id == "categorized_lifecycle":
        implementation = [
            note
            for note in notes
            if "defect" in note.content.casefold()
            or "change" in note.content.casefold()
        ]
        research = [
            note
            for note in notes
            if "source" in note.content.casefold()
            or "version" in note.content.casefold()
        ]
        checks = {
            "inspected_before_mutation": bool(tool_actions)
            and tool_actions[0] == "list",
            "two_distinct_notes": len(notes) == 2,
            "implementation_categories": len(implementation) == 1
            and bool(implementation[0].categories),
            "research_categories": len(research) == 1
            and bool(research[0].categories),
            "category_sets_differ": len(implementation) == 1
            and len(research) == 1
            and set(implementation[0].categories)
            != set(research[0].categories),
            "exact_mutations": note_mutations == ["note_added", "note_added"],
        }
    elif case.case_id == "revise_and_erase":
        text = "\n".join(
            f"{note.title}\n{note.content}" for note in notes
        ).casefold()
        checks = {
            "one_note_remains": len(notes) == 1,
            "identity_preserved": len(notes) == 1
            and bool(seeded_ids)
            and notes[0].note_id == seeded_ids[0],
            "stale_meaning_removed": "trust the documentation" not in text,
            "repair_preserved": "reproduc" in text
            and "stale" in text,
            "research_category": len(notes) == 1
            and any(
                "version" in category.casefold()
                or "research" in category.casefold()
                for category in notes[0].categories
            ),
            "used_replace_and_remove": note_mutations
            == ["note_added", "note_added", "note_replaced", "note_removed"],
        }
    else:
        assert case.case_id == "semantic_application_isolation"
        relevant, unrelated = seeded_ids
        semantic_selections = [
            payload
            for payload in selection_payloads
            if payload.get("semantic_selection") is True
        ]
        checks = {
            "notes_unchanged": len(notes) == 2 and not tool_actions,
            "relevant_marker_answered": "parser-alpha-731"
            in assistant_text.casefold(),
            "unrelated_marker_excluded": "report-beta-902"
            not in assistant_text.casefold(),
            "selector_isolated_note": any(
                payload.get("included_note_ids") == [relevant]
                and payload.get("omitted_note_ids") == [unrelated]
                for payload in semantic_selections
            ),
        }
    return {"passed": all(checks.values()), "checks": checks}


def run_note_behavior_benchmark(
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
    report_path = output_dir / "note_behavior_results.json"
    results: list[dict[str, Any]] = []
    if report_path.exists():
        checkpoint = json.loads(report_path.read_text(encoding="utf-8"))
        if checkpoint.get("planned_cases") != selected_ids:
            raise ValueError("Note checkpoint case list does not match this run")
        raw_results = checkpoint.get("results", [])
        if not isinstance(raw_results, list):
            raise ValueError("Note checkpoint results are invalid")
        results = [dict(item) for item in raw_results]
        checkpoint_identity = checkpoint.get("model_identity")
        if model_identity is not None and model_identity != checkpoint_identity:
            raise ValueError("Note checkpoint model identity does not match this run")
        if model_identity is None and len(results) == len(selected):
            probe_workspace = output_dir / "identity-probe" / "workspace"
            probe_workspace.mkdir(parents=True, exist_ok=True)
            probe_config = _case_config(
                base,
                sessions_root=output_dir / "identity-probe" / "sessions",
                workspace=probe_workspace,
            )
            model_identity = _model_identity(runtime_factory(probe_config))
            if model_identity != checkpoint_identity:
                raise ValueError("Note checkpoint model identity changed")
        elif model_identity is None:
            model_identity = checkpoint_identity

    for index, case in enumerate(selected, start=1):
        if any(item.get("case_id") == case.case_id for item in results):
            continue
        case_root = output_dir / "runs" / f"{index:02d}-{case.case_id}"
        workspace = case_root / "workspace"
        workspace.mkdir(parents=True, exist_ok=False)
        case_config = _case_config(
            base,
            sessions_root=case_root / "sessions",
            workspace=workspace,
        )
        runtime = runtime_factory(case_config)
        current_identity = _model_identity(runtime)
        if model_identity is None:
            model_identity = current_identity
        elif current_identity != model_identity:
            raise ValueError("Note runtime model identity changed")
        state = runtime.create_or_load_session()
        state, seeded_ids = _seed_case(case, runtime, state)
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
        events = runtime.history.read_history(state.session_id)
        tool_actions = [
            str(event.payload.get("tool_input", {}).get("action", ""))
            for event in events
            if event.event_type == "tool_called"
            and event.payload.get("tool_name") == "notes"
        ]
        note_mutations = [
            event.event_type
            for event in events
            if event.event_type
            in {"note_added", "note_replaced", "note_removed", "notes_compacted"}
        ]
        selection_payloads = [
            dict(event.payload)
            for event in events
            if event.event_type == "notes_selected"
        ]
        verification = _verify_case(
            case,
            seeded_ids=seeded_ids,
            notes=rebuilt.notes,
            tool_actions=tool_actions,
            note_mutations=note_mutations,
            selection_payloads=selection_payloads,
            assistant_text=assistant_text,
        )
        if error is not None:
            verification = {
                "passed": False,
                "checks": verification["checks"],
                "execution_error": error,
            }
        source_events = [
            event
            for event in events
            if event.event_type == "message_added"
            and isinstance(event.payload.get("message"), dict)
            and event.payload["message"].get("role") == "user"
            and event.payload["message"].get("content") == case.prompt
        ]
        results.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "session_id": state.session_id,
                "elapsed_seconds": time.monotonic() - started,
                "assistant_text": assistant_text,
                "source_prompt_sha256": sha256_text(case.prompt),
                "source_event_references": [
                    {
                        "session_id": event.session_id,
                        "sequence": event.sequence,
                        "event_id": event.id,
                        "event_hash": event.hash,
                    }
                    for event in source_events
                ],
                "seeded_note_ids": seeded_ids,
                "notes": _note_rows(rebuilt.notes),
                "tool_actions": tool_actions,
                "note_mutations": note_mutations,
                "note_selections": selection_payloads,
                "context_compilations": [
                    {"sequence": event.sequence, **dict(event.payload)}
                    for event in events
                    if event.event_type == "context_compiled"
                ],
                "verification": verification,
            }
        )
        report = {
            "benchmark": "note_behavior",
            "planned_cases": selected_ids,
            "model_identity": model_identity,
            "results": results,
            "total": len(selected),
            "completed": len(results),
            "passed": sum(
                1 for item in results if item["verification"]["passed"]
            ),
            "complete": len(results) == len(selected),
        }
        _write_report(report_path, report)

    return json.loads(report_path.read_text(encoding="utf-8"))
