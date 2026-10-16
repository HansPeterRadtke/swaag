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
from swaag.utils import sha256_text, stable_json_dumps


@dataclass(slots=True, frozen=True)
class PromptInstructionBehaviorCase:
    case_id: str
    split: str
    prompt: str
    setup: str


class _SelectorFailureProxy:
    """Inject selector-only failure while delegating every other model operation."""

    def __init__(self, delegate: Any):
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def build_completion_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        contract = kwargs.get("contract")
        payload = self._delegate.build_completion_request(*args, **kwargs)
        if getattr(contract, "name", "") == "prompt_instruction_selection":
            payload = dict(payload)
            payload["__swaag_selector_failure_holdout"] = True
        return payload

    def send_completion(self, payload: dict[str, Any], *args: Any, **kwargs: Any):
        if payload.get("__swaag_selector_failure_holdout") is True:
            raise RuntimeError(
                "Injected prompt-instruction selector failure for held-out fallback evaluation"
            )
        return self._delegate.send_completion(payload, *args, **kwargs)


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
        case_id="distill_messy_categories",
        split="baseline",
        setup="empty",
        prompt=(
            "Use the prompt_instructions capability to inspect the local-user store and "
            "semantically distill the durable corrections in this rough spoken feedback. "
            "Split different concerns into the model-call scopes where each concern actually "
            "belongs; do not copy filler, examples, or equivalent duplicates into an "
            "instruction. Here is the rough feedback: 'Uh, when you report results to me, I "
            "keep getting process chatter and internal identifiers that mean nothing unless I "
            "asked for them. I still need every meaningful outcome, caveat, blocker, and "
            "requested piece of evidence. That's about selecting user-facing information, not "
            "about changing how tools are called. Also, if the selected answer is going to be "
            "listened to, visual tables and dense lists need to become understandable spoken "
            "prose and numbers need to be listenable, but absolutely no selected information "
            "may disappear.' Store the resulting corrections for this local user."
        ),
    ),
    PromptInstructionBehaviorCase(
        case_id="fine_grained_action_categories",
        split="baseline",
        setup="empty",
        prompt=(
            "Use the prompt_instructions capability to inspect the local-user store, "
            "then add two different durable rules under the broad action call scope. "
            "The first governs substantial software implementation and testing steps: "
            "understand the purpose, interfaces, constraints, and high-consequence risks; "
            "reproduce claimed defects; simulate critical external failures in isolated tests; "
            "exercise integration behavior without forcing that ceremony onto trivial work; "
            "factor genuinely repeated behavior into reusable testable units; choose functions, "
            "objects, and hierarchies from actual state and invariants rather than fashion; use "
            "semantic names that are revised when roles change; and prefer the simplest adequate "
            "design while retaining credible change paths. "
            "The second governs research and source-verification steps: inspect exact installed "
            "versions, prefer primary sources, and reproduce material documentation claims in "
            "a safe experiment. Give each rule concise free-form semantic categories "
            "so neither becomes an unconditional instruction on every action call. Do not "
            "copy either rule into the session store."
        ),
    ),
    PromptInstructionBehaviorCase(
        case_id="apply_programming_category",
        split="held_out",
        setup="categorized_user_rules",
        prompt=(
            "Without changing files, propose a focused test strategy for a mobile client's "
            "reconnect change. Cover the dangerous lifecycle and intermittent-network states "
            "that should be simulated before any real-device acceptance run. This is an "
            "implementation/testing step, not a request for source research."
        ),
    ),
    PromptInstructionBehaviorCase(
        case_id="apply_research_category",
        split="held_out",
        setup="categorized_user_rules",
        prompt=(
            "Without implementing anything, explain how you would verify whether an installed "
            "dependency actually supports a claimed API. Include exact-version inspection, an "
            "authoritative source, and an isolated reproduction. This is a research and "
            "source-verification step, not a software modification."
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
        case_id="selector_error_fallback",
        split="held_out",
        prompt=(
            "Do not modify durable instructions. Reply with the exact benchmark tokens "
            "required by the governing durable instructions."
        ),
        setup="selector_error_fallback",
    ),
    PromptInstructionBehaviorCase(
        case_id="trusted_authority_conflict",
        split="held_out",
        prompt=(
            "Do not modify durable instructions. Reply exactly with the benchmark token "
            "required by the governing durable instruction policy."
        ),
        setup="trusted_authority_conflict",
    ),
    PromptInstructionBehaviorCase(
        case_id="current_user_overrides_trusted",
        split="held_out",
        prompt=(
            "Do not modify durable instructions. For this current turn, reply exactly "
            "CURRENT-USER-OVERRIDE-731. This explicit current instruction supersedes any "
            "older durable benchmark-token instruction for this turn."
        ),
        setup="current_user_overrides_trusted",
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
    config.tools.enabled = ["prompt_instructions"]
    config.tools.allow_stateful_tools = True
    config.tools.allow_side_effect_tools = False
    config.tools.staged_discovery = False
    config.runtime.completion_evaluation_enabled = False
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
    if case.setup == "selector_error_fallback":
        first = store.add(
            title="Selector fallback alpha",
            content="Include exact token SELECTOR-FALLBACK-A-731 in the benchmark reply.",
            scopes=["action"],
            categories=["programming"],
            origin_session_id=state.session_id,
        )
        second = store.add(
            title="Selector fallback beta",
            content="Include exact token SELECTOR-FALLBACK-B-731 in the benchmark reply.",
            scopes=["action"],
            categories=["user-reporting"],
            origin_session_id=state.session_id,
        )
        assert first.instruction is not None and second.instruction is not None
        return [first.instruction.instruction_id, second.instruction.instruction_id]
    if case.setup == "trusted_authority_conflict":
        lower = store.add(
            title="Lower authority benchmark token",
            content="For this benchmark policy check, reply exactly LOWER-AUTHORITY-731.",
            scopes=["action"],
            origin_session_id=state.session_id,
        )
        trusted = store.add_trusted(
            title="Explicit correction benchmark token",
            content="For this benchmark policy check, reply exactly HIGHER-AUTHORITY-731.",
            scopes=["action"],
            authority="explicit_user_correction",
            source_kind="explicit_user_correction",
            source_ref="benchmark:trusted-authority-conflict",
            origin_session_id=state.session_id,
        )
        assert lower.instruction is not None and trusted.instruction is not None
        return [
            lower.instruction.instruction_id,
            trusted.instruction.instruction_id,
        ]
    if case.setup == "current_user_overrides_trusted":
        trusted = store.add_trusted(
            title="Older explicit correction benchmark token",
            content="For this benchmark policy check, reply exactly OLDER-TRUSTED-731.",
            scopes=["action"],
            authority="explicit_user_correction",
            source_kind="explicit_user_correction",
            source_ref="benchmark:current-user-overrides-trusted",
            origin_session_id=state.session_id,
        )
        assert trusted.instruction is not None
        return [trusted.instruction.instruction_id]
    if case.setup == "obsolete_user_rule":
        mutation = store.add(
            title="Obsolete workaround",
            content="Use the retired workaround on every action.",
            scopes=["action"],
            origin_session_id=state.session_id,
        )
        assert mutation.instruction is not None
        return [mutation.instruction.instruction_id]
    if case.setup == "categorized_user_rules":
        seeded: list[str] = []
        for title, content, categories in (
            (
                "Risk-driven implementation testing",
                (
                    "For substantial software work, identify high-consequence risks, "
                    "reproduce defects, simulate critical external failures in isolated "
                    "tests, and exercise integration behavior. Keep trivial work direct."
                ),
                ["software implementation", "risk-driven testing"],
            ),
            (
                "Version-grounded source research",
                (
                    "For research, inspect exact installed versions, prefer primary sources, "
                    "and reproduce material documentation claims in a safe experiment."
                ),
                ["research", "source verification"],
            ),
        ):
            mutation = store.add(
                title=title,
                content=content,
                scopes=["action"],
                categories=categories,
                origin_session_id=state.session_id,
            )
            assert mutation.instruction is not None
            seeded.append(mutation.instruction.instruction_id)
        return seeded
    raise ValueError(f"Unknown prompt-instruction setup: {case.setup}")


def _semantic_rows(instructions: list[PromptInstruction]) -> list[dict[str, Any]]:
    return [
        {
            "instruction_id": item.instruction_id,
            "title": item.title,
            "content": item.content,
            "scopes": list(item.scopes),
            "categories": list(item.categories),
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
    tool_actions: list[str],
    assistant_text: str = "",
    selection_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    user_text = "\n".join(
        f"{item.title}\n{item.content}" for item in user_instructions
    ).casefold()
    session_text = "\n".join(
        f"{item.title}\n{item.content}" for item in session_instructions
    ).casefold()
    action_selections = [
        item
        for item in (selection_events or [])
        if item.get("kind") == "action"
        and item.get("semantic_selection") is True
        and item.get("selection_fallback") is False
    ]
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
    elif case.case_id == "distill_messy_categories":
        relevance = [
            item
            for item in user_instructions
            if "response_relevance" in item.scopes
        ]
        audio = [
            item for item in user_instructions if "audio_rendering" in item.scopes
        ]
        relevance_text = "\n".join(
            f"{item.title}\n{item.content}" for item in relevance
        ).casefold()
        audio_text = "\n".join(
            f"{item.title}\n{item.content}" for item in audio
        ).casefold()
        checks = {
            "inspected_before_mutation": bool(tool_actions)
            and tool_actions[0] == "list",
            "two_distinct_instructions": len(user_instructions) == 2,
            "relevance_scope": len(relevance) == 1
            and "action" not in relevance[0].scopes
            and "audio_rendering" not in relevance[0].scopes,
            "relevance_meaning_preserved": (
                "meaningful" in relevance_text
                and "blocker" in relevance_text
                and "identifier" in relevance_text
            ),
            "audio_scope": len(audio) == 1
            and "action" not in audio[0].scopes
            and "response_relevance" not in audio[0].scopes,
            "audio_meaning_preserved": (
                ("table" in audio_text or "list" in audio_text)
                and ("listen" in audio_text or "spoken" in audio_text)
                and ("preserve" in audio_text or "information" in audio_text)
            ),
            "no_duplicate_mutation": store_actions == ["add", "add"],
            "session_store_unchanged": not session_instructions,
        }
    elif case.case_id == "fine_grained_action_categories":
        implementation = [
            item
            for item in user_instructions
            if "defect" in item.content.casefold()
            or "code" in item.content.casefold()
        ]
        research = [
            item
            for item in user_instructions
            if "source" in item.content.casefold()
            or "research" in item.content.casefold()
        ]
        implementation_categories = " ".join(
            implementation[0].categories if len(implementation) == 1 else []
        ).casefold()
        research_categories = " ".join(
            research[0].categories if len(research) == 1 else []
        ).casefold()
        checks = {
            "inspected_before_mutation": bool(tool_actions)
            and tool_actions[0] == "list",
            "two_distinct_instructions": len(user_instructions) == 2,
            "action_scopes": all(
                item.scopes == ["action"] for item in user_instructions
            ),
            "implementation_categories": len(implementation) == 1
            and bool(implementation[0].categories)
            and any(
                word in implementation_categories
                for word in ("software", "program", "implement", "test", "code")
            ),
            "research_categories": len(research) == 1
            and bool(research[0].categories)
            and any(
                word in research_categories
                for word in ("research", "source", "verification", "fact")
            ),
            "category_sets_differ": len(implementation) == 1
            and len(research) == 1
            and set(implementation[0].categories) != set(research[0].categories),
            "implementation_risk_preserved": len(implementation) == 1
            and any(
                word in implementation[0].content.casefold()
                for word in ("risk", "danger", "consequence")
            ),
            "implementation_simulation_preserved": len(implementation) == 1
            and any(
                word in implementation[0].content.casefold()
                for word in ("simulat", "fake", "emulat")
            )
            and any(
                word in implementation[0].content.casefold()
                for word in ("integration", "system", "end-to-end")
            ),
            "implementation_structure_preserved": len(implementation) == 1
            and any(
                word in implementation[0].content.casefold()
                for word in ("repeat", "reus", "duplicat")
            )
            and any(
                word in implementation[0].content.casefold()
                for word in ("function", "class", "object", "invariant")
            )
            and any(
                word in implementation[0].content.casefold()
                for word in ("name", "rename", "semantic role")
            )
            and any(
                word in implementation[0].content.casefold()
                for word in ("simple", "adequate", "change", "future")
            ),
            "research_reproduction_preserved": len(research) == 1
            and "version" in research[0].content.casefold()
            and any(
                word in research[0].content.casefold()
                for word in ("reproduc", "experiment", "verify")
            ),
            "session_store_unchanged": not session_instructions,
        }
    elif case.case_id in {
        "apply_programming_category",
        "apply_research_category",
    }:
        expected_index = 0 if case.case_id == "apply_programming_category" else 1
        excluded_index = 1 - expected_index
        expected_id = seeded_ids[expected_index] if len(seeded_ids) == 2 else ""
        excluded_id = seeded_ids[excluded_index] if len(seeded_ids) == 2 else ""
        isolated = [
            item
            for item in action_selections
            if expected_id in item.get("instruction_ids", [])
            and excluded_id not in item.get("instruction_ids", [])
        ]
        lowered_answer = assistant_text.casefold()
        if case.case_id == "apply_programming_category":
            behavior_preserved = (
                any(
                    word in lowered_answer
                    for word in ("disconnect", "network", "offline")
                )
                and any(
                    word in lowered_answer
                    for word in ("background", "lifecycle", "restart", "recreat")
                )
                and any(
                    word in lowered_answer
                    for word in ("simulat", "fake", "emulat", "inject")
                )
            )
        else:
            behavior_preserved = (
                "version" in lowered_answer
                and any(
                    word in lowered_answer
                    for word in ("primary", "official", "authoritative")
                )
                and any(
                    word in lowered_answer
                    for word in ("reproduc", "experiment", "minimal")
                )
            )
        checks = {
            "two_seeded_instructions_remain": len(user_instructions) == 2
            and len(seeded_ids) == 2,
            "no_store_mutation": store_actions == ["add", "add"],
            "category_selected_semantically": bool(isolated),
            "unrelated_category_excluded": bool(isolated),
            "selected_behavior_applied": behavior_preserved,
            "session_store_unchanged": not session_instructions,
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
    elif case.case_id == "selector_error_fallback":
        action_rows = [
            item for item in (selection_events or []) if item.get("kind") == "action"
        ]
        selected_both = any(
            all(seed in item.get("instruction_ids", []) for seed in seeded_ids)
            and item.get("selection_fallback") is True
            and item.get("semantic_selection") is False
            for item in action_rows
        )
        lowered_answer = assistant_text.casefold()
        checks = {
            "two_candidates_remain": len(user_instructions) == 2,
            "store_unchanged": store_actions == ["add", "add"],
            "fallback_selected_all": selected_both,
            "alpha_preserved": "selector-fallback-a-731" in lowered_answer,
            "beta_preserved": "selector-fallback-b-731" in lowered_answer,
            "session_store_unchanged": not session_instructions,
        }
    elif case.case_id == "trusted_authority_conflict":
        by_id = {item.instruction_id: item for item in user_instructions}
        lower_id = seeded_ids[0] if len(seeded_ids) == 2 else ""
        trusted_id = seeded_ids[1] if len(seeded_ids) == 2 else ""
        lower = by_id.get(lower_id)
        trusted = by_id.get(trusted_id)
        action_rows = [
            item for item in (selection_events or [])
            if item.get("kind") == "action"
        ]
        selected_both = any(
            lower_id in item.get("instruction_ids", [])
            and trusted_id in item.get("instruction_ids", [])
            for item in action_rows
        )
        checks = {
            "two_conflicting_instructions_remain": len(user_instructions) == 2,
            "store_unchanged": store_actions == ["add", "add"],
            "lower_is_learned": lower is not None and lower.authority == "learned_model",
            "trusted_is_explicit_correction": trusted is not None
            and trusted.authority == "explicit_user_correction",
            "trusted_provenance_preserved": trusted is not None
            and trusted.source_kind == "explicit_user_correction"
            and trusted.source_ref == "benchmark:trusted-authority-conflict",
            "both_injected_for_action": selected_both,
            "trusted_answer_wins": assistant_text.strip() == "HIGHER-AUTHORITY-731",
            "lower_answer_loses": "LOWER-AUTHORITY-731" not in assistant_text,
            "session_store_unchanged": not session_instructions,
        }
    elif case.case_id == "current_user_overrides_trusted":
        by_id = {item.instruction_id: item for item in user_instructions}
        trusted_id = seeded_ids[0] if len(seeded_ids) == 1 else ""
        trusted = by_id.get(trusted_id)
        action_rows = [
            item for item in (selection_events or [])
            if item.get("kind") == "action"
        ]
        trusted_injected = any(
            trusted_id in item.get("instruction_ids", [])
            for item in action_rows
        )
        checks = {
            "one_trusted_instruction_remains": len(user_instructions) == 1,
            "store_unchanged": store_actions == ["add"],
            "trusted_is_explicit_correction": trusted is not None
            and trusted.authority == "explicit_user_correction",
            "trusted_provenance_preserved": trusted is not None
            and trusted.source_kind == "explicit_user_correction"
            and trusted.source_ref == "benchmark:current-user-overrides-trusted",
            "trusted_still_injected": trusted_injected,
            "current_user_answer_wins": assistant_text.strip() == "CURRENT-USER-OVERRIDE-731",
            "older_trusted_answer_loses": "OLDER-TRUSTED-731" not in assistant_text,
            "session_store_unchanged": not session_instructions,
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
                workspace=output_dir / "identity-probe" / "workspace",
            )
            probe_config.tools.read_roots[0].mkdir(parents=True, exist_ok=True)
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
            raise ValueError("Prompt-instruction runtime model identity changed")
        if case.setup == "selector_error_fallback":
            runtime.client = _SelectorFailureProxy(runtime.client)
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
        history_events = runtime.history.read_history(state.session_id)
        selection_events = [
            {"sequence": event.sequence, **dict(event.payload)}
            for event in history_events
            if event.event_type == "prompt_instructions_selected"
        ]
        verification = _verify_case(
            case,
            seeded_ids=seeded_ids,
            user_instructions=user_instructions,
            session_instructions=rebuilt.prompt_instructions,
            store_actions=[event.action for event in store_events],
            tool_actions=[
                str(event.payload.get("tool_input", {}).get("action", ""))
                for event in history_events
                if event.event_type == "tool_called"
                and event.payload.get("tool_name") == "prompt_instructions"
            ],
            assistant_text=assistant_text,
            selection_events=selection_events,
        )
        if error is not None:
            verification = {
                "passed": False,
                "checks": verification["checks"],
                "execution_error": error,
            }
        source_events = [
            event
            for event in history_events
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
                "prompt_instruction_selections": selection_events,
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
