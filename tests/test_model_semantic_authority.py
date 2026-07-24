from __future__ import annotations

from pathlib import Path

import pytest

from swaag.decision import DecisionValidationError, decision_from_payload
from swaag.memory_semantic import extract_from_event
from swaag.prompt_analyzer import analysis_from_payload
from swaag.strategy import strategy_from_payload
from swaag.types import HistoryEvent


ROOT = Path(__file__).resolve().parents[1]


def _production_sources() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "src" / "swaag").rglob("*"))
        if path.is_file() and path.suffix in {".py", ".txt", ".toml"}
    }


FORBIDDEN_FRAGMENTS = {
    "src/swaag/prompt_analyzer.py": [
        "analyze_prompt_emergency_fallback",
        "_VAGUE_MARKERS",
        "_GOAL_VERBS",
        "_OPEN_ENDED_NOUNS",
    ],
    "src/swaag/decision.py": [
        "decide_from_analysis_emergency_fallback",
        "normalized_model_decision",
        "analysis.task_type",
        "analysis.requires_",
    ],
    "src/swaag/expander.py": [
        "def expand_task(",
        "playable loop",
        "bounded first implementation",
    ],
    "src/swaag/planner.py": [
        "Untitled step",
        "Use the available context.",
        "expected_outputs[0]",
        "_default_done_condition",
        "Complete the task correctly and safely.",
        "If a step fails, replan from the latest valid state.",
    ],
    "src/swaag/failure.py": [
        "classify_failure_emergency_fallback",
        "FAILURE_KIND_DEFAULTS",
        "policy_for_kind",
    ],
    "src/swaag/strategy.py": [
        "select_strategy_emergency_default",
        "required_step_kinds\": [",
        "expected_flow\": [",
        "allowed_tools\": [",
        "missing_required",
        "disallowed_tools",
    ],
    "src/swaag/retrieval/embeddings.py": [
        "semantic_terms",
        "DegradedLexicalBackend",
        "LocalSemanticBackend",
        "degraded_lexical",
        "local_semantic",
    ],
    "src/swaag/memory_semantic.py": [
        "re.compile",
        "metadata.get(\"entities\"",
        "metadata.get(\"relationships\"",
        "metadata.get(\"facts\"",
    ],
    "src/swaag/environment/filesystem.py": [
        "_repair_missing_file_path",
        "_filename_bits",
        "rglob(requested_name)",
    ],
    "src/swaag/environment/environment.py": [
        "_repair_test_command_paths",
        "stem_bits",
        "candidate_bits",
    ],
    "src/swaag/prompts.py": [
        "missing mapping or handler",
        "mapping table or dispatch table",
    ],
    "src/swaag/tools/builtin.py": [
        "insert command here",
        "placeholder tokens",
        "without placeholder text",
        "todo\" in lowered",
        "\"/path/to/\" in lowered",
    ],
    "src/swaag/live_runtime_profiles.py": [
        "representative 30-task live subset",
        "completed the full live subset",
        "measured path",
    ],
    "src/swaag/benchmark/local_agent_runner.py": [
        "_extract_search_terms",
        "_rank_candidate",
        "_STOPWORDS",
        "_CODE_TOKENS",
        "difflib",
        "append fallback",
    ],
    "src/swaag/assets/prompts/task_decision_user.txt": [
        "vague tasks must request expansion",
        "structured complete tasks should not ask",
    ],
    "tests/helpers.py": [
        "_default_prompt_analysis",
        "Default deterministic choice",
        "Available subagents",
        "test scaffold action choice",
        "test scaffold subagent choice",
    ],
}


def test_production_source_rejects_named_semantic_authority_defects() -> None:
    production = _production_sources()
    combined = "\n".join(production.values())
    forbidden_global = [
        "_tools_equivalent",
        "_tool_names_equivalent",
        "allowed_followers",
        "capability_graph",
        "can_chain",
        "shortest_chain",
        "plan_tool_graph",
        "validate_tool_chain",
        "tool_graph_planned",
        "tool_graph_rejected",
        "_repair_declared_objective_conditions",
        "_repair_declared_response_semantic_conditions",
        "promoted_model_declared_objective_checks",
        "promoted_model_declared_response_semantic_checks",
        "plan_repaired",
        "create_direct_response_plan",
        "_create_model_selected_direct_response_plan",
        "semantic_direct_response",
    ]
    hits = [fragment for fragment in forbidden_global if fragment in combined]
    assert hits == []

    runtime_source = production["src/swaag/runtime.py"]
    assert "candidate_types=[" not in runtime_source
    assert '"candidate_types": ["' not in runtime_source
    assert "subagent_selection_contract(candidate_types)" in runtime_source
    assert "self._subagents.enabled_specs()" in runtime_source


def test_no_known_semantic_shortcut_surfaces_remain() -> None:
    hits: list[str] = []
    for relative_path, fragments in FORBIDDEN_FRAGMENTS.items():
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for fragment in fragments:
            if fragment in text:
                hits.append(f"{relative_path}: {fragment}")
    assert hits == []


def test_model_contract_parsers_do_not_repair_semantic_choices(make_config) -> None:
    analysis = analysis_from_payload(
        {
            "task_type": "vague",
            "completeness": "incomplete",
            "requires_expansion": False,
            "requires_decomposition": False,
            "confidence": 0.5,
            "detected_entities": [],
            "detected_goals": [],
        }
    )
    assert analysis.requires_expansion is False

    with pytest.raises(DecisionValidationError):
        decision_from_payload(
            {
                "split_task": False,
                "expand_task": True,
                "ask_user": False,
                "assume_missing": False,
                "generate_ideas": False,
                "direct_response": True,
                "execution_mode": "direct_response",
                "preferred_tool_name": "",
                "confidence": 0.9,
                "reason": "model returned contradictory machine fields",
            }
        )

    strategy = strategy_from_payload(
        {
            "task_profile": "coding",
            "strategy_name": "exploratory",
            "explore_before_commit": True,
            "tool_chain_depth": 2,
            "verification_intensity": 1.0,
            "reason": "model selected the label",
        }
    )
    assert not hasattr(strategy, "allowed_tools")
    assert strategy.required_step_kinds == []
    assert strategy.expected_flow == []

    event = HistoryEvent(
        id="evt_1",
        sequence=1,
        session_id="session",
        timestamp="2026-07-15T00:00:00Z",
        type="tool_result",
        version=1,
        payload={"tool_name": "calculator", "output": {"result": 42}},
        metadata={"trust_level": "trusted"},
        prev_hash="",
        hash="",
    )
    items, warning = extract_from_event(make_config(), event)
    assert warning is None
    assert len(items) == 1
    assert items[0].memory_kind == "event_snapshot"
    assert items[0].metadata["raw_event"]["payload"]["tool_name"] == "calculator"
    assert "entities" not in items[0].metadata
    assert "relationships" not in items[0].metadata
    assert "facts" not in items[0].metadata
