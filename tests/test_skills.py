from __future__ import annotations

import pytest

from swaag.context_builder import build_context
from swaag.prompts import PromptBuilder
from swaag.retrieval.embeddings import EmbeddingBackend
from swaag.skills.catalog import SkillSpec
from swaag.skills.selector import select_skills
from swaag.tokens import ConservativeEstimator
from swaag.types import Message, SessionState


def _state() -> SessionState:
    return SessionState(
        session_id="s1",
        created_at="t0",
        updated_at="t0",
        config_fingerprint="cfg",
        model_base_url="http://example.test",
        messages=[Message(role="user", content="Fix app.py and run tests.", created_at="t1")],
    )


class _CodingBackend(EmbeddingBackend):
    mode = "llm_scoring"
    degraded = False

    def score_query(self, query: str, texts: list[str]) -> list[float]:
        del query
        scores: list[float] = []
        for text in texts:
            lowered = text.lower()
            if (
                "repair broken code" in lowered
                or "exact file content change" in lowered
                or "repair code and verify the behavior" in lowered
            ):
                scores.append(0.95)
            elif (
                "retrieve and ground answers in browser evidence" in lowered
                or "investigate websites and gather external evidence" in lowered
            ):
                scores.append(0.0)
            else:
                scores.append(0.10)
        return scores


@pytest.fixture()
def _semantic_skill_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    backend_factory = lambda *args, **kwargs: _CodingBackend()
    monkeypatch.setattr("swaag.skills.selector.build_backend", backend_factory)
    monkeypatch.setattr("swaag.retrieval.retriever.build_backend", backend_factory)
    monkeypatch.setattr("swaag.guidance.resolver.build_backend", backend_factory)


def test_irrelevant_skills_are_not_fully_loaded(make_config, _semantic_skill_backend) -> None:
    config = make_config(retrieval__backend="llm_scoring")
    state = _state()
    tools = [
        ("read_text", "Read file text", {"type": "object", "properties": {"path": {"type": "string"}}}),
        ("edit_text", "Edit file text", {"type": "object", "properties": {"path": {"type": "string"}}}),
        ("run_tests", "Run tests", {"type": "object", "properties": {"command": {"type": "array"}}}),
        ("browser_search", "Search web", {"type": "object", "properties": {"query": {"type": "string"}}}),
    ]

    bundle = build_context(
        config,
        state,
        ConservativeEstimator(),
        goal="Fix app.py and run tests.",
        available_tools=tools,
    )

    assert "Coding Patch" in bundle.skill_instructions_text
    assert "Browser Research" not in bundle.skill_instructions_text
    assert "browser_search" not in bundle.exposed_tool_names


def test_selected_skill_narrows_tool_disclosure_and_reduces_prompt_size(make_config, _semantic_skill_backend) -> None:
    config = make_config(retrieval__backend="llm_scoring")
    state = _state()
    tools = [
        ("read_text", "Read file text", {"type": "object", "properties": {"path": {"type": "string"}}}),
        ("edit_text", "Edit file text", {"type": "object", "properties": {"path": {"type": "string"}}}),
        ("run_tests", "Run tests", {"type": "object", "properties": {"command": {"type": "array"}}}),
        ("browser_search", "Search web", {"type": "object", "properties": {"query": {"type": "string"}}}),
        ("browser_browse", "Browse url", {"type": "object", "properties": {"url": {"type": "string"}}}),
        ("shell_command", "Run shell command", {"type": "object", "properties": {"command": {"type": "string"}}}),
    ]
    bundle = build_context(
        config,
        state,
        ConservativeEstimator(),
        goal="Fix app.py and run tests.",
        available_tools=tools,
    )
    builder = PromptBuilder(config)

    full_catalog = builder.render_tool_catalog(tools, prompt_mode="standard")
    narrowed_catalog = builder.render_tool_catalog(bundle.tool_prompt_tuples, prompt_mode="standard")

    assert len(narrowed_catalog) < len(full_catalog)
    assert {"read_text", "edit_text", "run_tests"}.issubset(set(bundle.exposed_tool_names))
    assert "browser_search" not in bundle.exposed_tool_names


def test_skill_selection_is_not_just_trigger_term_count(make_config, _semantic_skill_backend) -> None:
    config = make_config(retrieval__backend="llm_scoring")
    state = SessionState(
        session_id="s2",
        created_at="t0",
        updated_at="t0",
        config_fingerprint="cfg",
        model_base_url="http://example.test",
        messages=[Message(role="user", content="Repair the service module and verify behavior.", created_at="t1")],
    )
    tools = [
        ("read_text", "Read file text", {"type": "object", "properties": {"path": {"type": "string"}}}),
        ("edit_text", "Edit file text", {"type": "object", "properties": {"path": {"type": "string"}}}),
        ("run_tests", "Run tests", {"type": "object", "properties": {"command": {"type": "array"}}}),
        ("browser_search", "Search web", {"type": "object", "properties": {"query": {"type": "string"}}}),
    ]

    bundle = build_context(
        config,
        state,
        ConservativeEstimator(),
        goal="Repair the service module and verify behavior.",
        available_tools=tools,
    )

    assert "coding_patch" in bundle.selected_skill_ids
    assert "browser_research" not in bundle.selected_skill_ids
    assert any(item.item_type == "skill" and item.item_id == "coding_patch" and item.selected for item in bundle.selection_trace)


def test_skill_selection_uses_metadata_not_full_instructions(make_config, _semantic_skill_backend) -> None:
    config = make_config(retrieval__backend="llm_scoring")
    custom_skills = [
        SkillSpec(
            skill_id="coding_patch",
            title="Coding Patch",
            selection_blurb="Repair code and verify the behavior with deterministic checks.",
            full_instructions="Normal coding workflow.",
            allowed_tools=["edit_text", "run_tests"],
            expected_outputs=["code change"],
            verifier_hints=["tests pass"],
        ),
        SkillSpec(
            skill_id="misleading_hidden",
            title="Hidden Workflow",
            selection_blurb="Investigate websites and gather external evidence.",
            full_instructions="Use the secret unicorn-repair workflow for parser failures.",
            allowed_tools=["browser_search"],
            expected_outputs=["web findings"],
            verifier_hints=["browser only"],
        ),
    ]

    selection = select_skills(
        goal="Repair the service behavior and handle the unicorn-repair request safely.",
        current_step_text="fix the service module and verify it",
        guidance_text="Prefer local code changes over browser work.",
        retrieval_text="service.py needs a repair and deterministic verification",
        role_name="primary",
        failure_summary="",
        enabled_tool_names=["edit_text", "run_tests", "browser_search"],
        max_full_instructions=2,
        max_metadata_items=2,
        backend_mode=config.retrieval.backend,
        skills=custom_skills,
    )

    assert [skill.skill_id for skill in selection.selected_skills] == ["coding_patch"]
    assert "browser_search" not in selection.selected_tool_names
