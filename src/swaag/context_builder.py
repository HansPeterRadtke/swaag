from __future__ import annotations

from dataclasses import dataclass, field

from swaag.budgeting import compute_call_budget, compute_section_budgets
from swaag.config import AgentConfig
from swaag.guidance import load_guidance_items, resolve_guidance
from swaag.notes import select_notes_for_prompt
from swaag.retrieval import HybridRetriever
from swaag.retrieval.trace import RetrievalTraceItem
from swaag.skills import SkillSelection, render_skill_instructions, render_skill_metadata, select_skills
from swaag.tokens import ConservativeEstimator
from swaag.types import HistoryEvent, Message, PromptComponent, SemanticMemoryItem, SessionState
from swaag.working_memory import build_working_memory


ContextTraceItem = RetrievalTraceItem

_MINIMAL_FRONTEND_KINDS = {"analysis", "task_decision", "expansion"}
_LIGHT_CONTEXT_KINDS = {"strategy", "failure", "subagent_selection", "control", "tool_input"}
_MINIMAL_FRONTEND_COMPONENT_NAMES = {"guidance", "environment"}


@dataclass(slots=True)
class ContextBundle:
    goal: str
    history_messages: list[Message]
    notes_text: str
    note_ids: list[str]
    omitted_note_ids: list[str]
    note_tokens: int
    note_tokens_exact: bool
    semantic_items: list[SemanticMemoryItem]
    working_memory_text: str
    strategy_text: str
    plan_text: str
    project_state_text: str
    environment_text: str
    relevant_files_text: str
    relevant_environment_files: list[tuple[str, str]]
    guidance_text: str
    guidance_sources: list[str]
    skill_metadata_text: str
    skill_instructions_text: str
    selected_skill_ids: list[str]
    exposed_tool_names: list[str]
    tool_prompt_tuples: list[tuple[str, str, dict]]
    retrieval_mode: str
    retrieval_degraded: bool
    recent_results_text: str
    active_entities_text: str
    components: list[PromptComponent] = field(default_factory=list)
    selection_trace: list[ContextTraceItem] = field(default_factory=list)



def _render_plan(state: SessionState) -> str:
    if state.active_plan is None:
        return ""
    parts = [f"Goal: {state.active_plan.goal}", "Steps:"]
    for step in state.active_plan.steps:
        tool_text = f" tool={step.expected_tool}" if step.expected_tool else ""
        parts.append(f"- [{step.status}] {step.title}{tool_text}")
    return "\n".join(parts)



def _render_strategy(state: SessionState) -> str:
    if state.active_strategy is None:
        return ""
    required_step_kinds = ", ".join(state.active_strategy.required_step_kinds) or "(none)"
    expected_flow = " -> ".join(state.active_strategy.expected_flow) or "(none)"
    return (
        f"Strategy: {state.active_strategy.strategy_name}\n"
        f"Task profile: {state.active_strategy.task_profile}\n"
        f"Reason: {state.active_strategy.reason}\n"
        f"Required step kinds: {required_step_kinds}\n"
        f"Expected flow: {expected_flow}\n"
        f"Explore before commit: {state.active_strategy.explore_before_commit}\n"
        f"Validate assumptions: {state.active_strategy.validate_assumptions}\n"
        f"Simplify if stuck: {state.active_strategy.simplify_if_stuck}\n"
        f"Switch on failure: {state.active_strategy.switch_on_failure}"
    )



def _render_project_state(state: SessionState, *, compact: bool = False) -> str:
    if (
        not state.project_state.files_seen
        and not state.project_state.relationships
        and not state.project_state.expected_artifacts
    ):
        return ""
    lines: list[str] = []
    if state.project_state.files_seen and not compact:
        lines.append("Files seen:")
        lines.extend(f"- {path}" for path in state.project_state.files_seen)
    if state.project_state.files_modified:
        lines.append("Files modified:")
        items = state.project_state.files_modified[:3] if compact else state.project_state.files_modified
        lines.extend(f"- {path}" for path in items)
    if state.project_state.relationships and not compact:
        lines.append("Relationships:")
        lines.extend(f"- {item}" for item in state.project_state.relationships)
    if state.project_state.expected_artifacts:
        lines.append("Expected artifacts:")
        items = state.project_state.expected_artifacts[:4] if compact else state.project_state.expected_artifacts
        lines.extend(f"- {item}" for item in items)
    if state.project_state.pending_artifacts:
        lines.append("Pending artifacts:")
        items = state.project_state.pending_artifacts[:4] if compact else state.project_state.pending_artifacts
        lines.extend(f"- {item}" for item in items)
    if state.project_state.completed_artifacts:
        lines.append("Completed artifacts:")
        items = state.project_state.completed_artifacts[:4] if compact else state.project_state.completed_artifacts
        lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)



def _render_environment(state: SessionState, *, compact: bool = False) -> str:
    workspace = state.environment.workspace
    shell = state.environment.shell
    if not workspace.root and not shell.cwd:
        return ""
    lines = []
    if workspace.root and workspace.cwd and workspace.root == workspace.cwd:
        lines.append(f"Workspace: {workspace.root}")
    else:
        if workspace.root:
            lines.append(f"Workspace root: {workspace.root}")
        if workspace.cwd:
            lines.append(f"Workspace cwd: {workspace.cwd}")
    if shell.last_command and not compact:
        lines.append(f"Last shell command: {shell.last_command}")
    if shell.last_exit_code is not None and not compact:
        lines.append(f"Last shell exit code: {shell.last_exit_code}")
    if workspace.created_files:
        lines.append("Created files:")
        items = workspace.created_files[:3] if compact else workspace.created_files
        lines.extend(f"- {item}" for item in items)
    if workspace.modified_files:
        lines.append("Modified files:")
        items = workspace.modified_files[:3] if compact else workspace.modified_files
        lines.extend(f"- {item}" for item in items)
    if workspace.deleted_files:
        lines.append("Deleted files:")
        items = workspace.deleted_files[:3] if compact else workspace.deleted_files
        lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)



def _render_environment_files(items: list[tuple[str, str]]) -> str:
    if not items:
        return ""
    parts = []
    for relative_path, content in items:
        parts.append(f"File: {relative_path}\n{content}")
    return "\n\n".join(parts)



def _render_memory_text(item_list: list[SemanticMemoryItem]) -> str:
    if not item_list:
        return ""
    return "\n".join(f"- ({item.memory_kind}) {item.content}" for item in item_list)



def _render_planning_notes(included_notes) -> str:
    if not included_notes:
        return ""
    return "\n".join(
        f"- [{note.note_id}] {note.title} (chars={len(note.content)})"
        for note in included_notes
    )



def build_context(
    config: AgentConfig,
    state: SessionState,
    counter,
    *,
    goal: str,
    call_kind: str = "decision",
    for_planning: bool = False,
    history_events: list[HistoryEvent] | None = None,
    available_tools: list[tuple[str, str, dict]] | None = None,
) -> ContextBundle:
    working_memory = build_working_memory(state)
    # Use the caller-provided counter so selection traces and intermediate
    # section budgets use exact token counts whenever the active model
    # exposes tokenization. Conservative estimates remain available only via
    # the runtime's explicit estimate fallback path.
    selection_counter = counter or ConservativeEstimator()
    call_budget = compute_call_budget(config, call_kind=call_kind)
    section_budgets = compute_section_budgets(
        config,
        call_kind=call_kind,
        safe_input_budget=call_budget.safe_input_budget,
    )
    dynamic_guidance_items = max(
        1,
        min(config.guidance.max_items, section_budgets.guidance_tokens // max(config.context_policy.guidance_item_token_hint, 1)),
    )
    dynamic_metadata_skills = max(
        1,
        min(config.skills.max_metadata_items, section_budgets.skills_tokens // max(config.context_policy.skill_metadata_token_hint, 1)),
    )
    dynamic_full_skills = max(
        1,
        min(config.skills.max_full_instructions, section_budgets.skills_tokens // max(config.context_policy.skill_instruction_token_hint, 1)),
    )
    dynamic_recent_results = max(
        1,
        min(
            config.context_builder.max_recent_results,
            len(working_memory.recent_results) or 1,
            section_budgets.history_tokens // max(config.context_policy.recent_result_token_hint, 1),
        ),
    )
    note_selection = select_notes_for_prompt(
        config,
        state.notes,
        selection_counter,
        max_tokens=section_budgets.notes_tokens,
    )
    notes_text = note_selection.rendered_text
    note_tokens = note_selection.tokens
    note_tokens_exact = note_selection.exact
    if for_planning:
        notes_text = _render_planning_notes(note_selection.included_notes)
        counted_notes = selection_counter.count_text(notes_text)
        note_tokens = counted_notes.tokens
        note_tokens_exact = counted_notes.exact
    current_step_text = working_memory.current_step_title or ""
    guidance_items = load_guidance_items(config, state)
    if call_kind in _MINIMAL_FRONTEND_KINDS:
        minimal_guidance_items = [item for item in guidance_items if item.always_on or item.layer == "task"]
        used_guidance_tokens = 0
        selected_guidance_items = []
        guidance_trace = []
        for item in sorted(minimal_guidance_items, key=lambda candidate: (candidate.priority, candidate.layer, candidate.source)):
            token_cost = selection_counter.count_text(item.text).tokens
            if used_guidance_tokens + token_cost > section_budgets.guidance_tokens:
                guidance_trace.append(
                    ContextTraceItem(
                        item_type="guidance",
                        item_id=item.source,
                        score=0.0,
                        reasons=["dropped:over_budget:minimal_frontend"],
                        selected=False,
                        token_cost=token_cost,
                        source=item.layer,
                        signals={"call_kind": call_kind},
                    )
                )
                continue
            selected_guidance_items.append(item)
            used_guidance_tokens += token_cost
            guidance_trace.append(
                ContextTraceItem(
                    item_type="guidance",
                    item_id=item.source,
                    score=100.0 - float(len(selected_guidance_items)),
                    reasons=["included:minimal_frontend"],
                    selected=True,
                    token_cost=token_cost,
                    source=item.layer,
                    signals={"call_kind": call_kind},
                )
            )
        guidance_text = "\n\n".join(f"[{item.layer}] {item.text}" for item in selected_guidance_items)
        guidance_sources = [item.source for item in selected_guidance_items]
        guidance_selection_trace = guidance_trace
    else:
        guidance_bundle = resolve_guidance(
            guidance_items,
            counter=selection_counter,
            max_items=dynamic_guidance_items,
            max_tokens=section_budgets.guidance_tokens,
            query_text="\n".join(part for part in [goal, current_step_text, state.active_role] if part.strip()),
            backend_mode=config.retrieval.backend,
            base_url=config.model.base_url,
            seed=config.model.seed,
            connect_timeout_seconds=config.model.connect_timeout_seconds,
            read_timeout_seconds=config.model.simple_timeout_seconds,
        )
        guidance_text = guidance_bundle.merged_text
        guidance_sources = [item.source for item in guidance_bundle.items]
        guidance_selection_trace = [
            ContextTraceItem(
                item_type="guidance",
                item_id=trace.source,
                score=float(index + 1 if trace.selected else 0),
                reasons=[trace.reason],
                selected=trace.selected,
                token_cost=trace.token_cost,
                source=trace.layer,
                signals={},
            )
            for index, trace in enumerate(guidance_bundle.trace)
        ]

    if call_kind in _MINIMAL_FRONTEND_KINDS | _LIGHT_CONTEXT_KINDS:
        history_messages = []
        semantic_items = []
        environment_files = []
        retrieval_mode = "llm_scoring" if config.retrieval.backend == "llm_scoring" else config.retrieval.backend
        retrieval_degraded = False
        retrieval_trace: list[ContextTraceItem] = []
    else:
        retriever = HybridRetriever(config)
        retrieval = retriever.retrieve(
            state,
            counter=selection_counter,
            goal=goal,
            current_step_text=current_step_text,
            environment_summary=_render_environment(state),
            guidance_summary=guidance_text,
            history_events=history_events,
            for_planning=for_planning,
            max_history_tokens=section_budgets.history_tokens,
            max_semantic_tokens=section_budgets.semantic_tokens,
            max_environment_tokens=section_budgets.environment_files_tokens,
            max_history_items=config.context_builder.max_history_messages,
            max_semantic_items=config.context_builder.max_semantic_items,
            max_environment_items=config.context_builder.max_environment_files,
        )
        history_messages = retrieval.history_messages
        semantic_items = retrieval.semantic_items
        environment_files = retrieval.environment_files
        retrieval_mode = retrieval.mode
        retrieval_degraded = retrieval.degraded
        retrieval_trace = retrieval.trace
    retrieval_text = "\n".join(
        [
            *(message.content for message in history_messages[: config.context_policy.retrieval_preview_items]),
            *(item.content for item in semantic_items[: config.context_policy.retrieval_preview_items]),
            *(
                f"{path}\n{content[: config.context_policy.retrieval_preview_chars]}"
                for path, content in environment_files[: config.context_policy.retrieval_preview_items]
            ),
        ]
    )
    enabled_tool_names = [name for name, _, _ in (available_tools or [])]
    if call_kind in _MINIMAL_FRONTEND_KINDS | _LIGHT_CONTEXT_KINDS or not enabled_tool_names:
        skill_selection = SkillSelection(
            selected_skills=[],
            metadata_skills=[],
            selected_tool_names=[],
            trace=[],
        )
    else:
        skill_selection = select_skills(
            goal=goal,
            current_step_text=current_step_text,
            guidance_text=guidance_text,
            retrieval_text=retrieval_text,
            role_name=state.active_role,
            failure_summary=state.metrics.last_reasoning_reason,
            enabled_tool_names=enabled_tool_names,
            max_full_instructions=dynamic_full_skills,
            max_metadata_items=dynamic_metadata_skills,
            backend_mode=config.retrieval.backend,
            nondiscriminative_delta=config.selection_policy.skill_nondiscriminative_delta,
            base_url=config.model.base_url,
            seed=config.model.seed,
            connect_timeout_seconds=config.model.connect_timeout_seconds,
            read_timeout_seconds=config.model.simple_timeout_seconds,
            max_text_chars=config.selection_policy.retrieval_scoring_text_chars,
        )
    recent_results = working_memory.recent_results[-dynamic_recent_results:]
    recent_results_text = "\n".join(f"- {item}" for item in recent_results)
    active_entities_text = "\n".join(f"- {item}" for item in working_memory.active_entities)
    compact_context = call_kind == "tool_input"
    plan_text = _render_plan(state)
    strategy_text = _render_strategy(state)
    project_state_text = _render_project_state(state, compact=compact_context)
    environment_text = _render_environment(state, compact=compact_context)
    relevant_files_text = _render_environment_files(environment_files)
    skill_metadata_text = render_skill_metadata(skill_selection.metadata_skills)
    skill_instructions_text = render_skill_instructions(skill_selection.selected_skills)
    tool_prompt_tuples = list(available_tools or [])
    if skill_selection.selected_tool_names:
        allowed = set(skill_selection.selected_tool_names)
        tool_prompt_tuples = [item for item in tool_prompt_tuples if item[0] in allowed]
    working_memory_text = (
        f"Active goal: {working_memory.active_goal}\n"
        f"Current step: {working_memory.current_step_title or '(none)'}\n"
    ).strip()
    bundle = ContextBundle(
        goal=goal,
        history_messages=history_messages,
        notes_text=notes_text,
        note_ids=[note.note_id for note in note_selection.included_notes],
        omitted_note_ids=note_selection.omitted_note_ids,
        note_tokens=note_tokens,
        note_tokens_exact=note_tokens_exact,
        semantic_items=semantic_items,
        working_memory_text=working_memory_text,
        strategy_text=strategy_text,
        plan_text=plan_text,
        project_state_text=project_state_text,
        environment_text=environment_text,
        relevant_files_text=relevant_files_text,
        relevant_environment_files=environment_files,
        guidance_text=guidance_text,
        guidance_sources=guidance_sources,
        skill_metadata_text=skill_metadata_text,
        skill_instructions_text=skill_instructions_text,
        selected_skill_ids=[item.skill_id for item in skill_selection.selected_skills],
        exposed_tool_names=[name for name, _, _ in tool_prompt_tuples],
        tool_prompt_tuples=tool_prompt_tuples,
        retrieval_mode=retrieval_mode,
        retrieval_degraded=retrieval_degraded,
        recent_results_text=recent_results_text,
        active_entities_text=active_entities_text,
        selection_trace=[
            *retrieval_trace,
            *guidance_selection_trace,
            *[
                ContextTraceItem(
                    item_type="skill",
                    item_id=trace.skill_id,
                    score=trace.score,
                    reasons=[trace.reason],
                    selected=trace.selected,
                    token_cost=0,
                    source="skill_catalog",
                    signals={},
                )
                for trace in skill_selection.trace
            ],
        ],
    )
    component_priorities = config.context_policy.component_priorities
    component_specs = [
        (float(component_priorities["working_memory"]), "always_on", PromptComponent(name="working_memory", category="working_memory", text=bundle.working_memory_text + "\n\n" if bundle.working_memory_text else "")),
        (float(component_priorities["plan"]), "always_on", PromptComponent(name="plan", category="plan", text=f"Active plan:\n{bundle.plan_text}\n\n" if bundle.plan_text else "")),
        (float(component_priorities["strategy"]), "always_on", PromptComponent(name="strategy", category="strategy", text=f"Execution strategy:\n{bundle.strategy_text}\n\n" if bundle.strategy_text else "")),
        (float(component_priorities["guidance"]), "llm_relevance", PromptComponent(name="guidance", category="guidance", text=f"Active guidance:\n{bundle.guidance_text}\n\n" if bundle.guidance_text else "")),
        (float(component_priorities["environment"]), "environment_state", PromptComponent(name="environment", category="environment", text=f"Environment state:\n{bundle.environment_text}\n\n" if bundle.environment_text else "")),
        (float(component_priorities["environment_files"]), "retrieval_selected", PromptComponent(name="environment_files", category="environment_files", text=f"Relevant workspace files:\n{bundle.relevant_files_text}\n\n" if bundle.relevant_files_text else "")),
        (float(component_priorities["semantic_memory"]), "retrieval_selected", PromptComponent(name="semantic_memory", category="semantic_memory", text=f"Relevant memory:\n{_render_memory_text(bundle.semantic_items)}\n\n" if bundle.semantic_items else "")),
        (float(component_priorities["skills"]), "skill_selected", PromptComponent(name="skills", category="skills", text=f"Selected skill instructions:\n{bundle.skill_instructions_text}\n\n" if bundle.skill_instructions_text else "")),
        (float(component_priorities["project_state"]), "project_context", PromptComponent(name="project_state", category="project_state", text=f"Project state:\n{bundle.project_state_text}\n\n" if bundle.project_state_text else "")),
        (float(component_priorities["skill_metadata"]), "skill_metadata", PromptComponent(name="skill_metadata", category="skills", text=f"Relevant skill metadata:\n{bundle.skill_metadata_text}\n\n" if bundle.skill_metadata_text else "")),
        (float(component_priorities["recent_results"]), "recent_progress", PromptComponent(name="recent_results", category="recent_results", text=f"Recent results:\n{bundle.recent_results_text}\n\n" if bundle.recent_results_text else "")),
        (float(component_priorities["active_entities"]), "entity_context", PromptComponent(name="active_entities", category="active_entities", text=f"Active entities:\n{bundle.active_entities_text}\n\n" if bundle.active_entities_text else "")),
        (float(component_priorities["notes"]), "selected_notes", PromptComponent(name="notes", category="notes", text=f"Working notes:\n{bundle.notes_text}\n\n" if bundle.notes_text else "")),
    ]
    if call_kind in _MINIMAL_FRONTEND_KINDS:
        component_specs = [
            (priority, reason, component)
            for priority, reason, component in component_specs
            if component.name in _MINIMAL_FRONTEND_COMPONENT_NAMES
        ]
    packed_components: list[PromptComponent] = []
    packed_tokens = 0
    for priority, reason, component in component_specs:
        if not component.text:
            continue
        counted = counter.count_text(component.text)
        if packed_tokens + counted.tokens <= call_budget.safe_input_budget:
            packed_components.append(component)
            packed_tokens += counted.tokens
            bundle.selection_trace.append(
                ContextTraceItem(
                    item_type="context_section",
                    item_id=component.name,
                    score=priority,
                    reasons=[f"included:{reason}"],
                    selected=True,
                    token_cost=counted.tokens,
                    source=component.category,
                    signals={"priority": priority},
                )
            )
        else:
            bundle.selection_trace.append(
                ContextTraceItem(
                    item_type="context_section",
                    item_id=component.name,
                    score=priority,
                    reasons=[f"dropped:over_budget:{reason}"],
                    selected=False,
                    token_cost=counted.tokens,
                    source=component.category,
                    signals={"priority": priority},
                )
            )
    bundle.components = packed_components
    return bundle
