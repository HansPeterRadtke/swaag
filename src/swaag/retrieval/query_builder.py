from __future__ import annotations

from dataclasses import dataclass, field

from swaag.retrieval.embeddings import semantic_terms
from swaag.types import SessionState


@dataclass(slots=True)
class RetrievalIntent:
    query_text: str
    goal: str
    current_step_text: str
    active_entities: list[str]
    unresolved_failures: list[str]
    environment_summary: str
    guidance_summary: str
    role_name: str
    purpose: str
    dependency_terms: list[str]
    terms: list[str] = field(default_factory=list)


def build_retrieval_intent(
    state: SessionState,
    *,
    goal: str,
    purpose: str,
    current_step_text: str,
    environment_summary: str,
    guidance_summary: str,
) -> RetrievalIntent:
    dependency_terms: list[str] = []
    if state.active_plan is not None and state.active_plan.current_step_id:
        current_step = next((item for item in state.active_plan.steps if item.step_id == state.active_plan.current_step_id), None)
        if current_step is not None:
            dependency_terms.extend(semantic_terms(" ".join(current_step.input_refs)))
            for dependency_id in current_step.depends_on:
                dependency_step = next((item for item in state.active_plan.steps if item.step_id == dependency_id), None)
                if dependency_step is None:
                    continue
                dependency_terms.extend(
                    semantic_terms(
                        f"{dependency_step.title} {dependency_step.goal} {dependency_step.expected_tool or ''} {' '.join(dependency_step.output_refs)}"
                    )
                )
    unresolved_failures: list[str] = []
    recent_messages = state.messages[-6:]
    for message in recent_messages:
        if message.role == "tool" and "tool_error" in message.content:
            unresolved_failures.append(message.content[:200])
    query_parts = [
        goal,
        current_step_text,
        " ".join(state.working_memory.active_entities),
        " ".join(unresolved_failures),
        environment_summary,
        guidance_summary,
        state.active_role,
        purpose,
    ]
    query_text = "\n".join(part for part in query_parts if part.strip())
    return RetrievalIntent(
        query_text=query_text,
        goal=goal,
        current_step_text=current_step_text,
        active_entities=list(state.working_memory.active_entities),
        unresolved_failures=unresolved_failures,
        environment_summary=environment_summary,
        guidance_summary=guidance_summary,
        role_name=state.active_role,
        purpose=purpose,
        dependency_terms=dependency_terms,
        terms=semantic_terms(query_text),
    )
