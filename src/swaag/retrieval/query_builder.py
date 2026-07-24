from __future__ import annotations

from dataclasses import dataclass, field

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
        dependency_terms=[],
        terms=[],
    )
