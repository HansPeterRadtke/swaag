from __future__ import annotations

from swaag.types import SessionState, WorkingMemory
from swaag.utils import utc_now_iso


def build_working_memory(state: SessionState) -> WorkingMemory:
    active_goal = (
        state.expanded_task.expanded_goal
        if state.expanded_task is not None
        else state.active_plan.goal
        if state.active_plan is not None
        else next((message.content for message in reversed(state.messages) if message.role == "user"), "")
    )
    current_step_id = state.active_plan.current_step_id if state.active_plan is not None else None
    current_step_title = ""
    if state.active_plan is not None and current_step_id is not None:
        for step in state.active_plan.steps:
            if step.step_id == current_step_id:
                current_step_title = step.title
                break

    recent_results = [
        message.content
        for message in state.messages
        if message.role == "tool"
    ][-4:]

    entities: list[str] = []
    entities.extend(sorted(state.file_views.keys())[-4:])
    entities.extend(note.title for note in state.notes[-4:])
    if state.active_plan is not None:
        for step in state.active_plan.steps:
            if step.step_id == current_step_id:
                entities.append(step.title)
                break

    deduped_entities: list[str] = []
    for item in entities:
        if item and item not in deduped_entities:
            deduped_entities.append(item)

    return WorkingMemory(
        active_goal=active_goal,
        current_step_id=current_step_id,
        current_step_title=current_step_title,
        recent_results=recent_results,
        active_entities=deduped_entities,
        updated_at=utc_now_iso(),
    )
