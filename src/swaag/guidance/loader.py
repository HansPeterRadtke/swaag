from __future__ import annotations

from pathlib import Path

from swaag.config import AgentConfig
from swaag.guidance.types import GuidanceItem
from swaag.types import SessionState


_ROLE_GUIDANCE = {
    "primary": "Stay goal-directed. Prefer verified evidence over intuition.",
    "planner": "Produce minimal valid plans with explicit verification and dependencies.",
    "verifier": "Reject unsupported or partial results. Deterministic failure is final.",
    "executor": "Use only the needed tools and preserve environment state.",
}


def load_guidance_items(config: AgentConfig, state: SessionState) -> list[GuidanceItem]:
    if not config.guidance.enabled:
        return []
    items: list[GuidanceItem] = []
    for raw_path in config.guidance.global_paths:
        path = Path(raw_path).expanduser()
        if path.exists():
            items.append(GuidanceItem(layer="global", source=str(path), text=path.read_text(encoding="utf-8").strip(), priority=10, always_on=True))
    root = Path(state.environment.workspace.root) if state.environment.workspace.root else None
    cwd = Path(state.environment.workspace.cwd) if state.environment.workspace.cwd else root
    if root is not None:
        for filename in config.guidance.filenames:
            candidate = root / filename
            if candidate.exists():
                items.append(GuidanceItem(layer="repo", source=str(candidate), text=candidate.read_text(encoding="utf-8").strip(), priority=20, always_on=True))
    if root is not None and cwd is not None:
        current = cwd
        while True:
            if current == root:
                break
            for filename in config.guidance.filenames:
                candidate = current / filename
                if candidate.exists():
                    items.append(GuidanceItem(layer="directory", source=str(candidate), text=candidate.read_text(encoding="utf-8").strip(), priority=30))
            if current.parent == current or root not in current.parents:
                break
            current = current.parent
    if state.expanded_task is not None:
        task_lines = [
            *(f"constraint: {item}" for item in state.expanded_task.constraints),
            *(f"expected_output: {item}" for item in state.expanded_task.expected_outputs),
            *(f"assumption: {item}" for item in state.expanded_task.assumptions),
        ]
        if task_lines:
            items.append(GuidanceItem(layer="task", source="expanded_task", text="\n".join(task_lines), priority=40))
    role_text = _ROLE_GUIDANCE.get(state.active_role, "")
    if role_text:
        items.append(GuidanceItem(layer="role", source=state.active_role, text=role_text, priority=50, always_on=True))
    return items
