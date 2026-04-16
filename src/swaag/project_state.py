from __future__ import annotations

from pathlib import Path

from swaag.types import ProjectState, SessionState
from swaag.utils import utc_now_iso


def build_project_state(state: SessionState) -> ProjectState:
    def _workspace_path(path_text: str) -> set[str]:
        root = state.environment.workspace.root
        if not root:
            return {path_text}
        path = Path(path_text)
        if path.is_absolute():
            return {str(path)}
        absolute = Path(root) / path
        return {path_text, str(absolute)}

    environment_files: set[str] = set()
    if state.environment.workspace.root:
        for item in state.environment.workspace.listed_files:
            environment_files |= _workspace_path(str(item))
        for item in state.environment.workspace.known_files:
            environment_files |= _workspace_path(str(item))
    files_seen = sorted(set(state.file_views.keys()) | environment_files)
    modified_environment_files: set[str] = set()
    for item in state.environment.workspace.modified_files:
        modified_environment_files |= _workspace_path(str(item))
    for item in state.environment.workspace.created_files:
        modified_environment_files |= _workspace_path(str(item))
    files_modified = sorted(
        set(
            path
            for path, view in state.file_views.items()
            if view.last_operation in {"edit_applied", "file_write_applied", "file_write_failed"} or view.content is not None
        )
        | modified_environment_files
    )
    directories = sorted({str(Path(path).parent) for path in files_seen})
    relationships: list[str] = []
    for directory in directories:
        members = sorted(path for path in files_seen if str(Path(path).parent) == directory)
        if len(members) > 1:
            relationships.append(f"directory:{directory} -> {', '.join(Path(path).name for path in members)}")
    expected_artifacts: list[str] = []
    pending_artifacts: list[str] = []
    completed_artifacts: list[str] = []
    if state.active_plan is not None:
        for step in state.active_plan.steps:
            labels = [item.strip() for item in [*step.output_refs, *step.expected_outputs] if item.strip()]
            if not labels:
                labels = [step.expected_output.strip()] if step.expected_output.strip() else [step.title]
            for label in labels:
                artifact_label = f"{step.step_id}:{label}"
                if artifact_label not in expected_artifacts:
                    expected_artifacts.append(artifact_label)
                if step.status == "completed":
                    if artifact_label not in completed_artifacts:
                        completed_artifacts.append(artifact_label)
                elif step.status in {"pending", "running"}:
                    if artifact_label not in pending_artifacts:
                        pending_artifacts.append(artifact_label)
    return ProjectState(
        files_seen=files_seen,
        files_modified=files_modified,
        directories=directories,
        relationships=relationships,
        expected_artifacts=expected_artifacts,
        pending_artifacts=pending_artifacts,
        completed_artifacts=completed_artifacts,
        last_updated=utc_now_iso(),
    )
