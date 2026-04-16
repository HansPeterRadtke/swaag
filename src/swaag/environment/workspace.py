from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from swaag.environment.filesystem import FilesystemManager
from swaag.environment.state import WorkspaceState
from swaag.utils import utc_now_iso


@dataclass(slots=True)
class WorkspaceSnapshot:
    root: str
    cwd: str
    files: dict[str, str]
    created_files: list[str]
    modified_files: list[str]
    deleted_files: list[str]
    captured_at: str


class WorkspaceManager:
    def __init__(self, filesystem: FilesystemManager):
        self.filesystem = filesystem

    def initialize_state(self) -> WorkspaceState:
        root = str(self.filesystem.workspace_root)
        return WorkspaceState(root=root, cwd=root, last_snapshot_at=utc_now_iso())

    def apply_delta(self, state: WorkspaceState, *, created: dict[str, str], modified: dict[str, str], deleted: list[str], cwd: str) -> WorkspaceState:
        known_files = dict(state.known_files)
        for key, value in created.items():
            known_files[key] = value
        for key, value in modified.items():
            known_files[key] = value
        for key in deleted:
            known_files.pop(key, None)
        listed_files = sorted(set(state.listed_files) | set(created) | set(modified))
        listed_files = [item for item in listed_files if item not in deleted]
        return WorkspaceState(
            root=state.root,
            cwd=cwd,
            known_files=known_files,
            listed_files=listed_files,
            modified_files=sorted(set(state.modified_files) | set(modified)),
            created_files=sorted(set(state.created_files) | set(created)),
            deleted_files=sorted(set(state.deleted_files) | set(deleted)),
            last_snapshot_at=utc_now_iso(),
        )

    def snapshot(self, *, before: dict[str, str], after: dict[str, str], cwd: str) -> WorkspaceSnapshot:
        delta = self.filesystem.compute_delta(before, after)
        return WorkspaceSnapshot(
            root=str(self.filesystem.workspace_root),
            cwd=cwd,
            files=after,
            created_files=list(delta["created_files"]),
            modified_files=list(delta["modified_files"]),
            deleted_files=list(delta["deleted_files"]),
            captured_at=utc_now_iso(),
        )
