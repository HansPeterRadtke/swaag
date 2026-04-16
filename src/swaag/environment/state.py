from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ProcessStatus = Literal["running", "completed", "failed", "killed", "timed_out"]


@dataclass(slots=True)
class WorkspaceState:
    root: str = ""
    cwd: str = ""
    known_files: dict[str, str] = field(default_factory=dict)
    listed_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    last_snapshot_at: str = ""


@dataclass(slots=True)
class ShellSessionState:
    cwd: str = ""
    env_overrides: dict[str, str] = field(default_factory=dict)
    unset_vars: list[str] = field(default_factory=list)
    command_count: int = 0
    last_command: str = ""
    last_exit_code: int | None = None
    updated_at: str = ""


@dataclass(slots=True)
class ProcessRecord:
    process_id: str
    command: list[str]
    cwd: str
    status: ProcessStatus
    pid: int | None = None
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_path: str = ""
    stderr_path: str = ""
    status_path: str = ""
    started_at: str = ""
    ended_at: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class EnvironmentState:
    workspace: WorkspaceState = field(default_factory=WorkspaceState)
    shell: ShellSessionState = field(default_factory=ShellSessionState)
    processes: dict[str, ProcessRecord] = field(default_factory=dict)
    waiting: bool = False
    waiting_reason: str = ""
    waiting_process_ids: list[str] = field(default_factory=list)
    last_updated: str = ""
