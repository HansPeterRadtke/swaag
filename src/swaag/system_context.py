from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from swaag.config import AgentConfig
from swaag.environment.environment import AgentEnvironment
from swaag.notes import render_notes
from swaag.note_context import NoteContextManager
from swaag.types import Note, PromptAssembly, SessionState
from swaag.utils import stable_json_dumps

if TYPE_CHECKING:
    from swaag.runtime import AgentRuntime

NOTE_SELECTED_IDS = "durable_notes.selected_ids"
NOTE_SELECTION_ATTEMPTED = "durable_notes.selection_attempted"


@dataclass(frozen=True)
class SystemContextSource:
    name: str
    category: str
    text: str
    introduction: str
    locator: dict[str, object]
    projection_source_label: str
    projection_header: str
    optional: bool = True
    projectable: bool = True

    def render(self, *, projection: str | None = None) -> str:
        body = self.text if projection is None else self.projection_header + projection
        return self.introduction + body + "\n\n"


def _tool_enabled(config: AgentConfig, *names: str) -> bool:
    enabled = set(config.tools.enabled)
    return any(name in enabled for name in names)


def _selected_notes(
    state: SessionState, context_state: dict[str, object] | None
) -> list[Note]:
    if not context_state or NOTE_SELECTED_IDS not in context_state:
        return list(state.notes)
    raw_ids = context_state.get(NOTE_SELECTED_IDS, [])
    selected_ids = {str(value) for value in raw_ids} if isinstance(raw_ids, list) else set()
    return [note for note in state.notes if note.note_id in selected_ids]


def reduce_system_context_for_overflow(
    runtime: "AgentRuntime",
    state: SessionState,
    assembly: PromptAssembly,
    context_state: dict[str, object],
) -> bool:
    enabled = set(runtime.config.tools.enabled)
    if "notes" not in enabled or not state.notes:
        return False
    if bool(context_state.get(NOTE_SELECTION_ATTEMPTED, False)):
        return False
    context_state[NOTE_SELECTION_ATTEMPTED] = True
    current_ids = {note.note_id for note in _selected_notes(state, context_state)}
    selected = NoteContextManager(runtime).select_for_action(state, assembly)
    selected_ids = {note.note_id for note in selected}
    context_state[NOTE_SELECTED_IDS] = [note.note_id for note in selected]
    return selected_ids != current_ids


def runtime_system_context_sources(
    config: AgentConfig,
    state: SessionState,
    *,
    context_state: dict[str, object] | None = None,
) -> list[SystemContextSource]:
    sources: list[SystemContextSource] = []

    if _tool_enabled(config, "list_files"):
        filesystem = AgentEnvironment(config, state).filesystem
        files = filesystem.list_files(".")
        sources.append(
            SystemContextSource(
                name="workspace_file_manifest",
                category="environment",
                text=stable_json_dumps(
                    {
                        "workspace_root": state.environment.workspace.root,
                        "files": files,
                        "count": len(files),
                    },
                    indent=2,
                ),
                introduction=(
                    "Workspace file manifest. Use the configured filesystem capability to recover "
                    "the exact current listing when needed:\n"
                ),
                locator={
                    "authoritative_source": "live_filesystem",
                    "workspace_root": state.environment.workspace.root,
                    "recovery_tool": "list_files",
                    "recovery_arguments": {"path": state.environment.workspace.root},
                },
                projection_source_label="complete current workspace file manifest",
                projection_header=(
                    "[SEMANTIC PROJECTION; the live filesystem remains authoritative]\n"
                ),
            )
        )

    if _tool_enabled(config, "notes"):
        note_text = render_notes(_selected_notes(state, context_state))
        if note_text:
            sources.append(
                SystemContextSource(
                    name="durable_notes",
                    category="notes",
                    text=note_text,
                    introduction=(
                        "Durable model-authored notes. These are navigation aids; verbatim user "
                        "messages and tool results remain authoritative:\n"
                    ),
                    locator={
                        "authoritative_source": "durable_note_events",
                        "session_id": state.session_id,
                        "recovery_tool": "notes",
                        "recovery_arguments": {"action": "list"},
                    },
                    projection_source_label="all exact durable model-authored notes",
                    projection_header=(
                        "[SEMANTIC PROJECTION; exact notes remain authoritative and retrievable]\n"
                    ),
                )
            )

    if state.attachments and _tool_enabled(config, "list_attachments", "read_attachment"):
        references: list[dict[str, Any]] = []
        for attachment in state.attachments:
            payload = asdict(attachment)
            payload.pop("storage_ref", None)
            references.append(payload)
        sources.append(
            SystemContextSource(
                name="attachment_references",
                category="attachments",
                text=stable_json_dumps(references, indent=2),
                introduction=(
                    "Raw attachments available to this task. These are references and cheap mechanical "
                    "facts only; decide whether and how to inspect them with an attachment capability:\n"
                ),
                locator={
                    "authoritative_source": "attachment_events_and_storage",
                    "session_id": state.session_id,
                    "recovery_tool": "list_attachments",
                    "recovery_arguments": {},
                },
                projection_source_label="attachment references",
                projection_header="",
                optional=False,
                projectable=False,
            )
        )

    return sources
