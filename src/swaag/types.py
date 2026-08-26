from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from swaag.environment.state import EnvironmentState


Role = Literal["user", "assistant", "tool", "summary"]
ContractMode = Literal["json_schema"]
ModelCallKind = Literal[
    "action",
    "summary",
    "tool_result_projection",
    "completion_evaluation",
    "caller_structured_output",
    "history_analysis",
    "doctor",
    "benchmark_quality_judge",
]
ToolAction = Literal["respond", "call_tool"]
ToolKind = Literal["pure", "stateful", "side_effect"]
SourceKind = Literal["file", "buffer"]
TrustLevel = Literal["trusted", "untrusted", "derived"]


@dataclass(slots=True)
class Message:
    role: Role
    content: str
    created_at: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Note:
    note_id: str
    title: str
    content: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReaderState:
    reader_id: str
    source_kind: SourceKind
    source_ref: str
    offset: int
    chunk_chars: int
    overlap_chars: int
    finished: bool
    last_chunk: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FileView:
    path: str
    content: str | None = None
    last_chunk_text: str = ""
    last_start_offset: int | None = None
    last_end_offset: int | None = None
    last_next_offset: int | None = None
    last_operation: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolInvocation:
    tool_name: str
    raw_input: dict[str, Any]
    validated_input: dict[str, Any]


@dataclass(slots=True)
class DerivedFileWrite:
    path: str
    content: str
    encoding: str = "utf-8"
    backup_content: str | None = None
    backup_suffix: str = ".bak"


@dataclass(slots=True)
class ToolGeneratedEvent:
    event_type: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    derived_writes: list[DerivedFileWrite] = field(default_factory=list)


@dataclass(slots=True)
class ToolExecutionResult:
    tool_name: str
    output: dict[str, Any]
    display_text: str
    generated_events: list[ToolGeneratedEvent] = field(default_factory=list)
    completed: bool = True


@dataclass(slots=True)
class PromptComponent:
    name: str
    text: str
    include_in_context: bool = True
    optional: bool = False
    category: str | None = None


@dataclass(slots=True)
class PromptAssembly:
    kind: ModelCallKind
    prompt_text: str
    components: list[PromptComponent]
    prompt_mode: str


@dataclass(slots=True)
class ContractSpec:
    name: str
    mode: ContractMode
    json_schema: dict[str, Any] | None = None


@dataclass(slots=True)
class BudgetComponentReport:
    name: str
    category: str
    tokens: int
    exact: bool
    include_in_context: bool
    optional: bool


@dataclass(slots=True)
class BudgetReport:
    context_limit: int
    input_tokens: int
    reserved_response_tokens: int
    safety_margin_tokens: int
    required_tokens: int
    non_context_tokens: int
    fits: bool
    exact: bool
    breakdown: list[BudgetComponentReport]


@dataclass(slots=True)
class CompletionResult:
    text: str
    raw_request: dict[str, Any]
    raw_response: dict[str, Any]
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    elapsed_seconds: float | None = None
    tokens_per_second: float | None = None
    first_token_seconds: float | None = None


@dataclass(slots=True)
class ToolDecision:
    action: ToolAction
    response: str
    tool_name: str
    tool_input: dict[str, Any]


@dataclass(slots=True)
class HistoryEvent:
    id: str
    sequence: int
    session_id: str
    timestamp: str
    type: str
    version: int
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    prev_hash: str | None = None
    hash: str = ""

    @property
    def event_type(self) -> str:
        return self.type


@dataclass(slots=True)
class SessionMetrics:
    model_calls: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    retries: int = 0
    budget_rejections: int = 0
    token_estimate_uses: int = 0
    action_count: int = 0
    input_tokens: int = 0
    reserved_response_tokens: int = 0
    successful_turns: int = 0
    failed_turns: int = 0
    tool_call_budget_hits: int = 0
    no_progress_stops: int = 0
    max_iteration_stops: int = 0
    model_request_progress_events: int = 0
    model_retry_events: int = 0
    unconstrained_contract_violations: int = 0
    server_schema_requests: int = 0
    failure_counts: dict[str, int] = field(default_factory=dict)
    tool_success_counts: dict[str, int] = field(default_factory=dict)
    tool_failure_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class DeferredTask:
    task_id: str
    text: str
    queued_at: str
    source: str = "control"


@dataclass(slots=True)
class CodeCheckpoint:
    checkpoint_id: str
    label: str
    created_at: str
    workspace_root: str
    storage_path: str
    file_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AttachmentReference:
    attachment_id: str
    original_name: str
    media_type: str
    size_bytes: int
    sha256: str
    storage_ref: str
    created_at: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionState:
    session_id: str
    created_at: str
    updated_at: str
    config_fingerprint: str
    model_base_url: str
    session_name: str = ""
    session_name_source: str = "placeholder"
    messages: list[Message] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    reader_states: dict[str, ReaderState] = field(default_factory=dict)
    file_views: dict[str, FileView] = field(default_factory=dict)
    pending_file_writes: dict[str, str] = field(default_factory=dict)
    environment: EnvironmentState = field(default_factory=EnvironmentState)
    deferred_tasks: list[DeferredTask] = field(default_factory=list)
    code_checkpoints: list[CodeCheckpoint] = field(default_factory=list)
    attachments: list[AttachmentReference] = field(default_factory=list)
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    turn_count: int = 0
    compaction_count: int = 0
    event_count: int = 0
    edit_count: int = 0
    last_event_hash: str | None = None


@dataclass(slots=True)
class NotePromptSelection:
    included_notes: list[Note]
    omitted_note_ids: list[str]
    rendered_text: str
    tokens: int
    exact: bool


@dataclass(slots=True)
class ReaderChunk:
    reader_id: str
    source_kind: SourceKind
    source_ref: str
    start_offset: int
    end_offset: int
    next_offset: int
    finished: bool
    text: str
