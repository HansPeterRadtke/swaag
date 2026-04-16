from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from swaag.environment.state import EnvironmentState

Role = Literal["user", "assistant", "tool", "summary"]
ContractMode = Literal["plain", "gbnf", "json_schema"]
ModelCallKind = Literal[
    "analysis",
    "task_decision",
    "expansion",
    "strategy",
    "failure",
    "action",
    "control",
    "decision",
    "tool_input",
    "summary",
    "doctor",
    "answer",
    "plan",
    "verification",
    "subagent_selection",
    "generation_decomposition",
    "overflow_recovery",
]
ToolAction = Literal["respond", "call_tool"]
ToolKind = Literal["pure", "stateful", "side_effect"]
SourceKind = Literal["file", "buffer"]
TrustLevel = Literal["trusted", "untrusted", "derived"]
PlanStepKind = Literal["tool", "respond", "read", "write", "reasoning", "note"]
PlanStepStatus = Literal["pending", "running", "completed", "failed", "skipped"]
PlanStatus = Literal["active", "completed", "failed"]
MemoryKind = Literal["semantic", "procedural"]
PromptTaskType = Literal["structured", "unstructured", "vague", "incomplete", "already_decomposed"]
PromptCompleteness = Literal["complete", "partial", "incomplete"]
ExecutionAction = Literal["execute_step", "retry_step", "replan", "wait", "stop", "answer_directly"]
FailureKind = Literal[
    "tool_failure",
    "reasoning_failure",
    "planning_failure",
    "missing_information",
    "verification_failure",
    "budget_failure",
    "state_inconsistency",
    "transient_external_wait",
    "retry_now",
    "retry_later_backoff",
    "deterministic_permanent",
    "side_effect_unsafe",
    "needs_replan",
    "needs_clarification",
    "blocked_external",
    "continue_other",
]
VerificationType = Literal["execution", "structural", "value", "composite", "llm_fallback"]


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
class PlanStep:
    step_id: str
    title: str
    goal: str
    kind: PlanStepKind
    expected_tool: str | None
    input_text: str
    expected_output: str
    done_condition: str
    success_criteria: str
    expected_outputs: list[str] = field(default_factory=list)
    verification_type: VerificationType = "llm_fallback"
    verification_checks: list[dict[str, Any]] = field(default_factory=list)
    required_conditions: list[str] = field(default_factory=list)
    optional_conditions: list[str] = field(default_factory=list)
    input_refs: list[str] = field(default_factory=list)
    output_refs: list[str] = field(default_factory=list)
    fallback_strategy: str = ""
    depends_on: list[str] = field(default_factory=list)
    status: PlanStepStatus = "pending"
    last_updated: str = ""

    def __post_init__(self) -> None:
        if not self.expected_outputs and self.expected_output:
            self.expected_outputs = [self.expected_output]
        if not self.verification_checks:
            if self.kind in {"tool", "read", "write", "note"}:
                self.verification_type = "composite"
                self.verification_checks = [
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": self.expected_tool or ""},
                    {"name": "output_nonempty", "check_type": "tool_output_nonempty"},
                    {"name": "output_schema_valid", "check_type": "tool_output_schema_valid"},
                ]
                self.required_conditions = [item["name"] for item in self.verification_checks]
                self.optional_conditions = []
            else:
                self.verification_type = "composite"
                self.verification_checks = [
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {
                        "name": "assistant_text_nonempty" if self.kind == "respond" else "reasoning_text_nonempty",
                        "check_type": "string_nonempty",
                        "actual_source": "assistant_text",
                    },
                ]
                self.required_conditions = [item["name"] for item in self.verification_checks]
                self.optional_conditions = []
        elif not self.required_conditions:
            self.required_conditions = [str(item.get("name", "")).strip() for item in self.verification_checks if str(item.get("name", "")).strip()]


@dataclass(slots=True)
class Plan:
    plan_id: str
    goal: str
    steps: list[PlanStep]
    success_criteria: str
    fallback_strategy: str
    status: PlanStatus
    created_at: str
    updated_at: str
    current_step_id: str | None = None


@dataclass(slots=True)
class WorkingMemory:
    active_goal: str = ""
    current_step_id: str | None = None
    current_step_title: str = ""
    recent_results: list[str] = field(default_factory=list)
    active_entities: list[str] = field(default_factory=list)
    updated_at: str = ""


@dataclass(slots=True)
class SemanticMemoryItem:
    memory_id: str
    memory_kind: MemoryKind
    content: str
    source_event_id: str
    trust_level: TrustLevel
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryEntity:
    entity_id: str
    name: str
    entity_type: str
    source_event_id: str
    trust_level: TrustLevel
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryRelationship:
    relationship_id: str
    source_entity_id: str
    relation_type: str
    target_entity_id: str
    source_event_id: str
    trust_level: TrustLevel
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryFact:
    fact_id: str
    fact_type: str
    content: str
    source_event_id: str
    trust_level: TrustLevel
    confidence: float
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
    grammar: str | None = None
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
class PromptAnalysis:
    task_type: PromptTaskType
    completeness: PromptCompleteness
    requires_expansion: bool
    requires_decomposition: bool
    confidence: float
    detected_entities: list[str] = field(default_factory=list)
    detected_goals: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DecisionOutcome:
    split_task: bool
    expand_task: bool
    ask_user: bool
    assume_missing: bool
    generate_ideas: bool
    confidence: float
    reason: str
    direct_response: bool = False
    execution_mode: Literal["full_plan", "single_tool", "direct_response"] = "full_plan"
    preferred_tool_name: str = ""


@dataclass(slots=True)
class ExpandedTask:
    original_goal: str
    expanded_goal: str
    scope: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StrategySelection:
    strategy_name: str
    explore_before_commit: bool
    validate_assumptions: bool
    simplify_if_stuck: bool
    switch_on_failure: bool
    reason: str
    mode: str = "conservative"
    tool_chain_depth: int = 1
    verification_intensity: float = 1.0
    retry_same_action_limit: int = 1
    replan_after_failures: int = 1
    confidence_floor: float = 0.5
    task_profile: str = "generic"
    allowed_tools: list[str] = field(default_factory=list)
    required_step_kinds: list[str] = field(default_factory=list)
    expected_flow: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SubagentSelectionDecision:
    spawn: bool
    subagent_type: str
    reason: str
    focus: str = ""


@dataclass(slots=True)
class SessionMetrics:
    model_calls: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    verification_failures: int = 0
    verification_passes: int = 0
    retries: int = 0
    replans: int = 0
    steps_started: int = 0
    steps_completed: int = 0
    steps_failed: int = 0
    budget_rejections: int = 0
    token_estimate_uses: int = 0
    strategy_switches: int = 0
    action_count: int = 0
    input_tokens: int = 0
    reserved_response_tokens: int = 0
    successful_turns: int = 0
    failed_turns: int = 0
    tool_call_budget_hits: int = 0
    no_progress_stops: int = 0
    max_iteration_stops: int = 0
    total_cost_units: float = 0.0
    verification_success_rate: float = 0.0
    verification_failure_rate: float = 0.0
    llm_fallback_rate: float = 0.0
    model_request_progress_events: int = 0
    model_retry_events: int = 0
    post_validate_fallbacks: int = 0
    server_schema_requests: int = 0
    verification_type_distribution: dict[str, int] = field(default_factory=dict)
    last_reasoning_status: str = ""
    last_reasoning_reason: str = ""
    stop_reason_counts: dict[str, int] = field(default_factory=dict)
    failure_counts: dict[str, int] = field(default_factory=dict)
    tool_success_counts: dict[str, int] = field(default_factory=dict)
    tool_failure_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class AgentRoleDefinition:
    role_name: str
    responsibilities: list[str] = field(default_factory=list)
    can_plan: bool = False
    can_execute_tools: bool = False
    can_verify: bool = False


@dataclass(slots=True)
class AgentEnvelope:
    sender_role: str
    recipient_role: str
    purpose: str
    content: str
    created_at: str


@dataclass(slots=True)
class ProjectState:
    files_seen: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    directories: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    expected_artifacts: list[str] = field(default_factory=list)
    pending_artifacts: list[str] = field(default_factory=list)
    completed_artifacts: list[str] = field(default_factory=list)
    last_updated: str = ""


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
    active_plan: Plan | None = None
    prompt_analysis: PromptAnalysis | None = None
    latest_decision: DecisionOutcome | None = None
    expanded_task: ExpandedTask | None = None
    active_strategy: StrategySelection | None = None
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    semantic_memory: list[SemanticMemoryItem] = field(default_factory=list)
    semantic_entities: dict[str, MemoryEntity] = field(default_factory=dict)
    semantic_relationships: list[MemoryRelationship] = field(default_factory=list)
    semantic_facts: list[MemoryFact] = field(default_factory=list)
    procedural_patterns: list[SemanticMemoryItem] = field(default_factory=list)
    project_state: ProjectState = field(default_factory=ProjectState)
    environment: EnvironmentState = field(default_factory=EnvironmentState)
    deferred_tasks: list[DeferredTask] = field(default_factory=list)
    code_checkpoints: list[CodeCheckpoint] = field(default_factory=list)
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    agent_roles: list[AgentRoleDefinition] = field(default_factory=list)
    active_role: str = "primary"
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
    chunk_chars: int
    overlap_chars: int
    finished: bool
    text: str
