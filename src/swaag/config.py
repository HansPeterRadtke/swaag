from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from swaag.utils import expand_env_in_value, sha256_text

@dataclass(slots=True)
class ModelConfig:
    base_url: str
    completion_endpoint: str
    tokenize_endpoint: str
    health_endpoint: str
    profile_name: str
    max_semantic_responsibilities_per_call: int
    model_identity: str
    provider_name: str
    api_key_env: str
    structured_output_mode: str
    cache_enabled: bool
    cache_mode: str
    cache_path: str
    timeout_seconds: int
    connect_timeout_seconds: int
    simple_timeout_seconds: int
    structured_timeout_seconds: int
    verification_timeout_seconds: int
    benchmark_timeout_seconds: int
    fail_safe_timeout_seconds: int
    progress_poll_seconds: float
    max_retries: int
    temperature: float
    top_p: float
    seed: int
    context_limit: int
    remote_context_limit_fallback: int
    stop: list[str]
    cache_max_bytes: int = 134217728
    cache_max_entries: int = 16384
    cache_lock_stripes: int = 128


@dataclass(slots=True)
class ContextConfig:
    reserved_response_tokens: int
    reserved_summary_tokens: int
    safety_margin_tokens: int
    max_compaction_rounds: int
    allow_estimate_fallback: bool
    compact_on_overflow: bool
    semantic_reduction_max_input_tokens: int = 0
    semantic_reduction_max_calls: int = 256
    max_token_count_cache_entries: int = 4096


@dataclass(slots=True)
class RuntimeConfig:
    tool_timeout_seconds: int
    lean_on_overflow: bool
    max_validation_recovery_cycles: int
    completion_evaluation_enabled: bool
    max_pending_controls: int = 256
    max_pending_inference: int = 256
    max_open_questions: int = 128
    max_open_question_chars: int = 65536


@dataclass(slots=True)
class SessionConfig:
    root: Path
    write_projections: bool


@dataclass(slots=True)
class AgentDataConfig:
    root: Path
    sandbox_backend: str
    python_executable: str
    command_timeout_seconds: int
    max_capture_chars: int


@dataclass(slots=True)
class EnvironmentConfig:
    shell_executable: str
    max_capture_chars: int
    track_shell_file_changes: bool = True


@dataclass(slots=True)
class ToolConfig:
    enabled: list[str]
    read_roots: list[Path]
    allow_stateful_tools: bool
    allow_side_effect_tools: bool
    staged_discovery: bool


@dataclass(slots=True)
class PromptConfig:
    standard_system_template: str
    lean_system_template: str
    action_template: str
    lean_action_template: str
    summary_system_template: str
    summary_template: str
    tool_result_projection_system_template: str
    tool_result_projection_template: str
    evidence_projection_system_template: str
    evidence_projection_template: str
    completion_evaluation_system_template: str
    completion_evaluation_template: str
    caller_structured_output_system_template: str
    caller_structured_output_template: str
    communication_status_system_template: str
    communication_status_template: str
    response_relevance_system_template: str
    response_relevance_template: str
    audio_rendering_system_template: str
    audio_rendering_template: str
    presentation_evaluation_system_template: str
    presentation_evaluation_template: str
    note_selection_system_template: str
    note_selection_template: str
    prompt_instruction_selection_system_template: str
    prompt_instruction_selection_template: str
    prompt_instruction_projection_system_template: str
    prompt_instruction_projection_template: str


@dataclass(slots=True)
class LoggingConfig:
    level: str
    file_path: Path
    queue_capacity: int
    max_bytes: int
    backup_count: int


@dataclass(slots=True)
class NotesConfig:
    max_notes: int
    max_note_chars: int
    max_total_chars: int


@dataclass(slots=True)
class PromptInstructionsConfig:
    max_instructions: int
    max_instruction_chars: int
    max_total_chars: int


@dataclass(slots=True)
class ReaderConfig:
    default_chunk_chars: int
    default_overlap_chars: int
    max_chunk_chars: int


@dataclass(slots=True)
class EditorConfig:
    create_backups: bool
    backup_suffix: str
    allow_writes: bool
    allowed_write_paths: list[str]


@dataclass(slots=True)
class CompressionConfig:
    """Reserved namespace for future non-semantic compression mechanics."""

    pass


@dataclass(slots=True)
class BudgetPolicyConfig:
    call_classes: dict[str, str]
    output_ratio: dict[str, float]
    output_floor_ratio: dict[str, float]
    output_ratio_by_kind: dict[str, float]
    output_floor_ratio_by_kind: dict[str, float]
    safety_ratio: dict[str, float]
    structured_output_json_factor_by_contract: dict[str, float]
    structured_output_json_floor_by_contract: dict[str, int]
    structured_output_json_factor_default: float
    structured_output_json_floor_tokens: int
    structured_output_schema_factor: float
    structured_output_schema_floor_tokens: int


@dataclass(slots=True)
class HistorySearchConfig:
    max_results: int
    token_score: int
    exact_score: int
    type_bonus: int
    preview_chars: int




@dataclass(slots=True)
class EmbeddingIndexConfig:
    enabled: bool
    base_url: str
    endpoint: str
    model: str
    timeout_seconds: float
    fields: list[str]
    max_results: int


@dataclass(slots=True)
class AttachmentConfig:
    max_upload_bytes: int
    preview_chars: int


@dataclass(slots=True)
class McpAuthorizationConfig:
    enabled: bool
    resource_uri: str
    authorization_servers: list[str]
    allowed_origins: list[str]
    introspection_url: str
    introspection_client_id: str
    introspection_client_secret_env: str
    introspection_client_secret: str = field(repr=False)
    required_scopes: list[str] = field(default_factory=list)
    timeout_seconds: float = 5.0


@dataclass(slots=True)
class McpConfig:
    enabled: bool
    transport: str
    authorization: McpAuthorizationConfig


@dataclass(slots=True)
class ExternalMcpServerConfig:
    enabled: bool
    optional: bool
    transport: str
    command: list[str]
    url: str
    header_env: dict[str, str]
    credential_command: list[str]
    credential_refresh_skew_seconds: float
    timeout_seconds: float


@dataclass(slots=True)
class ExternalToolsConfig:
    mcp_servers: dict[str, ExternalMcpServerConfig]


@dataclass(slots=True)
class A2AAuthorizationConfig:
    enabled: bool
    public_base_url: str
    bearer_token_env: str
    bearer_token: str = field(repr=False)


@dataclass(slots=True)
class A2APushConfig:
    enabled: bool
    credential_key_env: str
    allowed_hosts: list[str]
    timeout_seconds: float
    max_attempts: int
    retry_base_seconds: float


@dataclass(slots=True)
class A2AExtendedCardConfig:
    enabled: bool


@dataclass(slots=True)
class A2ACardSigningConfig:
    enabled: bool
    private_key_env: str
    key_id: str
    jwks_url: str


@dataclass(slots=True)
class OpenWebUiArtifactServingConfig:
    enabled: bool
    public_base_url: str
    signing_secret_env: str
    ttl_seconds: int


@dataclass(slots=True)
class CommunicationConfig:
    enabled: bool
    model_base_url: str
    model_profile_name: str
    model_identity: str
    remote_context_limit_fallback: int
    max_concurrent_requests: int
    status_max_output_tokens: int
    enabled_tools: list[str]
    host: str
    port: int
    poll_seconds: float
    max_active_workers: int = 128
    max_pending_requests: int = 128
    idle_work_mode: str = "finish_only"
    max_background_plans: int = 128
    model_routes: dict[str, str] = field(default_factory=dict)
    open_webui_artifacts: OpenWebUiArtifactServingConfig = field(
        default_factory=lambda: OpenWebUiArtifactServingConfig(False, "", "", 900)
    )

@dataclass(slots=True)
class ExternalBenchmarkTargetConfig:
    enabled: bool
    description: str
    workdir: str
    default_variables: dict[str, str]
    preflight_commands: list[list[str]]
    smoke_command: list[str]
    full_command: list[str]
    required_env: list[str]
    required_paths: list[str]
    allowed_path_literals: list[str]
    artifact_globs: list[str]


@dataclass(slots=True)
class ExternalBenchmarkAgentGenerationConfig:
    default_max_instances: int
    clone_timeout_seconds: int
    agent_timeout_seconds: int
    model_timeout_seconds: int
    model_structured_timeout_seconds: int
    allow_stateful_tools: bool
    allow_side_effect_tools: bool
    solver_max_attempts: int
    git_remote_base_url: str
    model_name_or_path: str
    prompt_template: str
    empty_patch_retry_prompt: str


@dataclass(slots=True)
class ExternalBenchmarkModelServerConfig:
    preflight_enabled: bool
    healthcheck_timeout_seconds: int
    retry_attempts: int
    retry_sleep_seconds: float


@dataclass(slots=True)
class ExternalBenchmarkTerminalBenchConfig:
    compose_probe_timeout_seconds: int
    compose_download_timeout_seconds: int
    allow_compose_download: bool


@dataclass(slots=True)
class ExternalBenchmarksConfig:
    root: Path
    smoke_timeout_seconds: int
    full_timeout_seconds: int
    model_server: ExternalBenchmarkModelServerConfig
    terminal_bench: ExternalBenchmarkTerminalBenchConfig
    agent_generation: ExternalBenchmarkAgentGenerationConfig
    targets: dict[str, ExternalBenchmarkTargetConfig]


@dataclass(slots=True)
class AgentConfig:
    model: ModelConfig
    context: ContextConfig
    runtime: RuntimeConfig
    sessions: SessionConfig
    agent_data: AgentDataConfig
    environment: EnvironmentConfig
    tools: ToolConfig
    prompts: PromptConfig
    logging: LoggingConfig
    notes: NotesConfig
    prompt_instructions: PromptInstructionsConfig
    reader: ReaderConfig
    editor: EditorConfig
    compression: CompressionConfig
    history_search: HistorySearchConfig
    embedding_index: EmbeddingIndexConfig
    attachments: AttachmentConfig
    mcp: McpConfig
    external_tools: ExternalToolsConfig
    a2a_authorization: A2AAuthorizationConfig
    a2a_push: A2APushConfig
    a2a_extended_card: A2AExtendedCardConfig
    a2a_card_signing: A2ACardSigningConfig
    communication: CommunicationConfig
    budget_policy: BudgetPolicyConfig
    external_benchmarks: ExternalBenchmarksConfig
    raw: dict[str, Any] = field(repr=False)
    sources: dict[str, str] = field(default_factory=dict, repr=False)

    def source_for(self, dotted_key: str) -> str:
        return self.sources.get(str(dotted_key).strip(), "unknown")

    def parameter_metadata(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        packaged_defaults = _load_packaged_defaults()
        for dotted in _leaf_paths(self.raw):
            value = _dotted_value(self.raw, dotted)
            default_value = _dotted_value(packaged_defaults, dotted)
            key = dotted.rsplit(".", 1)[-1]
            group = dotted.split(".", 1)[0]
            unit = _parameter_unit(key)
            security_sensitive = any(
                marker in dotted.casefold()
                for marker in ("authorization", "credential", "secret", "private_key", "allowed_hosts")
            )
            resource_sensitive = any(
                marker in key.casefold()
                for marker in ("timeout", "limit", "max_", "capacity", "bytes", "tokens", "port")
            )
            criticality = (
                "critical" if security_sensitive else "major" if resource_sensitive else "normal"
            )
            likelihood = (
                "deployment-specific"
                if any(marker in dotted for marker in ("host", "port", "base_url", "root", "path"))
                else "occasionally tuned" if resource_sensitive else "rare"
            )
            description, consequences = _parameter_semantics(
                dotted,
                key=key,
                group=group,
                security_sensitive=security_sensitive,
                resource_sensitive=resource_sensitive,
            )
            result[dotted] = {
                "group": group,
                "type": type(value).__name__,
                "unit": unit,
                "criticality": criticality,
                "change_likelihood": likelihood,
                "source": self.source_for(dotted),
                "default_value": default_value,
                "valid_range": _parameter_range(key, value),
                "description": description,
                "consequences": consequences,
            }
        return result

    def config_fingerprint(self) -> str:
        fingerprint_data = json.loads(json.dumps(self.raw))
        mcp_auth = fingerprint_data.get("mcp", {}).get("authorization", {})
        if isinstance(mcp_auth, dict) and "introspection_client_secret" in mcp_auth:
            mcp_auth["introspection_client_secret"] = "[CREDENTIAL]"
        a2a_auth = fingerprint_data.get("a2a", {}).get("authorization", {})
        if isinstance(a2a_auth, dict) and "bearer_token" in a2a_auth:
            a2a_auth["bearer_token"] = "[CREDENTIAL]"
        return sha256_text(json.dumps(fingerprint_data, sort_keys=True))


def _migrate_config_aliases(data: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(data)
    runtime = migrated.get("runtime")
    if isinstance(runtime, dict) and "max_repeated_action_occurrences" in runtime:
        runtime = dict(runtime)
        if "max_validation_recovery_cycles" not in runtime:
            runtime["max_validation_recovery_cycles"] = runtime["max_repeated_action_occurrences"]
        runtime.pop("max_repeated_action_occurrences", None)
        migrated["runtime"] = runtime
    return migrated


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_env_value(text: str) -> Any:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _apply_env_overrides(data: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    result = dict(data)
    prefix = "SWAAG__"
    for key, value in env.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("__")
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = _parse_env_value(value)
    return result


def _load_toml_file(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_packaged_defaults() -> dict[str, Any]:
    resource = resources.files("swaag").joinpath("assets/defaults.toml")
    with resource.open("rb") as handle:
        return tomllib.load(handle)


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _coerce_config(
    data: dict[str, Any],
    *,
    sources: dict[str, str] | None = None,
    secret_env: dict[str, str] | None = None,
) -> AgentConfig:
    data = expand_env_in_value(data)
    removed_runtime = {
        "tool_call_budget": "benchmark-only tool limits belong in the benchmark execution guard; production work remains cancelable and unbounded",
        "max_total_actions": "benchmark-only action limits belong in the benchmark execution guard; production work remains cancelable and unbounded",
        "verification_confidence_threshold": "semantic verification is model-owned and is not decided by a deterministic confidence threshold",
        "capture_model_io": "canonical execution evidence is retained by the runtime and cannot be disabled by a pseudo-toggle",
        "strict_budget": "the model context boundary is a mandatory invariant and cannot be disabled",
        "background_poll_seconds": "background dispatch uses communication.poll_seconds; this duplicate setting had no runtime effect",
    }
    runtime_input = data.get("runtime", {})
    if isinstance(runtime_input, dict):
        for key, reason in removed_runtime.items():
            if key in runtime_input:
                raise ValueError(f"runtime.{key} is obsolete: {reason}")
    if "archive" in data:
        raise ValueError(
            "archive configuration is obsolete: SWAAG has explicit worker/artifact archive "
            "operations, but no automatic age/count archive policy; do not expose settings "
            "that have no runtime effect"
        )
    environment_input = data.get("environment", {})
    if isinstance(environment_input, dict) and "command_timeout_seconds" in environment_input:
        raise ValueError(
            "environment.command_timeout_seconds is obsolete: shell/process execution uses "
            "runtime.tool_timeout_seconds as the single source of truth"
        )
    model = ModelConfig(**data["model"])
    context_data = dict(data["context"])
    context_data.pop("max_recent_messages", None)
    context_data.pop("workspace_manifest_max_files", None)
    context_data.pop("note_prompt_token_cap", None)
    context = ContextConfig(**context_data)
    runtime_data = dict(data["runtime"])
    runtime_data.pop("max_repeated_action_occurrences", None)
    runtime = RuntimeConfig(**runtime_data)
    sessions = SessionConfig(
        root=Path(data["sessions"]["root"]).expanduser(),
        write_projections=bool(data["sessions"]["write_projections"]),
    )
    agent_data = AgentDataConfig(
        root=Path(data["agent_data"]["root"]).expanduser(),
        sandbox_backend=str(data["agent_data"]["sandbox_backend"]),
        python_executable=str(data["agent_data"]["python_executable"]),
        command_timeout_seconds=int(data["agent_data"]["command_timeout_seconds"]),
        max_capture_chars=int(data["agent_data"]["max_capture_chars"]),
    )
    environment = EnvironmentConfig(**data["environment"])
    tools = ToolConfig(
        enabled=list(data["tools"]["enabled"]),
        read_roots=[Path(item).expanduser() for item in data["tools"]["read_roots"]],
        allow_stateful_tools=bool(data["tools"]["allow_stateful_tools"]),
        allow_side_effect_tools=bool(data["tools"]["allow_side_effect_tools"]),
        staged_discovery=bool(data["tools"].get("staged_discovery", True)),
    )
    prompts = PromptConfig(**data["prompts"])
    logging_cfg = LoggingConfig(
        level=str(data["logging"]["level"]),
        file_path=Path(data["logging"]["file_path"]).expanduser(),
        queue_capacity=int(data["logging"]["queue_capacity"]),
        max_bytes=int(data["logging"]["max_bytes"]),
        backup_count=int(data["logging"]["backup_count"]),
    )
    notes_data = dict(data["notes"])
    notes_data.pop("compact_target_chars", None)
    notes = NotesConfig(**notes_data)
    prompt_instructions = PromptInstructionsConfig(**data["prompt_instructions"])
    reader = ReaderConfig(**data["reader"])
    editor = EditorConfig(**data["editor"])
    compression = CompressionConfig()
    budget_policy = BudgetPolicyConfig(
        call_classes={str(key): str(value) for key, value in data["budget_policy"]["call_classes"].items()},
        output_ratio={str(key): float(value) for key, value in data["budget_policy"]["output_ratio"].items()},
        output_floor_ratio={str(key): float(value) for key, value in data["budget_policy"]["output_floor_ratio"].items()},
        output_ratio_by_kind={str(key): float(value) for key, value in data["budget_policy"]["output_ratio_by_kind"].items()},
        output_floor_ratio_by_kind={str(key): float(value) for key, value in data["budget_policy"]["output_floor_ratio_by_kind"].items()},
        safety_ratio={str(key): float(value) for key, value in data["budget_policy"]["safety_ratio"].items()},
        structured_output_json_factor_by_contract={
            str(key): float(value)
            for key, value in data["budget_policy"]["structured_output_json_factor_by_contract"].items()
        },
        structured_output_json_floor_by_contract={
            str(key): int(value)
            for key, value in data["budget_policy"].get("structured_output_json_floor_by_contract", {}).items()
        },
        structured_output_json_factor_default=float(data["budget_policy"]["structured_output_json_factor_default"]),
        structured_output_json_floor_tokens=int(data["budget_policy"]["structured_output_json_floor_tokens"]),
        structured_output_schema_factor=float(data["budget_policy"]["structured_output_schema_factor"]),
        structured_output_schema_floor_tokens=int(data["budget_policy"]["structured_output_schema_floor_tokens"]),
    )
    history_search = HistorySearchConfig(
        max_results=int(data["history_search"]["max_results"]),
        token_score=int(data["history_search"]["token_score"]),
        exact_score=int(data["history_search"]["exact_score"]),
        type_bonus=int(data["history_search"]["type_bonus"]),
        preview_chars=int(data["history_search"]["preview_chars"]),
    )
    embedding_index = EmbeddingIndexConfig(
        enabled=bool(data["embedding_index"]["enabled"]),
        base_url=str(data["embedding_index"]["base_url"]),
        endpoint=str(data["embedding_index"]["endpoint"]),
        model=str(data["embedding_index"]["model"]),
        timeout_seconds=float(data["embedding_index"]["timeout_seconds"]),
        fields=[str(item) for item in data["embedding_index"]["fields"]],
        max_results=int(data["embedding_index"]["max_results"]),
    )
    attachments = AttachmentConfig(
        max_upload_bytes=int(data["attachments"]["max_upload_bytes"]),
        preview_chars=int(data["attachments"]["preview_chars"]),
    )
    mcp_auth_data = data["mcp"].get("authorization", {})
    literal_mcp_secret = str(mcp_auth_data.get("introspection_client_secret", ""))
    if literal_mcp_secret:
        raise ValueError(
            "mcp.authorization.introspection_client_secret literal secrets are not permitted; configure introspection_client_secret_env and supply the secret through that environment variable"
        )
    mcp_secret_env = str(
        mcp_auth_data.get(
            "introspection_client_secret_env",
            "SWAAG_MCP_INTROSPECTION_CLIENT_SECRET",
        )
    ).strip()
    resolved_secret_env = dict(os.environ if secret_env is None else secret_env)
    mcp = McpConfig(
        enabled=bool(data["mcp"]["enabled"]),
        transport=str(data["mcp"]["transport"]),
        authorization=McpAuthorizationConfig(
            enabled=bool(mcp_auth_data.get("enabled", False)),
            resource_uri=str(mcp_auth_data.get("resource_uri", "")),
            authorization_servers=[str(item) for item in mcp_auth_data.get("authorization_servers", [])],
            allowed_origins=[str(item) for item in mcp_auth_data.get("allowed_origins", [])],
            introspection_url=str(mcp_auth_data.get("introspection_url", "")),
            introspection_client_id=str(mcp_auth_data.get("introspection_client_id", "")),
            introspection_client_secret_env=mcp_secret_env,
            introspection_client_secret=str(resolved_secret_env.get(mcp_secret_env, "")),
            required_scopes=[str(item) for item in mcp_auth_data.get("required_scopes", [])],
            timeout_seconds=float(mcp_auth_data.get("timeout_seconds", 5.0)),
        ),
    )
    external_tools_data = data.get("external_tools", {})
    raw_mcp_servers = external_tools_data.get("mcp_servers", {})
    if not isinstance(raw_mcp_servers, dict):
        raise ValueError("external_tools.mcp_servers must be a table")
    external_tools = ExternalToolsConfig(
        mcp_servers={
            str(name): ExternalMcpServerConfig(
                enabled=bool(payload.get("enabled", True)),
                optional=bool(payload.get("optional", True)),
                transport=str(payload.get("transport", "stdio")),
                command=[str(item) for item in payload.get("command", [])],
                url=str(payload.get("url", "")),
                header_env={str(k): str(v) for k, v in payload.get("header_env", {}).items()}
                if isinstance(payload.get("header_env", {}), dict)
                else {},
                credential_command=[str(item) for item in payload.get("credential_command", [])],
                credential_refresh_skew_seconds=float(
                    payload.get("credential_refresh_skew_seconds", 30.0)
                ),
                timeout_seconds=float(payload.get("timeout_seconds", 30.0)),
            )
            for name, payload in raw_mcp_servers.items()
            if isinstance(payload, dict)
        }
    )

    a2a_auth_data = data.get("a2a", {}).get("authorization", {})
    literal_bearer = str(a2a_auth_data.get("bearer_token", ""))
    if literal_bearer:
        raise ValueError(
            "a2a.authorization.bearer_token literal secrets are not permitted; "
            "configure bearer_token_env and supply the secret through that environment variable"
        )
    bearer_token_env = str(
        a2a_auth_data.get("bearer_token_env", "SWAAG_A2A_BEARER_TOKEN")
    ).strip()
    resolved_secret_env = dict(os.environ if secret_env is None else secret_env)
    a2a_authorization = A2AAuthorizationConfig(
        enabled=bool(a2a_auth_data.get("enabled", False)),
        public_base_url=str(a2a_auth_data.get("public_base_url", "")),
        bearer_token_env=bearer_token_env,
        bearer_token=str(resolved_secret_env.get(bearer_token_env, "")),
    )

    a2a_data = data.get("a2a", {})
    a2a_push_data = a2a_data.get("push", {})
    a2a_push = A2APushConfig(
        enabled=bool(a2a_push_data.get("enabled", False)),
        credential_key_env=str(a2a_push_data.get("credential_key_env", "")),
        allowed_hosts=[str(item) for item in a2a_push_data.get("allowed_hosts", [])],
        timeout_seconds=float(a2a_push_data.get("timeout_seconds", 15.0)),
        max_attempts=int(a2a_push_data.get("max_attempts", 5)),
        retry_base_seconds=float(a2a_push_data.get("retry_base_seconds", 1.0)),
    )
    a2a_extended_card = A2AExtendedCardConfig(
        enabled=bool(a2a_data.get("extended_card", {}).get("enabled", False))
    )
    signing_data = a2a_data.get("card_signing", {})
    a2a_card_signing = A2ACardSigningConfig(
        enabled=bool(signing_data.get("enabled", False)),
        private_key_env=str(signing_data.get("private_key_env", "")),
        key_id=str(signing_data.get("key_id", "")),
        jwks_url=str(signing_data.get("jwks_url", "")),
    )

    communication = CommunicationConfig(
        enabled=bool(data["communication"]["enabled"]),
        model_base_url=str(data["communication"]["model_base_url"]),
        model_profile_name=str(data["communication"].get("model_profile_name", "")),
        model_identity=str(data["communication"].get("model_identity", "")),
        remote_context_limit_fallback=int(data["communication"].get("remote_context_limit_fallback", 0)),
        max_concurrent_requests=int(data["communication"]["max_concurrent_requests"]),
        status_max_output_tokens=int(data["communication"].get("status_max_output_tokens", 192)),
        enabled_tools=[str(item) for item in data["communication"]["enabled_tools"]],
        host=str(data["communication"]["host"]),
        port=int(data["communication"]["port"]),
        poll_seconds=float(data["communication"]["poll_seconds"]),
        max_active_workers=data["communication"].get("max_active_workers", 128),
        max_pending_requests=data["communication"].get("max_pending_requests", 128),
        idle_work_mode=str(data["communication"].get("idle_work_mode", "finish_only")),
        max_background_plans=int(data["communication"].get("max_background_plans", 128)),
        model_routes={
            str(name): str(url)
            for name, url in data["communication"].get("model_routes", {}).items()
        },
        open_webui_artifacts=OpenWebUiArtifactServingConfig(
            enabled=bool(
                data["communication"].get("open_webui_artifacts", {}).get(
                    "enabled", False
                )
            ),
            public_base_url=str(
                data["communication"].get("open_webui_artifacts", {}).get(
                    "public_base_url", ""
                )
            ),
            signing_secret_env=str(
                data["communication"].get("open_webui_artifacts", {}).get(
                    "signing_secret_env", ""
                )
            ),
            ttl_seconds=int(
                data["communication"].get("open_webui_artifacts", {}).get(
                    "ttl_seconds", 900
                )
            ),
        ),
    )
    external_benchmarks = ExternalBenchmarksConfig(
        root=Path(data["external_benchmarks"]["root"]).expanduser(),
        smoke_timeout_seconds=int(data["external_benchmarks"]["smoke_timeout_seconds"]),
        full_timeout_seconds=int(data["external_benchmarks"]["full_timeout_seconds"]),
        model_server=ExternalBenchmarkModelServerConfig(
            preflight_enabled=bool(data["external_benchmarks"]["model_server"]["preflight_enabled"]),
            healthcheck_timeout_seconds=int(data["external_benchmarks"]["model_server"]["healthcheck_timeout_seconds"]),
            retry_attempts=int(data["external_benchmarks"]["model_server"]["retry_attempts"]),
            retry_sleep_seconds=float(data["external_benchmarks"]["model_server"]["retry_sleep_seconds"]),
        ),
        terminal_bench=ExternalBenchmarkTerminalBenchConfig(
            compose_probe_timeout_seconds=int(data["external_benchmarks"]["terminal_bench"]["compose_probe_timeout_seconds"]),
            compose_download_timeout_seconds=int(data["external_benchmarks"]["terminal_bench"]["compose_download_timeout_seconds"]),
            allow_compose_download=bool(data["external_benchmarks"]["terminal_bench"]["allow_compose_download"]),
        ),
        agent_generation=ExternalBenchmarkAgentGenerationConfig(
            default_max_instances=int(data["external_benchmarks"]["agent_generation"]["default_max_instances"]),
            clone_timeout_seconds=int(data["external_benchmarks"]["agent_generation"]["clone_timeout_seconds"]),
            agent_timeout_seconds=int(data["external_benchmarks"]["agent_generation"]["agent_timeout_seconds"]),
            model_timeout_seconds=int(data["external_benchmarks"]["agent_generation"]["model_timeout_seconds"]),
            model_structured_timeout_seconds=int(
                data["external_benchmarks"]["agent_generation"]["model_structured_timeout_seconds"]
            ),
            allow_stateful_tools=bool(data["external_benchmarks"]["agent_generation"]["allow_stateful_tools"]),
            allow_side_effect_tools=bool(data["external_benchmarks"]["agent_generation"]["allow_side_effect_tools"]),
            solver_max_attempts=int(data["external_benchmarks"]["agent_generation"]["solver_max_attempts"]),
            git_remote_base_url=str(data["external_benchmarks"]["agent_generation"]["git_remote_base_url"]),
            model_name_or_path=str(data["external_benchmarks"]["agent_generation"]["model_name_or_path"]),
            prompt_template=str(data["external_benchmarks"]["agent_generation"]["prompt_template"]),
            empty_patch_retry_prompt=str(data["external_benchmarks"]["agent_generation"]["empty_patch_retry_prompt"]),
        ),
        targets={
            str(target_id): ExternalBenchmarkTargetConfig(
                enabled=bool(target_payload["enabled"]),
                description=str(target_payload["description"]),
                workdir=str(target_payload["workdir"]),
                default_variables={
                    str(key): str(value)
                    for key, value in target_payload.get("default_variables", {}).items()
                },
                preflight_commands=[
                    [str(item) for item in command]
                    for command in target_payload.get("preflight_commands", [])
                ],
                smoke_command=[str(item) for item in target_payload["smoke_command"]],
                full_command=[str(item) for item in target_payload["full_command"]],
                required_env=[str(item) for item in target_payload["required_env"]],
                required_paths=[str(item) for item in target_payload["required_paths"]],
                allowed_path_literals=[str(item) for item in target_payload["allowed_path_literals"]],
                artifact_globs=[str(item) for item in target_payload["artifact_globs"]],
            )
            for target_id, target_payload in data["external_benchmarks"]["targets"].items()
        },
    )

    _validate_positive("model.context_limit", model.context_limit)
    _validate_positive(
        "model.max_semantic_responsibilities_per_call",
        model.max_semantic_responsibilities_per_call,
    )
    _validate_non_negative(
        "model.remote_context_limit_fallback",
        model.remote_context_limit_fallback,
    )
    _validate_positive("model.timeout_seconds", model.timeout_seconds)
    _validate_positive("model.connect_timeout_seconds", model.connect_timeout_seconds)
    _validate_positive("model.simple_timeout_seconds", model.simple_timeout_seconds)
    _validate_positive("model.structured_timeout_seconds", model.structured_timeout_seconds)
    _validate_positive("model.verification_timeout_seconds", model.verification_timeout_seconds)
    _validate_positive("model.benchmark_timeout_seconds", model.benchmark_timeout_seconds)
    _validate_positive("model.fail_safe_timeout_seconds", model.fail_safe_timeout_seconds)
    if model.progress_poll_seconds <= 0:
        raise ValueError("model.progress_poll_seconds must be positive")
    if model.structured_output_mode != "server_schema":
        raise ValueError("model.structured_output_mode must be server_schema")
    if not model.provider_name.strip():
        raise ValueError("model.provider_name must not be empty")
    if model.cache_mode not in {"record", "replay"}:
        raise ValueError("model.cache_mode must be record or replay")
    _validate_positive("context.reserved_response_tokens", context.reserved_response_tokens)
    _validate_positive("context.reserved_summary_tokens", context.reserved_summary_tokens)
    _validate_non_negative("context.safety_margin_tokens", context.safety_margin_tokens)
    _validate_non_negative(
        "context.semantic_reduction_max_input_tokens",
        context.semantic_reduction_max_input_tokens,
    )
    _validate_positive(
        "context.semantic_reduction_max_calls",
        context.semantic_reduction_max_calls,
    )
    _validate_positive("environment.max_capture_chars", environment.max_capture_chars)
    _validate_positive("runtime.tool_timeout_seconds", runtime.tool_timeout_seconds)
    _validate_positive("runtime.max_validation_recovery_cycles", runtime.max_validation_recovery_cycles)
    from swaag.capacity import positive_capacity
    for capacity_name, capacity_value in (
        ("model.cache_max_bytes", model.cache_max_bytes),
        ("model.cache_max_entries", model.cache_max_entries),
        ("model.cache_lock_stripes", model.cache_lock_stripes),
        ("context.max_token_count_cache_entries", context.max_token_count_cache_entries),
        ("runtime.max_pending_controls", runtime.max_pending_controls),
        ("runtime.max_pending_inference", runtime.max_pending_inference),
        ("communication.max_active_workers", communication.max_active_workers),
        ("communication.max_pending_requests", communication.max_pending_requests),
    ):
        positive_capacity(capacity_value, capacity_name)
    _validate_positive("runtime.max_open_questions", runtime.max_open_questions)
    _validate_positive("runtime.max_open_question_chars", runtime.max_open_question_chars)
    _validate_positive("notes.max_notes", notes.max_notes)
    _validate_positive("notes.max_note_chars", notes.max_note_chars)
    _validate_positive("notes.max_total_chars", notes.max_total_chars)
    _validate_positive(
        "prompt_instructions.max_instructions",
        prompt_instructions.max_instructions,
    )
    _validate_positive(
        "prompt_instructions.max_instruction_chars",
        prompt_instructions.max_instruction_chars,
    )
    _validate_positive(
        "prompt_instructions.max_total_chars",
        prompt_instructions.max_total_chars,
    )
    _validate_positive("reader.default_chunk_chars", reader.default_chunk_chars)
    _validate_non_negative("reader.default_overlap_chars", reader.default_overlap_chars)
    _validate_positive("reader.max_chunk_chars", reader.max_chunk_chars)
    _validate_positive("budget_policy.structured_output_json_floor_tokens", budget_policy.structured_output_json_floor_tokens)
    for contract_name, floor in budget_policy.structured_output_json_floor_by_contract.items():
        _validate_positive(
            f"budget_policy.structured_output_json_floor_by_contract.{contract_name}",
            floor,
        )
    _validate_positive("budget_policy.structured_output_schema_floor_tokens", budget_policy.structured_output_schema_floor_tokens)
    if budget_policy.structured_output_json_factor_default <= 0:
        raise ValueError("budget_policy.structured_output_json_factor_default must be positive")
    if budget_policy.structured_output_schema_factor <= 0:
        raise ValueError("budget_policy.structured_output_schema_factor must be positive")
    if not external_benchmarks.targets:
        raise ValueError("external_benchmarks.targets must not be empty")
    _validate_positive("external_benchmarks.smoke_timeout_seconds", external_benchmarks.smoke_timeout_seconds)
    _validate_positive("external_benchmarks.full_timeout_seconds", external_benchmarks.full_timeout_seconds)
    _validate_positive(
        "external_benchmarks.model_server.healthcheck_timeout_seconds",
        external_benchmarks.model_server.healthcheck_timeout_seconds,
    )
    _validate_positive(
        "external_benchmarks.model_server.retry_attempts",
        external_benchmarks.model_server.retry_attempts,
    )
    if external_benchmarks.model_server.retry_sleep_seconds < 0:
        raise ValueError("external_benchmarks.model_server.retry_sleep_seconds must be non-negative")
    _validate_positive(
        "external_benchmarks.terminal_bench.compose_probe_timeout_seconds",
        external_benchmarks.terminal_bench.compose_probe_timeout_seconds,
    )
    _validate_positive(
        "external_benchmarks.terminal_bench.compose_download_timeout_seconds",
        external_benchmarks.terminal_bench.compose_download_timeout_seconds,
    )
    _validate_positive(
        "external_benchmarks.agent_generation.default_max_instances",
        external_benchmarks.agent_generation.default_max_instances,
    )
    _validate_positive(
        "external_benchmarks.agent_generation.clone_timeout_seconds",
        external_benchmarks.agent_generation.clone_timeout_seconds,
    )
    _validate_positive(
        "external_benchmarks.agent_generation.agent_timeout_seconds",
        external_benchmarks.agent_generation.agent_timeout_seconds,
    )
    _validate_positive(
        "external_benchmarks.agent_generation.model_timeout_seconds",
        external_benchmarks.agent_generation.model_timeout_seconds,
    )
    _validate_positive(
        "external_benchmarks.agent_generation.model_structured_timeout_seconds",
        external_benchmarks.agent_generation.model_structured_timeout_seconds,
    )
    _validate_positive(
        "external_benchmarks.agent_generation.solver_max_attempts",
        external_benchmarks.agent_generation.solver_max_attempts,
    )
    if not external_benchmarks.agent_generation.git_remote_base_url:
        raise ValueError("external_benchmarks.agent_generation.git_remote_base_url must not be empty")
    if not external_benchmarks.agent_generation.model_name_or_path:
        raise ValueError("external_benchmarks.agent_generation.model_name_or_path must not be empty")
    if not external_benchmarks.agent_generation.prompt_template.strip():
        raise ValueError("external_benchmarks.agent_generation.prompt_template must not be empty")
    if not external_benchmarks.agent_generation.empty_patch_retry_prompt.strip():
        raise ValueError("external_benchmarks.agent_generation.empty_patch_retry_prompt must not be empty")
    if reader.default_overlap_chars >= reader.default_chunk_chars:
        raise ValueError("reader.default_overlap_chars must be smaller than reader.default_chunk_chars")
    if not tools.enabled:
        raise ValueError("tools.enabled must not be empty")
    _validate_positive("embedding_index.max_results", embedding_index.max_results)
    if embedding_index.enabled and (not embedding_index.base_url or not embedding_index.model):
        raise ValueError("embedding_index.base_url and embedding_index.model are required when embeddings are enabled")
    _validate_positive("attachments.max_upload_bytes", attachments.max_upload_bytes)
    _validate_positive("attachments.preview_chars", attachments.preview_chars)
    if communication.idle_work_mode not in {"finish_only", "authorized_backlog"}:
        raise ValueError("communication.idle_work_mode must be finish_only or authorized_backlog")
    _validate_positive("communication.max_background_plans", communication.max_background_plans)
    _validate_positive("communication.max_concurrent_requests", communication.max_concurrent_requests)
    _validate_positive(
        "communication.status_max_output_tokens",
        communication.status_max_output_tokens,
    )
    _validate_positive("communication.port", communication.port)
    if not 1 <= communication.port <= 65535:
        raise ValueError("communication.port must be between 1 and 65535")
    if communication.poll_seconds <= 0:
        raise ValueError("communication.poll_seconds must be positive")
    artifact_serving = communication.open_webui_artifacts
    if artifact_serving.enabled:
        parsed_artifact_url = urlparse(artifact_serving.public_base_url)
        host = (parsed_artifact_url.hostname or "").casefold()
        loopback = host in {"localhost", "127.0.0.1", "::1"}
        if (
            parsed_artifact_url.scheme not in {"http", "https"}
            or not parsed_artifact_url.netloc
            or parsed_artifact_url.username is not None
            or parsed_artifact_url.password is not None
            or parsed_artifact_url.query
            or parsed_artifact_url.fragment
        ):
            raise ValueError(
                "communication.open_webui_artifacts.public_base_url must be an absolute HTTP(S) URL without credentials, query, or fragment"
            )
        if parsed_artifact_url.scheme != "https" and not loopback:
            raise ValueError(
                "communication.open_webui_artifacts.public_base_url must use HTTPS unless it is loopback"
            )
        if not artifact_serving.signing_secret_env.strip():
            raise ValueError(
                "communication.open_webui_artifacts.signing_secret_env is required when enabled"
            )
        _validate_positive(
            "communication.open_webui_artifacts.ttl_seconds",
            artifact_serving.ttl_seconds,
        )
    for route_name, route_url in communication.model_routes.items():
        if not route_name.strip():
            raise ValueError("communication.model_routes names must not be empty")
        parsed_route = urlparse(route_url)
        if parsed_route.scheme not in {"http", "https"} or not parsed_route.netloc:
            raise ValueError(
                f"communication.model_routes.{route_name} must be an absolute HTTP(S) URL"
            )
    if mcp.transport not in {"stdio", "streamable_http", "both"}:
        raise ValueError(
            "mcp.transport must be stdio, streamable_http, or both"
        )
    if mcp.authorization.timeout_seconds <= 0:
        raise ValueError("mcp.authorization.timeout_seconds must be positive")
    if mcp.authorization.enabled:
        if not mcp.authorization.resource_uri.startswith(("http://", "https://")):
            raise ValueError("mcp.authorization.resource_uri must be an absolute HTTP(S) URI")
        if not mcp.authorization.authorization_servers:
            raise ValueError("mcp.authorization.authorization_servers must not be empty when enabled")
        if not all(item.startswith(("http://", "https://")) for item in mcp.authorization.authorization_servers):
            raise ValueError("mcp.authorization.authorization_servers must contain absolute HTTP(S) URIs")
        if not all(item.startswith(("http://", "https://")) for item in mcp.authorization.allowed_origins):
            raise ValueError("mcp.authorization.allowed_origins must contain absolute HTTP(S) origins")
        if not mcp.authorization.introspection_url.startswith(("http://", "https://")):
            raise ValueError("mcp.authorization.introspection_url must be an absolute HTTP(S) URI")
        if not mcp.authorization.introspection_client_id:
            raise ValueError("mcp.authorization.introspection_client_id is required when enabled")
        if not mcp.authorization.introspection_client_secret_env:
            raise ValueError(
                "mcp.authorization.introspection_client_secret_env is required when enabled"
            )
        if not mcp.authorization.introspection_client_secret:
            raise ValueError(
                "MCP introspection client secret is required when enabled; set environment variable "
                + mcp.authorization.introspection_client_secret_env
            )
    for server_name, server in external_tools.mcp_servers.items():
        if not server_name.strip():
            raise ValueError("external_tools.mcp_servers names must not be empty")
        if server.transport not in {"stdio", "streamable_http"}:
            raise ValueError(
                f"external_tools.mcp_servers.{server_name}.transport must be stdio or streamable_http"
            )
        if server.timeout_seconds <= 0:
            raise ValueError(
                f"external_tools.mcp_servers.{server_name}.timeout_seconds must be positive"
            )
        if server.transport == "stdio" and not server.command:
            raise ValueError(
                f"external_tools.mcp_servers.{server_name}.command is required for stdio"
            )
        if server.transport == "streamable_http":
            parsed_url = urlparse(server.url)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                raise ValueError(
                    f"external_tools.mcp_servers.{server_name}.url must be an absolute HTTP(S) URL"
                )
        for header_name, env_name in server.header_env.items():
            if not header_name.strip() or any(ch in header_name for ch in "\r\n:"):
                raise ValueError(
                    f"external_tools.mcp_servers.{server_name}.header_env contains an invalid header name"
                )
            if not env_name.strip():
                raise ValueError(
                    f"external_tools.mcp_servers.{server_name}.header_env values must be environment variable names"
                )
        if server.credential_command and server.transport != "streamable_http":
            raise ValueError(
                f"external_tools.mcp_servers.{server_name}.credential_command requires streamable_http transport"
            )
        if server.credential_refresh_skew_seconds < 0:
            raise ValueError(
                f"external_tools.mcp_servers.{server_name}.credential_refresh_skew_seconds must be non-negative"
            )

    if a2a_authorization.enabled:
        public_url = urlparse(a2a_authorization.public_base_url)
        if (
            public_url.scheme != "https"
            or not public_url.netloc
            or public_url.username is not None
            or public_url.password is not None
            or public_url.query
            or public_url.fragment
        ):
            raise ValueError(
                "a2a.authorization.public_base_url must be an absolute HTTPS URL without credentials, query, or fragment when enabled"
            )
        if not a2a_authorization.bearer_token_env:
            raise ValueError(
                "a2a.authorization.bearer_token_env is required when enabled"
            )
        if not a2a_authorization.bearer_token:
            raise ValueError(
                "A2A bearer credential is required when enabled; set environment variable "
                + a2a_authorization.bearer_token_env
            )

    if a2a_push.enabled:
        if not a2a_authorization.enabled:
            raise ValueError("a2a.push.enabled requires a2a.authorization.enabled")
        if not a2a_push.credential_key_env.strip():
            raise ValueError("a2a.push.credential_key_env is required when enabled")
        hosts = [item.strip().casefold() for item in a2a_push.allowed_hosts if item.strip()]
        if not hosts or len(hosts) != len(set(hosts)):
            raise ValueError("a2a.push.allowed_hosts must contain unique non-empty hostnames")
        for host in hosts:
            if host in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("a2a.push.allowed_hosts may not contain loopback hosts")
        _validate_positive("a2a.push.timeout_seconds", a2a_push.timeout_seconds)
        if a2a_push.max_attempts < 1:
            raise ValueError("a2a.push.max_attempts must be at least 1")
        _validate_positive("a2a.push.retry_base_seconds", a2a_push.retry_base_seconds)
    if a2a_extended_card.enabled and not a2a_authorization.enabled:
        raise ValueError("a2a.extended_card.enabled requires a2a.authorization.enabled")
    if logging_cfg.level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("logging.level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    _validate_positive("logging.queue_capacity", logging_cfg.queue_capacity)
    _validate_positive("logging.max_bytes", logging_cfg.max_bytes)
    _validate_non_negative("logging.backup_count", logging_cfg.backup_count)

    if agent_data.sandbox_backend != "bwrap":
        raise ValueError("agent_data.sandbox_backend currently supports only 'bwrap'")
    if not str(agent_data.root):
        raise ValueError("agent_data.root must not be empty")
    if not agent_data.python_executable.strip():
        raise ValueError("agent_data.python_executable must not be empty")
    _validate_positive(
        "agent_data.command_timeout_seconds", agent_data.command_timeout_seconds
    )
    _validate_positive("agent_data.max_capture_chars", agent_data.max_capture_chars)

    if a2a_card_signing.enabled:
        if not a2a_card_signing.private_key_env.strip():
            raise ValueError("a2a.card_signing.private_key_env is required when enabled")
        if not a2a_card_signing.key_id.strip():
            raise ValueError("a2a.card_signing.key_id is required when enabled")
        if a2a_card_signing.jwks_url:
            jwks = urlparse(a2a_card_signing.jwks_url)
            if (
                jwks.scheme != "https"
                or not jwks.netloc
                or jwks.username is not None
                or jwks.password is not None
                or jwks.fragment
            ):
                raise ValueError("a2a.card_signing.jwks_url must be an absolute HTTPS URL")

    return AgentConfig(
        model=model,
        context=context,
        runtime=runtime,
        sessions=sessions,
        agent_data=agent_data,
        environment=environment,
        tools=tools,
        prompts=prompts,
        logging=logging_cfg,
        notes=notes,
        prompt_instructions=prompt_instructions,
        reader=reader,
        editor=editor,
        compression=compression,
        history_search=history_search,
        embedding_index=embedding_index,
        attachments=attachments,
        mcp=mcp,
        external_tools=external_tools,
        a2a_authorization=a2a_authorization,
        a2a_push=a2a_push,
        a2a_extended_card=a2a_extended_card,
        a2a_card_signing=a2a_card_signing,
        communication=communication,
        budget_policy=budget_policy,
        external_benchmarks=external_benchmarks,
        raw=data,
        sources=dict(sources or {}),
    )


def _dotted_value(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _parameter_unit(key: str) -> str | None:
    lowered = key.casefold()
    if lowered in {"max_open_questions", "max_background_plans", "max_active_workers", "max_pending_controls", "max_pending_inference", "max_pending_requests", "cache_max_entries", "cache_lock_stripes", "max_token_count_cache_entries"}:
        return "count"
    if lowered.endswith("_seconds"):
        return "seconds"
    if lowered.endswith("_bytes"):
        return "bytes"
    if lowered.endswith("_tokens") or lowered == "context_limit":
        return "tokens"
    if lowered.endswith("_chars"):
        return "characters"
    if lowered.endswith("_ratio") or "ratio" in lowered:
        return "ratio"
    if lowered == "port" or lowered.endswith("_port"):
        return "tcp-port"
    return None


def _parameter_semantics(
    dotted: str,
    *,
    key: str,
    group: str,
    security_sensitive: bool,
    resource_sensitive: bool,
) -> tuple[str, str]:
    exact: dict[str, tuple[str, str]] = {
        "model.cache_max_bytes": (
            "Maximum serialized record/replay cassette bytes, checked before reading and replacing the file.",
            "At capacity the cache fails explicitly without deleting replay evidence; archive or choose another cache path before further recording.",
        ),
        "model.cache_max_entries": (
            "Maximum combined completion, prompt-rendering and token-count entries in one cassette.",
            "Limits retained cache objects; overflow leaves the previously committed cassette unchanged.",
        ),
        "model.cache_lock_stripes": (
            "Fixed number of request-lock files per cassette, shared by request hashes.",
            "More stripes reduce unrelated lock collisions but retain more lock files; change only when all users of this cassette are stopped.",
        ),
        "context.max_token_count_cache_entries": (
            "Maximum in-memory exact/estimated token-count memo entries per runtime.",
            "Oldest memo entries are evicted and recomputed when needed; authoritative tokenization history is retained.",
        ),
        "runtime.max_pending_controls": (
            "Maximum unprocessed control messages per session.",
            "Pressure rejects new controls without losing accepted ones; exact retries with the same control ID remain idempotent.",
        ),
        "runtime.max_pending_inference": (
            "Maximum queued, running, or suspended model requests per backend in this runtime store.",
            "Pressure rejects admission until a request becomes terminal; higher limits retain more pending work.",
        ),
        "communication.max_active_workers": (
            "Maximum queued, working, or canceling workers across shared worker managers.",
            "Distinct from execution thread concurrency; limits pending work and retained futures. Rejected starts leave worker state unchanged.",
        ),
        "communication.max_pending_requests": (
            "Maximum queued or processing communication requests.",
            "Pressure rejects additional requests without deleting accepted user requests; terminal records remain durable evidence.",
        ),
        "runtime.max_open_questions": (
            "Maximum unresolved questions retained in the active session projection.",
            "Higher limits consume more context; overflow rejects new questions without discarding existing evidence.",
        ),
        "runtime.max_open_question_chars": (
            "Maximum combined question, reason, and provisional-assumption characters in active question state.",
            "Higher limits consume more context; overflow requires resolving existing questions before adding more.",
        ),
        "communication.idle_work_mode": (
            "Whether explicitly authorized backlog plans can start after foreground work finishes.",
            "finish_only never dispatches backlog work; authorized_backlog opts into idle execution with recorded user provenance.",
        ),
        "communication.max_background_plans": (
            "Maximum pending or held authorized background plans.",
            "Higher limits retain a larger queue; overflow rejects additions without deleting queued work.",
        ),
        "model.base_url": (
            "llama.cpp/OpenAI-compatible endpoint used for model inference.",
            "Changing it changes the backend/model identity and can change capacity, latency, output behavior, and cache validity.",
        ),
        "model.context_limit": (
            "Offline/configured fallback context capacity in tokens; live backends are discovered and take precedence where supported.",
            "Too small causes unnecessary reduction; too large is unsafe unless the backend actually supports it.",
        ),
        "model.remote_context_limit_fallback": (
            "Explicit fallback context capacity for a remote backend when live capacity discovery is unavailable.",
            "A wrong value can either reject valid requests or send requests the remote backend cannot admit.",
        ),
        "context.reserved_response_tokens": (
            "Maximum response headroom reserved by context planning for ordinary calls.",
            "Increasing it reduces available input context; decreasing it raises output-starvation/retry risk.",
        ),
        "context.reserved_summary_tokens": (
            "Response headroom reserved for history-summary/compaction calls.",
            "Changing it trades source capacity against room for a faithful semantic reduction response.",
        ),
        "context.safety_margin_tokens": (
            "Additional token margin kept below the discovered model context boundary.",
            "Too small risks backend rejection from serialization/tokenizer drift; too large wastes usable context.",
        ),
        "context.max_compaction_rounds": (
            "Maximum remeasure-and-reduce rounds for one context-admission attempt.",
            "Higher values can recover more oversized contexts at additional model cost; lower values fail closed sooner.",
        ),
        "context.semantic_reduction_max_input_tokens": (
            "Resource-safety ceiling for the input working set of semantic reduction subcalls only.",
            "Lower values create more hierarchical fragments; higher values can increase latency/memory pressure or model-server OOM risk.",
        ),
        "context.semantic_reduction_max_calls": (
            "Hard ceiling on semantic calls within one hierarchical reduction tree.",
            "It bounds worst-case inference work; reducing it may make legitimately large evidence fail closed before convergence.",
        ),
        "sessions.root": (
            "Persistent root for durable session histories, inference state, communication state, and related runtime databases.",
            "Changing it changes the durability namespace; pointing at the wrong root makes existing sessions appear missing.",
        ),
        "agent_data.root": (
            "Persistent private root for agent-owned scratch data, notes, experiments, and the isolated Python environment.",
            "Changing it moves private agent state and must not overlap user project roots unless that exposure is intentional.",
        ),
        "agent_data.sandbox_backend": (
            "OS sandbox implementation used by the private agent workspace.",
            "Changing or disabling it changes filesystem/network isolation guarantees for scratch Python/shell work.",
        ),
        "tools.read_roots": (
            "Filesystem roots that project-facing read tools are allowed to inspect.",
            "Broadening this increases data exposure; narrowing it can make required project evidence inaccessible.",
        ),
        "tools.allow_stateful_tools": (
            "Global gate for tools that persist or mutate agent/runtime state without being classified as external side effects.",
            "Disabling it blocks stateful workflows; enabling it expands what the model can persist/control.",
        ),
        "tools.allow_side_effect_tools": (
            "Global gate for tools classified as external/project side effects.",
            "Enabling it permits destructive/external actions allowed by individual tool policies; disabling it fails those actions closed.",
        ),
        "editor.allow_writes": (
            "Master gate for project file editing/writing capabilities.",
            "Disabling it makes project writes fail closed; enabling it permits writes only within the configured editor/tool boundaries.",
        ),
        "logging.file_path": (
            "Structured operations JSONL destination.",
            "Changing it changes where startup/shutdown/errors/operational evidence is retained and must preserve writable permissions.",
        ),
        "logging.queue_capacity": (
            "Bounded in-memory operations-log queue depth.",
            "Too small increases explicit overflow/drop fallback events; too large increases memory retained during slow disk writes.",
        ),
        "logging.max_bytes": (
            "Maximum active operations-log file size before rotation.",
            "Lower values rotate more often; higher values increase disk use and single-file scan cost.",
        ),
        "logging.backup_count": (
            "Number of rotated operations-log files retained locally.",
            "Increasing it retains more historical operational evidence at greater disk cost; zero removes rotated history.",
        ),
        "mcp.authorization.introspection_client_secret_env": (
            "Environment-variable name containing the MCP introspection client secret; the secret itself is not stored in ordinary config.",
            "Changing it changes which deployment secret is read; a missing/wrong value makes protected introspection fail closed.",
        ),
        "a2a.authorization.bearer_token_env": (
            "Environment-variable name containing the A2A bearer credential; literal bearer secrets are rejected from config.",
            "Changing it changes the credential used to protect enabled A2A routes; a missing value prevents authenticated startup/use.",
        ),
        "a2a.push.allowed_hosts": (
            "Exact HTTPS callback hosts allowed for A2A push delivery.",
            "Broadening it expands outbound callback trust; narrowing it causes non-allowlisted callbacks to fail closed.",
        ),
        "communication.enabled": (
            "Enables the long-running user-facing communication/task protocol service.",
            "Turning it on exposes the configured loopback interfaces; turning it off removes task/protocol/status service availability.",
        ),
        "communication.model_base_url": (
            "Optional separate model endpoint for user-facing communication/status work; empty reuses the main model.",
            "A distinct endpoint avoids preempting the main worker; reusing the main endpoint requires control-priority preemption/replay.",
        ),
        "communication.status_max_output_tokens": (
            "Initial 192-token-scale response budget for one concise communication/status interpretation call.",
            "Lower values improve control-plane responsiveness but may trigger output-starvation recovery; higher values let status calls occupy a shared model slot longer before replaying worker inference.",
        ),
        "communication.host": (
            "Bind host for the raw communication listener.",
            "The service requires a loopback bind; changing exposure changes the network trust boundary and non-loopback binds fail closed.",
        ),
        "communication.port": (
            "TCP port for the loopback communication/protocol listener.",
            "Changing it changes all local client/proxy connection targets and can conflict with another listener.",
        ),
        "communication.open_webui_artifacts.public_base_url": (
            "External HTTPS base advertised for signed Open WebUI artifact links.",
            "A wrong origin produces unusable or unsafe links; non-loopback publication requires HTTPS and valid signing configuration.",
        ),
        "communication.open_webui_artifacts.signing_secret_env": (
            "Environment-variable name containing the HMAC secret for signed Open WebUI artifact URLs; the secret itself is not stored in ordinary config.",
            "Changing/rotating it invalidates previously signed links; a missing secret prevents enabled signed serving.",
        ),
    }
    if dotted in exact:
        return exact[dotted]
    lowered = key.casefold()
    words = key.replace("_", " ")
    if lowered.endswith("_timeout_seconds") or lowered == "timeout_seconds":
        return (
            f"Timeout in seconds for {group} {words.removesuffix(' timeout seconds').strip() or 'operations'}.",
            "Shorter timeouts fail slow operations sooner; longer timeouts increase blocking/wait time before failure is surfaced.",
        )
    if lowered == "enabled" or lowered.endswith("_enabled"):
        return (
            f"Enables or disables the {group} {words.removesuffix(' enabled').strip()} capability.",
            f"Changing it changes whether that {group} capability participates in runtime behavior.",
        )
    if security_sensitive:
        return (
            f"Security/trust-boundary setting for {dotted}.",
            "Changing it can alter credential use, authorization, or exposure boundaries and therefore requires deployment review.",
        )
    if resource_sensitive:
        return (
            f"Resource/capacity setting for {dotted}.",
            "Changing it can alter latency, capacity, memory/disk use, retry behavior, or fail-closed thresholds.",
        )
    return (
        f"Behavioral setting `{words}` in the {group} subsystem.",
        f"Changing it changes {group} behavior and should be validated against the affected workflow.",
    )


def _parameter_range(key: str, value: Any) -> str | None:
    lowered = key.casefold()
    if lowered == "idle_work_mode":
        return "finish_only|authorized_backlog"
    if lowered in {"max_open_questions", "max_open_question_chars", "max_background_plans", "max_active_workers", "max_pending_controls", "max_pending_inference", "max_pending_requests", "cache_max_entries", "cache_lock_stripes", "max_token_count_cache_entries"}:
        return "integer > 0"
    if isinstance(value, bool):
        return "true|false"
    if lowered.endswith("_seconds") or lowered.endswith("_bytes") or lowered.endswith("_tokens"):
        return ">= 0 unless a stricter cross-setting validation applies"
    if lowered == "context_limit":
        return "> 0"
    if lowered == "port" or lowered.endswith("_port"):
        return "1..65535"
    if "ratio" in lowered:
        return "usually 0..1; exact cross-setting validation applies"
    return None


def _leaf_paths(value: Any, prefix: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, dict):
        result: list[str] = []
        for key, child in value.items():
            result.extend(_leaf_paths(child, (*prefix, str(key))))
        return result
    return [".".join(prefix)] if prefix else []


def _mark_sources(
    ledger: dict[str, str], override: dict[str, Any], source: str
) -> None:
    for path in _leaf_paths(override):
        ledger[path] = source


def _env_override_sources(env: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    prefix = "SWAAG__"
    for key in env:
        if not key.startswith(prefix):
            continue
        dotted = ".".join(key[len(prefix):].lower().split("__"))
        result[dotted] = f"environment:{key}"
    return result


def load_config(
    config_paths: list[str | Path] | None = None,
    env: dict[str, str] | None = None,
) -> AgentConfig:
    env = dict(os.environ if env is None else env)
    merged = _load_packaged_defaults()
    sources = {path: "packaged_defaults" for path in _leaf_paths(merged)}

    # Documented precedence, lowest to highest after packaged defaults:
    # local project config, explicit config_paths in caller order, SWAAG_CONFIG,
    # and finally SWAAG__SECTION__KEY environment overrides.
    search_paths: list[tuple[Path, str]] = []
    local_default = Path.cwd() / "config/local.toml"
    if local_default.exists():
        search_paths.append((local_default, f"local_config:{local_default}"))
    if config_paths:
        for path in config_paths:
            resolved = Path(path)
            search_paths.append((resolved, f"explicit_config:{resolved}"))
    env_path = env.get("SWAAG_CONFIG")
    if env_path:
        resolved = Path(env_path)
        search_paths.append((resolved, f"SWAAG_CONFIG:{resolved}"))

    for path, source in search_paths:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        payload = _migrate_config_aliases(_load_toml_file(path))
        merged = _deep_merge(merged, payload)
        _mark_sources(sources, payload, source)

    env = dict(env)
    legacy_env = "SWAAG__RUNTIME__MAX_REPEATED_ACTION_OCCURRENCES"
    new_env = "SWAAG__RUNTIME__MAX_VALIDATION_RECOVERY_CYCLES"
    if legacy_env in env and new_env not in env:
        env[new_env] = env[legacy_env]
    env.pop(legacy_env, None)
    merged = _apply_env_overrides(merged, env)
    sources.update(_env_override_sources(env))
    return _coerce_config(merged, sources=sources, secret_env=env)
