from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from swaag.utils import expand_env_in_value, sha256_text


@dataclass(slots=True)
class ModelConfig:
    base_url: str
    completion_endpoint: str
    tokenize_endpoint: str
    health_endpoint: str
    profile_name: str
    model_identity: str
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
    progress_poll_seconds: float
    max_retries: int
    temperature: float
    top_p: float
    seed: int
    context_limit: int
    stop: list[str]


@dataclass(slots=True)
class ContextConfig:
    reserved_response_tokens: int
    reserved_summary_tokens: int
    safety_margin_tokens: int
    max_recent_messages: int
    max_compaction_rounds: int
    note_prompt_token_cap: int
    allow_estimate_fallback: bool
    compact_on_overflow: bool


@dataclass(slots=True)
class RuntimeConfig:
    max_tool_steps: int
    max_reasoning_steps: int
    tool_timeout_seconds: int
    background_poll_seconds: float
    tool_call_budget: int
    max_total_actions: int
    no_progress_failure_limit: int
    verification_confidence_threshold: float
    capture_model_io: bool
    lean_on_overflow: bool
    strict_budget: bool
    max_repeated_action_occurrences: int


@dataclass(slots=True)
class SessionConfig:
    root: Path
    write_projections: bool


@dataclass(slots=True)
class EnvironmentConfig:
    shell_executable: str
    command_timeout_seconds: int
    max_capture_chars: int
    aubro_entrypoint: str
    aubro_src: str
    aubro_timeout_seconds: int
    aubro_max_text_chars: int
    aubro_max_results: int
    aubro_max_links: int


@dataclass(slots=True)
class ToolConfig:
    enabled: list[str]
    read_roots: list[Path]
    allow_stateful_tools: bool
    allow_side_effect_tools: bool


@dataclass(slots=True)
class PromptConfig:
    standard_system_template: str
    lean_system_template: str
    action_template: str
    summary_system_template: str
    summary_template: str


@dataclass(slots=True)
class LoggingConfig:
    level: str


@dataclass(slots=True)
class NotesConfig:
    max_notes: int
    max_note_chars: int
    max_total_chars: int
    compact_target_chars: int


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


@dataclass(slots=True)
class CompressionConfig:
    max_messages_before_compress: int
    summary_chars: int


@dataclass(slots=True)
class BudgetPolicyConfig:
    call_classes: dict[str, str]
    output_ratio: dict[str, float]
    output_floor_ratio: dict[str, float]
    output_ratio_by_kind: dict[str, float]
    output_floor_ratio_by_kind: dict[str, float]
    safety_ratio: dict[str, float]
    fixed_overhead_ratio: dict[str, float]
    fixed_overhead_min_tokens: int
    structured_output_json_factor_by_contract: dict[str, float]
    structured_output_json_factor_default: float
    structured_output_json_floor_tokens: int
    structured_output_schema_factor: float
    structured_output_schema_floor_tokens: int
    safe_input_floor_tokens: int


@dataclass(slots=True)
class HistorySearchConfig:
    max_results: int
    token_score: int
    exact_score: int
    type_bonus: int
    preview_chars: int


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
    agent_context_limit: int
    model_timeout_seconds: int
    model_structured_timeout_seconds: int
    allow_stateful_tools: bool
    allow_side_effect_tools: bool
    runtime_max_reasoning_steps: int
    runtime_max_total_actions: int
    runtime_max_tool_steps: int
    runtime_tool_call_budget: int
    candidate_file_limit: int
    file_excerpt_char_limit: int
    issue_prompt_char_limit: int
    completion_max_tokens: int
    solver_max_attempts: int
    summary_max_chars: int
    find_max_chars: int
    replace_max_chars: int
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
    environment: EnvironmentConfig
    tools: ToolConfig
    prompts: PromptConfig
    logging: LoggingConfig
    notes: NotesConfig
    reader: ReaderConfig
    editor: EditorConfig
    compression: CompressionConfig
    history_search: HistorySearchConfig
    budget_policy: BudgetPolicyConfig
    external_benchmarks: ExternalBenchmarksConfig
    raw: dict[str, Any] = field(repr=False)

    def config_fingerprint(self) -> str:
        return sha256_text(json.dumps(self.raw, sort_keys=True))


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


def _coerce_config(data: dict[str, Any]) -> AgentConfig:
    data = expand_env_in_value(data)
    model = ModelConfig(**data["model"])
    context = ContextConfig(**data["context"])
    runtime = RuntimeConfig(**data["runtime"])
    sessions = SessionConfig(
        root=Path(data["sessions"]["root"]).expanduser(),
        write_projections=bool(data["sessions"]["write_projections"]),
    )
    environment = EnvironmentConfig(**data["environment"])
    tools = ToolConfig(
        enabled=list(data["tools"]["enabled"]),
        read_roots=[Path(item).expanduser() for item in data["tools"]["read_roots"]],
        allow_stateful_tools=bool(data["tools"]["allow_stateful_tools"]),
        allow_side_effect_tools=bool(data["tools"]["allow_side_effect_tools"]),
    )
    prompts = PromptConfig(**data["prompts"])
    logging_cfg = LoggingConfig(**data["logging"])
    notes = NotesConfig(**data["notes"])
    reader = ReaderConfig(**data["reader"])
    editor = EditorConfig(**data["editor"])
    compression = CompressionConfig(**data["compression"])
    budget_policy = BudgetPolicyConfig(
        call_classes={str(key): str(value) for key, value in data["budget_policy"]["call_classes"].items()},
        output_ratio={str(key): float(value) for key, value in data["budget_policy"]["output_ratio"].items()},
        output_floor_ratio={str(key): float(value) for key, value in data["budget_policy"]["output_floor_ratio"].items()},
        output_ratio_by_kind={str(key): float(value) for key, value in data["budget_policy"]["output_ratio_by_kind"].items()},
        output_floor_ratio_by_kind={str(key): float(value) for key, value in data["budget_policy"]["output_floor_ratio_by_kind"].items()},
        safety_ratio={str(key): float(value) for key, value in data["budget_policy"]["safety_ratio"].items()},
        fixed_overhead_ratio={str(key): float(value) for key, value in data["budget_policy"]["fixed_overhead_ratio"].items()},
        fixed_overhead_min_tokens=int(data["budget_policy"]["fixed_overhead_min_tokens"]),
        structured_output_json_factor_by_contract={
            str(key): float(value)
            for key, value in data["budget_policy"]["structured_output_json_factor_by_contract"].items()
        },
        structured_output_json_factor_default=float(data["budget_policy"]["structured_output_json_factor_default"]),
        structured_output_json_floor_tokens=int(data["budget_policy"]["structured_output_json_floor_tokens"]),
        structured_output_schema_factor=float(data["budget_policy"]["structured_output_schema_factor"]),
        structured_output_schema_floor_tokens=int(data["budget_policy"]["structured_output_schema_floor_tokens"]),
        safe_input_floor_tokens=int(data["budget_policy"]["safe_input_floor_tokens"]),
    )
    history_search = HistorySearchConfig(
        max_results=int(data["history_search"]["max_results"]),
        token_score=int(data["history_search"]["token_score"]),
        exact_score=int(data["history_search"]["exact_score"]),
        type_bonus=int(data["history_search"]["type_bonus"]),
        preview_chars=int(data["history_search"]["preview_chars"]),
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
            agent_context_limit=int(data["external_benchmarks"]["agent_generation"]["agent_context_limit"]),
            model_timeout_seconds=int(data["external_benchmarks"]["agent_generation"]["model_timeout_seconds"]),
            model_structured_timeout_seconds=int(
                data["external_benchmarks"]["agent_generation"]["model_structured_timeout_seconds"]
            ),
            allow_stateful_tools=bool(data["external_benchmarks"]["agent_generation"]["allow_stateful_tools"]),
            allow_side_effect_tools=bool(data["external_benchmarks"]["agent_generation"]["allow_side_effect_tools"]),
            runtime_max_reasoning_steps=int(data["external_benchmarks"]["agent_generation"]["runtime_max_reasoning_steps"]),
            runtime_max_total_actions=int(data["external_benchmarks"]["agent_generation"]["runtime_max_total_actions"]),
            runtime_max_tool_steps=int(data["external_benchmarks"]["agent_generation"]["runtime_max_tool_steps"]),
            runtime_tool_call_budget=int(data["external_benchmarks"]["agent_generation"]["runtime_tool_call_budget"]),
            candidate_file_limit=int(data["external_benchmarks"]["agent_generation"]["candidate_file_limit"]),
            file_excerpt_char_limit=int(data["external_benchmarks"]["agent_generation"]["file_excerpt_char_limit"]),
            issue_prompt_char_limit=int(data["external_benchmarks"]["agent_generation"]["issue_prompt_char_limit"]),
            completion_max_tokens=int(data["external_benchmarks"]["agent_generation"]["completion_max_tokens"]),
            solver_max_attempts=int(data["external_benchmarks"]["agent_generation"]["solver_max_attempts"]),
            summary_max_chars=int(data["external_benchmarks"]["agent_generation"]["summary_max_chars"]),
            find_max_chars=int(data["external_benchmarks"]["agent_generation"]["find_max_chars"]),
            replace_max_chars=int(data["external_benchmarks"]["agent_generation"]["replace_max_chars"]),
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
    _validate_positive("model.timeout_seconds", model.timeout_seconds)
    _validate_positive("model.connect_timeout_seconds", model.connect_timeout_seconds)
    _validate_positive("model.simple_timeout_seconds", model.simple_timeout_seconds)
    _validate_positive("model.structured_timeout_seconds", model.structured_timeout_seconds)
    _validate_positive("model.verification_timeout_seconds", model.verification_timeout_seconds)
    _validate_positive("model.benchmark_timeout_seconds", model.benchmark_timeout_seconds)
    if model.progress_poll_seconds <= 0:
        raise ValueError("model.progress_poll_seconds must be positive")
    if model.structured_output_mode != "server_schema":
        raise ValueError("model.structured_output_mode must be server_schema")
    if model.cache_mode not in {"record", "replay"}:
        raise ValueError("model.cache_mode must be record or replay")
    _validate_positive("context.reserved_response_tokens", context.reserved_response_tokens)
    _validate_positive("context.reserved_summary_tokens", context.reserved_summary_tokens)
    _validate_non_negative("context.safety_margin_tokens", context.safety_margin_tokens)
    _validate_non_negative("runtime.max_tool_steps", runtime.max_tool_steps)
    _validate_positive("runtime.max_reasoning_steps", runtime.max_reasoning_steps)
    _validate_positive("environment.command_timeout_seconds", environment.command_timeout_seconds)
    _validate_positive("environment.max_capture_chars", environment.max_capture_chars)
    _validate_positive("environment.aubro_timeout_seconds", environment.aubro_timeout_seconds)
    _validate_positive("environment.aubro_max_text_chars", environment.aubro_max_text_chars)
    _validate_positive("environment.aubro_max_results", environment.aubro_max_results)
    _validate_positive("environment.aubro_max_links", environment.aubro_max_links)
    _validate_positive("runtime.tool_timeout_seconds", runtime.tool_timeout_seconds)
    if runtime.background_poll_seconds < 0:
        raise ValueError("runtime.background_poll_seconds must be non-negative")
    _validate_positive("runtime.tool_call_budget", runtime.tool_call_budget)
    _validate_positive("runtime.max_total_actions", runtime.max_total_actions)
    _validate_positive("runtime.no_progress_failure_limit", runtime.no_progress_failure_limit)
    if not 0.0 <= runtime.verification_confidence_threshold <= 1.0:
        raise ValueError("runtime.verification_confidence_threshold must be between 0.0 and 1.0")
    _validate_positive("runtime.max_repeated_action_occurrences", runtime.max_repeated_action_occurrences)
    _validate_positive("notes.max_notes", notes.max_notes)
    _validate_positive("notes.max_note_chars", notes.max_note_chars)
    _validate_positive("notes.max_total_chars", notes.max_total_chars)
    _validate_positive("notes.compact_target_chars", notes.compact_target_chars)
    _validate_positive("reader.default_chunk_chars", reader.default_chunk_chars)
    _validate_non_negative("reader.default_overlap_chars", reader.default_overlap_chars)
    _validate_positive("reader.max_chunk_chars", reader.max_chunk_chars)
    _validate_positive("budget_policy.fixed_overhead_min_tokens", budget_policy.fixed_overhead_min_tokens)
    _validate_positive("budget_policy.structured_output_json_floor_tokens", budget_policy.structured_output_json_floor_tokens)
    _validate_positive("budget_policy.structured_output_schema_floor_tokens", budget_policy.structured_output_schema_floor_tokens)
    _validate_positive("budget_policy.safe_input_floor_tokens", budget_policy.safe_input_floor_tokens)
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
        "external_benchmarks.agent_generation.agent_context_limit",
        external_benchmarks.agent_generation.agent_context_limit,
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
        "external_benchmarks.agent_generation.runtime_max_reasoning_steps",
        external_benchmarks.agent_generation.runtime_max_reasoning_steps,
    )
    _validate_positive(
        "external_benchmarks.agent_generation.runtime_max_total_actions",
        external_benchmarks.agent_generation.runtime_max_total_actions,
    )
    _validate_positive(
        "external_benchmarks.agent_generation.runtime_max_tool_steps",
        external_benchmarks.agent_generation.runtime_max_tool_steps,
    )
    _validate_positive(
        "external_benchmarks.agent_generation.runtime_tool_call_budget",
        external_benchmarks.agent_generation.runtime_tool_call_budget,
    )
    _validate_positive(
        "external_benchmarks.agent_generation.candidate_file_limit",
        external_benchmarks.agent_generation.candidate_file_limit,
    )
    _validate_positive(
        "external_benchmarks.agent_generation.file_excerpt_char_limit",
        external_benchmarks.agent_generation.file_excerpt_char_limit,
    )
    _validate_positive(
        "external_benchmarks.agent_generation.issue_prompt_char_limit",
        external_benchmarks.agent_generation.issue_prompt_char_limit,
    )
    _validate_positive(
        "external_benchmarks.agent_generation.completion_max_tokens",
        external_benchmarks.agent_generation.completion_max_tokens,
    )
    _validate_positive(
        "external_benchmarks.agent_generation.solver_max_attempts",
        external_benchmarks.agent_generation.solver_max_attempts,
    )
    _validate_positive(
        "external_benchmarks.agent_generation.summary_max_chars",
        external_benchmarks.agent_generation.summary_max_chars,
    )
    _validate_positive(
        "external_benchmarks.agent_generation.find_max_chars",
        external_benchmarks.agent_generation.find_max_chars,
    )
    _validate_positive(
        "external_benchmarks.agent_generation.replace_max_chars",
        external_benchmarks.agent_generation.replace_max_chars,
    )
    if not external_benchmarks.agent_generation.git_remote_base_url:
        raise ValueError("external_benchmarks.agent_generation.git_remote_base_url must not be empty")
    if not external_benchmarks.agent_generation.model_name_or_path:
        raise ValueError("external_benchmarks.agent_generation.model_name_or_path must not be empty")
    if not external_benchmarks.agent_generation.prompt_template.strip():
        raise ValueError("external_benchmarks.agent_generation.prompt_template must not be empty")
    if not external_benchmarks.agent_generation.empty_patch_retry_prompt.strip():
        raise ValueError("external_benchmarks.agent_generation.empty_patch_retry_prompt must not be empty")
    _validate_positive("compression.max_messages_before_compress", compression.max_messages_before_compress)
    _validate_positive("compression.summary_chars", compression.summary_chars)
    if reader.default_overlap_chars >= reader.default_chunk_chars:
        raise ValueError("reader.default_overlap_chars must be smaller than reader.default_chunk_chars")
    if not tools.enabled:
        raise ValueError("tools.enabled must not be empty")
    if not model.stop:
        raise ValueError("model.stop must not be empty")

    return AgentConfig(
        model=model,
        context=context,
        runtime=runtime,
        sessions=sessions,
        environment=environment,
        tools=tools,
        prompts=prompts,
        logging=logging_cfg,
        notes=notes,
        reader=reader,
        editor=editor,
        compression=compression,
        history_search=history_search,
        budget_policy=budget_policy,
        external_benchmarks=external_benchmarks,
        raw=data,
    )


def load_config(config_paths: list[str | Path] | None = None, env: dict[str, str] | None = None) -> AgentConfig:
    env = dict(os.environ if env is None else env)
    merged = _load_packaged_defaults()

    search_paths: list[Path] = []
    if config_paths:
        search_paths.extend(Path(path) for path in config_paths)
    env_path = env.get("SWAAG_CONFIG")
    if env_path:
        search_paths.append(Path(env_path))
    local_default = Path.cwd() / "config/local.toml"
    if local_default.exists():
        search_paths.append(local_default)

    for path in search_paths:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        merged = _deep_merge(merged, _load_toml_file(path))

    merged = _apply_env_overrides(merged, env)
    return _coerce_config(merged)
