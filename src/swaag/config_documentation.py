"""Behavioral meaning for configuration families; never used to route user intent."""
from __future__ import annotations

# Each row describes a consumed setting and the practical consequence of changing it.
EXACT = {
    'model.completion_endpoint': ('HTTP path used to submit constrained generation requests.', 'Changing transport/path must match the provider API; a mismatch prevents inference.'),
    'model.tokenize_endpoint': ('HTTP path used to measure model input tokens.', 'An unavailable tokenizer requires disclosed estimation; a wrong tokenizer invalidates context accounting.'),
    'model.health_endpoint': ('HTTP path used to probe native model service health.', 'Changing it changes readiness evidence; a successful unrelated path is not evidence the model can generate.'),
    'model.profile_name': ('Backend model alias sent in OpenAI-compatible inference requests and recorded in request policy.', 'It must name the intended served model; changing it changes routing and replay identity.'),
    'model.max_semantic_responsibilities_per_call': ('Maximum independent semantic responsibilities allowed in one model call; one selects separated tool and terminal calls.', 'Lowering it separates decisions at extra latency; raising it permits fused action/status/response generation and requires quality measurement.'),
    'model.model_identity': ('Explicit model/revision identity used alongside discovered backend metadata.', 'Wrong identity can misattribute evidence and invalidate cache comparisons; it does not load or change the server model.'),
    'model.provider_name': ('Provider discriminator for telemetry and verified backend-specific adapters.', 'Use the actual backend; an incorrect discriminator can select the wrong observability or chat-template behavior.'),
    'model.api_key_env': ('Name of the environment variable holding the inference bearer credential.', 'Changing it changes which secret is read at request time; the secret itself must stay outside configuration.'),
    'model.structured_output_mode': ('Required server-side schema-constrained response mechanism.', 'Only server_schema is supported; disabling constraints is rejected rather than silently weakening action validation.'),
    'model.thinking_mode': ('Explicit vLLM/llama.cpp chat-template reasoning setting, or server_default to preserve backend behavior.', 'Generation and token measurement receive the same setting. Disabling reasoning can reduce latency but can change quality; unsupported provider overrides fail closed.'),
    'model.cache_mode': ('Record or replay behavior for the model-request cassette.', 'Replay requires matching recorded evidence; record can call the real backend for misses and durably retain responses.'),
    'model.cache_path': ('Exact request-cassette location; empty selects the sessions parent default.', 'Changing it selects a different replay evidence set and storage/locking namespace.'),
    'model.progress_poll_seconds': ('Interval between observations of backend or streamed inference progress.', 'Shorter intervals improve stall observation at additional polling cost; this is not an ordinary inference deadline.'),
    'model.max_retries': ('Retry allowance for model request and output-budget recovery paths.', 'Higher values can recover transient/structural failure with additional inference cost; retries do not authorize repeating external effects.'),
    'model.temperature': ('Sampling temperature sent to the model provider.', 'Changing it changes answer variability and instruction reliability; backend sampling domains apply.'),
    'model.top_p': ('Nucleus sampling probability sent to the model provider.', 'Lower values constrain token choices and can change quality; maintain comparable sampling settings in benchmarks.'),
    'model.seed': ('Base model-generation seed from which reproducible retry seeds are derived.', 'Changing it changes sampled responses; it does not guarantee identical output across runtimes, models or hardware.'),
    'model.stop': ('Additional literal generation stop strings, empty by default.', 'A matching string stops generation and may cut structured output short; avoid accidental stops inside valid content.'),
    'runtime.lean_on_overflow': ('Allows the compact prompt variant when the ordinary prompt exceeds context capacity.', 'It can preserve an otherwise admissible task with less prompt overhead; context accounting still enforces the live limit.'),
    'runtime.max_validation_recovery_cycles': ('Bound on retries after invalid structured actions or observed state changes.', 'Higher values spend more inference repairing failures; exhaustion returns a failure without applying a rejected action.'),
    'tools.enabled': ('Tool capability names exposed to this runtime.', 'Removing a tool prevents model selection of that capability; it must not disable enforcement of already stored instructions.'),
    'tools.staged_discovery': ('Exposes compact capability pointers and defers full tool schemas and workspace enumeration.', 'It reduces initial prompt size at the cost of explicit discovery calls; disabling it can greatly increase file/schema context.'),
    'context.allow_estimate_fallback': ('Permits disclosed conservative token estimates when exact server measurement is unavailable.', 'Disabling it refuses unmeasurable calls; enabling it retains safety margins but does not claim tokenizer exactness.'),
    'context.compact_on_overflow': ('Allows semantic context reduction after measured overflow.', 'Disabling it fails oversized calls sooner; enabling it adds model calls while retaining exact source references.'),
    'sessions.write_projections': ('Writes derived session state files alongside authoritative history.', 'Disabling projections changes read/startup cost; append-only history remains the recovery authority.'),
    'agent_data.python_executable': ('Absolute host Python executable used to create the shared sandbox virtual environment.', 'Changing it affects interpreter compatibility and installed packages; it does not grant host filesystem access to sandbox code.'),
    'agent_data.max_knowledge_bytes': ('Maximum retained content and metadata bytes across forge and Diane revisions.', 'Admission rejects excess content without evicting old revisions; this payload quota is not a filesystem-wide disk quota.'),
    'agent_data.max_knowledge_revisions': ('Maximum retained forge and Diane revision records.', 'Old revisions remain pinned and recoverable; reaching capacity refuses new revisions until an explicit archive/storage policy is applied.'),
    'environment.shell_executable': ('Shell used by project-facing command execution.', 'Changing it changes command syntax and startup behavior; it is separate from private sandbox shell isolation.'),
    'environment.max_capture_chars': ('Preview length for project command output.', 'Smaller previews reduce model context; exact retained artifacts remain the recovery source.'),
    'environment.track_shell_file_changes': ('Records project file changes around shell execution.', 'Disabling it reduces file scanning overhead but removes that automatic change-observation evidence.'),
    'logging.level': ('Minimum severity emitted by runtime logging.', 'More verbose logging improves diagnosis at increased disk/processing cost; filtering is not deletion of authoritative task history.'),
    'reader.default_chunk_chars': ('Default character count for paged text reads.', 'Larger pages reduce calls but use more model context; overlap must remain smaller than the chunk.'),
    'reader.default_overlap_chars': ('Repeated characters between adjacent default text pages.', 'Overlap helps continuity but costs context; zero removes overlap and values reaching chunk size are rejected.'),
    'reader.max_chunk_chars': ('Maximum accepted character count for one text read.', 'Larger limits permit larger tool responses; smaller limits require more continuation calls.'),
    'editor.create_backups': ('Creates backups before supported project text modifications.', 'Disabling it removes that local rollback copy; repository history remains a separate recovery mechanism.'),
    'editor.backup_suffix': ('Suffix used for project editor backup files.', 'Changing it changes backup locations and possible name collisions; it does not change the original target.'),
    'editor.allowed_write_paths': ('Explicit permitted project write paths in addition to the editor master gate.', 'Broadening it expands mutable project scope; narrowing it rejects writes outside the permitted paths.'),
    'history_search.max_results': ('Maximum history-search hits returned per request.', 'Larger values increase retrieval context; exact history remains available through paged reads.'),
    'history_search.token_score': ('Lexical score contribution for matching query tokens in the history lookup index.', 'Changing it changes mechanical search ranking, not semantic authority or instruction routing.'),
    'history_search.exact_score': ('Score contribution for an exact history-search phrase match.', 'Changing it changes retrieval ordering; a high-ranked result still requires source/context interpretation.'),
    'history_search.type_bonus': ('Per-event-type score adjustment in mechanical history lookup.', 'Changing it changes ranking of evidence classes; it never makes retrieved text authoritative instructions.'),
    'history_search.preview_chars': ('Maximum characters in each returned history-search preview.', 'Shorter previews reduce context but can require recovering the exact event before interpretation.'),
    'embedding_index.base_url': ('Endpoint of the optional embedding service.', 'Changing it changes retrieval representation/provider and may require rebuilding the index.'),
    'embedding_index.endpoint': ('Embedding API path appended to its service URL.', 'An incompatible path prevents indexing or retrieval; raw history remains authoritative.'),
    'embedding_index.model': ('Embedding model identifier for semantic retrieval vectors.', 'Changing the model changes vector meaning/dimension and requires a compatible index.'),
    'embedding_index.fields': ('Event fields selected for embedding-index input.', 'Changing them changes searchable evidence coverage and index contents; it does not rewrite raw history.'),
    'embedding_index.max_results': ('Maximum nearest-neighbor results returned by embedding retrieval.', 'Larger values increase retrieval and context cost; semantic similarity is not proof of factual relevance.'),
    'attachments.max_upload_bytes': ('Maximum accepted attachment payload size.', 'Raising it admits larger files and storage pressure; oversized uploads fail before normal ingestion.'),
    'attachments.preview_chars': ('Character limit for textual attachment previews.', 'Smaller previews require exact attachment reads for omitted content; original attachments remain retained.'),
    'perception.base_url': ('Endpoint of the configured image/perception model service.', 'Changing it changes image interpretation capability and data destination; benchmark the actual connected model.'),
    'perception.endpoint': ('API path used for perception requests.', 'It must match the chosen provider transport; a mismatch fails image interpretation.'),
    'perception.model': ('Model alias for perception requests.', 'Changing it changes visual capability and latency; it does not start a model service.'),
    'perception.max_output_tokens': ('Maximum requested tokens for a perception response.', 'Lower limits can truncate visual findings; higher limits increase latency and context consumption.'),
    'perception.benchmark_profile': ('Named benchmark profile describing the perception capability.', 'Changing it changes qualification/provenance expectations; a label alone does not certify visual accuracy.'),
    'perception.trust_note': ('Trust qualification supplied with perception-derived evidence.', 'It informs interpretation and limitations; model-observed content must not become user instructions.'),
    'mcp.transport': ('Enabled MCP transport selection: stdio, streamable HTTP, or both.', 'Changing it changes integration listeners and client compatibility; HTTP authorization remains separately enforced.'),
    'mcp.authorization.resource_uri': ('Canonical protected-resource URI used by MCP authorization.', 'It must match the published resource and token audience expectations; wrong identity rejects legitimate access or misstates the trust boundary.'),
    'mcp.authorization.authorization_servers': ('Advertised authorization-server origins for the MCP protected resource.', 'Changing them changes which authority clients use to obtain credentials.'),
    'mcp.authorization.allowed_origins': ('Explicit browser origins accepted by the protected MCP HTTP endpoint.', 'Broadening it admits more browser origins; narrowing it rejects previously allowed cross-origin clients.'),
    'mcp.authorization.introspection_url': ('Token-introspection endpoint used to verify MCP bearer credentials.', 'Changing it changes the credential-verification authority; unavailable or invalid introspection fails closed.'),
    'mcp.authorization.introspection_client_id': ('Client identifier used when authenticating to token introspection.', 'It must match the deployment credential pair; a mismatch prevents token verification.'),
    'mcp.authorization.required_scopes': ('Scopes required on accepted MCP access tokens.', 'Adding scopes tightens access; removing them broadens what credential holders can invoke.'),
    'a2a.authorization.public_base_url': ('Advertised public A2A origin.', 'It must match the authenticated deployment/proxy; a wrong URL misroutes clients and advertised task links.'),
    'a2a.push.credential_key_env': ('Environment-variable name holding the key used to encrypt stored push credentials.', 'Changing the key affects decryption of persisted callbacks; missing or short keys prevent enabled push delivery.'),
    'a2a.push.max_attempts': ('Maximum attempts for a queued A2A push callback.', 'Higher values retry transient failures longer; exhaustion leaves explicit failed delivery evidence.'),
    'a2a.push.retry_base_seconds': ('Base interval for push-delivery retry backoff.', 'Larger values reduce callback pressure but delay recovery; smaller values increase repeated delivery load.'),
    'a2a.card_signing.private_key_env': ('Environment-variable name containing the Agent Card ES256 signing key.', 'Changing it changes card signatures; unsupported curves or missing enabled keys fail startup.'),
    'a2a.card_signing.key_id': ('Public key identifier attached to signed Agent Cards.', 'It must resolve to the matching verifier key; rotating it changes client verification references.'),
    'a2a.card_signing.jwks_url': ('Published JWKS URL for Agent Card signature verification.', 'A wrong or unavailable URL prevents clients from resolving the advertised signing key.'),
    'communication.model_profile_name': ('Optional served-model alias overriding the main model for communication.', 'It must match the selected communication endpoint; an endpoint override alone does not infer its alias.'),
    'communication.model_identity': ('Optional exact model identity for communication evidence and caching.', 'Changing it changes model provenance; it does not prove capability or load a service.'),
    'communication.remote_context_limit_fallback': ('Explicit context capacity fallback for the communication backend.', 'Use only a verified deployed capacity; an oversized value can cause admission failure.'),
    'communication.enabled_tools': ('Capabilities exposed to the persistent foreground orchestrator.', 'Removing editing capabilities does not remove stored instruction enforcement; adding tools expands available actions within runtime permission gates.'),
    'communication.poll_seconds': ('Polling interval for background orchestration and foreground durable progress delivery.', 'Shorter intervals reduce delivery/scheduling delay at additional polling cost; model and tool threads continue independently of speech consumers.'),
    'communication.open_webui_artifacts.ttl_seconds': ('Validity duration for signed artifact access links.', 'Longer validity improves sharing duration but extends credential exposure; expired links fail verification.'),
    'external_benchmarks.root': ('Root for isolated external benchmark checkouts and retained run artifacts.', 'Changing it selects another execution/evidence namespace and disk footprint.'),
    'external_benchmarks.model_server.retry_attempts': ('Bound on model-server readiness retries in external benchmark setup.', 'Higher values tolerate longer startup delays; exhaustion fails preflight rather than reporting a benchmark result.'),
    'external_benchmarks.model_server.retry_sleep_seconds': ('Delay between external benchmark model readiness checks.', 'Longer delays reduce probe load but increase preflight latency; zero permits immediate retries.'),
    'external_benchmarks.terminal_bench.allow_compose_download': ('Permits downloading a missing Docker Compose helper during Terminal-Bench setup.', 'Enabling it authorizes that dependency download; disabling it requires an already available executable.'),
    'external_benchmarks.agent_generation.default_max_instances': ('Default number of benchmark instances selected for agent-generated attempts.', 'Increasing it increases total runtime, model use and retained artifacts.'),
    'external_benchmarks.agent_generation.allow_stateful_tools': ('State mutation permission for generated benchmark agent runtimes.', 'Enabling it permits their persistent state changes; it does not alter the foreground runtime permission gate.'),
    'external_benchmarks.agent_generation.allow_side_effect_tools': ('External/project side-effect permission for benchmark solver runtimes.', 'Enabling it permits edits and commands within each benchmark boundary; disabling it can make coding tasks intentionally inoperable.'),
    'external_benchmarks.agent_generation.solver_max_attempts': ('Maximum solver attempts for one benchmark instance.', 'Higher values permit more attempts and inference cost; each attempt retains its own evidence.'),
    'external_benchmarks.agent_generation.git_remote_base_url': ('Git origin prefix used to obtain benchmark project repositories.', 'Changing it changes source provenance and network destination; repositories must match the benchmark instance.'),
    'external_benchmarks.agent_generation.model_name_or_path': ('Model label or path supplied to the external benchmark solver integration.', 'Changing it changes reported solver identity and must match the actual configured backend.'),
    'external_benchmarks.agent_generation.prompt_template': ('Template used to give the benchmark issue and workspace context to the solver.', 'Changing it changes the evaluated task framing and prevents direct comparison with prior prompt conditions.'),
    'external_benchmarks.agent_generation.empty_patch_retry_prompt': ('Follow-up instruction used after a solver attempt produces no patch.', 'Changing it changes recovery behavior and additional inference; it must not fabricate a successful patch.'),
}


def parameter_documentation(dotted: str):
    if dotted in EXACT:return EXACT[dotted]
    parts=dotted.split('.')
    if parts[0]=='prompts':
        role='system instructions' if '_system_' in parts[-1] else 'operation instructions and evidence layout'
        return (f'Template resource for {parts[-1].replace("_", " ")}; supplies {role}.',
                'Changing this text changes model behavior and request identity; preserve authority boundaries and remeasure quality/context use.')
    if parts[0] in {'notes','prompt_instructions'} and parts[-1].startswith('max_'):
        scope='session notes' if parts[0]=='notes' else 'durable prompt instructions'
        kind={'max_notes':'record count','max_instructions':'record count','max_note_chars':'characters in one note','max_instruction_chars':'characters in one instruction','max_total_chars':'combined retained characters'}[parts[-1]]
        return (f'Maximum {kind} for {scope}.','Increasing this allows more durable context; exceeding the bound rejects additions instead of silently deleting authoritative records.')
    if parts[0]=='budget_policy':
        family=parts[1];subject=parts[-1].replace('_',' ')
        if family=='call_classes':return (f'Capacity class used for {subject} model calls.','Selects class defaults for output and safety reserves; per-kind overrides still take precedence.')
        if family.startswith('output_floor_ratio'):return (f'Context fraction used as the output-reserve floor for {subject}.','Higher floors reserve more generation capacity and reduce input space; oversized input still requires explicit admission/reduction.')
        if family.startswith('output_ratio'):return (f'Desired output-reserve fraction for {subject}.','Higher values leave more output room but can reduce input capacity; the compiler negotiates within measured capacity and operation minimums.')
        if family=='safety_ratio':return (f'Extra context safety fraction for the {subject} call class.','Larger margins tolerate accounting uncertainty at reduced usable context; they do not increase actual server capacity.')
        if 'factor' in family:return (f'Structured-output size estimation multiplier for {subject}.','Higher multipliers reserve more output room for constrained JSON/schema shape; lower values risk output starvation and retries.')
        if 'floor' in family:return (f'Minimum structured-output token reserve for {subject}.','Raising the floor protects JSON/schema completion but leaves less input room; it does not guarantee semantic correctness.')
    if dotted.startswith('external_benchmarks.targets.'):
        target=parts[2];field=parts[3]
        rows={
            'description':('Human-readable purpose of the benchmark target.','Changes presentation only, not verifier behavior.'),
            'workdir':('Working directory for target preflight and evaluator commands.','Changing it changes relative-path resolution and which checkout executes.'),
            'default_variables':('Default substitution value for this target command variable.','Changing it changes dataset, namespace or instance selection passed to the external evaluator.'),
            'preflight_commands':('Exact argument vectors run before benchmark evaluation.','Failures stop evaluation; changing commands changes readiness checks and executable effects.'),
            'smoke_command':('Exact argument vector for the smaller benchmark validation run.','Changing it changes the evaluated sample and resulting evidence; smoke success is not full-suite success.'),
            'full_command':('Exact argument vector for full target evaluation.','Changing it changes benchmark coverage, duration and executable effects.'),
            'required_env':('Environment-variable names required before evaluator execution.','Adding names tightens preflight; removing them can defer a missing prerequisite into runtime failure.'),
            'required_paths':('Paths required by benchmark preflight.','Changing them changes dependency validation, not the contents of the evaluator.'),
            'allowed_path_literals':('Explicit path literals allowed by benchmark command validation.','Broadening it expands executable path scope; keep it aligned with the isolated benchmark layout.'),
            'artifact_globs':('Patterns used to collect retained evaluator artifacts.','Changing them changes collected evidence coverage; omitted files may require manual recovery.'),
        }
        if field in rows:
            description,consequences=rows[field];return (f'{target}: {description}',consequences)
    return None

# Bounds mirrored from explicit configuration validators; backend-only domains are separate.
VALIDATE_POSITIVE_PARAMETERS = frozenset(['a2a.push.retry_base_seconds', 'a2a.push.timeout_seconds', 'agent_data.command_timeout_seconds', 'agent_data.max_capture_chars', 'agent_data.max_knowledge_bytes', 'agent_data.max_knowledge_revisions', 'agent_data.max_list_files', 'agent_data.max_output_bytes', 'attachments.max_upload_bytes', 'attachments.preview_chars', 'budget_policy.structured_output_json_floor_tokens', 'budget_policy.structured_output_schema_floor_tokens', 'communication.autonomous_idea_interval_seconds', 'communication.fast_max_input_tokens', 'communication.fast_max_user_message_chars', 'communication.fast_min_output_tokens', 'communication.fast_desired_output_tokens', 'communication.fast_recent_message_count', 'communication.fast_recent_history_chars', 'communication.fast_snapshot_max_age_seconds', 'communication.fast_notes_max_chars', 'communication.fast_policy_max_chars', 'communication.max_background_plans', 'communication.max_concurrent_requests', 'communication.open_webui_artifacts.ttl_seconds', 'communication.port', 'communication.status_max_output_tokens', 'context.reserved_response_tokens', 'context.reserved_summary_tokens', 'context.semantic_reduction_max_calls', 'embedding_index.max_results', 'environment.max_capture_chars', 'external_benchmarks.agent_generation.agent_timeout_seconds', 'external_benchmarks.agent_generation.clone_timeout_seconds', 'external_benchmarks.agent_generation.default_max_instances', 'external_benchmarks.agent_generation.model_structured_timeout_seconds', 'external_benchmarks.agent_generation.model_timeout_seconds', 'external_benchmarks.agent_generation.solver_max_attempts', 'external_benchmarks.full_timeout_seconds', 'external_benchmarks.model_server.healthcheck_timeout_seconds', 'external_benchmarks.model_server.retry_attempts', 'external_benchmarks.smoke_timeout_seconds', 'external_benchmarks.terminal_bench.compose_download_timeout_seconds', 'external_benchmarks.terminal_bench.compose_probe_timeout_seconds', 'logging.max_bytes', 'logging.queue_capacity', 'model.benchmark_timeout_seconds', 'model.connect_timeout_seconds', 'model.context_limit', 'model.fail_safe_timeout_seconds', 'model.max_semantic_responsibilities_per_call', 'model.simple_timeout_seconds', 'model.structured_timeout_seconds', 'model.timeout_seconds', 'model.verification_timeout_seconds', 'notes.max_note_chars', 'notes.max_notes', 'notes.max_total_chars', 'perception.max_output_tokens', 'perception.timeout_seconds', 'prompt_instructions.max_instruction_chars', 'prompt_instructions.max_instructions', 'prompt_instructions.max_total_chars', 'reader.default_chunk_chars', 'reader.max_chunk_chars', 'runtime.max_open_question_chars', 'runtime.max_open_questions', 'runtime.max_validation_recovery_cycles', 'runtime.tool_timeout_seconds'])
VALIDATE_NON_NEGATIVE_PARAMETERS = frozenset(['context.safety_margin_tokens', 'context.semantic_reduction_max_input_tokens', 'logging.backup_count', 'model.remote_context_limit_fallback', 'reader.default_overlap_chars'])
