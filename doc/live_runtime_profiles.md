# Live Runtime Profiles

Runtime profiles are transport/deployment configuration only. They must not encode benchmark answers, task-family semantic routing, or hidden recovery shortcuts.

## Core invariants

Every semantic control call uses provider-supported generation-time constrained decoding and local schema validation. The runtime never silently falls back to unconstrained control text. Model/context/profile choice is external configuration; the agent does not restart or swap model profiles inside its semantic loop unless the user/operator explicitly requests that behavior.

The active context limit is discovered from the configured server/model or declared explicitly in configuration. Prompt assembly performs token budgeting against that effective limit and reserves output headroom. Documentation must not assume stale machine-specific profiles such as a fixed 2K local context when the active server has changed.

## Sequential semantic execution

The SWAAG main-agent loop is sequential. `parallel` serving settings and GPU process concurrency are deployment concerns, not permission for the runtime to fan out semantic agent decisions. Independent tools/processes may run concurrently where their mechanics require it, but accepted semantic decisions are serialized through the main loop.

## Timeouts and observability

Long model calls are not automatically dead calls. Connect timeout, request timeout, progress polling, cancellation, and retry policy are independently configurable and recorded. Progress events are observability; they do not change semantic behavior.

## Effective configuration

Provider base URL, model identity, context limit, structured-output mode, timeouts, poll cadence, and other deployment settings must resolve through a single effective configuration with explicit precedence. Validation/benchmark commands must report that effective configuration rather than rely on undocumented machine defaults.

## Performance measurements

Performance measurements belong to dated benchmark artifacts or infra/model documentation, not timeless agent semantics. When hardware/model configuration changes, old throughput numbers must not be presented as the current runtime profile.
