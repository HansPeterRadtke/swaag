# History reuse across ordinary action delivery — 2026-10-16

The installed twelve-round diagnostic reaches an ordinary actor action at event
5827 and successfully loads the `questions` schema at event 5833. The next action
preparation returns to authoritative raw history (events 5840/5843), including
both large historical status results, rather than reusing the previous derived
view. The actual task, latest user authority and system instructions are unchanged.
The history identity includes loaded schemas, the corresponding constrained action
contract, and the initial-acknowledgement flag. These change during ordinary
schema discovery and after the first accepted action.

A controlled reproduction independently changes schema loading, acknowledgement,
and both. The unchanged baseline reduces the same history twice in all three
cases instead of reusing the first successful reduction. The initial fixture
omitted the required empty `required` schema list; its failures were fixture
errors, not this reproduction. Both that failed attempt and the corrected
three-failure baseline are retained. The baseline accepts an otherwise unused
optional argument as a test seam; its original fingerprint behavior is unchanged.

The ordinary combined action loop now supplies a semantic capability scope built
from the complete currently permitted tool catalogue, exact schemas, descriptions,
guidance, delegated specifications, and full action contract, including silent
completion policy. Loading an already permitted schema does not change that scope.
The actual request still exposes only loaded tool schemas and still enforces the
initial acknowledgement. Both are presentation/admission details, not changes to
the user's objective or permission to act.

The history identity retains the exact original request, later user interventions,
latest user source identity, validation feedback, selected system instructions,
operation kind and prompt mode. Changed permitted capabilities still invalidate
reuse. Raw history wins whenever it fits; newly appended observations and a fresh
runtime snapshot remain in the reused view. Other operation-specific callers keep
their previous exact contract scope. The new ordinary scope uses a new fingerprint
version, so old reductions are rederived once rather than silently trusted under
broader identity semantics.

All 41 final focused history and orchestrator-control checks pass in 61.27 seconds.
They include the three presentation changes, actual ordinary schema loading,
removal of an available capability, changed user/policy authority under both scope
modes, missing provenance, appended observations and durable replay. The earlier
focused run passed 41 checks and failed three invalid-fixture cases; it is not
reported as a successful run. The complete immutable suite and a further live
original-receipt continuation are separate gates before installation.

Evidence under `/data/var/swaag-benchmarks/foreground-routing-20261015` includes
`history-action-scope-red.txt`, `history-action-scope-failure-detail.txt`,
`history-action-scope-red-corrected-fixture.txt`,
`history-action-scope-focused.txt`, and
`history-action-scope-focused-final.txt`. This change does not qualify semantic
summaries as always accurate, remove locks, raise production reduction allowances,
or enlarge model context.
