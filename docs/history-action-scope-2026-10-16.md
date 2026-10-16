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


The complete immutable combined source 275dce1 passes **1,347 tests with three
optional skips in 232.39 seconds** on Thor. This includes the search-index
correction and all authority negative cases; it is distinct from the prior
1,332-test index-only release. The preceding installed twelve-round Q3 trial
stops at its 1,800-second deadline with only schema discovery performed and the
foreground question still open. All worker/control/file negative checks pass.
The combined source is eligible for installation after the stopped-process and
private rollback gates; the next installed original-receipt trial is separately
recorded in the foreground study.


The combined code is installed on Jetson and Thor from main 07f8b5d. All 195
package files match the immutable tested source and match across hosts. The final
installed smoke checks pass two cases per host (Jetson 2.52 seconds; Thor 0.51
seconds). The canonical installer temporarily removes test-only packages; their
previous exact versions are restored before final smoke testing. Both model
services retain their original PIDs and zero restarts. The installed original-receipt
continuation starts at sequence 6168 on the existing production BF16 endpoint with
staged discovery and the explicitly diagnostic twelve-round allowance. It is a
functional gate, not a paired latency comparison or a production-default change.

The installed BF16 continuation fails with `BudgetExceededError` after 1,151.63
seconds (supervisor 1,156.02), sequences 6168–6504. Thirteen returned calls consume
1,082.65 model seconds. Its last ordinary action compilation requires 19,363
tokens against the actual 8,192-token server limit. It performs no new actor or
tool action, emits no new semantic progress, and leaves the already answered
foreground question open. Worker inventory/run counts, controls and both file
hashes remain unchanged. The schema/acknowledgement reuse correction is not
reached: no version-two history projection is produced in this turn.

The trace isolates earlier semantic selection failures. The tool-result selector
retains almost every line of a 19,309-character historical status record, including
punctuation and repetitive history; its initial wide output truncates, followed by
two successful but nearly all-preserving splits. All four history windows then
receive `protect/unresolved_work`. The history selector prompt describes factual
material as protected while failing to explain the already supported `important`
and `ordinary` summary-plus-verbatim paths. Frozen read-only prompt studies test
these two decisions separately; neither candidate is adopted merely because it
produces valid JSON. Evidence is in `installed-history-scope-long-status-analysis.json`,
`selector-relevance-study`, and `history-selector-levels-study` under the study root.
