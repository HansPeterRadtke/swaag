# Model Semantic Authority and Project Continuation

## Non-negotiable architecture

SWAAG is a model-directed agent. The model owns every decision whose correct
answer depends on meaning. Deterministic code may execute, validate, constrain,
record, retry, replay, budget, and objectively verify; it may not substitute its
own semantic judgment for the model.

This rule has higher priority than benchmark scores, convenience, latency, or a
locally successful shortcut. A semantic shortcut is invalid when a synonymous
wording, different grammar, abbreviation, reordered request, unfamiliar domain,
or another language could change or break its behavior.

## Decisions that belong to the model

The model must decide user intent, completeness, ambiguity, task type, strategy,
decomposition, assumptions, semantic source and target roles, tool choice,
semantic tool arguments, content, prioritization, recovery, replanning, and the
final response. The model must also decide whether apparently similar tasks are
actually equivalent. These decisions must use structured model contracts when
machine validation is needed, rather than Python recreating the decision with
heuristics.

## Deterministic code that is allowed

Deterministic infrastructure is allowed when the same algorithm is correct
regardless of the user's wording or language. Examples include schema
validation, JSON parsing for an explicitly requested JSON contract, path
normalization, permission checks, sandboxing, tool execution, file byte
operations, append-only history, replay, token accounting, timeout and retry
limits, process polling, hash verification, syntactic validation, and objective
checks such as exit status or whether an explicitly declared test command
passed.

Deterministic code may reject an unsafe or malformed action, but it must not
silently invent a different semantic action. When model output is invalid, the
normal response is a bounded model retry with precise validation evidence, not
a hand-authored semantic repair.

## Forbidden patterns

Do not classify intent through regexes, keyword lists, English phrases,
filenames, extensions, directory names, benchmark task identifiers, or prompt
fragments. Do not install fixed semantic workflows before model planning. Do
not infer source and target roles from naming conventions. Do not select a tool
because a particular verb appeared. Do not compute requested business or domain
content in runtime code. Do not parse tests or errors to derive the expected
semantic answer. Do not rewrite model output according to benchmark-shaped
rules. Do not generate semantic final answers from templates. Do not preserve a
shortcut merely because current tests pass.

Mechanical parsers must remain mechanical. For example, parsing a validated
model-produced plan schema is allowed; scanning free-form user text to build
that plan in Python is not. Running a test command chosen by the model is
allowed; deriving the intended patch from assertion text is not.

## Required implementation pattern

Represent semantic decisions as explicit model contracts with schemas. Validate
the response mechanically. Feed validation, tool, or verification failures back
to the model. Let the model choose the corrected decision. Keep execution and
state transitions deterministic after the semantic decision is accepted.
Record enough evidence in history to explain which model decision caused each
action and which objective result verified it.

Semantic follow-up decisions must receive relevant observations, not only event
metadata. If a tool returns observed text, structured data, command output, or
verification evidence that is needed for planning, tool choice, recovery, or a
final response, prompt context must expose that evidence within the configured
budget. Deterministic code may trim, budget, escape, and mark trust boundaries,
but it must not replace relevant observed content with metadata-only summaries
that force the model to infer unseen facts.

Recovery evidence includes failed tool arguments, validation errors, verifier
evidence, and the latest known artifact snapshots. If a previous source
snippet, range, pattern, or other target no longer appears in the current
artifact, deterministic code may expose the current text and the failed target
as mechanical facts. It must not choose the semantic repair, treat a stale
pattern failure as already applied, or accept a broad partial match as the
requested final state. The model must decide the next repair, observation,
clarification, or blocker from the exposed evidence.

Fallback behavior must preserve model authority. A malformed model response may
trigger a bounded retry, a request for clarification, or a transparent failure.
It must not trigger a hidden Python classifier, fixed semantic plan, or normal
semantic final answer for work that has not been verified.

The runtime may emit a mechanical incomplete-status message when a watchdog,
budget, duplicate-action guard, or other safety boundary stops unresolved work.
That status is not semantic task content; it is an execution-status report that
must clearly say verified success was not reached.

## Constrained decoding and tool disclosure

Every live model call must use generation-time constrained decoding. Semantic
text outputs, including final answers and clarification requests, use a strict
closed JSON object contract such as `{"text": string}` and are unwrapped only
after validation. Live calls must not use plain text, generic JSON object mode,
raw grammars, or provider-specific shortcuts that are not implemented by the
active adapter.

Provider adapters are responsible for the transport shape:

- OpenRouter and OpenAI-compatible providers use Chat Completions Structured
  Outputs with `response_format.type=json_schema`,
  `response_format.json_schema.strict=true`,
  `response_format.json_schema.schema=<portable schema>`, and
  `provider.require_parameters=true`.
- `llama.cpp` uses the top-level `json_schema` payload field.

All generation schemas must stay within the portable subset: root object,
closed objects with `properties`, `required`, and
`additionalProperties:false`, every property required, `enum` for fixed string
values, arrays with `items`, nullable fields via `anyOf` with `null`, and no
open maps, conditionals, top-level arrays or strings, `const`, or min/max
validation keywords. Numeric, length, count, and operation-specific checks
belong in local validation after generation.

Semantic tool selection, planning, recovery, and delegation decisions must
receive the complete enabled tool registry. Each tool must be shown by
registered name, meaningful description, complete input schema, and registered
usage guidance. This applies to task decision, planning, direct/tool routing,
replanning, recovery, and subagent-selection calls. The tool decision call
chooses only whether to respond or which tool to call. A separate selected-tool
argument call then receives that selected tool's registered documentation and
schema. Python must not preselect a tool from skills, filenames, prompt words,
or expected plan labels.
Registered tool names are exact identities. Deterministic code must not define
tool equivalence tables, synonym sets, or special cases such as treating two
read tools as interchangeable. If execution produces a selected tool that does
not exactly match the model-authored `expected_tool`, runtime records the
mismatch and asks the model to correct the next constrained decision.

When the task-decision contract returns `execution_mode=single_tool`, that
choice only tells the planner model that the task may fit one tool call. The
runtime must not turn `preferred_tool_name` into a deterministic plan or generic
success criterion. A planner call with the complete registry must still declare
the step sequence and objective verification.

Plan validation is a mechanical boundary around model-owned planning. The
generation schema exposes the planner-supported verification type
(`composite`), and planning instructions list the supported check types. If a
schema-valid plan fails local validation, the runtime may make bounded
constrained planner retries with validation evidence. It must not fill missing
required conditions, invent check names, or rewrite step semantics. The bound
must be large enough for multiple distinct structural corrections; exhaustion is
a fail-closed model-contract failure, not permission for Python to synthesize a
plan.
Plan `input_text` is not executable tool input. It is model-authored instruction
context for the later step and may contain placeholder-looking text as inert
context. Actual selected-tool arguments are generated only by the selected-tool
input call after the step starts, with the selected tool's registered
documentation and complete schema.
Verification may expose the latest tool result under the current step's
model-declared `expected_output`, `expected_outputs`, and `output_refs` names.
That aliasing is mechanical graph bookkeeping so `artifact_present` can refer
to the model's own labels; Python must not use it to invent or transform
semantic content.
Tool definitions may declare required objective verification check types, such
as a resulting-file check for file-mutating tools. The model plan must declare a
matching check with `condition="required"`. If that check is missing or marked
optional, plan review rejects the plan with structured validation evidence and
asks the model for a corrected constrained plan. Python must not promote
optional checks, rewrite condition importance, or synthesize the expected text,
path, command, or criterion. Each model-authored check carries its own local
condition status; artifact, input, output, and expected-output labels remain in
their dedicated dataflow fields. For mutating tools, existence and
nonempty-output checks are mechanical evidence but not objective proof unless
the model also declares the tool's registered objective check type as required.
Original file contents, diffs, and write provenance are mechanical history
records. Visible backup files are disabled by default because creating extra
workspace files is a user-visible side effect that can violate tasks requiring
no unrelated changes.
For semantic response or reasoning checks, at least one model-declared
`criterion` check must carry `condition="required"`, unless the model declares a
required exact/string match against `assistant_text`. A criterion that is only
optional, or a plan whose required checks contain only dependency/nonempty
checks, cannot prove a semantic response and must be rejected before the step
executes.
For file-mutating tools, a passing objective check still does not by itself
prove semantic correctness. Runtime must run a constrained model-owned result
review with the step goal, success criteria, deterministic evidence, tool
output, diff, and current file text. The model returns the
`result_satisfies_step` verdict through the portable `verification` schema.
Python may enforce this gate and feed rejection evidence back to the next model
decision; it must not decide from substrings, filenames, or language-specific
patterns that a mutation was "close enough."
Required semantic review gates fail closed. If the semantic reviewer backend is
unavailable, degraded, or returns output that violates the structured contract,
the required reviewer perspective fails and recovery/replanning receives that
evidence. Exact literal `assistant_text` matches declared by the model can be
checked mechanically; all other semantic relevance belongs to a constrained
model call.
Final response success also needs model-owned whole-goal proof. Before exposing
a final answer as completed work, runtime asks the verifier model whether
`final_objective_satisfied` is true for the original user request, the candidate
answer, the active plan, recent tool and verification observations, and current
workspace evidence. A negative verdict is a verification failure that returns
to recovery/replanning. Python may assemble and bound the evidence, but it must
not infer that a weakened later plan or partial current artifact satisfies the
original request.
If an earlier model-declared objective check failed, a later recovery plan may
still choose to observe the current state and answer when the evidence already
appears sufficient. Python may record that the unresolved step-level check is
deferred, but it must not decide semantic sufficiency; only the constrained
`final_objective_satisfied` verifier can discharge the whole objective.
Live plan condition status is local to each verification check, so the model
cannot create a mismatched check-name cross-reference. The parser mechanically
translates those explicit statuses into internal required and optional lists for
execution and persisted history. Legacy saved plans may still be normalized for
structural dependency bookkeeping, but live plans with no required check are
invalid and the model must correct them. Python must not invent task-specific
checks, expected content, tools, success criteria, or condition importance.
Dependency validation is also mechanical. Python may reject self-dependencies
and cycles because they make a declared plan graph impossible to execute, but
it must return that evidence to the model rather than choosing a corrected
semantic order.

Plan-step action validation is mechanical. If the model-authored plan step
requires an `expected_tool`, a later `respond` or other non-tool action for that
step is invalid execution evidence, not a final answer. A later tool choice
that conflicts with `expected_tool` is also invalid for that step and is not
executed. The runtime may record the rejected action/tool and expected tool,
then ask the model again within repeated-action and watchdog bounds. It must
not choose the repair tool or synthesize the missing answer itself.

Repeated-action protection is also mechanical. The runtime may compare exact
closed structured actions within one step and stop executing byte-equivalent
repeats, even when previous side effects changed edit counters or history
metadata. After a failed preview, an exact repeated mutation hands off to the
normal verifier so failure evidence can drive model recovery. This guard must
not infer that two different actions are semantically equivalent, and it must
not choose replacement arguments.

Side-effect tool arguments must be concrete. Python may reject unresolved
artifact placeholders before mutation because that is a safety validation, but
it must not decide what the missing content should be. Objective containment
checks such as `file_contains` must declare concrete expected text through
`pattern`, `expected`, or `expected_json` and should be precise enough to reject
partial or corrupt edits. An empty target must fail.
For observed text edits, `edit_text` should use the portable `replace_exact`
operation with model-supplied `old_text` and `new_text`. The editor enforces an
exactly-one-match precondition, supports dry run, applies atomically through
the normal environment path, and returns explicit zero-match or multiple-match
errors. `replace_range` is allowed only as a low-level operation protected by
`expected_text`; raw offsets are not the preferred simple editing affordance.
Text-edit pattern absence must fail closed even if the replacement text is
already present somewhere in the current file. Treating that as "already
applied" would be a deterministic semantic inference about intent; the model
must see the current text and choose any recovery edit.

Skills are prompt text plus descriptive metadata only. A skill may add
instructions, information, examples, expected outputs, and verifier hints. Skill
metadata must never remove, hide, filter, or preselect tools for the model.

Semantic retrieval failures are handled mechanically. If a constrained
model-scored relevance call violates its schema after bounded attempts, SWAAG
records `semantic_retrieval_degraded` and rebuilds context with the neutral
`unavailable` backend. That fallback performs no keyword, filename, regex,
TF-IDF, language, or benchmark scoring; it only preserves execution continuity
with already recorded observations.

## Review procedure for every change

Before accepting a change, identify every branch that can alter intent,
strategy, decomposition, file roles, tool choice, tool arguments, content,
recovery, or final wording. For each branch, ask whether it would remain correct
for arbitrary paraphrases, synonyms, abbreviations, reordered clauses,
implicit references, unrelated domains, and other languages. If correctness
depends on words, names, patterns, or benchmark fixtures, move the decision to a
model contract or remove it.

Review both new code and code reached indirectly by the change. Search for
semantic regexes, keyword sets, phrase matching, filename-driven workflows,
fixed planners, deterministic answer builders, benchmark identifiers, and
special recovery functions. Inspect tests for fixtures that merely bless those
shortcuts. A regression test should prove model ownership and schema validation,
not encode one preferred English prompt.

## Mandatory checklist

A change is not complete until all of the following are true:

- Every semantic branch is backed by an explicit model decision.
- Deterministic branches are demonstrably semantics-independent.
- Invalid model output is retried or rejected, not semantically rewritten.
- Tool execution follows a model-selected action and validated arguments.
- Plan `input_text` is reviewed as instruction context, not executed as tool
  JSON.
- Output-ref artifact aliases point to observed tool results only.
- Registered tool objective-verification requirements are enforced without
  Python inventing the objective content or promoting optional checks into
  required checks.
- File-mutating tool steps cannot complete until constrained model-owned result
  review accepts the observed artifact state.
- Final answers cannot be exposed as success until constrained model-owned
  final objective verification accepts the original request, candidate answer,
  and concrete current evidence.
- Recovery plans that omit a repeated step-level objective check after a failed
  check are accepted only as far as the mandatory final objective proof; Python
  records the deferral and never treats it as semantic success by itself.
- Side-effect tool arguments contain concrete values, not unresolved artifact
  placeholders.
- Planning, tool choice, and selected-tool argument prompts include complete
  enabled tool descriptions and schemas where tools are relevant.
- Planning, tool choice, recovery, and final-answer prompts include relevant
  observed evidence from prior tools/history; content needed for the next
  semantic decision is not reduced to metadata-only summaries.
- `execution_mode=single_tool` still routes through model planning and
  objective verification; no deterministic direct-tool plan is created.
- Every live model call has a strict closed JSON-schema contract and the
  provider adapter enforces it at generation time.
- All generation schemas pass the portable-schema validator.
- Recovery and replanning return to the model.
- Plan validation rejects inconsistent semantic conditions with structured
  evidence; it never silently repairs model-authored importance or meaning.
- Objective verification checks fail when their declared machine-checkable
  target is empty or malformed.
- Final semantic content comes from the model and is exposed only after the
  response step and whole-goal final objective proof verify.
- Unresolved safety exits report incomplete status rather than semantic
  completion.
- No benchmark identifier, fixture wording, domain constant, filename pattern,
  or language-specific phrase controls behavior.
- Tests include materially different paraphrases and at least one non-English or
  language-neutral case where practical.
- Architecture scans and code review find no semantic leakage.
- Full unit and agent evaluations pass without weakening this invariant.

## Instructions for future contributors and Codex sessions

Start by reading this document, `README.md`, `doc/architecture.md`, and
`doc/runtime_loop.md`. Inspect the complete working-tree diff before editing.
Treat existing code as untrusted with respect to semantic authority, even when
it predates the current work or has strong test coverage. Prefer deleting a
semantic shortcut over generalizing its list of phrases. Do not add more
synonyms, more regexes, more language cases, or more benchmark branches; those
make the architecture less general, not more general.

When repairing a behavior, first define the model contract that should own the
decision, then define deterministic validation and execution around it. Add
architecture-focused tests before benchmark-specific tests. Run targeted tests,
the full deterministic suite, semantic-leakage searches, and only then bounded
live smoke validation before any broad benchmark. Report architecture
compliance separately from behavioral scores. Never claim completion from
benchmark success alone.

When pragmatic deterministic behavior is proposed, document why it is
semantics-independent, what invariant it enforces, and why arbitrary wording or
language cannot affect its correctness. If that argument cannot be made
precisely, the behavior belongs to the model.
