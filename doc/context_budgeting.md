# Context Budgeting

## Core rule

Prompt packing is now dynamic and call-type-aware.

The runtime no longer treats tiny absolute section caps as the main policy.
Instead it:

1. reads the active model context window,
2. classifies the call kind,
3. derives an output ceiling from that call kind,
4. derives a safety margin from context size plus call kind,
5. computes a safe input budget,
6. asks the semantic-selection layers what context is relevant,
7. packs the resulting sections by exact token cost until the budget is full.

The human-editable policy source is:

- `src/swaag/assets/defaults.toml`

The relevant groups are:

- `[budget_policy]` for call-class sizing, safety margins, fixed overhead, and
  structured-output reserve policy
- `[context_policy]` for packing hints and retrieval preview sizes
- `[selection_policy]` for retrieval/skill/detail-query tuning

Keep the layers separate:

- server/model facts
  - live in `[model]`
  - example: `context_limit`
- user-tunable policy
  - lives in `[budget_policy]`, `[context_policy]`, and `[selection_policy]`
- derived allocations
  - are calculated at runtime from those facts and policies
  - they are not stored as hand-maintained constants

## Call budget plan

`swaag.budgeting.compute_call_budget(...)` derives a `CallBudgetPlan`:

- `context_limit`
- `output_tokens`
- `safety_margin_tokens`
- `fixed_overhead_tokens`
- `safe_input_budget`

The output ceiling and safety margin scale with the active context size and
the call kind:

- `analysis`, `task_decision`, `action` => tiny output budget
- `strategy`, `failure`, `verification`, `tool_input` => small
- `expansion`, `plan`, `summary` => medium
- `answer` => large

The old config values such as `reserved_response_tokens` and
`reserved_summary_tokens` remain only as lower-bound guard rails for legacy
paths and tests. They are not the main packing policy anymore.

Structured-output reserve policy is also config-backed now:

- per-contract JSON reserve multipliers live under
  `[budget_policy.structured_output_json_factor_by_contract]`
- default JSON and grammar reserve factors live in `[budget_policy]`
- hard lower floors for structured reserves are explicit config values rather
  than hidden constants in code

## Section budgeting

`swaag.budgeting.compute_section_budgets(...)` splits the safe input
budget across optional context groups:

- history
- workspace files
- semantic memory
- guidance
- skills
- notes

These are proportional allocations derived from the safe input budget, not
small absolute caps tied to a single context size.

## Semantic selection first, exact packing second

The runtime separates two concerns:

1. Semantic relevance
   - retrieval picks relevant history, memory, and files
   - guidance resolution picks relevant guidance layers
   - skill selection picks relevant skills and tool exposure
2. Deterministic packing
   - exact token counting is used whenever possible
   - sections are packed until `safe_input_budget` is exhausted
   - lower-priority sections are dropped when they do not fit

This keeps semantic choice with the LLM-driven selectors and keeps exact
budget enforcement in deterministic code.

## Exact vs conservative counting

Exact mode:

- uses llama.cpp `/tokenize`
- records tokenize requests/results in history

Conservative fallback:

- uses the chars-per-token estimator
- is only used when exact tokenization fails and fallback is enabled
- records `token_estimate_used`

## Admission rule

A request is admitted only if:

```text
input_tokens + reserved_response_tokens + safety_margin_tokens <= context_limit
```

The `reserved_response_tokens` and `safety_margin_tokens` in the final
`BudgetReport` come from the dynamic call budget plan for that call kind.

## Compaction

When a request does not fit and compaction is enabled:

1. older history is considered for summary compression,
2. the largest summarizable prefix is found under the current budget policy,
3. the summary call itself is budgeted exactly like any other model call,
4. `summary_created` and `history_compressed` are appended to history,
5. prompt building retries with the compressed history state.

Original history is never deleted.

## Traceability

The runtime records:

- `prompt_built`
- `budget_checked`
- `budget_rejected`
- `context_built`
- retrieval / guidance / skill traces inside the context payload

Every admitted model call therefore has both:

- an exact or conservative token accounting record
- a trace of why each major context source was included or dropped
