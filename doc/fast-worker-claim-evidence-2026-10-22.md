# Fast worker-inventory claim evidence — 2026-10-22

## Original source failure

The short Coder response may receive no usable current worker world: the background state was stale, incomplete or unavailable, or durable revisions have changed. The fast prompt says not to invent current state, but its proposed text is untrusted semantic output and it previously passed direct reply normalization even for explicit current-count assertions such as "There are no active workers." when no current world existed. It also accepted a claim of three active workers against a fresh observed count of one. These are not verified state observations, yet the fast path could commit them as the canonical answer without inspecting the authoritative system.

New negative regression cases reproduced the observed source behavior before modification: current zero/one/three assertions passed with no observation; a worker-is-currently-running assertion passed with no observation; false active counts passed against one observed active worker; and claims passed against an explicitly incomplete snapshot.

## Boundary correction

This change is intentionally restricted to an **output evidence check**, not a Python user-message intent router. The first-stage model still decides whether to answer or delegate. Existing input authority, exact current message, original-request forwarding, and the ability to answer pure conceptual questions remain unchanged.

The guard recognizes a narrow present-tense worker-count assertion with an unambiguous integer or zero-to-twenty number word. Without a complete fresh worker-world observation, it delegates instead of treating that assertion or an explicit worker-is-running claim as a verified fact. With a complete current worker world, the claimed active-worker count must agree with the list of nonterminal workers and with a required explicit integer count. Invalid or missing inventory lists, boolean/null counts, invalid worker rows, and contradictions between the rows and the count all fail closed for live worker claims, even if a caller has incorrectly labeled the snapshot complete. An unsupported/contradicted claim is replaced by the preexisting uncertainty acknowledgement and the original user message goes to the full authoritative orchestrator.

The guard does **not** infer the meaning of "what are workers doing" from a regex on the user's request; it checks only the claim the model chose to make. It does not mechanically infer progress/ETA, workers started, or work completed. It does not claim to understand every language or every way of paraphrasing a status; model semantics and deeper evidence verification remain necessary. A separately supplied historical quotation such as "The warehouse worker has stopped" remains answerable when the user explicitly asks what a past statement claimed. An accurate count against fresh evidence remains eligible for one-call fast response.

## Test and production boundary

Focused tests include normalizer source-gating, fresh count agreement/disagreement, preservation of historical-report wording, zero-worker fast path, and end-to-end fallthrough from a fabricated worker count to the full action controller. The refined focused test set passed **sixty tests**, with no model server contact. Five new malformed/inconsistent-current-world inputs were exercised after a negative pre-fix reproduction, plus a positive well-formed zero-worker input.

The isolated installed-package candidate constructed before the malformed-snapshot refinement passed fifty-four focused tests with exact package byte parity; it must not be mistaken for acceptance of the later refinement until rebuilt. The complete isolated source regression suite then passed **1,724 tests with three optional skips and zero failures in 346.59 seconds**. The installed-package candidate used earlier in this investigation passed the first fifty-four tests before the malformed-snapshot refinement; a final installed candidate for the refined code still needs acceptance. This is not proof of real microphone-to-speaker delivery, cancellation safety, latency under GPU contention, or semantic accuracy on arbitrary natural language. The running voice gateway/communication service and backend models must not be restarted based on this narrow source test alone.


An isolated Thor Python microbenchmark ran 15,000 guard checks per case, reporting approximately 4.4 microseconds per valid count, 2.4 microseconds for a contradicted count, and 1.8 microseconds for a missing-snapshot delegation. This measures only the deterministic guard, not model inference or user response latency.
