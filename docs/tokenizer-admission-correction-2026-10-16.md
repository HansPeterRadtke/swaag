# Tokenizer admission correction — 2026-10-16

## Observed defect and experiment configuration error

The first nine-billion long-history trial used the OpenAI-compatible URL while
inheriting provider_name=vllm, completion_endpoint=/chat/completions and the
8,192-token remote fallback from the installed configuration. Setting
model.context_limit=32768 did not override that remote capacity resolution.
This was an experiment configuration error and is not a valid larger-context
comparison.

It also exposed an independent fail-open defect. The client sent a message-based
tokenizer request to llama.cpp's content-based /tokenize endpoint. The endpoint
returned HTTP 200 with {"tokens":[]}. SWAAG accepted that as an exact zero count
for a nonempty 1,170,491-character completion-review prompt. Prefix estimates
totaling roughly 390,000 tokens were canceled by the final recorded zero count,
and the request was incorrectly admitted. The server rejected 406,884 tokens
against its actual 32,768-token context. The event and exact request are retained;
this is a reproduced software defect, not merely slow model inference.

A small direct canary establishes the API distinction: the message payload gives
an empty token list, the plain-content payload gives six tokens, and applying the
native chat template followed by content tokenization gives twenty-eight tokens.
Local llama.cpp server documentation specifies /apply-template and content
tokenization; the native SWAAG adapter already implements that route.

## Correction and verification

Source 4793dcef0059982c54078350aa1a789123223efe, based on the immutable history-level
candidate fffc2d6, rejects malformed counts before exact caching and budget
admission. Counts must be nonnegative integers, excluding booleans; meaningful
nonempty content cannot have zero tokens. Token-ID lists must contain nonnegative
integer IDs. Empty text retains a valid zero count. Native prompt, remote chat,
output-tokenizer and generic exact/fallback counter boundaries use the same count
validation. Unsupported or invalid remote chat counting retains the explicitly
estimated fallback instead of inventing exactness. Exact-required callers fail.

The frozen regression baseline has eleven failures and three passes. After the
correction, all eighty-one focused context/tokenizer/cache/transport checks pass.
They cover malformed/zero responses, uncached fallback followed by a valid retry,
strict callers, valid empty text, native and chat tokenization, output counting,
connection reuse and existing replay behavior.

The exact previously rejected large request is replayed through counting only,
without generation. The compatible interface now estimates 390,164 tokens and
rejects it. The proper native interface discovers 32,768 from server properties,
counts exactly 406,884 tokens, and rejects it. This matches the generation server's
original error. Both preserve the complete request text. No context limit,
reserve, source authority, question state, or production configuration is relaxed.

The complete immutable candidate passes 1,372 tests with three optional skips
in 231.56 seconds on Thor (233.43 seconds including the supervisor). A native
ordinary-controller continuation is a separate behavioral gate. The latter keeps the original durable receipt and unchanged effect oracle,
uses the exact deployed native capacity, and permits twelve diagnostic reduction
rounds. It does not mutate the old question to obtain a passing result.

## Remaining work and evidence

The large historical review includes 562 durable evidence rows. Most remaining
bulk is actual historical snapshots, summaries, selected projections and exact
messages, with some nested accounting still inline. The tokenizer fix prevents
overflow admission; it does not establish that all semantic reductions are
efficient or reliable.

Evidence is retained under
/data/var/swaag-benchmarks/foreground-routing-20261015:
zero-token-accounting-reproduction.json, nine-billion-completion-overflow-request.json,
token-count-red.txt, token-count-focused-final.txt, token-count-live-verification.json,
completion-history-overflow-composition.json, and native-nine-billion-long-status.
The first nine-billion status candidate did not resolve the foreground question.
Its file/control/worker preservation checks remain distinct from functional success.


## Native whole-turn outcome and message-boundary correction

The correctly configured native continuation reaches its 720-second deadline:
one actor response and 58 evidence-projection responses return, taking 391.098
model seconds in total. No tool result or question resolution is produced.
Worker inventory/run counts, authoritative controls and both file hashes remain
unchanged; the foreground question remains open. There are no model-call failures
or oversized generation requests. The actual context is 32,768, but the configured
semantic-reduction input working set remains 8,192. A separately frozen follow-up
uses the full measured working set; it does not change production defaults.

Both native real-shell reruns initially stop before generation because the
Qwen3.5 chat template trims the final newline of a harness message. The strict
component-accounting guard correctly refuses the transformed string. A read-only
capture isolates exactly one missing boundary newline; the full message content
otherwise matches. This is distinct from the zero-token admission defect.

Candidate 44d29882043646cd977b2f53c12980625b1f7952 restores template-trimmed boundary
whitespace in the actual native raw prompt. Thus indentation and final newlines
remain exact both when counted and when sent to /completion. It does not merely
waive the accounting check. Other content transformations still fail, and native
protocol identity is versioned so old identities are not reused across the
serialization correction. Remote chat transport is unchanged.

The frozen baseline has five failing cases and one already-passing unchanged
message. After correction, sixty-nine focused checks pass. The captured real
prompt grows from 27,183 to 27,184 characters, preserves both messages completely,
and counts exactly 5,961 native tokens. The complete immutable candidate passes 1,378 tests with three optional skips
in 232.69 seconds (234.64 including the supervisor).

The corrected native shell trials both pass their unchanged real-effect oracles.
The successful command has first committed message/tool progress at 22.841 seconds
and a completed answer at 53.439 seconds; it creates only the exact receipt.
The expected-failure command has first progress at 18.542 seconds and completion
at 49.226 seconds; it runs once, returns the expected stderr and exit seven, and
changes no files. Both first actions contain an assistant message and a real
tool call. Manual final-answer review agrees with the observed effects. Initial
wording is unnecessarily verbose and does not pass a general concise audio-style
qualification. These tests do not override the small model's failed complex
routing cases.

The first owned GPU process is cleanly stopped with exit zero. Its entire
one-second memory record stays above 10.77 GiB available, and its PID and listener
are verified gone before follow-up admission. The separately guarded second
process is admitted with 18.5 GiB available, 10.1 GiB estimated footprint and the
unchanged four-GiB reserve. Its three-second warm startup is not compared with
the first two-minute cold load as a model-speed improvement. Existing production
model PIDs and restart counters remain unchanged.

Additional evidence: native-nine-billion-long-status-analysis.json,
native-template-diagnostic-rendering.json, native-template-diagnostic-analysis.json,
native-boundary-red.txt, native-boundary-focused.txt, native-boundary-live-replay.json,
boundary-fixed-nine-billion-shell-success and boundary-fixed-nine-billion-shell-expected_failure.


## Installed release

The combined code is merged and pushed on main at 3f58522221d72168dddd4b7726fe23c4a41493f7.
Both installed packages match all 195 tracked package files in the immutable
44d2988 candidate, and their aggregate file-hash maps match each other.
Installed smoke checks pass two cases on Jetson in 2.13 seconds and two on Thor
in 0.48 seconds. Canonical installation is performed only after all users of the
respective environment stop; exact test-only dependency versions are restored.
Private verified rollback copies preserve each previous package and metadata.

The new installed long-history diagnostic starts at sequence 7638, reuses the
same original receipt, and sets only the semantic-reduction input working set to
the measured 32,768-token native context. It retains twelve diagnostic rounds,
the original authority/effect oracle and the thirty-minute supervisor deadline.
All package bytes are rechecked against the tested candidate before this probe.
The production configuration remains unchanged. The second owned GPU runtime is
positively identified in NVIDIA process accounting at 6,492 MiB; readiness and a
fast HTTP response alone are not treated as GPU proof.
