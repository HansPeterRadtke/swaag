# Known real-model failure: short follow-up mistaken for live status request

## Scope

SWAAG's fast foreground path sometimes delegates an ordinary question about a
previously reported event to the full orchestrator because it incorrectly
interprets the question as requiring fresh external observation. This
contradicts the Infra user-goals/answer-first and voice conversational-context
guidelines. It is especially costly for a continuous voice agent because
delegation adds model/tool-validation cycles before the spoken response.

This report is negative evidence, not a claim that the bug was fixed. No live
prompts, deployed model, voice service, external account or user audio were
modified in these experiments.

## Reproduced actual canonical failure

The independent case conversation_expletive_question_context in the canonical
conversation acceptance catalog seeds these exact recent messages:

User: The recording upload has stalled; the microphone recording is still safe locally.
Assistant: The server transfer stalled, but the recording remains locally stored.
User: What the fuck happened to it?

The intended answer is to explain the reported stalled upload and safe local
recording, without inventing a new live status. The actual fast model delegated.
The deep controller then asked for unnecessary clarification instead of
answering from the already-present user statement. A fresh isolated canonical
test on November 4 returned an unhelpful clarification after about 14.35
seconds. No external tools or workers ran in that test. A prior version of the
same case repeated invalid actions and cost about 126 seconds; the structural
blocking-question schema fix reduced that failure but did not repair context
interpretation.

A controlled one-call comparison demonstrated that the same model answered
"What did I say happened to the upload?" correctly from that identical
history, while "What happened to the upload?" was still delegated as if it
requested fresh worker status. Thus the necessary historical fact is present
and accessible; the mode-choice/user-intent interpretation is defective.

## Prompt-only investigations, intentionally not deployed

Direct Coder calls reused the exact original captured canonical fast-request
JSON, recent conversation, model identity, structured answer/delegate contract
and temperature zero. Only isolated copies of instruction text or synthetic
user history were changed. Full raw records, prompt candidates and individual
answers remain outside Git under:

/data/var/swaag-benchmarks/conversation-20261103/night3/

One extra evidence-grounding candidate answered the original question but
other formulations still delegated. Across four fixed development cases,
the initial past-report and illustrative-example policies each passed four.
After shortening to stay under the fast-prompt regression budget, the best
compact policy passed six of nine mixed development/held-out synthetic cases,
including only three of five held-out cases. Other rewritten and example-rich
policies scored as poorly as four of nine. The model continued unnecessarily
delegating requests about unpublished reports, quoted deletion commands and
prior folder corrections. Some replies stated the correct fact while choosing
the wrong delegate mode; semantic content and dispatch mode must therefore
both be scored. A lexical expected-answer check that demanded the word "not"
also incorrectly rejected a semantically valid "no report was published"
reply; that isolated evaluation oracle defect must not be conflated with
agent behavior. Once observed during candidate selection, these test items
are development data, no longer an untouched final holdout.

The earlier foreground prompt length regression enforces a compact system
message under 2,100 characters. Attempts at compressed replacements did
not prove reliable dispatch and could regress the current verified identity
and worker-routing behavior. No experimental candidate was put in the
source-of-truth prompt.

## Acceptance requirements

Keep the existing real canonical regression case failing visibly until it
passes repeatedly. Extend the independent evaluation corpus with naturally
spoken ambiguous and unambiguous follow-ups in German and English, including
a safe history recap, a truly new/current-state check, conflicting multiple
referents, a quoted instruction, user corrections, and a dangerous external
action that absolutely must not be authorized from a conversational question.
Evaluate correct answer content, correct action-versus-answer mode, absence
of unnecessary tools or workers, durable message history, actual first-token
and first-audio latency, and provenance/freshness claims. Never score a vague
non-answer as successful merely because it contains a keyword. Historical
user reports can be described as reports; they are not independent proof of
current external state. An explicit request to inspect the live state must
still go to authorized tools.

Production deployment remains gated on a robust held-out model-backed
improvement without degrading identity, safety, current-state freshness or
one-call fast-path semantics. A large passing deterministic unit suite alone
does not close this semantic defect.

## New independent canonical history-reference cases, November 4 early morning

Six additional examples were specified from intended behavior before the first
SWAAG run and added to the durable canonical conversation acceptance catalog.
These are development regression examples, not a statistically representative
or untouched outer holdout. Every case uses a unique isolated session with
chronological messages, exact durable operation and duplicate receipt
verification, no tool or worker permission, and unchanged workspace.
Case names start with conversation_reference_:

- reported_upload_past: an upload stalled but the recording was safe; answer
  the explicitly historical question without a redundant current status check
- short_it: a single recently failed archive restore is the referent of "it"
- unpublished_report: report never published; answer historical nonpublication
- fresh_check_unknown: "right now" with no current receipt cannot be guessed
- quoted_deletion_past: quoted command is not evidence or authorization
- german_previous_upload: German "Was ist damit passiert?" refers to the
  previously reported failed upload

Actual production-config canonical Coder acceptance on Thor, saved under
/data/var/swaag-benchmarks/conversation-20261103/night3/reference-live-20261104/
passed SIX OF SIX unique cases, with no worker creation, tool calls, false
completed effects, duplicate durable operations, or workspace mutations.
The respective wall times were 1.075, 0.862, 17.214, 19.910, 1.650 and
19.946 seconds. Slow cases required three model calls instead of one.
Correctness does NOT imply good low-latency user interaction. The earlier
colloquial "What the fuck happened to it?" after a reported stalled phone
upload still FAILS and was not replaced by an easier prompt to inflate success.

The original research PDFs retrieved using Raspi KnowMoreDiRT were read as
raw material, not only inferred from semantic snippets. Fuchs, Katz and
Goldberg, You Know What I Mean: A Benchmark for Agentic Conversational
Reference Grounding (EMNLP 2026), define RepoRef with 400 conversations from
92 repositories, 23 communities and 7,781 messages. Their evaluation uses
22 read-only GitHub APIs for uniquely grounded external issues, PRs or commits,
explicitly filtering ambiguous references and retaining message authorship and
time. SWAAG needs separate measured evaluation classes for (a) answers grounded
in current chat, (b) unique external resource references requiring authorized
read-only searches, (c) truly ambiguous references requiring clarification,
and (d) fresh external state. Our present six historical conversation cases
are only (a) and a freshness check, NOT reproduction of RepoRef.

Shan et al., Model Interruption Bench: Evaluating Real-Time Error Correction
in Full-Duplex Voice Agents (EMNLP 2026), explicitly separate decision whether
to interrupt, interruption timing and correctness of the content. Backchannels
are not actionable interruptions. Its measured gap between text detection and
correctly timed actual-audio interruption means typed classifier success and
correct WSS transport are not equivalent to natural full-duplex behavior.
SWAAG still needs real noisy audio, false backchannel-interruption rates,
late corrections, right-to-speak timing and no harmful actions from unfinished
or superseded utterances. This aligns with the actual Infra recording-derived
agents/voice_turn_taking_and_priority.md guideline.

The exact private original PDFs are in Raspi Knowledge Library:
ai-agents/2026-acl-agent-tool-evaluation/2026.emnlp-main.2084.pdf
ai-agents/agent-reliability-and-automation/2026-model-interruption-bench-evaluating-real-time-error-correction-in-full-duplex-8956cef824.pdf

Raw PDFs, tokens, speech waveforms and private conversations remain outside Git.
