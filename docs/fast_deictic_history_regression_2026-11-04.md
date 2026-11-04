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
