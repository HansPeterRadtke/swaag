# Model-call decomposition study

## Purpose

This study evaluates the current Swaag LLM-call interfaces before any production redesign. The goal is not to make deterministic code judge semantic correctness. The goal is to characterize, externally and reproducibly, which semantic task shapes the currently available small model can perform reliably, which fused task shapes cause systematic failures, and whether proposed decompositions improve reliability without silently changing the semantic responsibility of the model.

No production call graph is changed during this study. The running model process is not started, stopped, restarted, or reconfigured by the study.

## Model and reproducibility protocol

The study target is the already-running Jetson llama.cpp endpoint at `http://127.0.0.1:14829`, currently serving SmolVLM2-2.2B-Instruct Q4_K_M with an 8192-token context.

Each case retains the exact prompt, exact JSON Schema, model request, structured output, context accounting, seed, temperature, timing, and expected semantic criterion under `/data/var/swaag/manual-tests/model-call-decomposition-study-20260829/`.

Two sampling regimes are kept separate:

- **Production replication:** temperature `0.0`, top-p `1.0`, seeds `17`, `42`, and `91`. Seed should not materially affect greedy decoding; disagreement across these runs would itself be unexpected evidence about the backend.
- **Robustness probe:** temperature `0.2`, top-p `1.0`, seeds `17`, `42`, and `91`. These runs are not current-production behavior. They test whether a result depends on one narrow greedy trajectory or remains stable under small sampling perturbations.

A hypothesis is not considered supported merely because one run succeeds. At minimum the exact production-regime result and all three robustness seeds are reported. Any disagreement is retained, not averaged away.

## What counts as evidence

The study harness may judge a case only because the case itself has externally specified ground truth or a narrowly defined expected semantic property. Production Swaag does not gain any mechanism for detecting semantic model errors from this study.

Mechanical failures such as malformed constrained output, impossible schemas, context overflow, invalid existing-object references, or backend contract violations are classified separately from semantic model behavior.

## Main design danger

Decomposition is not automatically safer. Splitting one semantic operation into several calls can introduce information loss between stages, inconsistent decisions, increased latency, additional context-budget pressure, loss of global reasoning, and correlated error cascades. A decomposition is useful only if the simpler calls are demonstrably more reliable and the interfaces between them preserve the semantic information needed by later stages.

The study therefore tests both the proposed simpler primitive and, where possible, the fused current call on the same scenario.

## Hypotheses and falsification criteria

### H1 — Action fusion overload

**Observation motivating the hypothesis:** SmolVLM2 can choose `calculator` and produce `{"expression":"37 * 19"}` under a simple tool-choice/arguments contract, and can populate a nested `tool_calls` array when that is its only responsibility. The full `agent_action` contract repeatedly emits no tool call.

**Hypothesis:** Reliability improves when action semantics are decomposed into smaller decisions instead of asking one call to simultaneously emit assistant prose, tool calls, continuation state, status, and blocking questions.

**Candidate decomposition:** `action class -> tool choice -> tool arguments`, with question text or final answer generated only on the branch that needs it.

**Danger:** The first-stage classifier may discard nuances needed for the later tool choice; multiple calls may disagree; latency and total tokens increase; a globally obvious action may become less reliable after decomposition.

**Falsified if:** the decomposed stages are not materially more reliable than the fused action across the seed matrix, or if information passed between stages is insufficient to recover the correct action.

### H2 — Mechanically derivable action fields should not be semantic outputs

**Hypothesis:** Fields whose values are mechanically implied by a semantic decision, especially `continue_loop` if it is fully determined by whether tool calls are emitted, needlessly burden weak models and can reduce action reliability.

**Danger:** The field may encode a semantic distinction not captured by the apparent mechanical relation; removing it could eliminate legitimate cases such as continued reasoning without a tool call or intentionally terminating after a tool result.

**Falsified if:** repository/runtime analysis finds valid states where the field cannot be mechanically derived, or isolated tests show no reliability difference when the redundant field is removed.

### H3 — Existing-object identity should be selected, not regenerated

**Observation:** The small model sometimes preserves the semantic task while mutating exact identifiers, such as changing `report.txt` to `hello.txt` or inventing evidence sequence numbers.

**Hypothesis:** When a semantic decision concerns an already existing object, asking the model to choose from constrained opaque IDs and letting deterministic code recover the exact underlying representation is more reliable than asking the model to regenerate paths, hashes, timestamps, sequence numbers, or other identifiers.

**Danger:** Candidate enumeration can make schemas large; an object may not yet have an ID; selection by opaque ID may reduce semantic cues; the wrong object can still be selected.

**Falsified if:** constrained-ID selection does not improve exact-reference accuracy or causes materially worse semantic selection.

### H4 — Large subset selection overloads weak models

**Observation:** note and prompt-instruction selection over-selected obviously irrelevant garden/audio/travel candidates.

**Hypothesis:** Binary relevance judgments, pairwise comparisons, or small-batch selection are more reliable than selecting an arbitrary subset from a heterogeneous candidate list.

**Danger:** Independent binary decisions can over-select even more because each candidate lacks competition; pairwise ranking can be non-transitive; many calls increase latency and may consume more total tokens than a fused selection.

**Falsified if:** binary/pairwise/small-batch variants do not improve precision and recall across seeds relative to the current subset-selection contract.

### H5 — Separate selection from rewriting for response relevance

**Observation:** the response-relevance call preserved obvious PID/hash/shell-command noise instead of filtering it.

**Hypothesis:** a first semantic call selecting relevant source units, followed by deterministic exact retention and an optional second prose-generation call, is more reliable than asking one call to simultaneously decide relevance and rewrite the answer.

**Danger:** chunk boundaries can destroy context; exact-unit selection may create incoherent prose; relevance can depend on relations across chunks; the second generation call may reintroduce omitted material or lose selected details.

**Falsified if:** selection-first does not improve retention of required facts and removal of irrelevant material, or produces unusably fragmented inputs for the second stage.

### H6 — Difference identification and acceptability judgment should be separated

**Observation:** presentation evaluation correctly described a missing DNS blocker but simultaneously emitted `acceptable=true`.

**Hypothesis:** asking one call to identify semantic differences and another small call to judge whether those identified differences are acceptable is more reliable than requiring internally consistent difference analysis plus verdict in one object.

**Danger:** the first call can miss the decisive difference, making the second stage confidently wrong; splitting removes the possibility that the acceptability judgment guides what differences matter.

**Falsified if:** the two-stage variant does not reduce contradictions or misses more material differences than the fused call.

### H7 — Semantic extraction before compression may preserve exact information better

**Observation:** large tool-result projection recognized categories such as ticket/deadline/constraint but lost the exact values.

**Hypothesis:** asking the model to select or extract relevant exact source spans/records before asking for compact synthesis can reduce identifier/value loss compared with direct large-source summarization.

**Danger:** exact-span selection is itself a semantic task; relevant information can be distributed across spans; extraction may consume more context and calls than direct compression; model-generated offsets are unsafe unless constrained to real source-unit IDs.

**Falsified if:** extraction-first does not improve required-fact preservation or produces materially higher total context/call cost without reliability gain.

### H8 — Completion judgment is already simple enough to remain fused

**Observation:** the initial complete/incomplete completion cases were mostly semantically correct.

**Hypothesis:** completion evaluation should remain a single semantic judgment unless further tests show a reproducible weakness; only exact evidence identity should be constrained mechanically.

**Danger:** more realistic evidence-rich cases may expose hidden overload; the current simple cases may be too easy.

**Falsified if:** completion correctness degrades significantly across seeds or richer scenarios, especially when evidence must be requested or reconciled.

### H9 — Communication-status reasoning may benefit from constrained evidence selection

**Observation:** the model correctly understood that a worker was still running but invented an evidence sequence and escalated severity.

**Hypothesis:** status interpretation can remain semantic, while evidence references should be selected only from actual available evidence IDs; severity may deserve a separate tiny judgment if it remains unstable.

**Danger:** constraining evidence IDs does not stop the model from citing irrelevant evidence; separating severity can disconnect it from the causal reasoning that should determine severity.

**Falsified if:** constrained evidence selection does not reduce fabricated references or separate severity judgment worsens correctness.

### H10 — Small semantic primitives may form a useful weak-model profile

**Hypothesis:** a weak-model execution profile can be built from empirically reliable primitives such as binary relevance, small-set choice, exact existing-ID selection, simple argument filling, and single-purpose prose generation, while stronger models can retain fused calls for efficiency.

**Danger:** capability profiles can overfit one model/version/quantization; model upgrades invalidate them; a large call graph may become operationally complex; correlated primitive errors can still compound.

**Falsified if:** reliability gains from decomposition are inconsistent across seeds/cases, or total latency/token/call complexity outweighs the measured improvement.

## Current production call inventory and planned isolated tests

| Call kind | Current semantic responsibility | Initial targeted study |
| --- | --- | --- |
| `action` | tool choice, arguments, assistant text, continuation, status, questions | fused vs action-class/tool/args ablations |
| `summary` | compress history while preserving task-relevant state | direct summary vs source-unit extraction then synthesis |
| `tool_result_projection` | reduce a large tool result for a specific objective | direct projection vs relevant-record selection then projection |
| `evidence_projection` | reduce communication evidence for a question | direct projection vs evidence-ID selection |
| `completion_evaluation` | decide complete/incomplete, remaining work, optional evidence request | simple and evidence-rich cases; likely remain fused |
| `caller_structured_output` | arbitrary caller-requested semantic transformation | representative extraction/classification/generation subcases; no single decomposition presumed |
| `communication_status` | answer status question, severity, citations | fused vs constrained evidence references; severity ablation |
| `response_relevance` | remove irrelevant material while preserving relevant answer | rewrite vs source-unit relevance selection + optional rewrite |
| `audio_rendering` | render source answer for spoken delivery | direct rendering; fact-preservation cases |
| `presentation_evaluation` | identify lost/changed info and decide acceptability | fused vs difference-only then acceptability-only |
| `history_analysis` | answer analytic question over exact history evidence | direct answer with constrained evidence IDs; simple vs causal cases |
| `notes_compaction` | consolidate notes without losing durable meaning | direct compaction; duplicate/conflict/exact-ID cases |
| `note_selection` | select applicable notes from candidates | arbitrary-subset vs binary relevance vs small-batch ranking |
| `prompt_instruction_selection` | select applicable durable instructions | arbitrary-subset vs binary relevance vs small-batch ranking |
| `prompt_instruction_projection` | compress applicable instructions after measured overflow | direct projection vs instruction-ID/exact-unit selection before synthesis |
| `doctor` | semantic diagnostics/advice | simple isolated diagnosis cases; no decomposition assumed yet |
| `benchmark_quality_judge` | benchmark-only semantic quality judgment | retained as evaluation infrastructure; not a production harness dependency |

## Reporting requirements

For every case and variant, the retained result must include:

- hypothesis ID and case ID;
- production call kind and whether the tested variant is fused or decomposed;
- exact prompt text;
- exact JSON Schema;
- exact model request parameters;
- seed, temperature, top-p, context limit and output reserve;
- raw structured output;
- timing and backend token counts when available;
- externally specified expected criterion;
- study judgment with a short rationale;
- whether the run supports, contradicts, or is inconclusive for the hypothesis.

The final report will explicitly separate: proven mechanical bugs, supported semantic-task-shape hypotheses, disproven hypotheses, inconclusive hypotheses, unexpected findings, latency/cost tradeoffs, and recommendations that remain speculation.

## Batch 1 results — 2026-08-29

Batch 1 tested H1/H2, H4, H5, and H6. Every variant was run with seeds 17, 42, and 91 at both production temperature 0.0 and robustness temperature 0.2. Exact prompts, schemas, requests, outputs, timing, and sampling settings are retained under `/data/var/swaag/manual-tests/model-call-decomposition-study-20260829/`.

The counts below are semantic pass counts against the explicitly stated study criterion. A run that exhausted its isolated-study output allowance before closing a free-form explanation is reported separately as a mechanical output-starvation run rather than silently scored as a semantic failure.

### H1/H2 — action fusion and mechanically redundant continuation

| Variant | Passes | Failures | Result |
| --- | ---: | ---: | --- |
| Current fused `agent_action` | 0/6 | 6/6 | Strongly fails this calculator/tool-use scenario. |
| Production prompt + `tool_calls` only | 6/6 | 0/6 | Strong support for reducing the action output burden. |
| Production prompt + `tool_calls` + `continue_loop` | 0/6 | 6/6 | Adding only `continue_loop` destroys the otherwise reliable tool call in this scenario. |
| Abstract `action_class` only | 0/6 | 6/6 | Disproves the proposed `action class -> tool -> args` first stage for this model/scenario. |

The code audit strengthens H2. `action_from_payload` rejects both `tool_calls != [] && continue_loop == false` and `continue_loop == true && tool_calls == []`. Therefore every accepted action satisfies `continue_loop == bool(tool_calls)`. Waiting must use a wait tool. In the current accepted-state model, `continue_loop` carries no independent semantic information. It is nevertheless emitted by the LLM, recorded in duplicate-action signatures, and used for terminal bookkeeping. The experiment shows that asking this model to reproduce that mechanically implied boolean can materially change its semantic tool choice.

**Supported:** the current fused action shape is a real weak point for this model, and a focused tool-call output is dramatically more reliable in the tested scenario.

**Disproved:** an abstract action-class classifier is not a good first decomposition stage here.

**Not yet proven:** that production should simply become `tool_calls`-only. Assistant response generation, questions, silent completion, status, multi-tool choice, and terminal-answer branches still require separate experiments before redesign.

### H4 — note and persistent-instruction selection

For notes, the current fused subset selector failed 0/6: it selected the relevant programming note together with irrelevant garden/audio notes. Independent binary relevance did not fix the problem. The relevant programming note was true 6/6, but both irrelevant garden and audio notes were also marked relevant 6/6.

This **disproves** the naive hypothesis that N independent yes/no note calls solve over-selection. They reproduce the same broad relevance bias at higher call cost.

Persistent instructions behaved differently. The current fused production selector selected only the applicable code-change instruction 6/6 across production and robustness seeds. That is evidence **against** redesigning persistent-instruction selection merely because note selection is weak.

Binary instruction relevance again performed poorly on irrelevant candidates. Some binary runs also entered long repetition loops inside the unconstrained `reason` string and exhausted the isolated output allowance. A direct 1024-token reproduction showed valid llama.cpp grammar and `stop_type=limit`: this was output starvation caused by repetitive free-form explanation, not malformed constrained decoding. A boolean-only relevance variant is therefore required before concluding whether the binary semantic judgment itself is weak.

### H5 — response relevance: rewrite vs exact-unit selection

The current fused response-relevance rewrite achieved the strict study criterion in only 1 completed run. It usually recognized operational details such as PID/hash/shell-command chatter as irrelevant but still retained some or all of that material in the generated answer.

The proposed exact-unit selection decomposition performed worse: 0/6 selected exactly the two relevant units. It frequently selected operational-noise units and omitted the DNS blocker.

**Disproved in its naive form:** source-unit selection before rewriting is not supported by this scenario/model. The response-relevance problem is real, but this proposed replacement is not the solution demonstrated by the data.

### H6 — presentation evaluation

The current fused evaluator failed 0/6. It frequently identified the missing DNS blocker in its explanation or `missing_or_changed_information` while simultaneously returning `acceptable=true`.

The difference-only variant succeeded 6/6: it reliably identified the missing DNS blocker and uncertainty when it was not also responsible for the verdict.

The verdict-only variant, given the material differences explicitly, succeeded only 3/6. Therefore the complete proposed two-stage replacement is **not proven**. What is proven on this scenario is narrower: difference identification can be isolated into a task this model performs reliably; binary acceptability remains unstable.

### Unexpected findings from batch 1

1. **Smaller is not universally easier.** `tool_calls`-only is dramatically better, while `action_class`-only, binary note relevance, binary instruction relevance, and exact-unit response selection are poor.
2. **Temperature zero did not make textual output perfectly seed-invariant.** Semantic choices were often stable, but rationale text differed across seeds on this backend/model. Three production-temperature seeds therefore remain useful.
3. **Free-form explanation fields can dominate a tiny classifier.** A boolean-plus-`reason` schema can enter a long repetition loop even though the boolean decision itself may be simple. Future primitive tests must compare reason-free and explanatory forms.
4. **Different persistent semantic stores behave differently.** Note selection is weak in this scenario; persistent-instruction selection is reproducibly good. They should not be redesigned as one undifferentiated “selection” problem.
5. **A model can correctly describe its own contradiction without resolving it.** The fused presentation evaluator often says the blocker is missing and still emits `acceptable=true`. Structured fields should not be assumed mutually consistent merely because they were produced in one constrained object.

### Mechanical bug discovered during the study

The study exposed an independent constrained-decoding integration defect: when completion evaluation had zero available evidence sources, Swaag constructed impossible `enum: []` fields. llama.cpp compiled those to zero-width grammar productions and could emit malformed JSON while reporting a normal stop. The patched contract omits mechanically impossible evidence requests, removes the analogous zero-tool empty enum, and the portable-schema validator now rejects empty enums. The focused completion/action/SQLite regression suite passed 80/80 after cleanup. Live zero-evidence and zero-tool llama.cpp probes produced valid constrained JSON with the patched schemas.

This mechanical fix is separate from the decomposition hypotheses and must not be interpreted as evidence about semantic model quality.

## Batch 2 results — 2026-08-29

Batch 2 extended the study to reason-free relevance primitives, completion evaluation, communication status, audio rendering, tool-result projection, prompt-instruction projection, and history summary. Total retained study runs after this batch: 198.

### H4b — removing free-form reasons from binary relevance

Removing the free-form `reason` field eliminated the repetition/output-starvation behavior, but it did not make binary relevance uniformly reliable.

- relevant code note: 5/6 correct;
- irrelevant garden note: 5/6 correct;
- irrelevant audio note: 6/6 correct;
- relevant code instruction: 5/6 correct;
- irrelevant audio instruction: 6/6 correct;
- irrelevant travel instruction: 0/6 correct.

This changes the H4 conclusion slightly. Free-form explanation was indeed a mechanical burden for the tiny classifier, but the underlying binary semantic judgment remains candidate-dependent and can be systematically wrong. Binary relevance is therefore not a generally safe replacement for fused selection. The current fused persistent-instruction selector remains stronger on the tested instruction set than the proposed binary replacement.

### H8 — completion evaluation

The simple incomplete case was semantically correct 6/6 and preserved `report.txt`, supporting the idea that the model can recognize obvious unfinished work.

The simple complete case returned `complete=true` 6/6 but also invented non-empty `remaining_work` such as `confirm`, `respond`, or `verify`. Under the strict contract criterion this is 0/6. This is another internally inconsistent multi-field object: the principal verdict is correct while a companion field contradicts it.

The evidence-backed complete case failed 0/6. Even though the supplied exact tool evidence showed `write_file` received `{"path":"report.txt","content":"hello"}` and wrote five bytes, the model repeatedly concluded that `report.txt` did not contain `hello` and requested the write again. This strongly **disproves H8 as originally stated**: completion evaluation is not proven simple enough to leave fused merely because trivial complete/incomplete cases looked reasonable.

A next experiment should separate the semantic completion verdict from generation of remaining-work prose and should test whether exact evidence is easier to consume when represented as a smaller explicit fact object rather than a larger production evidence block.

### H9 — communication status

The current fused status call understood the principal semantic state (`answer = no`, continuing implementation) in all six runs. Five of six also used only the real evidence IDs 10/11; one production seed generated `[-10, -11]`. Importance was consistently `major`, which may be over-escalated but was not `critical`; no run requested stronger-model escalation.

The proposed smaller `finished + constrained evidence IDs` contract was **worse**: it returned `finished=true` 6/6. Constraining references successfully prevented fabricated IDs, but simplifying the semantic output removed cues that apparently helped the model reach the correct status conclusion.

Therefore H9 is only partially supported. Constraining existing-object identities is mechanically attractive and prevents fabricated identities, but a stripped-down semantic verdict is not demonstrated to be reliable. We should test constrained evidence IDs inside the richer current status contract rather than replacing the rich semantic representation wholesale.

### Audio rendering

The current audio-rendering call preserved staging success, the DNS blocker, `api.example.test`, and uncertain ETA 6/6 across production and robustness seeds. On this simple factual case there is no evidence that audio rendering needs decomposition. It is currently one of the strongest call types tested.

### H7 — tool-result projection and exact-record selection

Direct semantic projection preserved all four required incident facts in 4/6 runs. Two runs collapsed the result to only `CHG-7419-Z`, losing deadline, negative constraint, and checksum cause. The failure is therefore real but not universal.

The proposed exact-record-ID selection performed 0/6: it selected all four records, including routine noise, every time. This **disproves the naive record-selection-first decomposition** on this scenario. It mirrors the note-selection over-inclusion behavior.

The useful conclusion is narrower: direct projection is imperfect but materially better than the tested ID-selection primitive. Future H7 variants should test different semantic interfaces, not simply “select relevant records.”

### Prompt-instruction projection

The current prompt-instruction projection failed catastrophically 0/6 on the simple four-instruction source. Outputs were merely `action` or `80 tokens`, preserving none of the operative instructions. This is a high-priority weak call type because prompt-instruction projection is used only after measured overflow, exactly where losing constraints can be costly.

No replacement is proven yet. Because fused persistent-instruction **selection** was strong while instruction **projection** was weak, a promising next hypothesis is to preserve exact selected instruction units/IDs for as long as possible and only semantically compress individual oversized instructions or a much smaller selected set. This remains speculation until isolated tests are run.

### History summary

The current summary call failed the strict preservation criterion 0/6. It consistently preserved staging success, DNS blocker, host, and uncertain ETA, but dropped the exact ticket `CHG-7419-Z` and the negative constraint `Never delete the source archive`. It also copied meta-instructions from the summary prompt into the summary itself (for example guidance about `preserve_recent_messages`).

This is a serious weak point for a small model because history summary is central to context overflow recovery. The next summary experiments should separate two questions: whether the model can identify which exact historical units must remain verbatim/recent, and whether it can summarize only the remainder. The existing `preserve_recent_messages` integer may itself be too coarse because important exact information need not be confined to the most recent N messages.

### Batch 2 surprises

1. **Reason-free binary classification helps some candidates but not all.** It removes pathological explanation loops but does not remove systematic semantic bias (travel instruction remained relevant 6/6).
2. **Rich semantic context can outperform a stripped-down verdict.** Communication status was semantically correct in the rich fused object and wrong 6/6 in the smaller `finished` contract.
3. **Exact-ID selection is not automatically easier.** Both response-unit selection and tool-record selection over-selected badly.
4. **Completion evaluation has the same multi-field contradiction pattern as action/presentation.** Correct `complete=true` can coexist with invented remaining work.
5. **Persistent-instruction selection and projection have radically different reliability.** Selection was 6/6 on the tested set; projection was 0/6.
6. **Audio rendering is unexpectedly robust on the tested factual case.** There is currently no evidence-based reason to decompose it.

## Batch 3 results — 2026-08-29

Batch 3 tested completion-verdict decomposition, notes compaction, history analysis, and single-instruction projection. The study continued to use seeds 17, 42, and 91 at production temperature 0.0 and robustness temperature 0.2.

### H8b — completion verdict only and evidence representation

A verdict-only contract containing only `complete: boolean` produced:

- simple complete case: 6/6 correct;
- simple incomplete case: 6/6 correct;
- evidence-backed complete case using the current production completion-evidence prompt representation: 0/6 correct;
- the same completed write represented as one compact exact deterministic fact: 6/6 correct.

This is one of the strongest decomposition findings in the study. Removing `reason` and `remaining_work` eliminates the contradictory companion fields for simple cases, but it does **not** by itself fix evidence-backed completion. The same semantic verdict becomes fully reliable when the evidence representation is simplified without changing the underlying exact fact.

**Supported:** both output-shape burden and evidence representation contribute independently to completion failure.

**Not yet proven:** how production evidence should be normalized. The compact fact used here was hand-constructed for the experiment; production must preserve exact provenance and must not let deterministic code make the semantic completion decision. A follow-up must derive compact exact evidence mechanically from existing tool-event fields while retaining source references.

### Notes compaction

The current notes-compaction shape preserved the full required fact set in 4/6 runs. One robustness run dropped the negative constraint, and the mechanically truncated robustness seed 91 completed with a larger output ceiling but retained only ticket/deadline/never-delete while dropping staging success, DNS blocker, hostname, and uncertainty.

Therefore notes compaction is materially stronger than history summary on this scenario, but it is not reliable enough to assume lossless semantic preservation for a weak model.

### History analysis

The current history-analysis contract failed to produce a completed structured object in 6/6 runs under the bounded study allowance. Replays with a much larger 4096-token ceiling entered very long constrained generation; the first history replay exceeded a five-minute HTTP read timeout. This is classified primarily as an operational/output-starvation failure, not a semantic score, because no complete object exists to judge.

The result is still important: even a four-event history plus a direct causal question can make this tiny model spend thousands of output tokens inside the rich history-analysis object. Before evaluating semantic quality, the contract itself needs targeted ablations to identify which free-form/list fields trigger the expansion.

### Single-instruction prompt projection

The hypothesis that prompt-instruction projection might become reliable if each instruction were projected independently was decisively disproved. Four different source instructions were each tested six times; all 24 outputs were simply `{"projection":"action"}`.

This means the failure is not caused merely by too many instructions being compressed together. The current projection prompt/operation is fundamentally mismatched to this model. Because current persistent-instruction selection is 6/6 while projection is 0/6 even for one instruction, a future hardening direction should investigate avoiding semantic rewriting of already-small selected instructions entirely and using exact retention/drop decisions or overflow-specific treatment. That remains a hypothesis, not a production recommendation yet.

### Batch 3 conclusions

1. Completion failure can be decomposed into at least two independent causes: multi-field output coordination and difficult evidence presentation.
2. Hand-simplifying exact evidence can change completion verdict accuracy from 0/6 to 6/6 without changing the underlying semantic fact.
3. Notes compaction is promising but not lossless.
4. History analysis currently has an output-shape/latency problem severe enough to prevent semantic evaluation on a tiny four-event case.
5. Prompt-instruction projection is not rescued by per-instruction decomposition; that hypothesis is disproved 0/24.

## Batch 4 results — 2026-08-29

Batch 4 added terminal-answer, blocking-question, two-tool, caller-structured-output, communication-evidence-projection, and a minimal doctor yes/no probe.

### Action branches beyond the calculator single-tool case

The current full `agent_action` contract is not universally unusable. For a trivial terminal response (`reply exactly pineapple; no tools`) the primary action was correct 6/6: exact `pineapple`, no tool calls, terminal. The answer-only variant was also 6/6. However, the full object filled auxiliary status fields with semantically useless values such as `unknown/unknown/unknown/critical`. This reinforces the repeated pattern that a principal semantic act can be correct while companion fields degrade independently.

Blocking clarification was poor in both forms. The full action failed the strict criterion 0/6. Most runs declared the underspecified delete task complete; two robustness runs emitted questions, but they were optional and did not ask for the missing file identity. A question-only contract also failed 0/6, repeatedly asking generic `what is the next step?` with optional criticality. Therefore isolating question generation alone does not solve underspecification handling for this prompt/model.

The independent two-tool task (`calculator` plus `time_now`) failed 0/6 under the fused action: it emitted no tools and self-calculated the multiplication, once incorrectly as 6930. The `tool_calls`-only variant also failed the strict two-call criterion 0/6, but for a different reason: every run emitted exactly one plausible tool call. Four runs chose calculator; two robustness runs chose `time_now`. This suggests the focused interface improves concrete tool use but does not make the model plan an arbitrary multi-call set.

A new hypothesis follows directly from this result: the weak-model action loop may work better if each semantic step chooses **one concrete next tool call**, executes it, receives the exact observation, recompiles the remaining objective, and chooses the next concrete action. This is more specific than the disproved abstract action-class decomposition and will be tested next on independent and dependent two-tool sequences.

### Caller-defined structured output

The representative exact structured-extraction case succeeded 6/6: ticket `CHG-7419-Z`, `blocked=true`, hostname `api.example.test`. This call shape is currently strong and provides no evidence-based reason for decomposition on simple extraction/classification tasks.

### Communication evidence projection

The current evidence-projection call failed catastrophically 0/6. Given exact evidence that tests passed **and implementation continues because TODO remains**, every run projected `worker is finished`. This is a direct semantic inversion, not merely loss of detail. Evidence projection is therefore a high-priority weak call type.

No replacement is proven yet. Because the richer communication-status call correctly understood “not finished” while the projection step reverses it, the failure may come from the compression instruction/task shape rather than inability to reason about the evidence itself.

### Doctor probe

A minimal constrained doctor-style yes/no probe returned `yes` 6/6. This probe is intentionally weak evidence: it validates that the basic constrained yes/no call path functions, but it is not representative enough to characterize the full doctor operation. Rich diagnostic cases remain required.

### Batch 4 conclusions

1. Full action can handle a trivial terminal response while still producing garbage auxiliary status.
2. Neither full action nor question-only decomposition reliably handles a simple blocking clarification.
3. Focused tool-call output encourages real tool use, but an arbitrary two-tool array still collapses to one tool.
4. Caller-defined simple structured extraction is robust 6/6.
5. Communication evidence projection is a reproducible 0/6 semantic inversion and is one of the clearest weak call types discovered.
6. The next action hypothesis should test sequential concrete next-tool decisions, not abstract action classes or multi-call planning.

## Batch 5/6 results — sequential actions and duplicate feedback

### A5 — one concrete tool call per semantic step

The proposed sequential weak-model action primitive was tested on two scenarios with six runs each.

- Independent requirements: calculator + current time. Every run selected calculator first and calculator again after receiving the exact result `37 * 19 = 693`. Advancement to `time_now`: **0/6**.
- Dependent requirements: `search_repo` then `read_file`. Every run selected `search_repo` first and selected `search_repo` again after receiving the exact path `src/swaag/action.py`. Advancement to `read_file`: **0/6**.

This decisively **disproves A5 in its naive form**. A focused single-tool schema makes concrete tool construction reliable, but exact tool-result history alone does not make this model track which subgoal has already been satisfied.

### A6 — former runtime duplicate-rejection experiment (invalid production assumption)

At the time of this experiment, Swaag rejected repeated tool-bearing actions and repeated observation calls by comparing exact tool names/arguments and recently visible results. The experiment treated that as mechanically knowable no-progress detection. Subsequent review showed that assumption is invalid for external observations: an identical read/search/poll call may be meaningful because external state can change independently of Swaag. Production runtime therefore must not infer semantic uselessness from call identity alone.

The study replayed the same A5 second-step situations while adding that former duplicate-rejection wording: the exact prior invocation was rejected as immediately repeated, the exact rejected call was shown, and the model was instructed to choose materially different arguments or a different tool using the evidence already returned.

Result: **0/6** advancement for calculator→time and **0/6** advancement for search→read. The model ignored the feedback. For search it sometimes changed arguments (for example regex flags, path, or max_matches) while still repeating the same semantic observation.

Therefore A6 remains useful only as historical evidence about this weak model: explicit duplicate feedback did not teach the next semantic subgoal. It is **not** evidence for production duplicate rejection. Production Swaag now permits repeated external observations; finite-loop cutoffs belong to benchmarks that own the task oracle and evaluation boundary.

### Important production-context caveat: staged tool discovery

The isolated action experiments deliberately loaded the domain tool schemas directly in order to study action-contract shape. Production Swaag defaults to staged discovery: the first action initially has only the exact `load_tools` schema plus a compact capability index; selected domain schemas become available on later actions.

This means the study has characterized the semantic action contract after relevant schemas are loaded, but it has not yet fully characterized the **production discovery stage itself**. Before recommending an action redesign, the study must separately test whether the tiny model can use `load_tools` reliably, whether it over-loads capabilities, and whether staged discovery changes the later action failures observed above.

## Recording-driven design constraints added 2026-08-30

The latest relevant agent recording was reviewed before continuing the study. It reinforces several design constraints that matter to interpretation of these experiments:

- The agent/harness should remain **general-purpose**, not a task-specific workflow engine. The system may enforce mechanical constraints and expose capabilities, but semantic task decisions belong to the model.
- The preferred ideal is a **small number of constrained model-call categories** rather than a proliferation of bespoke model calls. A model call should receive the exact relevant context and choose from mechanically valid structured alternatives; deterministic code should not secretly perform the semantic choice.
- Tool/functionality discovery is therefore central. The model should be able to discover the relevant capability from a large general tool set without every full schema permanently occupying context. This makes the current staged `load_tools` design an especially important hypothesis to test rather than an implementation detail to assume works.
- Complete durable history is authoritative, but it cannot always fit. Semantic compression/summarization is unavoidable when context grows; the model must decide what information can be lost. The recording explicitly treats context sizing, output reserve, and compression level as unresolved research questions rather than facts that deterministic code can solve semantically.
- Structured constrained decoding is valuable because it can make mechanically invalid output impossible, but it does not make semantic field choices correct. The study's repeated multi-field contradictions are therefore directly relevant.
- Exact token/context budgeting is a first-class concern. The recording considers rough semantic output-size categories as a possible model task, but notes the recursive problem that output needs depend on input context. This remains an open hypothesis and should not be hard-coded without evidence.

These constraints make the study's negative results more important: a decomposition is not acceptable merely because it improves one benchmark if it turns the harness into a task-specific deterministic planner. The target remains a general agent whose semantic choices are made by the model, with deterministic code limited to mechanically knowable validation, budgeting, persistence, and execution.

## Staged tool-discovery study

Production defaults to staged discovery, so a dedicated study tested the capability-selection stage itself. The first full production-shaped first-action probes were already semantically poor and extremely slow: calculator self-answered while requesting `load_tools` itself and violating continuation semantics; README reading asked the user for a path already supplied. Because the full fused action confounds discovery with the already-known action-output burden, a focused `tool_names`-only discovery contract was then run for four tasks across the same six seed/regime settings.

Focused discovery results:

- `read_file` task: exact relevant set 0/6. All six completed runs included `read_file` but selected 13–14 capabilities, including unrelated terminal, notes, tests, process/wait/wakeup tools. This is reproducible severe over-selection.
- `search_repo` then `read_file` task: exact relevant set 0/6; four completed runs selected 14–18 capabilities and two runs exhausted the 256-token study output while emitting long arrays. Again severe over-selection.
- calculator + time task: three runs selected exactly `calculator,time_now`; one robustness run selected an unrelated five-tool wait/test set; two runs exhausted the small output allowance. This is mixed, not reliable.
- calculator-only task: one robustness run selected exactly `calculator`; five runs exhausted the 256-token output allowance while generating an overlong array/string. This call shape has an output-starvation/over-selection problem even when only one obvious capability is relevant.

Therefore the current arbitrary-subset capability-discovery primitive is **not proven reliable and is strongly weak on file/search cases**. The pattern matches note selection and record-ID selection: this model often understands that a relevant item belongs in the set but fails to stop adding irrelevant items.

The recording's requirement for general-purpose capability discovery means the response should not be a deterministic task router. The next hypotheses should instead test model-semantic alternatives that reduce arbitrary-subset burden while retaining generality—for example hierarchical/category discovery followed by a concrete capability choice, and one-next-capability selection with an explicit `none/answer` alternative. Those alternatives must themselves be tested because the earlier abstract action-class experiment failed.

### Discovery alternatives: one-next-capability and hierarchical category

Two general-purpose alternatives were tested across the same four discovery tasks and six seed/regime settings.

`one_next_capability` removed arbitrary-subset over-selection entirely, but reliability remained task-dependent. It chose `read_file` 6/6 and chose calculator as the first concrete capability for calculator+time 6/6. Calculator-only was 5/6 (one production seed chose `schedule_wakeup`). Repository search was 0/6 against the intended `search_repo`; it systematically chose `search_in_file` instead. Overall strict score: 17/24.

`category_only` also removed over-selection. It chose `compute_time` 6/6 for calculator, `workspace_read_search` 6/6 for README reading, and `compute_time` 6/6 for calculator+time. Repository-search routing was 0/6: every run chose `interactive_shell` instead of `workspace_read_search`. Overall strict score: 18/24.

Therefore both alternatives are **mixed**, not proven. They demonstrate that reducing arbitrary-subset burden can dramatically improve stopping behavior, but they can introduce stable semantic routing errors. Hierarchical discovery is promising enough for further study, but production adoption would require broader tasks and a second-stage within-category capability test. It must not be treated as a deterministic router.

# Final synthesis — 2026-08-30

The isolated study is now closed with **548 retained result files**. All original hypotheses H1–H10 have direct replicated evidence. Unless explicitly noted, the replicated semantic tests use seeds 17, 42, and 91 at production temperature 0.0 and robustness temperature 0.2. The study did not start, stop, restart, or reconfigure the Jetson model process and did not install a decomposition architecture into production Swaag.

## Final hypothesis ledger

### H1 — Action fusion overload: **PARTIALLY SUPPORTED, proposed decomposition DISPROVED**

The motivating failure is real. On an explicit calculator task the full current action object used no tool 0/6, while the identical action prompt constrained to `tool_calls` only produced the correct calculator invocation 6/6. The full action also failed the independent two-tool task 0/6 and blocking clarification 0/6, while it could still perform a trivial terminal answer 6/6.

However, the originally proposed `action class -> tool -> arguments` decomposition is disproved: action-class-only chose `answer` instead of tool 0/6. A `tool_calls`-only arbitrary multi-tool array chose only one tool 0/6. Sequential one-tool calls repeated the first tool instead of advancing 0/6 on both independent calculator→time and dependent search→read tasks. Existing mechanical duplicate feedback did not solve this 0/6. Question-only clarification also failed 0/6. Staged discovery is itself weak, and one-choice/category alternatives are mixed.

**Conclusion:** the fused action shape genuinely burdens this model, especially for concrete tool use, but no complete general replacement architecture is proven. The reliable narrow primitive is concrete single-tool argument construction when the correct domain tool set is already in scope. Production action redesign requires a stronger next-subgoal/discovery solution than any tested here.

### H2 — Mechanically derivable action fields should not be semantic outputs: **SUPPORTED**

Repository/runtime analysis proves that every accepted tool-bearing action satisfies `continue_loop == bool(tool_calls)`: tool calls with `continue_loop=false` are rejected and `continue_loop=true` without tools is rejected. Waiting must use a wait tool. Therefore `continue_loop` carries no independent semantic state in accepted actions.

Experimentally, `tool_calls`-only succeeded 6/6 on the calculator task, while adding only `continue_loop` changed the result to 0/6. This is unusually strong evidence because both code invariants and model behavior point in the same direction.

**Recommendation:** stop asking the LLM to regenerate `continue_loop`; derive it mechanically from accepted tool calls. Internal bookkeeping may retain the field if needed, but it should not consume semantic model capacity.

### H3 — Existing-object identity should be selected, not regenerated: **STRONGLY SUPPORTED, NOT PERFECT**

Two direct comparisons isolate the hypothesis:

- Existing file target: free path regeneration was exact 0/6, inventing `/home/user/report.txt`, `path/to/output.txt`, or selecting `hello.txt`; opaque existing-object selection chose the correct `f1` 5/6.
- Existing evidence event: free integer regeneration was exact 0/6 and returned `1000000000000000` in all six runs; opaque evidence-ID selection chose the correct existing event 5/6.

The rich communication-status experiment strengthens this result: constraining evidence sequences to actual available IDs preserved the correct semantic status 6/6 while mechanically preventing fabricated references.

**Recommendation:** wherever the semantic question is “which already-existing object/event/instruction/note/evidence source?”, let the model select a constrained existing identity and let deterministic code recover the exact path/hash/sequence/value. This does not move the semantic choice into code. It only prevents regeneration of facts the system already possesses. Selection can still be semantically wrong, as the 5/6 results show.

### H4 — Large subset selection overloads weak models; binary/small alternatives are better: **DISPROVED IN THE PROPOSED GENERAL FORM**

Current note subset selection over-selected badly. But independent binary note relevance also marked irrelevant garden/audio notes relevant 6/6. Removing free-form `reason` prevented repetition loops and improved some candidates, but the irrelevant travel instruction remained relevant 6/6. Exact-record and exact-source-unit selection showed the same over-selection pattern.

Persistent-instruction selection is a counterexample: the current fused selector chose only the applicable code instruction 6/6. Therefore “subset selection is generally too hard” is false; reliability depends strongly on task representation and candidate set.

**Conclusion:** do not replace arbitrary subsets globally with N binary calls. The resulting cost is higher and the same semantic bias can remain.

### H5 — Separate relevance selection from rewriting: **DISPROVED FOR THE TESTED DESIGN**

Current response-relevance rewriting is weak: it often recognized operational noise yet retained it. But exact source-unit selection was worse, selecting irrelevant units and missing the DNS blocker 0/6.

**Conclusion:** the response-relevance problem is real, but the tested selection-first replacement is not a solution.

### H6 — Difference identification and acceptability judgment should be separated: **MIXED / PARTIALLY SUPPORTED**

The current fused presentation evaluator failed 0/6 and frequently described the missing DNS blocker while returning `acceptable=true`. Difference-identification-only succeeded 6/6. A subsequent acceptability-only call given the known differences succeeded only 3/6.

**Conclusion:** isolated semantic difference detection is a reliable primitive on this case and may be useful. The complete proposed two-stage replacement is not proven because verdict conversion remains unstable.

### H7 — Semantic extraction before compression preserves exact information better: **DISPROVED FOR THE TESTED EXTRACTION PRIMITIVES**

Direct tool-result projection preserved all required planted facts 4/6. Exact-record selection before projection selected all records 6/6 and therefore scored 0/6. Response source-unit selection similarly failed. Fact-inventory history extraction also suffered omissions and severe output-starvation.

**Conclusion:** “extract/select exact units first” is not a generally reliable compression primitive for this model. Direct semantic projection can still lose identifiers, but the tested extraction-first alternatives are worse.

### H8 — Completion judgment is already simple enough to remain fused: **FALSIFIED**

Simple incomplete completion was correct 6/6. But simple complete outputs paired `complete=true` with invented remaining work, and evidence-backed completed writes were judged incomplete 0/6 under the production evidence representation. Therefore the original assumption that completion was already simple enough is false.

A verdict-only contract is 6/6 on trivial complete/incomplete cases. Evidence representation matters substantially: one hand-simplified successful-write fact plus verdict-only was 6/6 in the first ablation. Broader testing shows this is not a universal fix. Compact fact-only completion was 6/6 for running process, passed tests, and failed tests, but only 2/6 for write success and 2/6 for write failure. Adding the candidate answer made write success 6/6 but write failure 0/6, tests passed 2/6, and tests failed 0/6. The candidate answer can therefore bias the model toward the claimed completion state instead of the authoritative tool facts.

**Conclusion:** completion evaluation needs redesign, but no generally reliable semantic completion primitive is proven. The best evidence-supported direction is to reduce contradictory output fields and present mechanically exact evidence more compactly, while preserving the model as the semantic judge and treating deterministic tool facts as authoritative. More capable models may be required for reliable evidence reconciliation.

### H9 — Communication status should retain semantic reasoning while constraining evidence identities: **SUPPORTED IN THE REFINED FORM**

The current rich status call correctly understood “not finished” in all six original runs but one run fabricated negative evidence sequences. The earlier stripped `finished + evidence IDs` contract was semantically wrong 6/6, disproving the idea that status should be reduced to a tiny verdict.

The fair refined experiment retained the rich current status schema and constrained only `evidence_sequences` to actual available IDs. It preserved the correct `answer=no`, continuing implementation, and remaining TODO semantics **6/6**, while fabricated evidence references became mechanically impossible.

**Recommendation:** keep rich semantic status reasoning, but constrain references to existing evidence identities. Do not strip the semantic representation merely to make the object smaller. Severity remains less well calibrated (`major` throughout the tested case) and should not be redesigned without a separate externally grounded severity study.

### H10 — A generic weak-model profile can be built from small semantic primitives: **DISPROVED IN THE BROAD FORM; OPERATION-SPECIFIC PROFILING REMAINS PLAUSIBLE**

The study repeatedly shows that smaller calls are not uniformly easier. Strong examples include concrete single-tool arguments, exact existing-ID selection, simple structured extraction, audio rendering, and some completion/test-state judgments. Weak examples include action-class classification, binary relevance for some candidates, question-only clarification, exact-unit selection, record selection, single-instruction projection, state enums for worker status, history-analysis primitives, and instruction prioritization.

The same nominal primitive can be strong or catastrophic depending on content. For example, binary-like completion works perfectly on process/test cases and poorly on file-write reconciliation; fused persistent-instruction selection is strong while single-instruction projection is 0/24.

**Conclusion:** do not create a global “tiny model uses primitives X/Y/Z” profile. If weak models are supported, capability must be profiled **per operation/task representation**, with exact model/version/quantization identity and regression cases. Otherwise decomposition complexity will exceed the demonstrated reliability gain.

## Additional completed findings outside H1–H10

### Staged tool discovery: **CURRENT FORM WEAK; ALTERNATIVES MIXED**

Production's arbitrary-subset `load_tools` discovery is strongly prone to over-selection. README reading selected 13–14 capabilities 6/6 rather than only `read_file`; repository search/read selected 14–18 capabilities in completed runs. Calculator-only often exhausted its output allowance by generating excessive selections.

`one_next_capability` removed over-selection and scored 17/24 across four tasks: `read_file` 6/6, calculator+time first capability 6/6, calculator-only 5/6, repository-search intended `search_repo` 0/6 because it systematically chose `search_in_file`.

Hierarchical `category_only` scored 18/24: calculator/file/time categories were 6/6 each, repository search was 0/6 and systematically routed to `interactive_shell`.

**Conclusion:** reducing arbitrary-subset burden is useful, but no general discovery replacement is proven. Hierarchical discovery deserves broader future study; deterministic task routing would violate the general-purpose design requirement from the recordings.

### Evidence projection: **SEMANTIC FAILURE, NOT JUST PROSE COMPRESSION**

Current evidence projection inverted “implementation continues because TODO remains” into “worker is finished” 0/6. Replacing free-form projection with a single `finished|continuing|failed|unknown` enum did not help: the model returned `finished` for every scenario. Only genuinely finished evidence was therefore correct 6/6; continuing, failed, and unknown were each 0/6.

**Conclusion:** this model has a strong `finished` bias on this status framing. Do not rely on evidence projection for semantic status compression with this model. The richer communication-status call itself is substantially better and should consume exact evidence directly whenever it fits.

### History analysis: **OPERATIONALLY AND SEMANTICALLY UNSUITABLE FOR THIS MODEL**

The rich current history-analysis object failed to finish 6/6 under bounded study output and could run for thousands of generated tokens. Field ablations prove this is not only output shape: claim-correctness-only was 0/6 (`unclear` instead of wrong), strategy-only was 0/6 (`declare_complete` instead of verify blocker), and strongest-evidence sequence was 0/6 (sequence 10 instead of blocker sequence 12).

**Conclusion:** no tested decomposition rescues history analysis. For this model, exposing `history_analyze` as a reliable semantic diagnostic capability is not supported by evidence.

### History summary: **WEAK; SIMPLE ALTERNATIVES DO NOT FIX IT**

Current summary failed strict preservation 0/6, consistently losing the exact ticket and negative constraint. A summary-only contract without `preserve_recent_messages` improved to only 1/6 strict preservation, with one mechanical output-starvation run. Generic fact inventory scored 0/6 strict: five runs starved output and the only completed run retained staging/blocker/uncertainty but lost ticket/constraint.

**Conclusion:** removing the companion integer does not solve semantic history compression. No lossless general summary primitive is proven for this model. Since the recordings correctly identify semantic compression as unavoidable once exact history no longer fits, this remains a fundamental weak-model limitation rather than something deterministic code can safely “fix” by deciding what facts are unimportant.

### Prompt-instruction projection: **CATASTROPHICALLY WEAK**

Current projection was 0/6 on four simple instructions, usually returning `action` or `80 tokens`. Projecting one instruction at a time was 0/24 and always returned `action`. Exact-retention prioritization under a two-instruction budget was also 0/6, overwhelmingly selecting `audio` for a code-edit task and often selecting it twice.

**Conclusion:** neither semantic rewriting nor tested budget prioritization is reliable. Current persistent-instruction selection itself is strong on the tested case (6/6), so production should preserve exact selected instruction text whenever it fits and avoid invoking projection except as a last-resort overflow path. There is no proven safe semantic overflow projection for this model.

### Notes compaction: **MIXED, 4/6 strict**

Notes compaction is materially better than history summary on the planted scenario but is not lossless. It should not be treated as a guaranteed preservation primitive.

### Caller structured output: **STRONG ON REPRESENTATIVE EXTRACTION, 6/6**

Simple caller-defined exact structured extraction succeeded 6/6. No redesign is justified from current evidence.

### Audio rendering: **STRONG ON REPRESENTATIVE FACTUAL RENDERING, 6/6**

The tested audio call preserved all required factual content 6/6. No decomposition is justified from current evidence.

### Doctor: **MECHANICAL PROBE VALIDATED, NOT A SEMANTIC DIAGNOSTIC MODEL CALL**

Production doctor intentionally performs health, tokenizer, and literal constrained-output checks and asks only for `{"answer":"yes"}`. That model call succeeded 6/6. Rich artificial doctor semantics are therefore not required to characterize the production operation.

## Proven mechanical defects found during the study

The study exposed invalid `enum: []` constrained schemas when no completion evidence identities (and analogously no tools) were available. llama.cpp can compile such impossible enums into zero-width grammar paths and produce malformed output while stopping normally. The code was corrected so impossible evidence/tool choices are omitted rather than delegated to the model, and portable-schema validation now rejects empty enums.

This is a genuine harness defect, not a semantic model failure. Focused completion/action/SQLite regression tests previously passed 80/80 after the fix, and live zero-evidence/zero-tool constrained probes returned valid JSON.

The study also confirmed that long unconstrained/free-form string fields can make this tiny model repeat text until the output limit, leaving an otherwise grammar-valid JSON object unfinished. This is mechanical output starvation, not a constrained-decoding schema violation. Removing gratuitous explanation fields can reduce this operational failure mode, but does not necessarily improve semantic correctness.

## Context-size and output-reserve conclusion

The recordings ask the central question correctly: a general harness cannot deterministically know the exact semantic output length a model will need before the model performs the semantic task. Asking the same weak model to predict an exact token count first also creates a recursive dependency: the estimate depends on what input is supplied, but the amount of input that can be supplied depends on the reserved output size. This study provides no evidence that this tiny model can be trusted as an exact token estimator, and several supposedly tiny calls unexpectedly generated hundreds or thousands of repetitive tokens.

The evidence-supported context strategy is therefore:

1. **Use the real model tokenizer for every mechanically known input.** Count the complete serialized prompt, system text, constrained schema/grammar representation as appropriate to the backend, exact retained history, instructions, tool schemas, evidence, and any fixed protocol overhead. Estimation is only a fallback when the backend tokenizer is unavailable.
2. **Compute hard feasibility mechanically.** `input_tokens + minimum_valid_output_reserve + safety/backend overhead <= context_limit`. Minimum valid output reserve comes from the contract shape/protocol, not from semantic speculation.
3. **Use an empirical desired output reserve per call kind/model profile, not an alleged exact prediction.** The desired reserve can come from measured distributions of successful outputs for that exact model/version/quantization and call representation. It is headroom, not a claim about semantic truth.
4. **If a measured prompt does not fit, reduce semantic sources only after measured overflow.** Preserve exact authoritative source material whenever possible; invoke semantic compression only for the source category that causes the measured overflow. Do not pre-summarize merely because history “looks large.”
5. **Recover from output starvation adaptively.** If the backend reaches the output limit while the constrained object is unfinished, recompile with more output reserve when context allows. If more output reserve requires input reduction, apply the same measured-overflow policy rather than arbitrary fixed percentages.
6. **Never let deterministic budgeting decide semantic importance.** Code may decide byte/token accounting, what exact source regions exist, and how much must be removed. The model must decide which semantic information can be compressed or omitted when such a decision is unavoidable.
7. **Do not require a preliminary LLM output-size-estimation call.** A rough semantic size category could be studied later as an optional hint, but it is not proven and must never be required to make the original call fit. The current evidence favors exact counting + empirical reserves + adaptive retry.

This resolves the apparent “hen-and-egg” problem without pretending the system can know semantic answer length exactly. The system knows exact **input size**, exact **hard context limit**, exact **mechanical minimum output space**, and measured historical output distributions. The remaining uncertainty is handled with desired headroom and adaptive recovery, not false precision.

## Production recommendations supported by evidence

1. Derive `continue_loop` mechanically from accepted tool calls rather than asking the model to emit it.
2. Constrain references to already-existing identities wherever the model is choosing an existing object/event/evidence source; deterministic code should recover exact underlying values after selection.
3. In communication status, keep the rich semantic call but constrain `evidence_sequences` to actual available evidence IDs.
4. Remove gratuitous free-form rationale fields from tiny classifiers/semantic probes when the rationale is not consumed, because they can cause severe output starvation.
5. Preserve exact selected prompt instructions whenever they fit. Treat prompt-instruction projection as an unreliable emergency path for this model, not a routine transformation.
6. Keep exact durable history authoritative and only trigger semantic compression after measured context overflow.
7. Use exact tokenizer accounting, empirical per-call desired reserves, and adaptive output recovery for context budgeting.

## Changes explicitly NOT justified by the study

- Do not replace the action loop with `action_class -> tool -> args`.
- Do not assume one-tool-per-step will advance multi-step work.
- Do not rely on duplicate feedback to teach semantic progress.
- Do not replace subset selection globally with binary relevance calls.
- Do not replace relevance/compression with arbitrary exact-unit or record-ID selection.
- Do not install a deterministic task router for tools/capabilities.
- Do not reduce rich communication status to a tiny `finished` boolean.
- Do not trust evidence projection or history analysis with this model as currently represented.
- Do not assume compact completion evidence makes completion judgment generally reliable.
- Do not ask this model to predict exact required output tokens as a prerequisite for context fitting.
- Do not treat constrained decoding as semantic validation; it guarantees shape/mechanical domains, not correctness.

## Model-specific weak/strong call profile from this study

For **SmolVLM2-2.2B-Instruct Q4_K_M on the tested llama.cpp endpoint**, strong or promising operations include concrete single-tool argument construction with an already-known tool, existing-object ID selection (5/6 in direct tests), rich communication status with constrained evidence IDs (6/6), simple caller structured extraction (6/6), tested audio rendering (6/6), current persistent-instruction selection on the tested candidate set (6/6), notes compaction on the planted case (4/6), and some simple process/test completion states.

Consistently weak operations include abstract action classification, blocking-question generation, multi-tool planning, sequential subgoal tracking, arbitrary subset discovery, note relevance in some candidate sets, response-unit selection, record selection, presentation acceptability verdict, evidence projection, history analysis, history summary exact preservation, prompt-instruction projection, instruction prioritization under hard budget, and evidence-rich completion reconciliation.

This profile is **model/version/quantization/task-representation specific**. It must not be generalized to stronger models or future quantizations without rerunning the retained cases.

## Final answer to the architectural question

The data does **not** support replacing Swaag with a large graph of tiny semantic calls. That would make the harness slower and more complex while many of those primitives are themselves systematically wrong. The data also does **not** support leaving every current fused call untouched.

The supported hardening principle is narrower:

**Move only mechanically knowable information out of the model's output burden; constrain choices over already-existing identities; preserve rich semantic context where it demonstrably helps; and simplify individual model-call representations only when replicated tests show that the simpler representation is actually more reliable.**

For this tiny model, some important general-agent capabilities remain beyond reliable performance regardless of the tested decomposition. The harness should bound those failures mechanically and preserve evidence/history for recovery, but should not pretend deterministic code can detect or repair arbitrary semantic mistakes. A stronger model is still necessary for reliable general history analysis, semantic compression, multi-step planning, and evidence reconciliation.
