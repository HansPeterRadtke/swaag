# Ordinary orchestrator messages and real commands — 2026-10-14

## Question and scope

The user clarified that functional orchestration matters more than a particular latency: does the ordinary orchestrator produce a user message together with tool calls or system commands, and then actually execute them correctly? The earlier approximately twenty-second result measured a separate status reader. It was not a measurement of this combined action.

This follow-up uses the installed SWAAG package matching source bf38d9d (package implementation fe70bef), the actual ordinary TCP orchestrator.message operation, and the existing Thor Qwen3.8-27B UD-Q3_K_XL endpoint. There is no independent conversation model, status question, alternate action loop or early-answer prototype in these trajectories. Default generation, output budgets, reasoning configuration, locking and completion verification remain in force. Each trajectory has a fresh workspace and durable session store; response caching and idle work are disabled. The only enabled execution tool is shell_command, with side effects explicitly enabled for the authorized isolated test.

The caller requests a short pre-action message, real command execution and a report based on actual output. Model decisions, persisted progress, actual shell start/result, filesystem effects and every client response line are recorded separately. The trace observer polls durable histories at one-tenth-second intervals; observed elapsed times have that sampling granularity, while inter-event intervals use the durable event timestamps.

## Successful command and expected command failure

| Ordinary request | First validated combined action | Tool dispatch | Final response at client | Actual execution |
|---|---:|---:|---:|---|
| Create and read a test receipt | 59.735 s | 59.848 s | 116.981 s | Exactly one shell call, exit zero, exact expected output and file bytes |
| Run a diagnostic that deliberately fails | 48.994 s | 49.107 s | 126.435 s | Exactly one shell call, exit seven, exact expected stderr, no created/modified/deleted files |

Both first actions contain a nonempty assistant_message and the executable shell_command call in the same validated action object. The runtime then persists assistant_progress and dispatches the command immediately, without another model call between them. The successful case creates only receipt.txt in its isolated workspace and reads its actual contents. The diagnostic case reports the real nonzero exit code and stderr; it neither retries nor falsely reports command success. Final model answers were checked against the actual process results. No pending user controls remain in either completed trajectory.

These measurements answer the combined-action question positively for the tested operations. They do not support treating twenty seconds as a fixed reaction time. The initial interaction gate, main action generation and occasional repair calls contribute to the delay before dispatch.

## Message production versus delivery

The ordinary JSONL caller receives exactly one line, after the tool loop and completion evaluation. The earlier pre-action message is present in canonical history but is not sent on that request connection. Source inspection agrees: orchestrator.message awaits the full synchronous turn, and the ordinary foreground CLI also prints only the returned final answer. Existing worker protocol streams are a different interface and were not substituted for the orchestrator entry point in this test.

Thus the model/runtime can generate a message and command together and execute the command correctly, while the ordinary caller still does not receive that initial message at dispatch time. This is a concrete delivery limitation of the current unary interface, not evidence of a missing tool call or a requirement for a second model. No TTS playback timing was measured, and no new streaming contract or client behavior is silently introduced by this diagnostic.

## Configuration failure retained

The first shell setup inherited tools.allow_side_effect_tools=false. Listing shell_command as enabled did not override that explicit permission gate, so the actual model schema contained no shell tool. The model initially returned only an intention; the independent completion evaluator correctly rejected completion because nothing had executed. The owned probe was terminated rather than repeatedly requesting an unavailable operation. No command ran and no test receipt was created. This setup is excluded from command-execution timing claims but retained in full.

The corrected experiments enable the authorized side effect only in isolated runtime configurations. Production defaults and the unrelated Infra host configuration are unchanged. This distinction is essential: a test cannot demonstrate shell execution while its effective tool policy forbids it.

## Additional measured overhead

The successful-command model invented an exact word-count constraint even though the user requested only a brief message. Existing semantic repair removed that unsupported constraint before execution, taking 9.917 seconds. The diagnostic response likewise required a 14.367-second constraint repair before its final answer. The first diagnostic action happened to satisfy an invented fifteen-word count, so no repair was triggered for that action. These are retained model-output defects and their measured recovery costs, not evidence that exact-word-count repair can be removed generally.

Completion evaluation costs 22.384 seconds for the successful command and 24.620 seconds for the diagnostic. The disabled-tool baseline demonstrates why completion checking has functional value: it rejected an intention-only answer without actual execution. No verification bypass is used to improve the reported timing.

## Cancellation while action inference is running

A third real ordinary request asks for the same file-creation command. At 13.726 seconds the harness observes the first action inference in canonical history; at 13.729 seconds it sends a second ordinary finalized request cancelling that command and forbidding any shell call or file creation. The shared planner request slot is still occupied.

The original model computation returns a concrete shell-call proposal, but it does not execute. The recorded agent_action_rejected event explicitly identifies the newer control missing from the original context snapshot and retains the rejected shell proposal. The runtime reconciles the newer authoritative instruction and accepts a cancellation action with no tool calls. Both exact user messages are marked processed in canonical history. Both request connections receive the same cancellation result at approximately 117.95 seconds after the initial request, about 104.24 seconds after cancellation was sent. The final completion is shared, not a second execution.

The independently checked result is zero shell calls, no receipt file and no pending controls. This establishes cancellation correctness for a newer finalized instruction arriving during action inference; it does not claim immediate interruption of model computation, rapid cancellation acknowledgement or cancellation of a command that has already started.

## Operational verification

All three completed trajectory harnesses exit successfully. The initial disabled-tool setup is separately retained as an intentionally terminated invalid execution setup. No production source, model setting, service or workload is changed by these experiments. Across 2,041 host-memory samples, minimum available memory was 18.667 GiB, with zero samples below the experimental eight-GiB threshold. All 190 installed package files still match source; both existing vLLM service process identities and zero restart counts are unchanged. Owned forwarding/monitoring processes have exited. Exact final memory counts, installed-package identity and service process/restart checks are retained with the artifacts.

## Evidence

All scripts, exact configurations and prompts, client timelines, complete durable model/action/tool histories, assertions, memory observations and read-back evidence are retained under /data/var/swaag-benchmarks/orchestrator-action-delivery-20261014 on Jetson. The prior status/transport study remains separate. This follow-up changes documentation and experimental artifacts, not production code or model configuration.
