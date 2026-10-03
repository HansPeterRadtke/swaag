# Context, execution mode and inference follow-up - October 3, 2026

This follows the user's challenge to the service deployment, unnecessary context and nineteen-minute production call. It does not certify universal agent correctness. The September 30 full Infra audit remains the reading-coverage record: all 139 original recordings were covered; vendor/binary/credential inspections were separately identified. The current fetched agent recordings contain no new agent requirements relative to that snapshot. Original recordings remain unchanged.

## What the recordings require

The August 4 and August 5 agent recordings make necessary information and measured context capacity central responsibilities of the harness. The August 25 reasoning recording warns that unnecessary prompt material can reduce quality and forbids deterministic semantic selection. The August 29 whole-source reading recording requires complete requested source when it fits, and discusses staged tool discovery as a possible approach. The September 16 clarification recording distinguishes important unresolved questions, provisional assumptions and continued authorized work. The September 17 knowledge recording distinguishes the private agent environment from project files. None of the checked recordings instructs installation of a boot-started system service.

A short requested answer can require a large input. Output length is not a relevance filter. Source discovery and preservation are different responsibilities: the model may choose which filesystem tools to invoke; once source is explicitly supplied or retrieved, exact content remains present when it fits. Overflow handling retains the existing model-owned projection and exact recovery lineage.

## Execution mode

SWAAG has foreground `ask` and `chat` commands and an explicit optional `communication serve` command. The Infra communication service was introduced on August 26 as a deployment choice for persistent network clients, not as a requirement derived from the recordings. On October 3 its idle Jetson service was disabled and stopped and removed from default enabled units. Histories, workers and plans were preserved. The unit remains available for an explicitly chosen network deployment. The model server has an independent lifecycle.

## Context changes

Default staged discovery now provides an exact recoverable workspace reference instead of scanning and injecting a complete filesystem inventory before the model has requested it. Complete automatic manifests remain available with staged discovery disabled. The capability index retains names and purposes; execution guidance accompanies selected tool schemas. Trusted instructions, user input, history, attachments, notes and observed tool results are not shortened by these changes.

Regression cases exercise a direct no-tool answer without scanning the workspace, model-selected discovery and reading of a complete 22,431-character source, and preservation of exact execution guidance after schema loading. The initial focused run had 107 passes and a test assertion that incorrectly expected raw rather than JSON-escaped newlines; correcting that assertion required no production-code change. All three new regressions then passed. The full frozen-source run finished with 1,001 passes, three optional external-tool skips and one ten-second delegated-tool wait timeout while compilation competed for the command runner's shared two-core quota. With only the compiler paused, that unchanged test passed in 9.22 seconds. All 1,002 non-skipped cases therefore passed across the full run and isolated rerun; the full-run timeout remains recorded rather than relabeled as a clean single run. The built wheel matches all 184 package source files exactly.

## What the nineteen-minute task actually did

The September 30 production Pipe request was `Reply exactly OPENWEBUI-PRODUCTION-OK and do not use tools.` It completed in 1,169.120 seconds without tool use. The action prompt had 8,509 tokens, including 4,371 for the automatically injected workspace inventory and 1,977 for the capability index. Prompt evaluation took 1,056.623 seconds; generation of 91 structured tokens took 42.735 seconds. The completion check processed 1,121 prompt tokens in 39.666 seconds and generated 63 tokens in 12.953 seconds. Queue waits were 0.078 and 0.086 seconds.

The matching llama.cpp log contains no concurrent inference during these requests; the preceding model request ended almost four hours earlier. It was not a model-slot queue delay. Historical host GPU/CPU telemetry is unavailable, so unrelated compute load at that historical instant cannot be ruled out. Current investigations checked live processes, model slots and GPU users; existing Whisper servers were left running.

## Backend defect and controlled measurements

The deployed CUDA library was built with `GGML_CUDA_FA=OFF`, but the profile requested Flash Attention. Disassembly of the actually loaded `ggml_cuda_flash_attn_ext_supported` function shows an unconditional false return; llama.cpp's backend support logic consequently routes that operation away from CUDA. Kernel symbol names alone do not prove runtime support.

With the context fix and old attention configuration, a real SWAAG turn had 3,103 action input tokens and took 515.040 seconds end to end: 390.171 seconds for action prefill, 46.910 for action generation, 64.614 for completion prefill and 10.094 for completion generation. It returned the exact answer with no tools. All result caching was disabled.

The exact saved action and completion requests were then replayed with explicit non-Flash attention. An 8,192-token f16/f16 configuration took 147.756 seconds for both calls. A 12,288-token q4_0/f16 configuration took 140.482 seconds; its action prefill took 73.794 seconds at 42.049 tokens/second and output generation ran at 5.348 tokens/second. These tests changed cache representation and available window as well as attention mode; they are not a pure single-variable Flash Attention benchmark. Both produced the correct action and completion decisions.

A 32,768-token non-Flash test failed during first inference with CUDA allocation exhaustion. It is not a successful deployment candidate. Cache-only accounting admitted startup but did not bound that configuration's extra CUDA workspace. Conservative all-layer admission remains for non-Flash/automatic attention. Explicit Flash Attention can use the recorded dense/recurrent layer split, with conservative fallback for missing intervals/custom masks and a separate MTP reservation. The four-GiB host reserve remains unchanged.

The launcher now sends explicit `--flash-attn off` for disabled settings, supports a versioned backend binary per profile, and stops restarting after admission/occupied-port exit status 75. A separately built CUDA Flash Attention backend is being validated before selection. Its build provenance records unchanged Release objects reused from the same source commit and attention translation units rebuilt with the changed macro; no upstream source modification was made.

## Acceptance boundaries

Final deployment and live required-source evidence are pending this working report's final update. Browser upload/download/control acceptance, protected external MCP/A2A provider deployments, broad authorization-conflict trajectories and long-horizon semantic evaluation remain separate open acceptance work. A native GUI or voice frontend is not provided by the core foreground program. Existing scheduler, durable history, notes, questions, authorized backlog, tool execution and control mechanisms are described in the full audit; neither these changes nor a passing local suite imply that every future semantic context choice is correct.
