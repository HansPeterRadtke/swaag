# SWAAG fast first-speech worker-state fence at receipt commit — 2026-10-22

## Verified failure and boundary

The live fast user-facing response is generated using a recent worker/plan snapshot and a durable worker revision. Before this correction, the code validated the worker revision **before** entering the orchestrator's final session control lock. A worker may commit a real status or question change while the request waits to acquire that lock. The final lock protects the user message receipt from concurrent replacement but does not make the separate worker database immutable.

A deterministic two-model fake-client reproduction first captured a worker with status created. It then caused that worker to become working **after the earlier revision check and immediately as the final control lock was acquired**, without refreshing the precomputed world. The original code committed the model's now-stale "created and not executing" response as a finished canonical answer. This was observed as an actual failing regression, not inferred from code alone.

## Source correction

The existing meaningful worker revision and precomputed-world comparison have been moved to **inside the final control-lock section**, immediately after authoritative message/policy freshness checks and before either direct response completion or a fast delegate's first speech is committed. The deterministic worker revision is still read using the project's indexed status/question cursors, worker lifecycle rows and stored summary: it does not replay every worker history or run another inference to check freshness.

On mismatch or a failed revision lookup, the optional fast response is discarded and the original durable user message continues into the full orchestration controller. Nothing in the fast stage is interpreted as authorization to execute a tool, and no interrupted external operation is restarted just because the fast answer was rejected. The service lock still does not create a cross-database transaction; a worker may advance again immediately after this observation, so this change narrows a **reproduced** race rather than promising an impossible globally atomic snapshot.

The revision query now runs during the final control-lock hold. It must remain read-only, bounded by existing indexed metadata, and measured for latency under concurrent worker updates before any claim about production responsiveness. This is a correctness-before-speed trade-off, not a reason to extend the lock over model inference.

## Tests and release boundary

New integration regression uses the actual durable WorkerStore transition injected precisely on receipt-lock acquisition. Existing regression coverage checks stale status/question updates earlier in inference, controls arriving during the fast stage, exact receipt replay, normal direct speech, and autonomous orchestration fallback.

The focused suite passed **sixty-four tests** after this correction. A separate non-editable installation of the exact candidate matched all **212 of 212** source-package files with SQLite **3.53.1**; **forty-one focused tests passed** when run against that installed package rather than the source-tree imports. The complete source regression then passed **1,725 tests with three optional skips and zero failures in 360.43 seconds** under the model-free Thor test environment. The test runner exited successfully. This does not qualify any new physical Android, model-content or deployed-service behavior. All tests use disposable local histories and fake model responses. These are source-level assertions, not proof of actual Coder model semantic quality, physical Android playback, active process code identity, or shared-backend preemption.

Live Thor service and unrelated voice-agent worktrees must remain unchanged until the separate installed-package, end-to-end integration and safe release conditions are satisfied.
