# Full Infra source-linked verification audit — October twenty-first

The earlier report covered agent infrastructure, not every required topic. This verifies source and evidence **coverage**, not every feature. Passage counts cannot be interpreted as a percent of compliance. Source rules, ideas, researched alternatives, and bundled multi-claim paragraphs differ in scope.

There are **2,230 recorded source passages** from **78 guideline documents** and **168 originals**. Only **30 passages** have manually reviewed evidence links, and **no complete passage** has universal compliance proof. Evidence outcomes: two scope-limited tested mechanisms, sixteen partly tested, ten not accepted, one tested but failing, and one preserved-source-only.

## Per-project review boundaries

| Responsible domain | Passages | Limited evidence links | Not yet linked |
| --- | ---: | ---: | ---: |
| Client UI / Infra web / Android | 718 | 6 | 712 |
| Evidence appendix | 578 | 0 | 578 |
| Games project | 159 | 3 | 156 |
| Infra policy, research or documentation | 126 | 0 | 126 |
| Infra project/workflow | 14 | 1 | 13 |
| KnowMoreDiRT / SWAAG integration | 16 | 1 | 15 |
| Robot controller project | 13 | 1 | 12 |
| SWAAG + every applicable project | 306 | 11 | 295 |
| SWAAG agent/runtime | 177 | 4 | 173 |
| Trading project | 123 | 3 | 120 |

## Explicit failing or nonaccepted requirements

- **voice-editor-and-manual** — gui/voice_interfaces.md: Actual Android source and UI checks across selection/cut/paste, auto-send disable, full playback/seek controls.
- **bidirectional-referents** — gui/bidirectional_ai_ui_and_live_design.md: Selection, pointer, screen/build state, ambiguous deictic requests, reversible UI preview and tests.
- **network-stream-semantics** — network_transport_and_testing.md: Profile and test durable files vs live media vs replaceable telemetry vs exact control receipts.
- **purpose-before-code** — programming/README.md: Review concrete project/module purposes, invariants, failure branches, algorithm choice and expected side effects individually.
- **kmd-source-provenance** — kmd/knowledge_and_memory_design.md: Reconcile branch versions; independent unseen discourse-scope, negation, referent and cross-chunk heldouts; production quality.
- **options-collateral** — trading/options.md: Do not promote past profit claims until exact live/broker-equivalent settlement and original-bid-size tests pass.
- **save-at-any-moment** — games/playability_realism_and_saving.md: Save/reload open quests, active physics, time state, generated world and ongoing goals with exact resumed state.
- **trained-habits-real-world-rollouts** — games/learned_habits_and_automation.md: Real task-boundary examples, training, different scenes, physical success/failure, manual fallback and ability to reject/retrain.
- **robot-sensors-and-clock** — robotics/algorithm_selection_and_learning.md: Test calibration, timestamps, frames, action constraints, model-free versus model-based planners with heldout simulator and physical safety.
- **warning-color-contract** — gui/core/color.md: Warn only on true severity; test accessibility, colorblind behavior and the actual error/ready states.
- **backtest-results-ui** — gui/domains/backtest_results.md: Real screens must answer robustness and validity first; measure loads and test tooltips/charts/parameters, source data and stale/failed states.

## What this result does and does not authorize

SWAAG has tested durable receipts, worker state, recovery, constrained model output and some audio emulation. These do not establish finished physical Android operation, guaranteed semantic relevance, OS-level sandboxing, cross-host quotas, or correctness in trading, robotics, KMD, gaming or every GUI. A historical research instruction is not automatically a core SWAAG feature. All downstream projects must provide their own tests and observed results. Original statements and research alternatives remain available through the source-linked JSONL records.

The next acceptance passes must classify each binding proposition by owner, define its observable effect, verify or falsify it in the appropriate repository and actual environment, and preserve failed evidence. User decisions are needed only for truly unresolved material ambiguities, not to replace experiments.
