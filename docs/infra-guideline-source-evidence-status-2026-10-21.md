# Full Infra requirement evidence status — October 21, 2026

This replaces the incomplete earlier coverage count, which indexed only the seventy-eight Markdown guides. The current reproducible scope also includes two plain-text policy documents and Infra root operating instructions. Every original nonempty line is retained exactly once as a body or heading reference. This is **source coverage**, not implementation certification.

**Current source inventory:** 82 policy and supporting documents, 2,471 body source passages, 593 independently indexed Markdown headings, and 168 unchanged original recording files. The separate preservation checker reports 168 originals, 168 ledger entries, 168 verbatim appendix sections and zero integrity errors.

**Evidence:** Thirty narrow reviewed source claims remain attached to their exact original source content hashes. No complete passage has been certified for all possible implementations or inputs. Two have tested mechanisms with limited scope, sixteen are partly tested, ten remain unaccepted, one is tested and failing, and one has source preservation only. The other 2,441 body passages lack a source-specific effect-acceptance record. This is not a compliance percentage: many passages are illustrations, past research, examples or bundled rules rather than independent binary features.

## Original source and authority boundaries

The owner-provided root Infra instructions apply to Infra-owned repositories and account services; they are not global commands to rewrite unrelated projects. The earlier `tests.txt` explicitly defers to the current recorded testing guidelines on conflict. `constraint_decoding.txt` describes strict OpenRouter/OpenAI-compatible provider contracts and should not be imposed on incompatible local endpoints without verification. Quoted fenced examples stay source data; heading directives remain discoverable, not automatically accepted instructions.

## Verified source-link coverage by responsible system

| Responsible domain | Source passages | Linked specific evidence | Unlinked passages |
| --- | ---: | ---: | ---: |
| Client UI / Infra web / Android | 718 | 6 | 712 |
| Evidence appendix | 578 | 0 | 578 |
| Games project | 159 | 3 | 156 |
| Infra policy, research or documentation | 126 | 0 | 126 |
| Infra project/workflow | 14 | 1 | 13 |
| Infra repository/workflow | 53 | 0 | 53 |
| KnowMoreDiRT / SWAAG integration | 16 | 1 | 15 |
| OpenAI-compatible structured-output provider integration | 8 | 0 | 8 |
| Robot controller project | 13 | 1 | 12 |
| SWAAG + every applicable project | 486 | 11 | 475 |
| SWAAG agent/runtime | 177 | 4 | 173 |
| Trading project | 123 | 3 | 120 |

## Known negative and unaccepted outcomes

- **voice-editor-and-manual** (gui/voice_interfaces.md, NOT_ACCEPTED): Actual Android source and UI checks across selection/cut/paste, auto-send disable, full playback/seek controls.
- **bidirectional-referents** (gui/bidirectional_ai_ui_and_live_design.md, NOT_ACCEPTED): Selection, pointer, screen/build state, ambiguous deictic requests, reversible UI preview and tests.
- **network-stream-semantics** (network_transport_and_testing.md, NOT_ACCEPTED): Profile and test durable files vs live media vs replaceable telemetry vs exact control receipts.
- **purpose-before-code** (programming/README.md, NOT_ACCEPTED): Review concrete project/module purposes, invariants, failure branches, algorithm choice and expected side effects individually.
- **kmd-source-provenance** (kmd/knowledge_and_memory_design.md, TESTED_BUT_FAILING): Reconcile branch versions; independent unseen discourse-scope, negation, referent and cross-chunk heldouts; production quality.
- **options-collateral** (trading/options.md, NOT_ACCEPTED): Do not promote past profit claims until exact live/broker-equivalent settlement and original-bid-size tests pass.
- **save-at-any-moment** (games/playability_realism_and_saving.md, NOT_ACCEPTED): Save/reload open quests, active physics, time state, generated world and ongoing goals with exact resumed state.
- **trained-habits-real-world-rollouts** (games/learned_habits_and_automation.md, NOT_ACCEPTED): Real task-boundary examples, training, different scenes, physical success/failure, manual fallback and ability to reject/retrain.
- **robot-sensors-and-clock** (robotics/algorithm_selection_and_learning.md, NOT_ACCEPTED): Test calibration, timestamps, frames, action constraints, model-free versus model-based planners with heldout simulator and physical safety.
- **warning-color-contract** (gui/core/color.md, NOT_ACCEPTED): Warn only on true severity; test accessibility, colorblind behavior and the actual error/ready states.
- **backtest-results-ui** (gui/domains/backtest_results.md, NOT_ACCEPTED): Real screens must answer robustness and validity first; measure loads and test tooltips/charts/parameters, source data and stale/failed states.

## Release and continuation requirements

The full SWAAG source suite verifies core mechanics, not KMD semantic reasoning, trading profitability or execution equivalence, robotics controller safety, 3D game consistency or every Android user interaction. Each binding rule needs a responsible source implementation, a demonstrated reference behavior and representative negative tests, and deployed or physical-effect evidence when applicable. Unknown/unobserved findings stay unaccepted rather than auto-promoted from model prose, schema validity or a nearby test. Historical research proposals are not silently universalized.

Rebuild the source trace with `scripts/infra_guideline_requirement_trace.py --guidelines /data/infra/docs/guidlines --infra-root /data/infra --output <audit-directory>`, then verify manually reviewed evidence with `scripts/infra_guideline_evidence_overlay.py`. Generated audit artifacts, original recording bytes and the existing pinned thirty reviewed cases remain unchanged by this report.
