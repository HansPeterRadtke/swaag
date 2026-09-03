# Benchmark methodology

Swaag benchmarks must measure the behavior they claim to measure without moving benchmark knowledge into production runtime policy. Production runtime enforces mechanically knowable protocol, authorization, resource, persistence, and state-consistency facts. Benchmarks may impose finite task-specific action, tool-call, repetition, wall-clock, and retry boundaries because the benchmark owns the task oracle and finite evaluation boundary. Those limits terminate and score benchmark work; they must never be injected into production `AgentConfig` semantics.

## Durable benchmark execution

Long-running benchmark families use checkpointed durable output under `/data/var`. Checkpoints are atomically replaced after completed units and are configuration-signed with the output-affecting task/model/schema/profile settings needed to prevent incompatible runs from being silently merged. A compatible restart skips completed work. `--clean` is the explicit destructive opt-in.

Every model-comparable study records exact seeds and output-affecting schema/model identity. Seeded repetition is required when variance matters. A benchmark result is evidence only for the exact model/configuration represented by its retained artifacts.

## Structural validity is not semantic correctness

Generation-time constrained decoding and semantic correctness are separate measurements.

The `constraint-decoding` benchmark exercises real production `ContractSpec` schemas, including state-dependent tool unions, zero-tool states, completion evidence unions, closed enums, terminal responses, capability selection, and instruction/note selection. For every call it records the exact schema hash, the schema actually present in the transport request, seed/repetition, parse validity, schema validity, transport/grammar failures, timing/token facts, and a response hash.

A call is structurally valid only when the expected state-dependent schema was actually sent and the returned JSON validates against it. Semantic task correctness is scored separately. A schema-valid but wrong answer is therefore structural success and semantic failure, not a successful task.

This benchmark is intended to measure the practical robustness of generation-time constrained decoding, especially on local quantized models where provider-native free-form function calling may be unreliable. It does not prove that the model chose the correct legal action.

## Long-horizon context and provenance

The `long-horizon-context` benchmark aggregates two production-path mechanisms instead of assuming one context strategy explains all behavior.

Repeated compaction preservation plants exact authoritative facts spanning dates, identifiers, user and negative constraints, causality, unresolved questions, promises, paths/references, tool outcomes, and completion state. Later turns inject explicitly untrusted contradictory decoys. After every semantic compaction cycle, the benchmark separately measures:

- exact authoritative-value preservation in retained state;
- source references and recoverability back to durable authoritative events;
- semantic retrieval of the exact authoritative values by a separately constrained model probe;
- resistance to later contradictory decoys, measured by authoritative semantic retrieval rather than by requiring the decoy text to disappear.

Independent measured-overflow trials then require evidence that the unprojected request really exceeded context capacity, semantic projection was actually used, projection lineage matches the source event, the raw source remains recoverable, required facts survive, and the final request fits.

The aggregate benchmark reports these dimensions independently. One dimension cannot hide failure in another. Exact preservation does not prove semantic retrieval; successful retrieval does not prove source provenance; provenance does not prove overflow handling; and decoy visibility by itself is not a failure.

## Current evidence boundary

The deterministic benchmark implementations and fake-client regression suites establish that these mechanisms exist and that their measurements are separated correctly. They do not establish the size of Swaag's practical advantage over other harnesses.

The decisive live studies still require:

- repeated long-horizon trials that exceed context multiple times and cross process restarts;
- delayed retrieval where early constraints become relevant only much later;
- multiple model sizes and quantization levels;
- thousands of constrained-decoding calls across the state-dependent contract matrix;
- explicit reporting of structural validity, semantic correctness, transport/grammar failures, latency, and token cost;
- retained exact seeds, schema hashes, model identity, checkpoint state, and source-event provenance.

Do not use short frontier-model coding results as evidence for these long-horizon/local-model claims, and do not use structural-validity results as evidence of semantic correctness.
