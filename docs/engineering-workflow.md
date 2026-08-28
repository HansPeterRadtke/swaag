# Evidence-driven engineering workflow

This is an implementation-neutral engineering contract derived from the
programming recordings through 2026-08-28. It governs substantial software
work without turning every trivial edit into a heavyweight process or a fixed
deterministic workflow.

## Scale the work semantically

An LLM decides how much planning, research, and verification the specific task
needs. A fully specified, reversible hello-world change may go directly from
inspection to implementation, test, and report. A substantial or uncertain
change should first make its purpose, inputs, outputs, interfaces, invariants,
hardware/runtime constraints, performance needs, and acceptance evidence
explicit. Deterministic code may enforce declared permissions, schemas,
budgets, commands, and verifier results; it must not infer task complexity or
risk from filenames, keywords, or a fixed checklist.

Risk priority follows consequences and the conditions threatened, not line
count or apparent algorithmic complexity. Identify the failures that could
lose state, violate security or safety, corrupt data, strand a user, create an
irreversible effect, or fail only in a production environment. Turn each
material risk into observable evidence before implementation where practical.

## Choose structure by responsibility and likely change

Factor genuinely repeated behavior into one named, reusable unit so its
contract and failure cases can be tested once rather than trusting many nearly
identical copies. Do not create an abstraction merely because two short blocks
look alike: first establish that they represent the same responsibility and
should change together.

Use a function for a stateless operation when that expresses the contract
clearly. Bundle data and behavior in an object when instances own meaningful
state, identity, lifecycle, or an invariant. A class used only as a static
namespace can improve local organization, but it is not automatically better
than a module and functions. Likewise, do not force a deep object hierarchy
onto a task whose required representation is a small value, enum, record, or
flat collection. Model only the detail the current behavior needs while
retaining a credible path for changes the task actually signals.

Names describe the present semantic role, unit, and scope rather than creation
order or incidental implementation. Rename a stale `counter` instead of adding
`counter2`; use the repository's prevailing grouping and naming conventions
rather than imposing one universal word order. Public interfaces deserve extra
care because names and structure become compatibility commitments.

There is no fixed line-count or field-count threshold for functions, classes,
or hierarchies. The LLM weighs clarity, test isolation, runtime cost, current
requirements, local conventions, and plausible changeability. Prefer the
simplest adequate design, not the smallest design at any future cost and not a
speculative architecture for detail no requirement uses.

## Question and evidence loop

1. Establish the intended behavior and observable success criteria.
2. Inspect the real source, tests, configuration, runtime, hardware, and recent
   changes before proposing an edit.
3. Reproduce a reported defect or uncertain technical claim in the smallest
   safe environment that preserves the relevant behavior.
4. Research exact installed versions and their primary specifications when
   behavior is version-sensitive. Treat documentation as a hypothesis until a
   focused probe confirms the behavior that matters here; record discrepancies.
5. Choose libraries and tools by the task's fidelity, isolation, maintenance,
   runtime, and performance needs. Do not select a heavyweight familiar stack
   automatically.
6. Make the smallest general change supported by the evidence. Re-run the
   focused reproduction, then broader tests whose scope can expose interactions.
7. Inspect failures and traces as new evidence. Repair general behavior rather
   than expected strings, visible fixtures, or one tuned parameter set.
8. Report what was observed, changed, verified, and still uncertain. A passing
   unit test is not proof of deployment, real-device behavior, or every
   operational condition.

## Simulate critical conditions

Do not make the user perform avoidable real-world failure tests. Design
boundaries so clocks, networks, storage, process lifecycle, remote peers, user
input, and device state can be controlled by tests. Combine deterministic
fakes for fast isolated evidence with integration or whole-system simulators
that exercise the actual components and serialization boundaries.

A risk-driven matrix may include slow, lost, duplicated, reordered, or restored
connections; unavailable or inconsistent counterparts; restart and recovery;
backgrounding, process destruction, and time jumps; partial writes and stale
state; concurrent user controls; and repeated transitions. Select only the
conditions relevant to the task, but include combinations when interactions are
the risk. Parameterize ranges and hold out variants so a solution cannot pass by
tuning one visible value.

Simulation does not eliminate higher-fidelity acceptance. Validate on the real
runtime, hardware, device class, or service boundary when emulation cannot
faithfully prove a required property. Record the version, configuration, seed,
commands, environment, and raw evidence needed to reproduce each result.

## Researched foundations

- NIST SSDF PW.8 requires teams to scope, design, perform, and document testing,
  consider the production technology stack, and retain discovered issues and
  remediations: <https://csrc.nist.gov/pubs/sp/800/218/final>.
- Android's current testing strategy separates unit, component, feature,
  application, and release-candidate layers by scope and fidelity instead of
  presenting one universal test environment:
  <https://developer.android.com/training/testing/fundamentals/strategies>.
- Android documents deterministic lifecycle manipulation with
  `ActivityScenario`, forced Doze/App Standby states with `adb`, and emulator
  network latency/speed controls. These can reproduce background destruction
  and intermittent-network conditions before a real-device run:
  <https://developer.android.com/guide/components/activities/testing>,
  <https://developer.android.com/training/monitoring-device-state/doze-standby>,
  and <https://developer.android.com/studio/run/emulator-console>.
- Linux `netem` can inject delay, loss, corruption, duplication, reordering, and
  rate limits, while its documented timer, queueing, and TCP-placement
  limitations are reasons to verify the actual experiment rather than trust a
  nominal setting: <https://man7.org/linux/man-pages/man8/tc-netem.8.html>.
- pytest's current `monkeypatch` and `tmp_path` facilities provide automatically
  restored dependency/environment substitution and isolated filesystem state;
  their availability and exact version still need to be checked in the target
  environment: <https://docs.pytest.org/en/stable/how-to/monkeypatch.html> and
  <https://docs.pytest.org/en/stable/how-to/tmp_path.html>.
- Python's class tutorial defines classes as a way to bundle data and behavior
  and points to data classes for record-like values; PEP 8 makes usage and local
  consistency the overriding naming principles. The C++ Core Guidelines
  independently recommend single-operation functions for testing/reuse and
  classes when related data has an invariant. These corroborate the recording's
  context-sensitive boundary rather than establishing a language-independent
  class threshold: <https://docs.python.org/3/tutorial/classes.html>,
  <https://peps.python.org/pep-0008/>, and
  <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines>.

These sources corroborate techniques, not a mandatory stack. The selected LLM
instruction/notes mechanism keeps programming, research, reporting, and other
semantic guidance separately authored and selected for a call. The
`prompt-instructions` live benchmark includes both authoring and held-out
category-application cases so this document does not substitute for behavior.
