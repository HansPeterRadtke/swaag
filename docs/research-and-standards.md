# Research and standards notes

External work informs Swaag but is not architectural authority. Re-check versions and specifications before implementation because these projects evolve quickly.

Anthropic's context-engineering work treats context as a finite resource curated for every inference from instructions, tools, external data, and history. Letta similarly treats context management as a central harness responsibility. Recent externalized-history work such as Scroll explores keeping lossless events outside the prompt and projecting selected information into working context. Preserve authoritative source state outside context and benchmark compaction rather than copying one memory framework.

Relevant evaluation families include FollowBench for multi-constraint instruction following, AgentIF and AgentIF-OneDay for agentic instruction following, OR-Bench for unnecessary refusal, and long-horizon benchmarks for persistence and premature completion. Prompt optimization systems such as Promptfoo illustrate a useful baseline/variant/held-out-evaluation workflow. Keep prompt evaluation model-specific.

AG-UI defines rich streamed agent-to-UI events. A2A defines durable task-oriented interoperability including long-running task states. Open WebUI exposes status/progress, user input and confirmation, files, citations, and custom events and can be supported through an adapter. MCP is useful for capability discovery and structured tools. These solve different layers; keep Swaag's internal event/state model transport-independent.

Docling is a strong broad document conversion/extraction system. Jetson also has the separate all2text project with multiple format-specific and provider-backed conversion paths. Neither is a universal semantic ingestion standard. Preserve raw files and let an LLM choose whether and how to inspect them based on user intent.

llama.cpp supports continuous batching and parallel serving configurations, but cancellation/preemption behavior must be tested for the deployed version. vLLM provides alternative scheduling mechanisms. Swaag's runtime abstraction should be stronger than any backend and must not claim true suspend/resume unless the backend actually provides it.

OpenTelemetry has GenAI semantic conventions for model, agent, and tool telemetry. Prefer those conventions for operational traces and metrics while keeping telemetry distinct from durable semantic execution history.

When a design task says to research an open question, actually search documentation, specifications, papers, implementations, and the local machine where relevant; compare alternatives; perform safe experiments; and record what passed, failed, and remains uncertain. Do not convert an open research instruction into an unsupported architectural assertion.
