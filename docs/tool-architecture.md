# Tool and capability architecture

SWAAG separates agent behavior into three dependency layers. This boundary is architectural, not a packaging preference.

## Layer 1 - core agent harness

The core owns model calls, constrained decoding, context compilation and accounting, append-only history and replay, durable worker/session state, cancellation, retries, provenance, semantic reduction, completion evaluation, and transport-neutral orchestration. Layer 1 must not know that OCR, web search, a database vendor, an image generator, or any other domain capability exists.

## Layer 2 - SWAAG system tools

System tools are repository-owned capabilities that expose SWAAG state or generic host operations the harness needs to be useful: history search/windows, notes, prompt instructions, shared state, raw attachment references and bounded reads, filesystem/workspace operations, shell/process/terminal access, artifacts, scheduling/control, and similarly generic utilities. They remain tools rather than hidden planner code, and every one remains independently configurable/disableable. A deployment may disable all of them and run history-only model chat, but SWAAG does not need domain-specific built-ins to become useful.

A system tool may depend on SWAAG internals or on generic operating-system interfaces. It must not embed a particular external service/provider merely because that provider is commonly useful. Shell is deliberately powerful and can itself discover or invoke programs available to the service account; that host-level fallback does not make those programs part of SWAAG.

## Layer 3 - external/open-world tools

Everything domain-specific belongs outside the SWAAG repository: browser automation, web search providers, OCR/document conversion, speech systems, databases, proprietary APIs, image generation, remote services, and future capabilities that do not yet exist. External tools can live on the same machine or another machine and can have arbitrary dependencies, credentials, permissions, sandboxes, installation requirements, and failure modes.

SWAAG consumes external capabilities through generic portable descriptions. MCP is the preferred standards-based connector when available. Configured MCP servers can use local stdio or remote Streamable HTTP. Remote request headers use header-to-environment-variable mappings so credentials stay outside portable schemas and durable semantic configuration. For expiring credentials, a configured credential-provider command can return ephemeral headers plus an expiry timestamp; SWAAG caches and refreshes those headers and performs one forced refresh after a 401/403. The helper owns OAuth/CIMD/browser/token-storage semantics, so the core does not become an authorization client UI or credential vault. SWAAG discovers `tools/list`, projects ordinary MCP JSON Schema into its stricter constrained-decoding subset while retaining the exact original server schema, stages only semantically selected schemas into the model context, executes selected MCP tools, and records the result through the normal durable tool-history path. Client/UI-provided delegated tools use the same model-facing external catalog but retain their own executor lifecycle.

External tool names may not shadow system tools or tools from another external catalog. Provider-specific routing, defaults, authentication semantics, installation checks, and execution logic stay with the provider/connector, not the core registry.

## Optional external integration tests

Unit tests for the generic external-tool/MCP machinery are mandatory. Tests that require a particular independently installed external repository, browser, network, model, credential, or service are explicitly optional integration tests. They must be separately marked and skipped by the normal deterministic suite. Running an optional suite can expose a broken external environment, but absence of Aubro, all2text, or any other example provider must not mark the SWAAG core as failed.

The Jetson development host currently uses Automated Browser (Aubro) and all2text as real external MCP examples because their repositories are under our control. Their MCP servers and provider-specific tests belong in those repositories. SWAAG's optional integration suite only proves that a standards-based external server can be discovered and called; it does not promote either project into layers 1 or 2. The optional conformance harness can additionally launch those independent servers through the pinned official MCP client SDK to verify negotiation and schema/call decoding without adding the SDK to SWAAG runtime dependencies.
