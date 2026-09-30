# Security and permission boundaries

Swaag treats model output, user-provided attachments, retrieved pages, and tool output as untrusted semantic data. Prompts can guide model behavior, but prompts are not an authorization mechanism. Deterministic code validates schemas, checks the enabled tool set and effective tool kind again at dispatch, enforces filesystem boundaries, and verifies durable identities and content hashes.

## Deployment boundary

The raw communication service remains mechanically loopback-only; wildcard, LAN, VPN, DNS, and public binds are rejected. Protocol-specific authenticated exposure is layered on top of that local boundary. A2A bearer protection is optional and, when enabled, resolves the bearer credential only from the configured secret environment-variable name; literal bearer secrets in TOML/config overlays are rejected. Open WebUI artifact URLs are independently HMAC signed. Non-local deployment terminates TLS at a reverse proxy and forwards only the authenticated protocol routes rather than exposing the plaintext listener.

MCP Streamable HTTP is separately disabled unless `mcp.enabled` selects `streamable_http` or `both`. Its single `/mcp` endpoint accepts POST only, rejects non-loopback browser origins before reading the body, requires both response media types, bounds the body, and verifies protocol version, method, name, and declared primitive routing headers against the parsed request before dispatch. It does not mint protocol sessions. These checks reduce local DNS-rebinding and confused-deputy risk but do not authenticate another same-host process.

Runtime state belongs under the configured sessions root. Session-scoped filesystem identifiers accept only bounded ASCII storage IDs, and their resolved paths must remain below that root. This check covers active history, archived shards, exact artifacts, and persistent terminals. User-facing session names remain data and are never interpreted as paths. Attachment and artifact locators are integrity-checked before reads.

## Capability boundary

- `tools.enabled` is the capability allowlist. Discovery never grants a capability that dispatch would reject.
- `allow_stateful_tools` and `allow_side_effect_tools` are independent mechanical gates, rechecked against the validated operation's effective kind.
- `edit_text` and `write_file` additionally require `editor.allow_writes`; both stay inside configured project/workspace roots and honor an exact resolved-path allowlist when configured. The separate `agent_workspace` capability is not a project write bypass: it is bound to the configurable agent-owned data root and cannot address project paths.
- Side-effect tools with a deterministic effect verifier are checked after their history-backed writes are committed and before a successful tool result is exposed. Failed verification is durable failure evidence and the action loop cannot treat the call as successful.
- Read and write path checks resolve symlinks before comparing roots. Runtime-owned session/cache snapshots are excluded from ordinary workspace discovery.
- Raw attachments are stored without automatic parsing. SWAAG provides bounded raw reads only; domain-specific parsing/conversion belongs to external tools or explicitly enabled shell execution, so parser/provider isolation is owned by that external deployment boundary.
- MCP, A2A, AG-UI, Open WebUI, and direct task calls are adapters. They do not bypass registry, worker, attachment, or history checks.

## Deliberately powerful tools

Enabling project-facing `shell_command` or creating/sending a persistent `terminal` grants arbitrary command execution with the Swaag service account's operating-system permissions. Workspace read roots, editor write allowlists, and attachment limits do **not** sandbox those project-facing capabilities. For general calculations, doability tests, private notes and experiments that should not touch the user repository, use `agent_workspace`: it executes ordinary Python/shell work inside a bubblewrap mount/network sandbox rooted at `agent_data.root`, with no host-project or `/etc` visibility and no network. Its `pip_install` operation is separately classified as a side effect and deliberately enables network only for that explicit installation step.

External MCP servers, browser automation, attachment converters, databases, and other layer-three providers have their own dependency, network, credential, parser, and privilege attack surfaces. Their results are evidence, not trusted instructions. Local stdio servers inherit their configured process account/environment; remote Streamable HTTP servers require their own TLS/authentication boundary. Secrets required by providers must remain in the provider/connector environment and must not be copied into model prompts or durable semantic events.

## Audited residual risks

- The localhost transport authenticates by OS/network locality only; same-host callers that can reach the port are trusted clients.
- The service account and repository contents share one trust domain unless deployment adds stronger OS isolation.
- Time-of-check/time-of-use races against a concurrently malicious same-account process cannot be eliminated by path normalization alone.
- Parsing hostile complex inputs is an external-tool concern; use provider-specific sandboxing and least privilege where those inputs are untrusted.
- Optional Open WebUI text-artifact serving exists only behind signed, expiring, worker/session/artifact-scoped URLs. The loopback listener re-verifies artifact size/hash/path ownership, compares HMACs in constant time, returns private/no-store + nosniff + attachment-disposition headers, and never exposes internal filesystem paths. Non-loopback advertised artifact URLs require HTTPS and the signing key is read from a named environment variable.

These are explicit deployment constraints, not semantic decisions delegated to a model.
