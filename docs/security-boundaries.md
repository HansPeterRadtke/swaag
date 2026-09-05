# Security and permission boundaries

Swaag treats model output, user-provided attachments, retrieved pages, and tool output as untrusted semantic data. Prompts can guide model behavior, but prompts are not an authorization mechanism. Deterministic code validates schemas, checks the enabled tool set and effective tool kind again at dispatch, enforces filesystem boundaries, and verifies durable identities and content hashes.

## Deployment boundary

The communication service has no network authentication. It therefore fails closed unless its bind host is an explicit IPv4/IPv6 loopback address or `localhost`; wildcard, LAN, VPN, DNS, and public binds are rejected. The deployed systemd unit remains localhost-only. A reverse proxy must not expose this listener until a separately designed authenticated authorization boundary exists.

MCP Streamable HTTP is separately disabled unless `mcp.enabled` selects `streamable_http` or `both`. Its single `/mcp` endpoint accepts POST only, rejects non-loopback browser origins before reading the body, requires both response media types, bounds the body, and verifies protocol version, method, name, and declared primitive routing headers against the parsed request before dispatch. It does not mint protocol sessions. These checks reduce local DNS-rebinding and confused-deputy risk but do not authenticate another same-host process.

Runtime state belongs under the configured sessions root. Session-scoped filesystem identifiers accept only bounded ASCII storage IDs, and their resolved paths must remain below that root. This check covers active history, archived shards, exact artifacts, and persistent terminals. User-facing session names remain data and are never interpreted as paths. Attachment and artifact locators are integrity-checked before reads.

## Capability boundary

- `tools.enabled` is the capability allowlist. Discovery never grants a capability that dispatch would reject.
- `allow_stateful_tools` and `allow_side_effect_tools` are independent mechanical gates, rechecked against the validated operation's effective kind.
- `edit_text` and `write_file` additionally require `editor.allow_writes`; both stay inside the workspace and honor an exact resolved-path allowlist when configured.
- Side-effect tools with a deterministic effect verifier are checked after their history-backed writes are committed and before a successful tool result is exposed. Failed verification is durable failure evidence and the action loop cannot treat the call as successful.
- Read and write path checks resolve symlinks before comparing roots. Runtime-owned session/cache snapshots are excluded from ordinary workspace discovery.
- Raw attachments are stored without automatic parsing. SWAAG provides bounded raw reads only; domain-specific parsing/conversion belongs to external tools or explicitly enabled shell execution, so parser/provider isolation is owned by that external deployment boundary.
- MCP, A2A, AG-UI, Open WebUI, and direct task calls are adapters. They do not bypass registry, worker, attachment, or history checks.

## Deliberately powerful tools

Enabling `shell_command` or creating/sending a persistent `terminal` grants arbitrary command execution with the Swaag service account's operating-system permissions. Workspace read roots, editor write allowlists, and attachment limits do **not** sandbox a shell. A command can read or change anything that account can access, use the network, start children, and invoke other programs. These capabilities must be disabled for untrusted tasks or isolated with OS/service controls; Swaag does not claim an in-process shell sandbox.

External MCP servers, browser automation, attachment converters, databases, and other layer-three providers have their own dependency, network, credential, parser, and privilege attack surfaces. Their results are evidence, not trusted instructions. Local stdio servers inherit their configured process account/environment; remote Streamable HTTP servers require their own TLS/authentication boundary. Secrets required by providers must remain in the provider/connector environment and must not be copied into model prompts or durable semantic events.

## Audited residual risks

- The localhost transport authenticates by OS/network locality only; same-host callers that can reach the port are trusted clients.
- The service account and repository contents share one trust domain unless deployment adds stronger OS isolation.
- Time-of-check/time-of-use races against a concurrently malicious same-account process cannot be eliminated by path normalization alone.
- Parsing hostile complex inputs is an external-tool concern; use provider-specific sandboxing and least privilege where those inputs are untrusted.
- No artifact-serving endpoint is exposed. Adding one requires authenticated per-task authorization and content-disposition/type controls rather than publishing internal paths.

These are explicit deployment constraints, not semantic decisions delegated to a model.
