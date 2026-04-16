from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

from swaag.config import AgentConfig
from swaag.types import SemanticMemoryItem, TrustLevel

TRUSTED_TOOLS = frozenset({"calculator", "time_now"})
UNTRUSTED_TOOLS = frozenset({"echo", "read_text", "read_file", "list_files", "notes", "edit_text", "write_file", "run_tests", "shell_command"})



def combine_trust_levels(levels: Iterable[TrustLevel]) -> TrustLevel:
    normalized = list(levels)
    if not normalized:
        return "derived"
    if any(level == "untrusted" for level in normalized):
        return "untrusted"
    if any(level == "derived" for level in normalized):
        return "derived"
    return "trusted"



def trust_level_for_event(event_type: str, payload: dict[str, Any], metadata: dict[str, Any] | None = None) -> TrustLevel:
    metadata = {} if metadata is None else metadata
    explicit = metadata.get("trust_level")
    if explicit in {"trusted", "untrusted", "derived"}:
        return explicit
    if event_type == "message_added":
        role = str(payload.get("message", {}).get("role", ""))
        return "untrusted" if role == "user" else "derived"
    if event_type in {"file_chunk_read", "buffer_chunk_read", "file_read_for_edit", "file_read_requested", "buffer_read_requested", "filesystem_read", "filesystem_listed", "shell_command_completed", "process_polled", "process_completed", "process_timed_out", "workspace_snapshot"}:
        return "untrusted"
    if event_type == "tool_result":
        tool_name = str(payload.get("tool_name", ""))
        if tool_name in TRUSTED_TOOLS:
            return "trusted"
        if tool_name in UNTRUSTED_TOOLS:
            return "untrusted"
    if event_type in {"model_response_received", "decision_parsed", "tool_called"}:
        return "untrusted"
    return "derived"



def provenance_for_event(event_type: str, payload: dict[str, Any], source_event_id: str | None = None) -> dict[str, Any]:
    provenance = {"event_type": event_type}
    if source_event_id is not None:
        provenance["source_event_id"] = source_event_id
    if "tool_name" in payload:
        provenance["tool_name"] = payload["tool_name"]
    if "path" in payload:
        provenance["path"] = payload["path"]
    return provenance



def with_trust_level(event_type: str, payload: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    result = dict({} if metadata is None else metadata)
    result.setdefault("trust_level", trust_level_for_event(event_type, payload, result))
    result.setdefault("provenance", provenance_for_event(event_type, payload))
    return result



def sanitize_external_text(config: AgentConfig, text: str) -> str:
    cleaned = text.replace("\x00", "")
    if len(cleaned) > config.security.max_external_text_chars:
        return cleaned[: config.security.max_external_text_chars]
    return cleaned



def sanitize_payload(config: AgentConfig, payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            result[key] = sanitize_external_text(config, value)
        elif isinstance(value, dict):
            result[key] = sanitize_payload(config, value)
        elif isinstance(value, list):
            result[key] = [sanitize_payload(config, item) if isinstance(item, dict) else sanitize_external_text(config, item) if isinstance(item, str) else item for item in value]
        else:
            result[key] = value
    return result



def should_promote_to_semantic(config: AgentConfig, *, trust_level: TrustLevel) -> bool:
    if config.security.block_untrusted_semantic and trust_level == "untrusted":
        return False
    return True



def semantic_item_is_trusted(config: AgentConfig, item: SemanticMemoryItem) -> bool:
    return should_promote_to_semantic(config, trust_level=item.trust_level)
