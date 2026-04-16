from __future__ import annotations

from dataclasses import dataclass

from swaag.config import AgentConfig
from swaag.types import Message, SessionState


@dataclass(slots=True)
class CompressionDecision:
    should_compress: bool
    source_message_count: int



def decide_history_compression(config: AgentConfig, state: SessionState) -> CompressionDecision:
    total_messages = len(state.messages)
    if total_messages <= config.compression.max_messages_before_compress:
        return CompressionDecision(should_compress=False, source_message_count=0)
    keep = min(config.context.max_recent_messages, total_messages)
    source_count = max(0, total_messages - keep)
    return CompressionDecision(should_compress=source_count > 0, source_message_count=source_count)



def summary_message_payload(summary_text: str, *, source_message_count: int, created_at: str) -> dict:
    return {
        "role": "summary",
        "content": summary_text,
        "created_at": created_at,
        "metadata": {"source_message_count": source_message_count},
    }
