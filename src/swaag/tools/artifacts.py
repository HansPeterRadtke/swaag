from __future__ import annotations

from typing import Any

from swaag.environment.artifacts import TextArtifactStore
from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.types import ToolExecutionResult, ToolGeneratedEvent
from swaag.utils import stable_json_dumps


class ReadArtifactTool(Tool):
    repeated_observation_is_redundant = True
    name = "read_artifact"
    description = "Read an exact bounded slice of a durable text artifact produced by tools, such as full shell/test output that was too large for the immediate context."
    usage_guidance = "Use the exact stdout_artifact_id or stderr_artifact_id returned by the producing tool as artifact_id; never use a filename, command, or guessed label. Because tool calls in one action cannot consume results from earlier calls in that same action, run the producing tool first and read its returned artifact_id in a later action. Advance start_offset with next_offset until finished, and use offsets strategically rather than rereading the same slice."
    kind = "pure"
    input_schema = {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string"},
            "start_offset": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "max_chars": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        },
        "required": ["artifact_id", "start_offset", "max_chars"],
        "additionalProperties": False,
    }

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        artifact_id = raw_input.get("artifact_id")
        start_offset = raw_input.get("start_offset")
        start_offset = 0 if start_offset is None else start_offset
        max_chars = raw_input.get("max_chars")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ToolValidationError("read_artifact.artifact_id must be a non-empty string")
        if not isinstance(start_offset, int) or isinstance(start_offset, bool) or start_offset < 0:
            raise ToolValidationError("read_artifact.start_offset must be a non-negative integer")
        if max_chars is not None and (not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0):
            raise ToolValidationError("read_artifact.max_chars must be a positive integer")
        return {"artifact_id": artifact_id.strip(), "start_offset": start_offset, "max_chars": max_chars}

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"artifact_read"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        max_chars = validated_input["max_chars"] or context.config.reader.default_chunk_chars
        max_chars = min(int(max_chars), int(context.config.reader.max_chunk_chars))
        store = TextArtifactStore(context.config.sessions.root, context.session_state.session_id)
        artifact_id = validated_input["artifact_id"]
        alias_key = {
            "stdout": "stdout_artifact_id",
            "latest_stdout": "stdout_artifact_id",
            "stderr": "stderr_artifact_id",
            "latest_stderr": "stderr_artifact_id",
        }.get(artifact_id.casefold(), artifact_id)
        if alias_key in {"stdout_artifact_id", "stderr_artifact_id", "artifact_id"}:
            resolved = ""
            for message in reversed(context.session_state.messages):
                if message.role != "tool" or not isinstance(message.metadata, dict):
                    continue
                output_meta = message.metadata.get("output", {})
                if not isinstance(output_meta, dict):
                    continue
                candidate = output_meta.get(alias_key)
                if isinstance(candidate, str) and candidate.strip():
                    resolved = candidate.strip()
                    break
            if resolved:
                artifact_id = resolved
        output = store.read(
            artifact_id,
            start_offset=validated_input["start_offset"],
            max_chars=max_chars,
        )
        event = ToolGeneratedEvent(
            "artifact_read",
            {
                "artifact_id": output["artifact_id"],
                "start_offset": output["start_offset"],
                "end_offset": output["end_offset"],
                "finished": output["finished"],
            },
        )
        return ToolExecutionResult(
            tool_name=self.name,
            output=output,
            display_text=f"read_artifact result: {stable_json_dumps(output, indent=2)}",
            generated_events=[event],
        )


ARTIFACT_TOOLS = [ReadArtifactTool()]
