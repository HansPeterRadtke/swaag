from __future__ import annotations

from importlib import resources
from typing import Iterable

from swaag.compression import summary_provenance_text
from swaag.config import AgentConfig
from swaag.types import Message, ModelCallKind, PromptAssembly, PromptComponent
from swaag.utils import stable_json_dumps

LLAMA3_BEGIN = "<|begin_of_text|>"
LLAMA3_SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
LLAMA3_USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
LLAMA3_ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA3_EOT = "<|eot_id|>"


class PromptBuilder:
    """Build only the model/tool-loop action prompt and history-compaction prompt."""

    def __init__(self, config: AgentConfig):
        self._config = config

    def _load_template(self, template_name: str) -> str:
        resource = resources.files("swaag").joinpath(f"assets/prompts/{template_name}")
        return resource.read_text(encoding="utf-8").strip()

    def system_text(self, prompt_mode: str) -> str:
        template_name = (
            self._config.prompts.lean_system_template
            if prompt_mode == "lean"
            else self._config.prompts.standard_system_template
        )
        return self._load_template(template_name)

    def render_tool_catalog(self, tools: Iterable[tuple]) -> str:
        lines: list[str] = []
        for item in tools:
            name = str(item[0])
            description = str(item[1])
            schema = item[2]
            guidance = str(item[3]).strip() if len(item) > 3 else ""
            lines.extend(
                [
                    f"- name: {name}",
                    f"  description: {description}",
                    f"  input_schema: {stable_json_dumps(schema)}",
                ]
            )
            if guidance:
                lines.append(f"  usage_guidance: {guidance}")
        return "\n".join(lines)

    def render_capability_index(self, capabilities: Iterable[tuple[str, str, str]]) -> str:
        lines: list[str] = []
        for name, description, guidance in capabilities:
            lines.append(f"- {name}: {description}")
            if str(guidance).strip():
                lines.append(f"  guidance: {str(guidance).strip()}")
        return "\n".join(lines)

    def render_messages(self, messages: list[Message]) -> str:
        if not messages:
            return "(none)"
        rendered: list[str] = []
        for message in messages:
            label = message.role.upper()
            if message.name:
                label = f"{label}:{message.name}"
            rendered.append(
                f"[{label}]\n{summary_provenance_text(message)}{message.content.strip()}"
            )
        return "\n\n".join(rendered)

    def message_prompt_components(
        self,
        messages: list[Message],
        *,
        prefix: str,
        category: str,
        header: str,
        optional: bool = False,
        tool_result_projections: dict[int, str] | None = None,
    ) -> list[PromptComponent]:
        components = [
            PromptComponent(
                name=f"{prefix}_header",
                category=category,
                text=header + "\n",
                optional=optional,
            )
        ]
        if not messages:
            components.append(
                PromptComponent(
                    name=f"{prefix}_empty",
                    category=category,
                    text="(none)\n\n",
                    optional=optional,
                )
            )
            return components
        for index, message in enumerate(messages, start=1):
            label = message.role.upper()
            if message.name:
                label = f"{label}:{message.name}"
            event_sequence = message.metadata.get("source_event_sequence")
            event_hash = message.metadata.get("source_event_hash")
            provenance = ""
            body = message.content.strip()
            component_name = f"{prefix}_{index}"
            if message.role == "tool" and isinstance(event_sequence, int):
                component_name = f"{prefix}_tool_event_{event_sequence}"
                provenance = f"[SOURCE EVENT sequence={event_sequence} hash={event_hash or 'unknown'}]\n"
                if tool_result_projections and event_sequence in tool_result_projections:
                    body = (
                        "[SEMANTIC PROJECTION; raw source remains authoritative and retrievable]\n"
                        + tool_result_projections[event_sequence].strip()
                    )
            elif message.role == "summary":
                provenance = summary_provenance_text(message)
            components.append(
                PromptComponent(
                    name=component_name,
                    category="tool_result" if message.role == "tool" else category,
                    text=f"[{label}]\n{provenance}{body}\n\n",
                    optional=optional,
                )
            )
        return components

    def partition_turn(self, messages: list[Message]) -> tuple[list[Message], Message | None, list[Message]]:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "user":
                return messages[:index], messages[index], messages[index + 1 :]
        return messages, None, []

    def _assemble(
        self,
        kind: ModelCallKind,
        prompt_mode: str,
        user_components: list[PromptComponent],
    ) -> PromptAssembly:
        return self._assemble_with_system(
            kind,
            prompt_mode,
            self.system_text(prompt_mode),
            user_components,
        )

    def _assemble_with_system(
        self,
        kind: ModelCallKind,
        prompt_mode: str,
        system_instruction: str,
        user_components: list[PromptComponent],
    ) -> PromptAssembly:
        components = [
            PromptComponent(name="llama3_begin", category="wrapper", text=LLAMA3_BEGIN),
            PromptComponent(name="system_header", category="wrapper", text=LLAMA3_SYSTEM_HEADER),
            PromptComponent(
                name="system_prompt",
                category="system_prompt",
                text=system_instruction.strip(),
            ),
            PromptComponent(name="system_eot", category="wrapper", text=LLAMA3_EOT),
            PromptComponent(name="user_header", category="wrapper", text=LLAMA3_USER_HEADER),
            *user_components,
            PromptComponent(name="user_eot", category="wrapper", text=LLAMA3_EOT),
            PromptComponent(name="assistant_header", category="wrapper", text=LLAMA3_ASSISTANT_HEADER),
        ]
        return PromptAssembly(
            kind=kind,
            prompt_mode=prompt_mode,
            prompt_text="".join(component.text for component in components),
            components=components,
        )

    def build_semantic_operation_prompt(
        self,
        *,
        kind: ModelCallKind,
        system_instruction: str,
        components: list[PromptComponent],
        prompt_mode: str = "lean",
    ) -> PromptAssembly:
        return self._assemble_with_system(
            kind,
            prompt_mode,
            system_instruction,
            components,
        )

    def build_agent_action_prompt(
        self,
        messages: list[Message],
        tools: Iterable[tuple],
        *,
        original_request: str,
        pending_user_messages: list[str],
        prompt_mode: str,
        context_components: list[PromptComponent] | None = None,
        capability_index: Iterable[tuple[str, str, str]] | None = None,
        tool_result_projections: dict[int, str] | None = None,
        validation_feedback: str = "",
    ) -> PromptAssembly:
        history, current_user, turn_transcript = self.partition_turn(messages)
        components = [
            PromptComponent(
                name="original_user_request",
                category="current_user",
                text=f"Original user request, verbatim and authoritative:\n{original_request}\n\n",
            ),
            *self.message_prompt_components(
                history,
                prefix="conversation_history",
                category="history",
                header="Previous conversation messages, verbatim:",
                optional=True,
                tool_result_projections=tool_result_projections,
            ),
            PromptComponent(
                name="current_user_turn",
                category="current_user",
                text=(
                    "Current user message, verbatim:\n"
                    f"{current_user.content if current_user is not None else original_request}\n\n"
                ),
            ),
            *self.message_prompt_components(
                turn_transcript,
                prefix="current_turn",
                category="turn_context",
                header="Exact assistant actions and tool results since the current user message:",
                tool_result_projections=tool_result_projections,
            ),
        ]
        if pending_user_messages:
            pending = "\n\n".join(
                f"[USER INTERVENTION {index}]\n{text}"
                for index, text in enumerate(pending_user_messages, start=1)
            )
            components.append(
                PromptComponent(
                    name="pending_user_interventions",
                    category="current_user",
                    text=f"New user interventions, verbatim and authoritative:\n{pending}\n\n",
                )
            )
        if context_components:
            components.extend(context_components)
        if capability_index is not None:
            components.append(
                PromptComponent(
                    name="compact_capability_index",
                    category="tool_descriptions",
                    text=(
                        "Capabilities that may be loaded when semantically relevant. Their full schemas are intentionally omitted until selected:\n"
                        f"{self.render_capability_index(capability_index) or '(none)'}\n\n"
                    ),
                )
            )
        components.append(
            PromptComponent(
                name="loaded_tool_schemas",
                category="tool_descriptions",
                text=f"Exact tool schemas available for this call:\n{self.render_tool_catalog(tools) or '(none)'}\n\n",
            )
        )
        if validation_feedback:
            components.append(
                PromptComponent(
                    name="mechanical_validation_feedback",
                    category="instruction",
                    text=(
                        "The previous constrained output was mechanically invalid. Correct only that structural error:\n"
                        f"{validation_feedback}\n\n"
                    ),
                )
            )
        components.append(
            PromptComponent(
                name="agent_action_instruction",
                category="instruction",
                text=self._load_template(self._config.prompts.action_template),
            )
        )
        return self._assemble("action", prompt_mode, components)

    def build_completion_evaluation_prompt(
        self,
        *,
        original_request: str,
        assistant_message: str,
        status_json: str,
        tool_evidence: str = "",
        tool_evidence_rows: list[dict] | None = None,
        tool_result_projections: dict[int, str] | None = None,
    ) -> PromptAssembly:
        system_prompt = self._load_template(self._config.prompts.completion_evaluation_system_template)
        components = [
            PromptComponent(name="llama3_begin", category="wrapper", text=LLAMA3_BEGIN),
            PromptComponent(name="system_header", category="wrapper", text=LLAMA3_SYSTEM_HEADER),
            PromptComponent(name="system_prompt", category="system_prompt", text=system_prompt),
            PromptComponent(name="system_eot", category="wrapper", text=LLAMA3_EOT),
            PromptComponent(name="user_header", category="wrapper", text=LLAMA3_USER_HEADER),
            PromptComponent(
                name="completion_objective",
                category="current_user",
                text=f"Original user objective:\n{original_request}\n\n",
            ),
            PromptComponent(
                name="completion_candidate",
                category="turn_context",
                text=f"Candidate final answer:\n{assistant_message}\n\n",
            ),
            PromptComponent(
                name="completion_status",
                category="turn_context",
                text=f"Current action status:\n{status_json}\n\n",
            ),
            PromptComponent(
                name="completion_tool_evidence_header",
                category="tool_result",
                text="Deterministic/tool evidence from this turn:\n",
            ),
            *self._completion_evidence_components(
                prefix="completion",
                tool_evidence=tool_evidence,
                rows=tool_evidence_rows or [],
                projections=tool_result_projections or {},
            ),
            PromptComponent(
                name="completion_instruction",
                category="instruction",
                text="\n" + self._load_template(
                    self._config.prompts.completion_evaluation_template
                ),
            ),
            PromptComponent(name="user_eot", category="wrapper", text=LLAMA3_EOT),
            PromptComponent(name="assistant_header", category="wrapper", text=LLAMA3_ASSISTANT_HEADER),
        ]
        return PromptAssembly(kind="completion_evaluation", prompt_mode="lean", prompt_text="".join(c.text for c in components), components=components)

    def build_caller_structured_output_prompt(
        self,
        *,
        original_request: str,
        assistant_message: str,
        tool_evidence_rows: list[dict],
        tool_result_projections: dict[int, str] | None = None,
    ) -> PromptAssembly:
        system_prompt = self._load_template(
            self._config.prompts.caller_structured_output_system_template
        )
        components = [
            PromptComponent(name="llama3_begin", category="wrapper", text=LLAMA3_BEGIN),
            PromptComponent(name="system_header", category="wrapper", text=LLAMA3_SYSTEM_HEADER),
            PromptComponent(name="system_prompt", category="system_prompt", text=system_prompt),
            PromptComponent(name="system_eot", category="wrapper", text=LLAMA3_EOT),
            PromptComponent(name="user_header", category="wrapper", text=LLAMA3_USER_HEADER),
            PromptComponent(
                name="caller_output_objective",
                category="current_user",
                text=f"Original user objective:\n{original_request}\n\n",
            ),
            PromptComponent(
                name="caller_output_candidate",
                category="turn_context",
                text=f"Verified worker answer:\n{assistant_message}\n\n",
            ),
            PromptComponent(
                name="caller_output_evidence_header",
                category="tool_result",
                text="Exact tool evidence from this turn:\n",
            ),
            *self._completion_evidence_components(
                prefix="caller_output",
                tool_evidence="",
                rows=tool_evidence_rows,
                projections=tool_result_projections or {},
            ),
            PromptComponent(
                name="caller_output_instruction",
                category="instruction",
                text="\n"
                + self._load_template(
                    self._config.prompts.caller_structured_output_template
                ),
            ),
            PromptComponent(name="user_eot", category="wrapper", text=LLAMA3_EOT),
            PromptComponent(name="assistant_header", category="wrapper", text=LLAMA3_ASSISTANT_HEADER),
        ]
        return PromptAssembly(
            kind="caller_structured_output",
            prompt_mode="lean",
            prompt_text="".join(component.text for component in components),
            components=components,
        )

    @staticmethod
    def _completion_evidence_components(
        *,
        prefix: str,
        tool_evidence: str,
        rows: list[dict],
        projections: dict[int, str],
    ) -> list[PromptComponent]:
        if not rows:
            return [
                PromptComponent(
                    name=f"{prefix}_tool_evidence_empty" if not tool_evidence else f"{prefix}_tool_evidence_legacy",
                    category="tool_result",
                    text=(tool_evidence or "(none)") + "\n",
                )
            ]
        components: list[PromptComponent] = []
        for index, row in enumerate(rows, start=1):
            sequence = row.get("source_event_sequence")
            source_hash = str(row.get("source_event_hash", ""))
            name = f"{prefix}_tool_evidence_{index}"
            body = stable_json_dumps(row, indent=None)
            provenance = ""
            if isinstance(sequence, int):
                name = f"{prefix}_tool_event_{sequence}"
                provenance = f"[SOURCE EVENT sequence={sequence} hash={source_hash or 'unknown'}]\n"
                if sequence in projections:
                    body = (
                        "[SEMANTIC PROJECTION; raw source remains authoritative and retrievable]\n"
                        + projections[sequence].strip()
                    )
            components.append(
                PromptComponent(
                    name=name,
                    category="tool_result",
                    text=f"{provenance}{body}\n",
                )
            )
        return components

    def build_tool_result_projection_prompt(
        self,
        *,
        original_request: str,
        tool_name: str,
        raw_tool_result: str,
        source_event_sequence: int,
        source_event_hash: str,
        target_tokens: int,
    ) -> PromptAssembly:
        system_prompt = self._load_template(self._config.prompts.tool_result_projection_system_template)
        user_text = self._load_template(self._config.prompts.tool_result_projection_template).format(
            original_request=original_request,
            tool_name=tool_name,
            raw_tool_result=raw_tool_result,
            source_event_sequence=int(source_event_sequence),
            source_event_hash=source_event_hash,
            target_tokens=max(1, int(target_tokens)),
        )
        components = [
            PromptComponent(name="llama3_begin", category="wrapper", text=LLAMA3_BEGIN),
            PromptComponent(name="system_header", category="wrapper", text=LLAMA3_SYSTEM_HEADER),
            PromptComponent(name="system_prompt", category="system_prompt", text=system_prompt),
            PromptComponent(name="system_eot", category="wrapper", text=LLAMA3_EOT),
            PromptComponent(name="user_header", category="wrapper", text=LLAMA3_USER_HEADER),
            PromptComponent(name="projection_task", category="tool_result", text=user_text),
            PromptComponent(name="user_eot", category="wrapper", text=LLAMA3_EOT),
            PromptComponent(name="assistant_header", category="wrapper", text=LLAMA3_ASSISTANT_HEADER),
        ]
        return PromptAssembly(
            kind="tool_result_projection",
            prompt_mode="lean",
            prompt_text="".join(component.text for component in components),
            components=components,
        )

    def build_evidence_projection_prompt(
        self,
        *,
        purpose: str,
        source_label: str,
        raw_evidence: str,
        target_tokens: int,
    ) -> PromptAssembly:
        system_prompt = self._load_template(
            self._config.prompts.evidence_projection_system_template
        )
        user_text = self._load_template(
            self._config.prompts.evidence_projection_template
        ).format(
            purpose=purpose,
            source_label=source_label,
            raw_evidence=raw_evidence,
            target_tokens=max(1, int(target_tokens)),
        )
        return self._assemble_with_system(
            "evidence_projection",
            "lean",
            system_prompt,
            [
                PromptComponent(
                    name="evidence_projection_task",
                    category="history",
                    text=user_text,
                )
            ],
        )

    def build_communication_status_prompt(
        self,
        *,
        question: str,
        mechanical_status: dict,
        evidence_rows: list[dict],
        runtime_semantic_evidence: dict | None = None,
        evidence_projection: str = "",
        validation_feedback: str = "",
    ) -> PromptAssembly:
        components = [
            PromptComponent(
                name="communication_question",
                category="current_user",
                text=f"User status/history question, verbatim:\n{question}\n\n",
            ),
            PromptComponent(
                name="mechanical_runtime_status",
                category="runtime_state",
                text=(
                    "Deterministic runtime state (facts, not semantic interpretation):\n"
                    + stable_json_dumps(mechanical_status, indent=None)
                    + "\n\n"
                ),
            ),
            PromptComponent(
                name="status_evidence_header",
                category="history",
                text="Authoritative target-worker evidence, oldest to newest:\n",
            ),
        ]
        if evidence_projection:
            components.append(
                PromptComponent(
                    name="status_evidence_projection",
                    category="history",
                    text=(
                        "[SEMANTIC PROJECTION; exact durable events remain authoritative and retrievable]\n"
                        + evidence_projection.strip()
                        + "\n\n"
                    ),
                )
            )
        else:
            if runtime_semantic_evidence:
                components.append(
                    PromptComponent(
                        name="runtime_semantic_evidence",
                        category="history",
                        text=(
                            "[EXACT RUNTIME SEMANTIC SOURCE; not a durable event sequence]\n"
                            + stable_json_dumps(runtime_semantic_evidence, indent=None)
                            + "\n\n"
                        ),
                    )
                )
        if not evidence_projection and evidence_rows:
            for row in evidence_rows:
                sequence = int(row["sequence"])
                components.append(
                    PromptComponent(
                        name=f"status_event_{sequence}",
                        category="history",
                        text=(
                            f"[SOURCE EVENT sequence={sequence} hash={row['hash']}]\n"
                            + stable_json_dumps(row, indent=None)
                            + "\n\n"
                        ),
                    )
                )
        elif not evidence_projection and not runtime_semantic_evidence:
            components.append(
                PromptComponent(
                    name="status_evidence_empty",
                    category="history",
                    text="(none)\n\n",
                )
            )
        if validation_feedback.strip():
            components.append(
                PromptComponent(
                    name="communication_status_validation_feedback",
                    category="validation_feedback",
                    text=(
                        "The previous response was rejected by mechanical validation. "
                        "Correct the stated error using the same authoritative evidence:\n"
                        + validation_feedback.strip()
                        + "\n\n"
                    ),
                )
            )
        components.append(
            PromptComponent(
                name="communication_status_instruction",
                category="instruction",
                text=self._load_template(
                    self._config.prompts.communication_status_template
                ),
            )
        )
        return self._assemble_with_system(
            "communication_status",
            "lean",
            self._load_template(
                self._config.prompts.communication_status_system_template
            ),
            components,
        )

    def build_summary_prompt(
        self,
        messages: list[Message],
        *,
        prompt_mode: str = "lean",
        maximum_preserve_recent_messages: int = 0,
        target_summary_tokens: int = 0,
    ) -> PromptAssembly:
        history_block = self.render_messages(messages)
        system_prompt = self._load_template(self._config.prompts.summary_system_template)
        user_text = self._load_template(self._config.prompts.summary_template).format(
            history_block=history_block,
            maximum_preserve_recent_messages=max(0, int(maximum_preserve_recent_messages)),
            target_summary_tokens=max(1, int(target_summary_tokens)),
        )
        components = [
            PromptComponent(name="llama3_begin", category="wrapper", text=LLAMA3_BEGIN),
            PromptComponent(name="system_header", category="wrapper", text=LLAMA3_SYSTEM_HEADER),
            PromptComponent(name="system_prompt", category="system_prompt", text=system_prompt),
            PromptComponent(name="system_eot", category="wrapper", text=LLAMA3_EOT),
            PromptComponent(name="user_header", category="wrapper", text=LLAMA3_USER_HEADER),
            PromptComponent(name="summary_history", category="history", text=user_text),
            PromptComponent(name="user_eot", category="wrapper", text=LLAMA3_EOT),
            PromptComponent(name="assistant_header", category="wrapper", text=LLAMA3_ASSISTANT_HEADER),
        ]
        return PromptAssembly(
            kind="summary",
            prompt_mode=prompt_mode,
            prompt_text="".join(component.text for component in components),
            components=components,
        )
