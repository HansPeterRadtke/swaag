from __future__ import annotations

from importlib import resources
from typing import Iterable

from swaag.compression import summary_provenance_text
from swaag.config import AgentConfig
from swaag.types import (
    Message,
    ModelCallKind,
    PromptArtifact,
    PromptAssembly,
    PromptComponent,
    PromptMessageRange,
)
from swaag.utils import sha256_text, stable_json_dumps

FALLBACK_SYSTEM_PREFIX = "[SYSTEM MESSAGE]\n"
FALLBACK_MESSAGE_SEPARATOR = "\n[USER MESSAGE]\n"
FALLBACK_GENERATION_SUFFIX = "\n[ASSISTANT RESPONSE]\n"


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

    def _prompt_artifacts(
        self,
        *,
        kind: ModelCallKind,
        system_instruction: str,
        template_names: tuple[str, ...],
    ) -> list[PromptArtifact]:
        artifacts = [
            PromptArtifact(
                source="prompt_protocol:explicit_text_fallback_v1",
                sha256=sha256_text(
                    FALLBACK_SYSTEM_PREFIX
                    + "{system}"
                    + FALLBACK_MESSAGE_SEPARATOR
                    + "{user}"
                    + FALLBACK_GENERATION_SUFFIX
                ),
            ),
            PromptArtifact(
                source=f"rendered_system:{kind}",
                sha256=sha256_text(system_instruction.strip()),
            ),
        ]
        seen = {artifact.source for artifact in artifacts}
        for template_name in template_names:
            source = f"assets/prompts/{template_name}"
            if source in seen:
                continue
            seen.add(source)
            artifacts.append(
                PromptArtifact(
                    source=source,
                    sha256=sha256_text(self._load_template(template_name)),
                )
            )
        return artifacts

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
                        "[SEMANTIC PROJECTION; raw source remains authoritative and retrievable "
                        f"with history_window from active-session sequence {event_sequence}]\n"
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
        template_names: tuple[str, ...] = (),
    ) -> PromptAssembly:
        system_template = (
            self._config.prompts.lean_system_template
            if prompt_mode == "lean"
            else self._config.prompts.standard_system_template
        )
        return self._assemble_with_system(
            kind,
            prompt_mode,
            self._load_template(system_template),
            user_components,
            template_names=(system_template, *template_names),
        )

    def _assemble_with_system(
        self,
        kind: ModelCallKind,
        prompt_mode: str,
        system_instruction: str,
        user_components: list[PromptComponent],
        template_names: tuple[str, ...] = (),
    ) -> PromptAssembly:
        components = [
            PromptComponent(
                name="fallback_system_prefix",
                category="wrapper",
                text=FALLBACK_SYSTEM_PREFIX,
            ),
            PromptComponent(
                name="system_prompt",
                category="system_prompt",
                text=system_instruction.strip(),
            ),
            PromptComponent(
                name="fallback_message_separator",
                category="wrapper",
                text=FALLBACK_MESSAGE_SEPARATOR,
            ),
            *user_components,
            PromptComponent(
                name="fallback_generation_suffix",
                category="wrapper",
                text=FALLBACK_GENERATION_SUFFIX,
            ),
        ]
        user_end = 3 + len(user_components)
        return PromptAssembly(
            kind=kind,
            prompt_mode=prompt_mode,
            prompt_text="".join(component.text for component in components),
            components=components,
            prompt_artifacts=self._prompt_artifacts(
                kind=kind,
                system_instruction=system_instruction,
                template_names=template_names,
            ),
            message_ranges=[
                PromptMessageRange(role="system", component_start=1, component_end=2),
                PromptMessageRange(role="user", component_start=3, component_end=user_end),
            ],
        )

    def build_semantic_operation_prompt(
        self,
        *,
        kind: ModelCallKind,
        system_instruction: str,
        components: list[PromptComponent],
        prompt_mode: str = "lean",
        template_names: tuple[str, ...] = (),
    ) -> PromptAssembly:
        return self._assemble_with_system(
            kind,
            prompt_mode,
            system_instruction,
            components,
            template_names=template_names,
        )

    def template_text(self, template_name: str) -> str:
        return self._load_template(template_name)

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
        return self._assemble(
            "action",
            prompt_mode,
            components,
            template_names=(self._config.prompts.action_template,),
        )


    def build_agent_capability_selection_prompt(
        self,
        messages: list[Message],
        *,
        original_request: str,
        pending_user_messages: list[str],
        capability_index: Iterable[tuple[str, str, str]],
        context_components: list[PromptComponent] | None = None,
        validation_feedback: str = "",
    ) -> PromptAssembly:
        history, current_user, turn_transcript = self.partition_turn(messages)
        capabilities = list(capability_index)
        rendered_capabilities = "\n".join(
            f"- {name}: {description} {guidance}".strip()
            for name, description, guidance in capabilities
            if str(name) != "load_tools"
        )
        components = [
            PromptComponent(
                name="original_user_request", category="current_user",
                text=f"Original user request, verbatim and authoritative:\n{original_request}\n\n",
            ),
            *self.message_prompt_components(
                history, prefix="conversation_history", category="history",
                header="Previous conversation messages, verbatim:", optional=True,
            ),
            PromptComponent(
                name="current_user_turn", category="current_user",
                text=("Current user message, verbatim:\n" +
                      f"{current_user.content if current_user is not None else original_request}\n\n"),
            ),
            *self.message_prompt_components(
                turn_transcript, prefix="current_turn", category="turn_context",
                header="Exact assistant actions and tool results since the current user message:",
            ),
        ]
        if pending_user_messages:
            pending = "\n\n".join(
                f"[USER INTERVENTION {index}]\n{text}"
                for index, text in enumerate(pending_user_messages, start=1)
            )
            components.append(PromptComponent(
                name="pending_user_interventions", category="current_user",
                text=f"New user interventions, verbatim and authoritative:\n{pending}\n\n",
            ))
        if context_components:
            components.extend(context_components)
        components.append(PromptComponent(
            name="capability_selection_index", category="tool_descriptions",
            text=f"Existing capability identities and descriptions:\n{rendered_capabilities}\n\n",
        ))
        if validation_feedback:
            components.append(PromptComponent(
                name="mechanical_validation_feedback", category="instruction",
                text=f"Previous selection was mechanically invalid: {validation_feedback}\n\n",
            ))
        components.append(PromptComponent(
            name="agent_capability_selection_instruction", category="instruction",
            text=(
                "Choose exactly one existing capability identity that is the best concrete NEXT capability needed "
                "for the user's current task. This call has one semantic responsibility: capability selection. "
                "Choose none only when no not-yet-loaded capability should be loaded now. Do not choose a loader, "
                "do not choose several capabilities, do not plan future steps, and do not write user-facing prose."
            ),
        ))
        return self._assemble(
            "action_capability_selection", "standard", components, template_names=()
        )

    def build_agent_tool_call_prompt(
        self,
        messages: list[Message],
        tools: Iterable[tuple],
        *,
        original_request: str,
        pending_user_messages: list[str],
        context_components: list[PromptComponent] | None = None,
        capability_index: Iterable[tuple[str, str, str]] | None = None,
        tool_result_projections: dict[int, str] | None = None,
        completed_tool_calls: list[dict] | None = None,
        validation_feedback: str = "",
    ) -> PromptAssembly:
        assembly = self.build_agent_action_prompt(
            messages,
            tools,
            original_request=original_request,
            pending_user_messages=pending_user_messages,
            prompt_mode="standard",
            context_components=context_components,
            capability_index=capability_index,
            tool_result_projections=tool_result_projections,
            validation_feedback=validation_feedback,
        )
        components = [
            component
            for component in assembly.components[3:-1]
            if component.name != "agent_action_instruction"
        ]
        if completed_tool_calls:
            lines = [
                "Previously completed tool calls whose corresponding results are present in the current authoritative transcript:"
            ]
            for item in completed_tool_calls:
                lines.append(
                    "- "
                    + stable_json_dumps(
                        {
                            "tool_name": item.get("tool_name"),
                            "arguments": item.get("arguments", {}),
                        },
                        indent=None,
                    )
                    + f" -> recorded execution; corresponding result/error evidence source event sequence={item.get('source_event_sequence', 'unknown')}."
                )
            components.append(
                PromptComponent(
                    name="completed_tool_calls",
                    category="tool_result",
                    text="\n".join(lines) + "\n\n",
                )
            )
        components.append(
            PromptComponent(
                name="agent_tool_call_instruction",
                category="instruction",
                text=(
                    "Choose concrete tool calls and their exact arguments for the next useful mechanical work. "
                    "This call has one semantic responsibility: tool selection/arguments. Do not write user-facing "
                    "prose, status, questions, completion judgments, or continuation flags. Return an empty tool_calls "
                    "array when no currently loaded tool should be executed. If a relevant capability is listed but "
                    "not loaded, select load_tools for that capability instead of guessing its schema. Calls in one "
                    "response must be independent; a later call must not depend on an earlier call's result. Treat the "
                    "previously-completed-tool-calls section, when present, as factual history only. Repetition may be "
                    "semantically useful when external state or freshness matters; decide that from the task and evidence."
                ),
            )
        )
        return self._assemble(
            "action_tool_call",
            "standard",
            components,
            template_names=(),
        )

    def build_agent_terminal_response_prompt(
        self,
        messages: list[Message],
        *,
        original_request: str,
        pending_user_messages: list[str],
        context_components: list[PromptComponent] | None = None,
        tool_result_projections: dict[int, str] | None = None,
        allow_silent_completion: bool = False,
        validation_feedback: str = "",
    ) -> PromptAssembly:
        history, current_user, turn_transcript = self.partition_turn(messages)
        components = [
            PromptComponent(
                name="original_user_request", category="current_user",
                text=f"Original user request, verbatim and authoritative:\n{original_request}\n\n",
            ),
            *self.message_prompt_components(
                history, prefix="conversation_history", category="history",
                header="Previous conversation messages, verbatim:", optional=True,
                tool_result_projections=tool_result_projections,
            ),
            PromptComponent(
                name="current_user_turn", category="current_user",
                text=("Current user message, verbatim:\n" +
                      f"{current_user.content if current_user is not None else original_request}\n\n"),
            ),
            *self.message_prompt_components(
                turn_transcript, prefix="current_turn", category="turn_context",
                header="Exact assistant actions and tool results since the current user message:",
                tool_result_projections=tool_result_projections,
            ),
        ]
        if pending_user_messages:
            pending = "\n\n".join(
                f"[USER INTERVENTION {index}]\n{text}"
                for index, text in enumerate(pending_user_messages, start=1)
            )
            components.append(PromptComponent(
                name="pending_user_interventions", category="current_user",
                text=f"New user interventions, verbatim and authoritative:\n{pending}\n\n",
            ))
        if context_components:
            components.extend(context_components)
        if validation_feedback:
            components.append(PromptComponent(
                name="mechanical_validation_feedback", category="instruction",
                text=("The previous constrained output was mechanically invalid. Correct only that structural error:\n"
                      + validation_feedback + "\n\n"),
            ))
        components.append(PromptComponent(
            name="agent_terminal_response_instruction", category="instruction",
            text=(
                "Produce the complete terminal user-facing response for the current request using only authoritative "
                "request/history/tool evidence. This call has one semantic responsibility: final response generation. "
                "Do not choose tools, plan another action, emit status, or judge completion. The message must be non-empty "
                + ("unless the enclosing protocol explicitly requires silence; set silent_completion accordingly."
                   if allow_silent_completion else "and silent_completion must be false.")
            ),
        ))
        return self._assemble("action_terminal_response", "standard", components, template_names=())

    def build_completion_verdict_prompt(
        self,
        *,
        original_request: str,
        assistant_message: str,
        status_json: str,
        tool_evidence: str = "",
        tool_evidence_rows: list[dict] | None = None,
        tool_result_projections: dict[int, str] | None = None,
        historical_evidence: str = "",
        historical_evidence_projection: str = "",
        reexpanded_evidence_rows: list[dict] | None = None,
        reexpanded_evidence_projections: dict[str, str] | None = None,
    ) -> PromptAssembly:
        base = self.build_completion_evaluation_prompt(
            original_request=original_request,
            assistant_message=assistant_message,
            status_json=status_json,
            tool_evidence=tool_evidence,
            tool_evidence_rows=tool_evidence_rows,
            tool_result_projections=tool_result_projections,
            historical_evidence=historical_evidence,
            historical_evidence_projection=historical_evidence_projection,
            evidence_source_inventory=[],
            reexpanded_evidence_rows=reexpanded_evidence_rows,
            reexpanded_evidence_projections=reexpanded_evidence_projections,
        )
        user_components = [
            component
            for component in base.components[3:-1]
            if component.name not in {
                "completion_evidence_source_inventory",
                "completion_instruction",
            }
        ]
        user_components.append(
            PromptComponent(
                name="completion_verdict_instruction",
                category="instruction",
                text=(
                    "\nDecide only whether the user's objective is actually complete from the supplied authoritative evidence. "
                    "Return complete=true only when no meaningful requested work remains. Do not generate a reason, "
                    "remaining-work list, evidence request, plan, status, or user-facing prose. Return only the constrained JSON object.\n"
                ),
            )
        )
        return self._assemble_with_system(
            "completion_evaluation",
            "lean",
            (
                "You are an independent completion evaluator. Make exactly one semantic decision: whether the user's "
                "actual objective is complete from the supplied evidence. Deterministic tool/test evidence is authoritative "
                "for what mechanically happened. Do not reward final-looking prose and do not invent missing evidence. "
                "Return only the constrained JSON object."
            ),
            user_components,
            template_names=(),
        )

    def build_completion_evaluation_prompt(
        self,
        *,
        original_request: str,
        assistant_message: str,
        status_json: str,
        tool_evidence: str = "",
        tool_evidence_rows: list[dict] | None = None,
        tool_result_projections: dict[int, str] | None = None,
        historical_evidence: str = "",
        historical_evidence_projection: str = "",
        evidence_source_inventory: list[dict] | None = None,
        reexpanded_evidence_rows: list[dict] | None = None,
        reexpanded_evidence_projections: dict[str, str] | None = None,
    ) -> PromptAssembly:
        system_prompt = self._load_template(self._config.prompts.completion_evaluation_system_template)
        user_components = [
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
            *(
                [
                    PromptComponent(
                        name="completion_evidence_source_inventory",
                        category="tool_result",
                        text=(
                            "\nExact evidence sources available for semantic re-expansion. "
                            "Request only a source whose complete content is needed to decide completion:\n"
                            + stable_json_dumps(
                                evidence_source_inventory or [], indent=None
                            )
                            + "\n"
                        ),
                    )
                ]
                if evidence_source_inventory
                else []
            ),
            *self._completion_reexpanded_evidence_components(
                reexpanded_evidence_rows or [],
                reexpanded_evidence_projections or {},
            ),
            *(
                [
                    PromptComponent(
                        name="completion_historical_evidence",
                        category="history",
                        text=(
                            "\nDurable evidence from before the current user turn:\n"
                            + (
                                "[SEMANTIC PROJECTION; exact events remain authoritative "
                                "and retrievable]\n"
                                + historical_evidence_projection
                                if historical_evidence_projection
                                else historical_evidence
                            )
                            + "\n"
                        ),
                    )
                ]
                if historical_evidence or historical_evidence_projection
                else []
            ),
            PromptComponent(
                name="completion_instruction",
                category="instruction",
                text="\n" + self._load_template(
                    self._config.prompts.completion_evaluation_template
                ),
            ),
        ]
        return self._assemble_with_system(
            "completion_evaluation",
            "lean",
            system_prompt,
            user_components,
            template_names=(
                self._config.prompts.completion_evaluation_system_template,
                self._config.prompts.completion_evaluation_template,
            ),
        )

    @staticmethod
    def _completion_reexpanded_evidence_components(
        rows: list[dict],
        projections: dict[str, str],
    ) -> list[PromptComponent]:
        components: list[PromptComponent] = []
        for index, row in enumerate(rows, start=1):
            source_key = f"{row.get('source_kind', '')}:{row.get('source_id', '')}"
            body = stable_json_dumps(row, indent=None)
            if source_key in projections:
                projected = dict(row)
                projected.pop("text", None)
                projected["semantic_projection"] = projections[source_key].strip()
                projected["projection_notice"] = (
                    "Derived view only; the exact integrity-checked source remains authoritative."
                )
                body = stable_json_dumps(projected, indent=None)
            components.append(
                PromptComponent(
                    name=f"completion_reexpanded_evidence_{index}",
                    category="tool_result",
                    text=(
                        "\nSemantically requested exact evidence source:\n"
                        f"{body}\n"
                    ),
                )
            )
        return components

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
        user_components = [
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
        ]
        return self._assemble_with_system(
            "caller_structured_output",
            "lean",
            system_prompt,
            user_components,
            template_names=(
                self._config.prompts.caller_structured_output_system_template,
                self._config.prompts.caller_structured_output_template,
            ),
        )

    def build_response_relevance_prompt(
        self,
        *,
        original_request: str,
        source_answer: str,
        validation_feedback: str = "",
    ) -> PromptAssembly:
        system_template = self._config.prompts.response_relevance_system_template
        user_template = self._config.prompts.response_relevance_template
        user_text = self._load_template(user_template).format(
            original_request=original_request,
            source_answer=source_answer,
            validation_feedback=(
                "The previous candidate was independently rejected. Correct these issues:\n"
                + validation_feedback.strip()
                + "\n"
                if validation_feedback.strip()
                else ""
            ),
        )
        return self._assemble_with_system(
            "response_relevance",
            "lean",
            self._load_template(system_template),
            [
                PromptComponent(
                    name="response_relevance_task",
                    category="turn_context",
                    text=user_text,
                )
            ],
            template_names=(system_template, user_template),
        )

    def build_audio_rendering_prompt(
        self,
        *,
        original_request: str,
        source_answer: str,
        validation_feedback: str = "",
    ) -> PromptAssembly:
        system_template = self._config.prompts.audio_rendering_system_template
        user_template = self._config.prompts.audio_rendering_template
        user_text = self._load_template(user_template).format(
            original_request=original_request,
            source_answer=source_answer,
            validation_feedback=(
                "The previous candidate was independently rejected. Correct these issues:\n"
                + validation_feedback.strip()
                + "\n"
                if validation_feedback.strip()
                else ""
            ),
        )
        return self._assemble_with_system(
            "audio_rendering",
            "lean",
            self._load_template(system_template),
            [
                PromptComponent(
                    name="audio_rendering_task",
                    category="turn_context",
                    text=user_text,
                )
            ],
            template_names=(system_template, user_template),
        )

    def build_presentation_evaluation_prompt(
        self,
        *,
        mode: str,
        original_request: str,
        source_answer: str,
        candidate_answer: str,
    ) -> PromptAssembly:
        system_template = self._config.prompts.presentation_evaluation_system_template
        user_template = self._config.prompts.presentation_evaluation_template
        user_text = self._load_template(user_template).format(
            mode=mode,
            original_request=original_request,
            source_answer=source_answer,
            candidate_answer=candidate_answer,
        )
        return self._assemble_with_system(
            "presentation_evaluation",
            "lean",
            self._load_template(system_template),
            [
                PromptComponent(
                    name="presentation_evaluation_task",
                    category="turn_context",
                    text=user_text,
                )
            ],
            template_names=(system_template, user_template),
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
                        "[SEMANTIC PROJECTION; raw source remains authoritative and retrievable "
                        f"with history_window from active-session sequence {sequence}]\n"
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
        return self._assemble_with_system(
            "tool_result_projection",
            "lean",
            system_prompt,
            [PromptComponent(name="projection_task", category="tool_result", text=user_text)],
            template_names=(
                self._config.prompts.tool_result_projection_system_template,
                self._config.prompts.tool_result_projection_template,
            ),
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
            template_names=(
                self._config.prompts.evidence_projection_system_template,
                self._config.prompts.evidence_projection_template,
            ),
        )

    def build_prompt_instruction_projection_prompt(
        self,
        *,
        call_kind: ModelCallKind,
        source_instructions: str,
        source_sha256: str,
        source_tokens: int,
        overflow_tokens: int,
        target_tokens: int,
    ) -> PromptAssembly:
        system_template = (
            self._config.prompts.prompt_instruction_projection_system_template
        )
        user_template = self._config.prompts.prompt_instruction_projection_template
        system_prompt = self._load_template(system_template)
        user_text = self._load_template(user_template).format(
            call_kind=call_kind,
            source_instructions=source_instructions,
            source_sha256=source_sha256,
            source_tokens=max(1, int(source_tokens)),
            overflow_tokens=max(1, int(overflow_tokens)),
            target_tokens=max(1, int(target_tokens)),
        )
        return self._assemble_with_system(
            "prompt_instruction_projection",
            "lean",
            system_prompt,
            [
                PromptComponent(
                    name="prompt_instruction_projection_task",
                    category="system_prompt_instruction",
                    text=user_text,
                )
            ],
            template_names=(system_template, user_template),
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
            template_names=(
                self._config.prompts.communication_status_system_template,
                self._config.prompts.communication_status_template,
            ),
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
        return self._assemble_with_system(
            "summary",
            prompt_mode,
            system_prompt,
            [PromptComponent(name="summary_history", category="history", text=user_text)],
            template_names=(
                self._config.prompts.summary_system_template,
                self._config.prompts.summary_template,
            ),
        )
