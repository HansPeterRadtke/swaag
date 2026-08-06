from __future__ import annotations

from importlib import resources
from typing import Iterable

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

    def render_messages(self, messages: list[Message]) -> str:
        if not messages:
            return "(none)"
        rendered: list[str] = []
        for message in messages:
            label = message.role.upper()
            if message.name:
                label = f"{label}:{message.name}"
            rendered.append(f"[{label}]\n{message.content.strip()}")
        return "\n\n".join(rendered)

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
        components = [
            PromptComponent(name="llama3_begin", category="wrapper", text=LLAMA3_BEGIN),
            PromptComponent(name="system_header", category="wrapper", text=LLAMA3_SYSTEM_HEADER),
            PromptComponent(name="system_prompt", category="system_prompt", text=self.system_text(prompt_mode)),
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

    def build_agent_action_prompt(
        self,
        messages: list[Message],
        tools: Iterable[tuple],
        *,
        original_request: str,
        pending_user_messages: list[str],
        prompt_mode: str,
        context_components: list[PromptComponent] | None = None,
        validation_feedback: str = "",
    ) -> PromptAssembly:
        history, current_user, turn_transcript = self.partition_turn(messages)
        components = [
            PromptComponent(
                name="original_user_request",
                category="current_user",
                text=f"Original user request, verbatim and authoritative:\n{original_request}\n\n",
            ),
            PromptComponent(
                name="conversation_history",
                category="history",
                text=f"Previous conversation messages, verbatim:\n{self.render_messages(history)}\n\n",
                optional=True,
            ),
            PromptComponent(
                name="current_user_turn",
                category="current_user",
                text=(
                    "Current user message, verbatim:\n"
                    f"{current_user.content if current_user is not None else original_request}\n\n"
                ),
            ),
            PromptComponent(
                name="current_turn_transcript",
                category="turn_context",
                text=(
                    "Exact assistant actions and tool results since the current user message:\n"
                    f"{self.render_messages(turn_transcript)}\n\n"
                ),
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
        components.append(
            PromptComponent(
                name="complete_enabled_tool_registry",
                category="tool_descriptions",
                text=f"Available tools for this call:\n{self.render_tool_catalog(tools) or '(none)'}\n\n",
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

    def build_summary_prompt(
        self,
        messages: list[Message],
        *,
        prompt_mode: str = "lean",
        maximum_preserve_recent_messages: int = 0,
    ) -> PromptAssembly:
        history_block = self.render_messages(messages)
        system_prompt = self._load_template(self._config.prompts.summary_system_template)
        user_text = self._load_template(self._config.prompts.summary_template).format(
            history_block=history_block,
            maximum_preserve_recent_messages=max(0, int(maximum_preserve_recent_messages)),
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
