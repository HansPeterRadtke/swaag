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

    def render_tool_catalog(self, tools: Iterable[tuple[str, str, dict]], *, prompt_mode: str) -> str:
        lines: list[str] = []
        for name, description, schema in tools:
            if prompt_mode == "lean":
                properties = sorted((schema.get("properties") or {}).keys())
                arg_text = ", ".join(properties) if properties else "no arguments"
                lines.append(f"- {name}: {arg_text}")
            else:
                lines.append(f"- {name}: {description}")
                lines.append(f"  schema={stable_json_dumps(schema)}")
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
        current_user_index = None
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "user":
                current_user_index = index
                break
        if current_user_index is None:
            return messages, None, []
        return messages[:current_user_index], messages[current_user_index], messages[current_user_index + 1 :]

    def _assemble(self, kind: ModelCallKind, prompt_mode: str, user_components: list[PromptComponent]) -> PromptAssembly:
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

    def build_decision_prompt(
        self,
        messages: list[Message],
        tools: Iterable[tuple[str, str, dict]],
        *,
        prompt_mode: str,
        notes_block: str = "",
        context_components: list[PromptComponent] | None = None,
    ) -> PromptAssembly:
        history, current_user, turn_context = self.partition_turn(messages)
        history_block = self.render_messages(history)
        current_user_block = current_user.content if current_user else ""
        turn_context_block = self.render_messages(turn_context)
        # Decision-time tool exposure should stay narrow and metadata-first.
        # Full schemas are reserved for the later tool_input call.
        tool_catalog = self.render_tool_catalog(tools, prompt_mode="lean")
        user_text = self._load_template(self._config.prompts.decision_template)
        user_components = [
            PromptComponent(name="history", category="history", text=f"Conversation history:\n{history_block}\n\n"),
            PromptComponent(name="current_user_turn", category="current_user", text=f"Current user request:\n{current_user_block}\n\n"),
            PromptComponent(name="current_turn_context", category="turn_context", text=f"Current-turn tool context:\n{turn_context_block}\n\n"),
        ]
        if context_components:
            user_components.extend(context_components)
        if notes_block:
            user_components.append(PromptComponent(name="notes", category="notes", text=f"Working notes:\n{notes_block}\n\n"))
        if tool_catalog:
            user_components.append(PromptComponent(name="tool_descriptions", category="tool_descriptions", text=f"Available tools:\n{tool_catalog}\n\n"))
        user_components.append(PromptComponent(name="decision_instruction", category="instruction", text=user_text))
        return self._assemble("decision", prompt_mode, user_components)

    def build_tool_input_prompt(
        self,
        messages: list[Message],
        *,
        tool_name: str,
        prompt_mode: str,
        context_components: list[PromptComponent] | None = None,
    ) -> PromptAssembly:
        history, current_user, turn_context = self.partition_turn(messages)
        history_block = self.render_messages(history)
        current_user_block = current_user.content if current_user else ""
        turn_context_block = self.render_messages(turn_context)
        template = self._load_template("tool_input_user.txt").format(tool_name=tool_name)
        extra_instruction = ""
        if tool_name == "shell_command":
            extra_instruction = (
                "Return a single non-interactive shell command.\n"
                "Do not return `bash`, `sh`, `python`, or `python3` by themselves.\n"
                "The command must perform the current step and print concise stdout evidence.\n"
                "The command must be directly executable now with no manual substitution.\n"
                "Do not use placeholders such as `<file>`, `<patch_file>`, `/path/to/...`, `...`, or TODO markers.\n"
                "When editing code, prefer a concrete `python3 - <<'PY'` script that opens a real repo-relative path and writes the change.\n"
                "If you use a heredoc, put the script body on following lines and end with `PY` on its own line.\n"
            )
        elif tool_name == "run_tests":
            extra_instruction = (
                "Return a single non-interactive test command.\n"
                "Prefer the narrowest relevant test invocation.\n"
            )
        elif tool_name == "edit_text":
            extra_instruction = (
                "\n"
                "Return arguments for one concrete source-file edit.\n"
                "Set `path` to a real file path, never `.` or a directory.\n"
                "Set `operation` to one of: replace_pattern_once, replace_pattern_all, replace_range, insert_at, delete_range.\n"
                "For ordinary line replacements, prefer `replace_pattern_once`.\n"
                "Base `pattern` and `replacement` on the actual source preview in the context, not on the issue text.\n"
                "For `replace_pattern_once` or `replace_pattern_all`, include both `pattern` and `replacement`.\n"
                "If one nearby source line can anchor the fix, replace that short anchor and insert the new code around it instead of replacing a large block.\n"
                "When adding one missing mapping or handler, use one existing nearby entry line as the full `pattern`, and set `replacement` to that same line plus the new adjacent line.\n"
                "If the preview shows a mapping table or dispatch table, patch that table directly instead of unrelated fallback return code.\n"
                "Prefer the smallest exact source change that fixes the bug.\n"
            )
        elif tool_name == "write_file":
            extra_instruction = (
                "\n"
                "Return arguments for one concrete file write.\n"
                "Required fields: path (string), content (string), create (boolean).\n"
                "Set `path` to the exact repo-relative or absolute file path to write.\n"
                "Set `content` to the complete final file contents — not a diff, not a patch, not a summary.\n"
                "Set `create` to true only when the file should be created if it does not already exist.\n"
                "Use write_file only when replacing the entire file is the correct action.\n"
                "If only a portion of a file needs editing, use edit_text instead.\n"
            )
        user_components = [
            PromptComponent(name="history", category="history", text=f"Conversation history:\n{history_block}\n\n"),
            PromptComponent(name="current_user_turn", category="current_user", text=f"Current user request:\n{current_user_block}\n\n"),
            PromptComponent(name="current_turn_context", category="turn_context", text=f"Current-turn tool context:\n{turn_context_block}\n\n"),
        ]
        if context_components:
            user_components.extend(context_components)
        user_components.append(PromptComponent(name="tool_input_instruction", category="instruction", text=template))
        if extra_instruction:
            user_components.append(
                PromptComponent(name="tool_input_tool_specific_instruction", category="instruction", text=extra_instruction)
            )
        return self._assemble("tool_input", prompt_mode, user_components)

    def build_analysis_prompt(
        self,
        user_text: str,
        *,
        prompt_mode: str,
        context_components: list[PromptComponent] | None = None,
    ) -> PromptAssembly:
        user_components = [
            PromptComponent(name="current_user_turn", category="current_user", text=f"Current user request:\n{user_text}\n\n"),
        ]
        if context_components:
            user_components.extend(context_components)
        user_components.append(
            PromptComponent(
                name="analysis_instruction",
                category="instruction",
                text=self._load_template(self._config.prompts.analysis_template),
            )
        )
        return self._assemble("analysis", prompt_mode, user_components)

    def build_task_decision_prompt(
        self,
        user_text: str,
        analysis_json: str,
        *,
        prompt_mode: str,
        context_components: list[PromptComponent] | None = None,
    ) -> PromptAssembly:
        user_components = [
            PromptComponent(name="current_user_turn", category="current_user", text=f"Current user request:\n{user_text}\n\n"),
            PromptComponent(name="analysis", category="analysis", text=f"Prompt analysis:\n{analysis_json}\n\n"),
        ]
        if context_components:
            user_components.extend(context_components)
        user_components.append(
            PromptComponent(
                name="task_decision_instruction",
                category="instruction",
                text=self._load_template(self._config.prompts.task_decision_template),
            )
        )
        return self._assemble("task_decision", prompt_mode, user_components)

    def build_task_expansion_prompt(
        self,
        user_text: str,
        analysis_json: str,
        decision_json: str,
        *,
        prompt_mode: str,
        context_components: list[PromptComponent] | None = None,
    ) -> PromptAssembly:
        user_components = [
            PromptComponent(name="current_user_turn", category="current_user", text=f"Current user request:\n{user_text}\n\n"),
            PromptComponent(name="analysis", category="analysis", text=f"Prompt analysis:\n{analysis_json}\n\n"),
            PromptComponent(name="task_decision", category="decision", text=f"Task decision:\n{decision_json}\n\n"),
        ]
        if context_components:
            user_components.extend(context_components)
        user_components.append(
            PromptComponent(
                name="expansion_instruction",
                category="instruction",
                text=self._load_template(self._config.prompts.expansion_template),
            )
        )
        return self._assemble("expansion", prompt_mode, user_components)

    def build_active_session_control_prompt(
        self,
        *,
        session_goal: str,
        active_step: str,
        waiting_reason: str,
        queued_message: str,
        prompt_mode: str,
        context_components: list[PromptComponent] | None = None,
    ) -> PromptAssembly:
        user_components = [
            PromptComponent(name="session_goal", category="current_user", text=f"Current session goal:\n{session_goal or '(none)'}\n\n"),
            PromptComponent(name="active_step", category="turn_context", text=f"Current active step:\n{active_step or '(none)'}\n\n"),
            PromptComponent(name="waiting_reason", category="turn_context", text=f"Waiting state:\n{waiting_reason or '(not waiting)'}\n\n"),
            PromptComponent(name="queued_control_message", category="current_user", text=f"New control-plane message:\n{queued_message}\n\n"),
        ]
        if context_components:
            user_components.extend(context_components)
        user_components.append(
            PromptComponent(
                name="control_instruction",
                category="instruction",
                text=self._load_template(self._config.prompts.control_template),
            )
        )
        return self._assemble("control", prompt_mode, user_components)

    def build_answer_prompt(
        self,
        messages: list[Message],
        *,
        prompt_mode: str,
        notes_block: str = "",
        context_components: list[PromptComponent] | None = None,
    ) -> PromptAssembly:
        history, current_user, turn_context = self.partition_turn(messages)
        history_block = self.render_messages(history)
        current_user_block = current_user.content if current_user else ""
        turn_context_block = self.render_messages(turn_context)
        user_text = self._load_template(self._config.prompts.answer_template)
        user_components = [
            PromptComponent(name="history", category="history", text=f"Conversation history:\n{history_block}\n\n"),
            PromptComponent(name="current_user_turn", category="current_user", text=f"Current user request:\n{current_user_block}\n\n"),
            PromptComponent(name="current_turn_context", category="turn_context", text=f"Current-turn tool context:\n{turn_context_block}\n\n"),
        ]
        if context_components:
            user_components.extend(context_components)
        if notes_block:
            user_components.append(PromptComponent(name="notes", category="notes", text=f"Working notes:\n{notes_block}\n\n"))
        user_components.append(PromptComponent(name="answer_instruction", category="instruction", text=user_text))
        return self._assemble("answer", prompt_mode, user_components)

    def build_summary_prompt(self, messages: list[Message], *, prompt_mode: str = "lean") -> PromptAssembly:
        history_block = self.render_messages(messages)
        system_prompt = self._load_template(self._config.prompts.summary_system_template)
        user_text = self._load_template(self._config.prompts.summary_template).format(history_block=history_block)
        user_components = [
            PromptComponent(name="summary_history", category="history", text=user_text),
        ]
        components = [
            PromptComponent(name="llama3_begin", category="wrapper", text=LLAMA3_BEGIN),
            PromptComponent(name="system_header", category="wrapper", text=LLAMA3_SYSTEM_HEADER),
            PromptComponent(name="system_prompt", category="system_prompt", text=system_prompt),
            PromptComponent(name="system_eot", category="wrapper", text=LLAMA3_EOT),
            PromptComponent(name="user_header", category="wrapper", text=LLAMA3_USER_HEADER),
            *user_components,
            PromptComponent(name="user_eot", category="wrapper", text=LLAMA3_EOT),
            PromptComponent(name="assistant_header", category="wrapper", text=LLAMA3_ASSISTANT_HEADER),
        ]
        return PromptAssembly(
            kind="summary",
            prompt_mode=prompt_mode,
            prompt_text="".join(component.text for component in components),
            components=components,
        )

    def build_plan_prompt(
        self,
        goal: str,
        *,
        prompt_mode: str,
        context_components: list[PromptComponent],
        tools: Iterable[tuple[str, str, dict]],
        replan_reason: str = "",
        replan_attempt: int = 0,
        max_replans: int = 0,
    ) -> PromptAssembly:
        # Planning only needs tool names and argument surfaces, not full JSON
        # schemas. Keeping this lean avoids wasting context on tool details
        # before the runtime has selected a concrete tool.
        tool_catalog = self.render_tool_catalog(tools, prompt_mode="lean")
        user_components = [
            PromptComponent(name="planning_goal", category="current_user", text=f"Task goal:\n{goal}\n\n"),
            *context_components,
        ]
        if replan_reason:
            attempt_hint = (
                f" (attempt {replan_attempt} of {max_replans}; prefer a simpler approach if previous attempts failed)"
                if replan_attempt > 1 and max_replans > 0
                else ""
            )
            user_components.append(PromptComponent(name="replan_reason", category="instruction", text=f"Replan reason{attempt_hint}:\n{replan_reason}\n\n"))
        if tool_catalog:
            user_components.append(PromptComponent(name="tool_descriptions", category="tool_descriptions", text=f"Available tools:\n{tool_catalog}\n\n"))
        user_components.append(
            PromptComponent(
                name="planning_instruction",
                category="instruction",
                text=self._load_template(self._config.prompts.planning_template),
            )
        )
        return self._assemble("plan", prompt_mode, user_components)

    def build_verification_prompt(
        self,
        *,
        step_title: str,
        step_goal: str,
        expected_outputs: list[str],
        success_criteria: str,
        assistant_text: str,
        criteria: list[dict[str, str]],
        evidence: dict[str, object],
        prompt_mode: str,
        context_components: list[PromptComponent] | None = None,
    ) -> PromptAssembly:
        user_components = [
            PromptComponent(name="verification_step_title", category="current_user", text=f"Step title:\n{step_title}\n\n"),
            PromptComponent(name="verification_step_goal", category="current_user", text=f"Step goal:\n{step_goal}\n\n"),
            PromptComponent(
                name="verification_expected_outputs",
                category="instruction",
                text=f"Expected outputs:\n{stable_json_dumps(expected_outputs)}\n\n",
            ),
            PromptComponent(
                name="verification_success_criteria",
                category="instruction",
                text=f"Success criteria:\n{success_criteria}\n\n",
            ),
            PromptComponent(
                name="verification_assistant_text",
                category="turn_context",
                text=f"Candidate result:\n{assistant_text}\n\n",
            ),
            PromptComponent(
                name="verification_evidence",
                category="turn_context",
                text=f"Deterministic evidence:\n{stable_json_dumps(evidence)}\n\n",
            ),
            PromptComponent(
                name="verification_criteria",
                category="instruction",
                text=f"Criteria:\n{stable_json_dumps(criteria)}\n\n",
            ),
        ]
        if context_components:
            user_components.extend(context_components)
        user_components.append(
            PromptComponent(
                name="verification_instruction",
                category="instruction",
                text=self._load_template(self._config.prompts.verification_template),
            )
        )
        return self._assemble("verification", prompt_mode, user_components)
