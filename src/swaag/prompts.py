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

    def render_tool_catalog(self, tools: Iterable[tuple], *, prompt_mode: str) -> str:
        del prompt_mode
        lines: list[str] = []
        for item in tools:
            name = str(item[0])
            description = str(item[1])
            schema = item[2]
            usage_guidance = str(item[3]) if len(item) > 3 else ""
            lines.append(f"- {name}")
            lines.append(f"  description: {description}")
            lines.append(f"  input_schema: {stable_json_dumps(schema)}")
            if usage_guidance.strip():
                lines.append(f"  usage_guidance: {usage_guidance.strip()}")
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
        tools: Iterable[tuple],
        *,
        prompt_mode: str,
        notes_block: str = "",
        context_components: list[PromptComponent] | None = None,
    ) -> PromptAssembly:
        history, current_user, turn_context = self.partition_turn(messages)
        history_block = self.render_messages(history)
        current_user_block = current_user.content if current_user else ""
        turn_context_block = self.render_messages(turn_context)
        tool_catalog = self.render_tool_catalog(tools, prompt_mode=prompt_mode)
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
        tool_spec: tuple,
        prompt_mode: str,
        context_components: list[PromptComponent] | None = None,
    ) -> PromptAssembly:
        tool_name = str(tool_spec[0])
        tool_description = str(tool_spec[1])
        tool_schema = tool_spec[2]
        usage_guidance = str(tool_spec[3]) if len(tool_spec) > 3 else ""
        history, current_user, turn_context = self.partition_turn(messages)
        history_block = self.render_messages(history)
        current_user_block = current_user.content if current_user else ""
        turn_context_block = self.render_messages(turn_context)
        template = self._load_template("tool_input_user.txt").format(tool_name=tool_name)
        user_components = [
            PromptComponent(name="history", category="history", text=f"Conversation history:\n{history_block}\n\n"),
            PromptComponent(name="current_user_turn", category="current_user", text=f"Current user request:\n{current_user_block}\n\n"),
            PromptComponent(name="current_turn_context", category="turn_context", text=f"Current-turn tool context:\n{turn_context_block}\n\n"),
            PromptComponent(
                name="selected_tool_documentation",
                category="tool_documentation",
                text=(
                    f"Selected tool:\n"
                    f"name: {tool_name}\n"
                    f"description: {tool_description}\n"
                    f"input_schema: {stable_json_dumps(tool_schema)}\n"
                    f"usage_guidance: {usage_guidance.strip() or '(none)'}\n\n"
                ),
            ),
        ]
        if context_components:
            user_components.extend(context_components)
        user_components.append(PromptComponent(name="tool_input_instruction", category="instruction", text=template))
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
        tools: Iterable[tuple] = (),
        context_components: list[PromptComponent] | None = None,
        previous_rejected_decision: str = "",
        semantic_review_feedback: str = "",
    ) -> PromptAssembly:
        tool_catalog = self.render_tool_catalog(tools, prompt_mode=prompt_mode)
        user_components = [
            PromptComponent(name="current_user_turn", category="current_user", text=f"Current user request:\n{user_text}\n\n"),
            PromptComponent(name="analysis", category="analysis", text=f"Prompt analysis:\n{analysis_json}\n\n"),
        ]
        if context_components:
            user_components.extend(context_components)
        if tool_catalog:
            user_components.append(
                PromptComponent(
                    name="tool_descriptions",
                    category="tool_descriptions",
                    text=f"Available tools:\n{tool_catalog}\n\n",
                )
            )
        user_components.append(
            PromptComponent(
                name="task_decision_instruction",
                category="instruction",
                text=self._load_template(self._config.prompts.task_decision_template),
            )
        )
        if previous_rejected_decision:
            user_components.append(
                PromptComponent(
                    name="previous_rejected_decision",
                    category="turn_context",
                    text=f"\nPrevious rejected task decision JSON:\n{previous_rejected_decision}\n",
                )
            )
        if semantic_review_feedback:
            user_components.append(
                PromptComponent(
                    name="task_decision_semantic_review_feedback",
                    category="instruction",
                    text=(
                        "\nTask-decision correction requirements from all previous attempts:\n"
                        f"{semantic_review_feedback}\n\n"
                        "Return one corrected decision now. The correction requirements above override the rejected fields. "
                        "Keep already-valid fields, but change every field named by the accumulated feedback.\n"
                    ),
                )
            )
        return self._assemble("task_decision", prompt_mode, user_components)

    def build_task_decision_semantic_review_prompt(
        self,
        *,
        user_text: str,
        analysis_json: str,
        decision_json: str,
        tools: Iterable[tuple],
        prompt_mode: str,
        context_components: list[PromptComponent] | None = None,
    ) -> PromptAssembly:
        tool_catalog = self.render_tool_catalog(tools, prompt_mode=prompt_mode)
        components = [
            PromptComponent(
                name="current_user_turn",
                category="current_user",
                text=f"Current user request:\n{user_text}\n\n",
            ),
            PromptComponent(
                name="analysis",
                category="analysis",
                text=f"Prompt analysis:\n{analysis_json}\n\n",
            ),
            PromptComponent(
                name="candidate_task_decision",
                category="decision",
                text=f"Candidate task decision:\n{decision_json}\n\n",
            ),
        ]
        if context_components:
            components.extend(context_components)
        components.append(
            PromptComponent(
                name="complete_enabled_tool_registry",
                category="tool_descriptions",
                text=f"Complete enabled tool registry:\n{tool_catalog or '(none)'}\n\n",
            )
        )
        components.append(
            PromptComponent(
                name="task_decision_semantic_review_instruction",
                category="instruction",
                text=(
                    "Audit the candidate decision before runtime acts. Return only the strict review object.\n"
                    "List every distinct evidence source explicitly required by the user request, such as each named file, URL, "
                    "artifact, test target, or external state source. Do not merge distinct sources unless the selected tool's "
                    "published input schema can consume them in one call. A tool with one scalar path can cover only one named file "
                    "per call; an array field may cover multiple sources in one call.\n"
                    "minimum_evidence_call_count is the smallest number of calls needed under the candidate execution mode and "
                    "preferred tool, using only capabilities explicitly present in the complete registry.\n"
                    "selected_mode_and_tool_can_cover_declared_count is true only when the candidate mode, preferred tool, evidence "
                    "flag, and declared count can actually cover every listed source.\n"
                    "decision_matches_request is false when any explicit instruction is dropped. decision_is_internally_consistent "
                    "is false when the reason, mode, preferred tool, evidence flag, or count disagree. Put actionable correction "
                    "details in feedback.\n"
                ),
            )
        )
        return self._assemble("verification", prompt_mode, components)


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
        tools: Iterable[tuple],
        replan_reason: str = "",
        previous_rejected_plan: str = "",
        replan_attempt: int = 0,
        max_replans: int = 0,
    ) -> PromptAssembly:
        tool_catalog = self.render_tool_catalog(tools, prompt_mode=prompt_mode)
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
        if previous_rejected_plan:
            user_components.append(
                PromptComponent(
                    name="previous_rejected_plan",
                    category="turn_context",
                    text=(
                        "Previous rejected plan JSON:\n"
                        f"{previous_rejected_plan}\n\n"
                        "Correct this model-authored plan rather than regenerating unrelated fields. Apply the smallest "
                        "change required by the validation evidence. Preserve valid step IDs, tools, dependencies, dataflow, "
                        "verification checks, and objective criteria unless the reported validation error requires changing them.\n\n"
                    ),
                )
            )
        if tool_catalog:
            user_components.append(PromptComponent(name="tool_descriptions", category="tool_descriptions", text=f"Available tools:\n{tool_catalog}\n\n"))
        user_components.append(
            PromptComponent(
                name="planning_instruction",
                category="instruction",
                text=self._load_template(self._config.prompts.planning_template).format(
                    max_plan_steps=self._config.planner.max_plan_steps
                ),
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
        criteria: list[dict[str, object]],
        evidence: dict[str, object],
        prompt_mode: str,
        allowed_candidate_excerpts: list[str] | None = None,
        context_components: list[PromptComponent] | None = None,
        previous_rejected_verification: str = "",
        verification_feedback: str = "",
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
            PromptComponent(
                name="verification_allowed_candidate_excerpts",
                category="instruction",
                text=(
                    "Allowed candidate excerpts:\n"
                    f"{stable_json_dumps(allowed_candidate_excerpts or [])}\n\n"
                ),
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
        if previous_rejected_verification:
            user_components.append(
                PromptComponent(
                    name="previous_rejected_verification",
                    category="turn_context",
                    text=f"\nPrevious rejected verification JSON:\n{previous_rejected_verification}\n",
                )
            )
        if verification_feedback:
            user_components.append(
                PromptComponent(
                    name="verification_correction_feedback",
                    category="instruction",
                    text=(
                        "\nVerification protocol correction requirements from all previous attempts:\n"
                        f"{verification_feedback}\n\n"
                        "Return one corrected verification object now. Preserve valid judgments, but correct every field "
                        "named by the accumulated protocol feedback.\n"
                    ),
                )
            )
        return self._assemble("verification", prompt_mode, user_components)
