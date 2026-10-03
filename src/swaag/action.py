from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


class ActionValidationError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class AgentToolCall:
    tool_name: str
    arguments: dict[str, Any]


@dataclass(slots=True, frozen=True)
class AgentQuestion:
    question: str
    criticality: str
    reason: str
    assumption_if_unanswered: str
    importance: str = "normal"


@dataclass(slots=True, frozen=True)
class AgentStatus:
    situation: str
    action: str
    reason: str
    importance: str

    @property
    def importance_rank(self) -> int:
        return {"minor": 1, "normal": 2, "major": 3, "critical": 4}[self.importance]


@dataclass(slots=True, frozen=True)
class AgentResponseConstraints:
    exact_word_count: int | None = None


@dataclass(slots=True, frozen=True)
class AgentAction:
    assistant_message: str
    tool_calls: list[AgentToolCall]
    continue_loop: bool
    silent_completion: bool
    status: AgentStatus
    questions: list[AgentQuestion]
    response_constraints: AgentResponseConstraints = field(
        default_factory=AgentResponseConstraints
    )

    @property
    def calls_tools(self) -> bool:
        return bool(self.tool_calls)


def action_from_payload(payload: dict[str, Any], *, enabled_tool_names: Iterable[str]) -> AgentAction:
    if not isinstance(payload, dict):
        raise ActionValidationError("Agent action payload must be an object")

    assistant_message = payload.get("assistant_message")
    tool_calls_payload = payload.get("tool_calls")
    continue_loop = payload.get("continue_loop")
    silent_completion = payload.get("silent_completion", False)
    status_payload = payload.get("status")
    questions_payload = payload.get("questions", [])
    response_constraints_payload = payload.get(
        "response_constraints", {"exact_word_count": None}
    )
    if status_payload is None:
        # Backward compatibility for pre-status stored actions and test fixtures.
        status_payload = {"situation": "", "action": "", "reason": "", "importance": "normal"}

    if not isinstance(assistant_message, str):
        raise ActionValidationError("assistant_message must be a string")
    if not isinstance(tool_calls_payload, list):
        raise ActionValidationError("tool_calls must be an array")
    if not isinstance(continue_loop, bool):
        raise ActionValidationError("continue_loop must be a boolean")
    if not isinstance(silent_completion, bool):
        raise ActionValidationError("silent_completion must be a boolean")
    if not isinstance(status_payload, dict):
        raise ActionValidationError("status must be an object")
    required_status = {"situation", "action", "reason", "importance"}
    if set(status_payload) != required_status:
        raise ActionValidationError("status must contain exactly situation, action, reason, importance")
    for key in ("situation", "action", "reason"):
        if not isinstance(status_payload.get(key), str):
            raise ActionValidationError(f"status.{key} must be a string")
    importance = status_payload.get("importance")
    if importance not in {"minor", "normal", "major", "critical"}:
        raise ActionValidationError("status.importance must be one of minor, normal, major, critical")
    status = AgentStatus(
        situation=status_payload["situation"],
        action=status_payload["action"],
        reason=status_payload["reason"],
        importance=importance,
    )

    if not isinstance(response_constraints_payload, dict):
        raise ActionValidationError("response_constraints must be an object")
    if set(response_constraints_payload) != {"exact_word_count"}:
        raise ActionValidationError(
            "response_constraints must contain exactly exact_word_count"
        )
    exact_word_count = response_constraints_payload.get("exact_word_count")
    if exact_word_count is not None and (
        isinstance(exact_word_count, bool)
        or not isinstance(exact_word_count, int)
        or exact_word_count <= 0
    ):
        raise ActionValidationError(
            "response_constraints.exact_word_count must be a positive integer or null"
        )
    response_constraints = AgentResponseConstraints(
        exact_word_count=exact_word_count
    )

    if not isinstance(questions_payload, list):
        raise ActionValidationError("questions must be an array")
    questions: list[AgentQuestion] = []
    for index, item in enumerate(questions_payload):
        if not isinstance(item, dict):
            raise ActionValidationError(f"questions[{index}] must be an object")
        required_question = {"question", "criticality", "reason", "assumption_if_unanswered"}
        if set(item) not in (required_question, required_question | {"importance"}):
            raise ActionValidationError(f"questions[{index}] must contain exactly question, criticality, reason, assumption_if_unanswered")
        if item.get("criticality") not in {"optional", "blocking"}:
            raise ActionValidationError(f"questions[{index}].criticality must be optional or blocking")
        for key in ("question", "reason", "assumption_if_unanswered"):
            if not isinstance(item.get(key), str):
                raise ActionValidationError(f"questions[{index}].{key} must be a string")
        question = item["question"].strip()
        reason = item["reason"].strip()
        assumption = item["assumption_if_unanswered"].strip()
        criticality = item["criticality"]
        if not question:
            raise ActionValidationError(f"questions[{index}].question must not be empty")
        if not reason:
            raise ActionValidationError(f"questions[{index}].reason must not be empty")
        if criticality == "optional" and not assumption:
            raise ActionValidationError(
                f"questions[{index}].assumption_if_unanswered must state the provisional assumption for an optional question"
            )
        if criticality == "blocking" and assumption:
            raise ActionValidationError(
                f"questions[{index}].assumption_if_unanswered must be empty for a blocking question"
            )
        importance = item.get("importance", "critical" if criticality == "blocking" else "normal")
        if importance not in {"minor", "normal", "major", "critical"}:
            raise ActionValidationError("question importance is invalid")
        questions.append(
            AgentQuestion(
                question=question,
                criticality=criticality,
                reason=reason,
                assumption_if_unanswered=assumption,
                importance=importance,
            )
        )

    enabled = set(enabled_tool_names)
    tool_calls: list[AgentToolCall] = []
    for index, item in enumerate(tool_calls_payload):
        if not isinstance(item, dict):
            raise ActionValidationError(f"tool_calls[{index}] must be an object")
        tool_name = item.get("tool_name")
        arguments = item.get("arguments")
        if not isinstance(tool_name, str) or tool_name not in enabled:
            raise ActionValidationError(f"tool_calls[{index}].tool_name is not enabled: {tool_name!r}")
        if not isinstance(arguments, dict):
            raise ActionValidationError(f"tool_calls[{index}].arguments must be an object")
        tool_calls.append(AgentToolCall(tool_name=tool_name, arguments=arguments))

    if any(question.criticality == "blocking" for question in questions) and tool_calls:
        raise ActionValidationError(
            "blocking questions cannot accompany tool calls in the same action; gather safe evidence first, then emit the blocking question as the next step"
        )

    for index, question in enumerate(questions):
        if question.question not in assistant_message:
            raise ActionValidationError(
                f"assistant_message must disclose questions[{index}].question verbatim"
            )
        if (
            question.criticality == "optional"
            and question.assumption_if_unanswered not in assistant_message
        ):
            raise ActionValidationError(
                f"assistant_message must disclose questions[{index}].assumption_if_unanswered verbatim"
            )

    if response_constraints.exact_word_count is not None:
        actual_word_count = len(assistant_message.split())
        if actual_word_count != response_constraints.exact_word_count:
            raise ActionValidationError(
                "assistant_message has "
                f"{actual_word_count} whitespace-delimited words but "
                "response_constraints.exact_word_count requires "
                f"{response_constraints.exact_word_count}"
            )

    if tool_calls and not continue_loop:
        raise ActionValidationError(
            "continue_loop must be true when tool_calls are present so their exact results are observed"
        )
    if continue_loop and not tool_calls:
        raise ActionValidationError(
            "continue_loop=true requires at least one tool call; use a wait tool when waiting is required"
        )
    if silent_completion and (continue_loop or tool_calls):
        raise ActionValidationError("silent_completion=true is valid only for a terminal action with no tool calls")
    if not continue_loop and not assistant_message.strip() and not silent_completion:
        raise ActionValidationError(
            "Terminal actions require a non-empty assistant_message unless silent_completion=true because the user/protocol explicitly requested no user-facing response"
        )
    return AgentAction(
        assistant_message=assistant_message,
        tool_calls=tool_calls,
        continue_loop=continue_loop,
        silent_completion=silent_completion,
        status=status,
        questions=questions,
        response_constraints=response_constraints,
    )
