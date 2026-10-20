"""One short, self-contained and replayable user-facing inference before deep planning.

The quick result is a semantic proposal, NOT a tool authorization. The central
orchestrator still owns all work, consequences and final evidence. When the
context is too rich for a safe short reply, this layer declines entirely.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from swaag.runtime import AgentRuntime, PreparedCall
from swaag.types import ContractSpec, PromptComponent, SessionState
from swaag.utils import stable_json_dumps


@dataclass(frozen=True, slots=True)
class FastForegroundResponse:
    mode: str
    reply: str
    budget_input_tokens: int


def generate_fast_foreground_response(
    runtime: AgentRuntime,
    state: SessionState,
    message: str,
    *,
    observed_world: dict[str, Any] | None = None,
) -> FastForegroundResponse | None:
    """One LLM call or an explicit refusal to attempt an incomplete context.

    The present user message is never shortened. Existing non-core durable
    instructions belong to the full controller rather than an unsafe fast
    summary. The recent message allowance is a quick-stage admission boundary,
    not loss of canonical history: beyond it, return None and run the full path.
    """
    if not isinstance(message, str) or not message.strip() or len(message) > 2500:
        return None
    external_instructions = [item for item in state.prompt_instructions
                             if item.source_kind != "swaag_core"]
    # This stage deliberately does not compress durable user policy, relevant
    # remembered facts, unread attachments, or unresolved user decisions.
    # Those require the original full authority/context constructor.
    if (external_instructions or state.notes or state.open_questions
            or state.deferred_tasks or state.attachments):
        return None
    for contributor in runtime.system_prompt_contributors:
        shared_store = getattr(contributor, "store", None)
        if shared_store is not None and shared_store.has_any_events():
            # Even a learned/default user-wide instruction might matter.
            # Never drop its authority merely to shave prompt tokens.
            return None
    conversation = [m for m in state.messages
                    if m.role in {"user", "assistant"}
                    and not (isinstance(m.metadata, dict) and m.metadata.get("internal_action"))]
    # Take the most recent WHOLE messages that fit; never slice the source
    # halfway through a negation or correction to win a token budget. Older
    # history remains exact and available to the authoritative controller.
    # On a long-lived real session the latest six turns may exceed the budget;
    # admitting the newest two or three is much better than disabling all
    # fast responses for that entire session.
    newest = []
    recent_chars = 0
    for previous in reversed(conversation[-6:]):
        if recent_chars + len(previous.content) > 3500:
            break
        newest.append(previous)
        recent_chars += len(previous.content)
    recent = list(reversed(newest))
    history = "".join(f"{m.role.upper()}: {m.content}\n" for m in recent)
    # Reuse an immutable cached world observation if available. Do not perform a
    # synchronous global worker-history scan just to speak a short reply.
    current = observed_world or {}
    summary = {
        "assistant_role": "SWAAG user-facing orchestrator",
        "foreground_requested_model": runtime.config.model.model_identity,
    }
    summary["older_conversation_messages_available_only_to_full_orchestrator"] = max(
        0, len(conversation) - len(recent)
    )
    if current and isinstance(current.get("captured_at"), str):
        summary["world_snapshot_captured_at"] = current["captured_at"]
        for label in ("active_workers", "active_plans"):
            items = current.get(label)
            if isinstance(items, list):
                summary[label + "_count_at_capture"] = len(items)
    # Never insert large or unverified source content into the quick prompt.
    assembly = runtime.prompts.build_semantic_operation_prompt(
        kind="orchestrator_interaction",
        system_instruction=runtime.prompts._load_template("foreground_fast_system.txt"),
        components=[
            PromptComponent(name="fast_trusted_background", category="environment",
                            text="Trusted runtime background (captured facts, not evidence of an effect):\n"
                                 + stable_json_dumps(summary, indent=None) + "\n\n"),
            PromptComponent(name="fast_recent_conversation", category="history",
                            text="Recent exact conversation (earlier turns still exist in full orchestrator; delegate if needed):\n"
                                 + (history or "(no prior messages)\n") + "\n"),
            PromptComponent(name="fast_authoritative_user", category="current_user",
                            text="Final authoritative current user message, unchanged:\n"
                                 + message + "\n"),
        ],
        template_names=("foreground_fast_system.txt",),
    )
    contract = ContractSpec(name="fast_foreground_response", mode="json_schema", json_schema={
        "type": "object", "additionalProperties": False,
        "properties": {
            "mode": {"type": "string", "enum": ["answer", "delegate"]},
            "reply": {"type": "string"},
        },
        "required": ["mode", "reply"],
    })
    compiled = runtime._compile_context(
        state, assembly, contract,
        minimum_output_tokens=64, desired_output_tokens=144,
        include_prompt_instructions=False,
    )
    if not compiled.report.fits:
        return None
    runtime._record_prompt_built(state, assembly, contract, compiled.report)

    def validate(payload: dict) -> FastForegroundResponse:
        if (set(payload) != {"mode", "reply"}
                or payload.get("mode") not in {"answer", "delegate"}
                or not isinstance(payload.get("reply"), str)
                or not payload["reply"].strip()):
            raise ValueError("Fast foreground needs mode=answer|delegate and a nonempty reply")
        result = payload["reply"].strip()
        mode = payload["mode"]
        if mode == "answer":
            if re.search(
                r"\b(?:I\s+(?:will|need to|am going to|shall)|I['’]ll|let me|"
                r"checking|starting|stopping|running|investigating)\b",
                result, flags=re.IGNORECASE,
            ):
                # This is a consistency/fail-closed gate over the model's own
                # generated reply, not a deterministic interpretation of the
                # user's instruction. Incomplete action claims go to SWAAG.
                mode = "delegate"
        return FastForegroundResponse(
            mode=mode, reply=result,
            budget_input_tokens=compiled.report.input_tokens,
        )

    return runtime._execute_structured_call(
        state,
        PreparedCall(assembly, compiled.report, "lean", contract),
        validator=validate,
    )
