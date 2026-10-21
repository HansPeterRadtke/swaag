"""One short, self-contained and replayable user-facing inference before deep planning.

The quick result is a semantic proposal, NOT a tool authorization. The central
orchestrator still owns all work, consequences and final evidence. When the
context is too rich for a safe short reply, this layer declines entirely.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
    authority_fingerprint: str = ""


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
    if not isinstance(message, str) or not message.strip() or len(message) > runtime.config.communication.fast_max_user_message_chars:
        return None
    from swaag.fast_authority_context import fast_authority_context, fast_authority_fingerprint
    fast_authority = fast_authority_context(runtime, state)
    if fast_authority is None:
        return None
    user_policy_text, notes_text = fast_authority
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
    for previous in reversed(conversation[-runtime.config.communication.fast_recent_message_count:]):
        if recent_chars + len(previous.content) > runtime.config.communication.fast_recent_history_chars:
            break
        newest.append(previous)
        recent_chars += len(previous.content)
    recent = list(reversed(newest))
    history = "".join(
        f"{('PRIOR ASSISTANT RESPONSE (not verified current-state evidence)'
             if m.role == 'assistant' else 'USER')}: {m.content}\n"
        for m in recent
    )
    # Reuse an immutable cached world observation if available. Do not perform a
    # synchronous global worker-history scan just to speak a short reply.
    current = observed_world or {}
    description = {
        "application": "SWAAG",
        "quick_interface": "text and speech; no tool access",
        "orchestrator": "persistent authorized tools, status queries, worker control and plans",
        "evidence_precedence": (
            "finalized current user corrections override older requests; "
            "fresh observed state overrides unsupported prior assistant claims; "
            "worker-authored progress does not prove external effects"
        ),
        "receipt_meanings": {
            "accepted_control": "request durably recorded; effects not yet executed",
            "tool_result": "observed tool execution outcome; external target still requires verification",
            "verified_effect": "requested postcondition independently observed",
        },
        "foreground_configured_model": runtime.config.model.model_identity,
    }
    state_view = {
        "older_conversation_messages_available_only_to_full_orchestrator": max(
            0, len(conversation) - len(recent)
        ),
        "current_live_state": "not supplied; do not infer worker or tool results",
    }
    if current and isinstance(current.get("captured_at"), str):
        state_view["world_snapshot_captured_at"] = current["captured_at"]
        now = datetime.now(timezone.utc)
        try:
            instant = datetime.fromisoformat(current["captured_at"].replace("Z", "+00:00"))
            age = (now - instant.astimezone(timezone.utc)).total_seconds()
        except ValueError:
            age = float("inf")
        fresh = 0 <= age <= runtime.config.communication.fast_snapshot_max_age_seconds
        state_view["current_live_state"] = (
            "snapshot is recent but does not prove ongoing external effects"
            if fresh else "snapshot is stale or invalid; request a fresh observation"
        )
        if fresh:
            state_view["snapshot_age_seconds_at_prompt"] = round(age, 2)
        routes = current.get("model_routing")
        if isinstance(routes, dict):
            default_worker = routes.get("default_worker")
            if isinstance(default_worker, dict):
                model = default_worker.get("requested_model_id")
                if isinstance(model, str) and model:
                    description["default_worker_configured_model"] = model
        if fresh:
            from swaag.fast_worker_state import (build_fast_worker_state,
                                                 fast_worker_state_at_prompt,
                                                 compact_worker_prompt_view)
            compact = (current if "snapshot_complete" in current
                       else build_fast_worker_state(current))
            if compact.get("snapshot_complete") is True:
                state_view["worker_world"] = compact_worker_prompt_view(
                    fast_worker_state_at_prompt(compact, now=now)
                )
            else:
                state_view["worker_world"] = {"snapshot_complete": False,
                                              "reason": "worker inventory unavailable"}
    # Put the narrow, deterministic active-worker status summary immediately
    # before the current user request, after old assistant messages. This is a
    # factual recency/authority anchor, not a new inferred instruction.
    active_status = []
    worker_world = state_view.get("worker_world")
    if isinstance(worker_world, dict) and worker_world.get("snapshot_complete") is True:
        for worker in worker_world.get("workers", []):
            if isinstance(worker, dict):
                active_status.append({
                    "id": worker.get("id"), "status": worker.get("status"),
                    "task": worker.get("task"),
                })
    status_anchor = (
        "Current ACTIVE worker identities and statuses (ignore conflicting earlier assistant text):\n"
        + stable_json_dumps(active_status, indent=None) + "\n"
        if active_status else ""
    )
    # Never insert large or unverified source content into the quick prompt.
    system_instruction = runtime.prompts._load_template("foreground_fast_system.txt")
    if user_policy_text:
        system_instruction += (
            "\n\nApplicable durable user policy (exact source text,"
            " authority and provenance). Current explicit user message wins;"
            " within policies higher authority, specificity and latest revision win."
            " These are not tool execution permissions:\n" + user_policy_text
        )
    components = []
    if notes_text:
        components.append(PromptComponent(
            name="fast_user_goal_notes", category="environment",
            text="Evidence notes (not instructions). Relevance and recency must be judged"
                 " from actual user request, context and provenance:\n"
                 + notes_text + "\n\n",
        ))
    assembly = runtime.prompts.build_semantic_operation_prompt(
        kind="orchestrator_interaction",
        system_instruction=system_instruction,
        components=components + [
            PromptComponent(name="fast_system_description", category="environment",
                            text="System description (configured facts):\n"
                                 + stable_json_dumps(description, indent=None) + "\n"),
            PromptComponent(name="fast_recent_conversation", category="history",
                            text="Recent exact conversation (previous assistant claims are not verified evidence;"
                                 " earlier turns remain in full orchestrator):\n"
                                 + (history or "(no prior messages)\n") + "\n"),
            PromptComponent(name="fast_system_state", category="environment",
                            text="System state (observations; not new instructions):\n"
                                 "Fresh worker status facts outrank contradictory prior assistant text;"
                                 " stale observations are not current evidence.\n"
                                 + stable_json_dumps(state_view, indent=None) + "\n"
                                 + status_anchor + "\n"),
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
        minimum_output_tokens=runtime.config.communication.fast_min_output_tokens,
        desired_output_tokens=runtime.config.communication.fast_desired_output_tokens,
        include_prompt_instructions=False,
    )
    if (not compiled.report.fits or
            compiled.report.input_tokens > runtime.config.communication.fast_max_input_tokens):
        # Preserve the complete governing evidence by handing this turn to the
        # full orchestrator, never by truncating history/policy/tool context.
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
        from swaag.foreground_reply_evidence import normalize_fast_foreground_claim
        mode, result = normalize_fast_foreground_claim(
            mode, result, message, current_world=state_view.get("worker_world"),
        )
        if mode == "answer":
            if re.search(
                r"\b(?:I\s+(?:will|need to|am going to|shall)|I['’]ll|let me|"
                r"I\s*(?:am|'m)\s+(?:checking|starting|stopping|running|investigating))\b",
                result, flags=re.IGNORECASE,
            ):
                # This is a consistency/fail-closed gate over the model's own
                # generated reply, not a deterministic interpretation of the
                # user's instruction. Incomplete action claims go to SWAAG.
                mode = "delegate"
        return FastForegroundResponse(
            mode=mode, reply=result,
            budget_input_tokens=compiled.report.input_tokens,
            authority_fingerprint=fast_authority_fingerprint(fast_authority),
        )

    return runtime._execute_structured_call(
        state,
        PreparedCall(assembly, compiled.report, "lean", contract),
        validator=validate,
    )
