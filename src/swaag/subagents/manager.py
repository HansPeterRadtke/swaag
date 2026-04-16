"""Subagent orchestration.

Subagents are spawned with isolated, narrowly-scoped context. The decision
of *whether* to spawn a subagent is semantic and is owned by the main LLM;
this module is the deterministic mechanism that performs the spawn,
collects evidence, validates the structural shape of the result, and hands
back a typed :class:`SubagentReport`.

There is **no** keyword vocabulary or substring matching in this module.
The reviewer subagent uses only:

* deterministic verification outcome (``verification.passed`` and the
  presence of structured evidence),
* structural step-kind expectations (a ``tool`` step must have produced a
  tool result; a ``respond`` step must have produced assistant text),
* exact-equality comparison against ``step.expected_output`` when (and
  only when) ``expected_output`` is a literal exact-match contract.

Any semantic equivalence judgement (e.g. "the model said something
synonymous with the expected answer") must be made elsewhere by the LLM.
"""

from __future__ import annotations

import copy

from swaag.retrieval.embeddings import build_backend
from swaag.strategy import StrategyValidationError, validate_plan_against_strategy
from swaag.context_builder import ContextBundle
from swaag.subagents.protocol import SubagentArtifact, SubagentReport
from swaag.subagents.specs import SubagentSpec, default_subagent_specs
from swaag.types import Plan, PlanStep, SessionState
from swaag.verification import VerificationOutcome


class SubagentManager:
    def __init__(
        self,
        *,
        backend_mode: str = "llm_scoring",
        base_url: str | None = None,
        seed: int = 11,
        connect_timeout_seconds: int = 10,
        read_timeout_seconds: int = 60,
    ) -> None:
        self._specs = default_subagent_specs()
        self._semantic_backend = build_backend(
            backend_mode,
            base_url=base_url,
            seed=seed,
            connect_timeout_seconds=connect_timeout_seconds,
            read_timeout_seconds=read_timeout_seconds,
        )

    def spec(self, subagent_type: str) -> SubagentSpec:
        return copy.deepcopy(self._specs[subagent_type])

    def scoped_state(self, state: SessionState) -> SessionState:
        return copy.deepcopy(state)

    def retrieve_context(self, state: SessionState, *, goal: str, bundle: ContextBundle) -> SubagentReport:
        spec = self.spec("retriever")
        scoped_state = self.scoped_state(state)
        scoped_query = "\n".join(
            part
            for part in [
                goal,
                scoped_state.active_role,
                scoped_state.working_memory.current_step_title,
                "\n".join(scoped_state.working_memory.active_entities),
            ]
            if part
        )
        focus_candidates: list[tuple[str, str, str]] = []
        for index, message in enumerate(bundle.history_messages):
            focus_candidates.append((f"history_message:message:{index}", "history_message", message.content))
        for index, item in enumerate(bundle.semantic_items):
            focus_candidates.append((f"semantic_memory:item:{index}", "semantic_memory", item.content))
        for relative_path, content in bundle.relevant_environment_files:
            focus_candidates.append((f"environment_file:{relative_path}", "environment_file", f"{relative_path}\n{content}"))
        if bundle.guidance_text.strip():
            focus_candidates.append(("guidance:active", "guidance", bundle.guidance_text))
        scores = (
            self._semantic_backend.score_query(scoped_query, [text for _, _, text in focus_candidates])
            if focus_candidates
            else []
        )
        ranked = sorted(
            [
                (score, item_id, item_type, text)
                for (item_id, item_type, text), score in zip(focus_candidates, scores, strict=True)
            ],
            key=lambda item: (-item[0], item[1]),
        )
        focused = ranked[: min(5, len(ranked))]
        focused_ids = [item_id for _score, item_id, _item_type, _text in focused]
        coverage = {
            "history": sum(1 for _score, _item_id, item_type, _text in focused if item_type.startswith("history")),
            "memory": sum(1 for _score, _item_id, item_type, _text in focused if item_type == "semantic_memory"),
            "guidance": sum(1 for _score, _item_id, item_type, _text in focused if item_type == "guidance"),
            "environment": sum(1 for _score, _item_id, item_type, _text in focused if item_type == "environment_file"),
        }
        focus_lines = []
        for score, item_id, item_type, text in focused:
            preview = text.replace("\n", " ")[:80].strip()
            focus_lines.append(f"- {item_type} {item_id} (semantic_focus={score:.3f}) {preview}")
        focus_summary = "\n".join(focus_lines)
        evidence = {
            "goal": goal,
            "active_role": scoped_state.active_role,
            "selected_item_count": len(focus_candidates),
            "focused_item_ids": focused_ids,
            "coverage": coverage,
            "retrieval_mode": bundle.retrieval_mode,
            "retrieval_degraded": bundle.retrieval_degraded,
            "scoped_query": scoped_query,
        }
        accepted = bool(focused_ids)
        reason = "retrieval_focus_ready" if accepted else "no_relevant_context_selected"
        return SubagentReport(
            spec=spec,
            accepted=accepted,
            reason=reason,
            evidence=evidence,
            recommended_action="continue" if accepted else "retrieve_more",
            artifacts=[
                SubagentArtifact(
                    artifact_type="retrieval_focus",
                    content={
                        "focus_summary": focus_summary,
                        "focused_item_ids": focused_ids,
                        "coverage": coverage,
                        "scoped_query": scoped_query,
                    },
                )
            ],
        )

    def review_plan(self, state: SessionState, plan: Plan) -> SubagentReport:
        spec = self.spec("planner")
        scoped_state = self.scoped_state(state)
        completed_step_kinds = (
            [step.kind for step in scoped_state.active_plan.steps if step.status == "completed"]
            if scoped_state.active_plan
            else []
        )
        evidence = {
            "step_count": len(plan.steps),
            "has_final_respond": bool(plan.steps and plan.steps[-1].kind == "respond"),
            "verification_complete": all(step.verification_checks and step.required_conditions for step in plan.steps),
            "completed_step_kinds": completed_step_kinds,
        }
        accepted = evidence["has_final_respond"] and evidence["verification_complete"]
        reason = "plan_review_passed" if accepted else "plan_missing_required_review_properties"
        if state.active_strategy is not None:
            try:
                validate_plan_against_strategy(plan, state.active_strategy, completed_step_kinds=completed_step_kinds)
            except StrategyValidationError as exc:
                accepted = False
                reason = str(exc)
                evidence["strategy_validation_error"] = str(exc)
        return SubagentReport(
            spec=spec,
            accepted=accepted,
            reason=reason,
            evidence=evidence,
            recommended_action="continue" if accepted else "replan",
            artifacts=[SubagentArtifact(artifact_type="plan_review", content=evidence)],
        )

    def replan(self, state: SessionState, *, goal: str, current_plan: Plan | None, failure_reason: str) -> SubagentReport:
        spec = self.spec("planner")
        scoped_state = self.scoped_state(state)
        pending_steps = []
        failed_steps = []
        if current_plan is not None:
            pending_steps = [step.step_id for step in current_plan.steps if step.status in {"pending", "running"}]
            failed_steps = [step.step_id for step in current_plan.steps if step.status == "failed"]
        evidence = {
            "goal": goal,
            "failure_reason": failure_reason,
            "pending_steps": pending_steps,
            "failed_steps": failed_steps,
            "active_role": scoped_state.active_role,
        }
        guidance = (
            "Rebuild the plan around the failed step, preserve already completed verified work, "
            "and tighten verification for the next attempt."
        )
        if failed_steps:
            guidance += f" Failed steps: {', '.join(failed_steps)}."
        if pending_steps:
            guidance += f" Remaining pending steps: {', '.join(pending_steps)}."
        return SubagentReport(
            spec=spec,
            accepted=True,
            reason="replan_requested",
            evidence=evidence,
            recommended_action="replan",
            artifacts=[SubagentArtifact(artifact_type="replan_request", content={**evidence, "replan_guidance": guidance})],
        )

    def review_result(
        self,
        state: SessionState,
        step: PlanStep,
        *,
        verification: VerificationOutcome,
        subsystem_result,
    ) -> SubagentReport:
        """Reviewer subagent.

        Acceptance is based ONLY on:

        * the deterministic verification outcome,
        * the *structural* shape that the step kind requires (tool steps
          must have a tool result; respond/reasoning steps must have
          assistant text),
        * an exact-equality check between assistant text and
          ``step.expected_output`` when expected_output is a non-empty
          literal — this is a deterministic exact-match contract, not a
          fuzzy comparison or fragment search.

        No keyword vocabulary, substring search, or "fragment hits" is
        used. Semantic equivalence judgements belong to the LLM.
        """

        spec = self.spec("reviewer")
        _ = self.scoped_state(state)
        latest_tool = subsystem_result.tool_results[-1] if subsystem_result.tool_results else None
        latest_output = latest_tool.output if latest_tool is not None else {}
        text_candidate = subsystem_result.assistant_text.strip()
        if not text_candidate and latest_tool is not None:
            text_candidate = str(
                latest_output.get("text")
                or latest_output.get("content")
                or latest_output.get("result")
                or ""
            ).strip()
        normalized_candidate = " ".join(text_candidate.split())
        literal_expected_output = (
            step.expected_output.strip() if isinstance(step.expected_output, str) else ""
        )
        normalized_literal = " ".join(literal_expected_output.split())
        literal_exact_match = bool(
            normalized_candidate and normalized_literal and normalized_candidate == normalized_literal
        )
        evidence = {
            "verification_passed": verification.passed,
            "verification_reason": verification.reason,
            "tool_result_count": len(subsystem_result.tool_results),
            "assistant_text_present": bool(subsystem_result.assistant_text.strip()),
            "expected_literal": literal_expected_output,
            "literal_exact_match": literal_exact_match,
        }
        accepted = verification.passed and bool(verification.evidence)
        if step.kind in {"tool", "read", "write", "note"}:
            accepted = accepted and bool(subsystem_result.tool_results)
        if step.kind in {"respond", "reasoning"}:
            accepted = accepted and bool(subsystem_result.assistant_text.strip())
        reason = "review_passed" if accepted else "verification_or_result_evidence_insufficient"
        return SubagentReport(
            spec=spec,
            accepted=accepted,
            reason=reason,
            evidence=evidence,
            recommended_action="continue" if accepted else "retry_or_replan",
            artifacts=[SubagentArtifact(artifact_type="result_review", content=evidence)],
        )
