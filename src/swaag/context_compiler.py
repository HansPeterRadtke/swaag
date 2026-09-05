from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from swaag.budgeting import CallBudgetPlan, compute_call_budget, structured_output_token_floor
from swaag.config import AgentConfig
from swaag.tokens import TokenCounter, build_budget
from swaag.types import BudgetReport, ContractSpec, PromptAssembly, PromptComponent
from swaag.utils import sha256_text, stable_json_dumps


@dataclass(slots=True)
class ContextCompilation:
    report: BudgetReport
    plan: CallBudgetPlan
    structured_output_floor_tokens: int
    minimum_output_tokens: int
    desired_output_tokens: int
    context_limit_source: str
    serialized_prompt_chars: int
    serialized_prompt_sha256: str
    serialized_components: list[dict[str, Any]]

    @property
    def overflow_tokens(self) -> int:
        return max(0, self.report.required_tokens - self.report.context_limit)

    @property
    def available_input_tokens(self) -> int:
        return max(
            0,
            self.report.context_limit
            - self.report.reserved_response_tokens
            - self.report.safety_margin_tokens,
        )

    def accounting(self) -> dict[str, Any]:
        return {
            "call_kind": self.plan.call_kind,
            "budget_class": self.plan.budget_class,
            "context_limit": self.report.context_limit,
            "available_input_tokens": self.available_input_tokens,
            "input_tokens": self.report.input_tokens,
            "reserved_response_tokens": self.report.reserved_response_tokens,
            "safety_margin_tokens": self.report.safety_margin_tokens,
            "required_tokens": self.report.required_tokens,
            "overflow_tokens": self.overflow_tokens,
            "non_context_tokens": self.report.non_context_tokens,
            "exact": self.report.exact,
            "fits": self.report.fits,
            "structured_output_floor_tokens": self.structured_output_floor_tokens,
            "minimum_output_tokens": self.minimum_output_tokens,
            "desired_output_tokens": self.desired_output_tokens,
            "context_limit_source": self.context_limit_source,
            "policy": asdict(self.plan),
            "serialized_prompt_chars": self.serialized_prompt_chars,
            "serialized_prompt_sha256": self.serialized_prompt_sha256,
            "components": self.serialized_components,
        }


class ContextCompiler:
    """Mechanical compiler for one model request.

    It owns exact/conservative accounting and hard admission. It deliberately
    does not decide semantic relevance. Candidate semantic material must be
    selected or reduced by an LLM before being handed to this compiler.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def plan(self, *, call_kind: str, context_limit: int | None = None) -> CallBudgetPlan:
        return compute_call_budget(self.config, call_kind=call_kind, context_limit=context_limit)

    def compile(
        self,
        assembly: PromptAssembly,
        contract: ContractSpec,
        counter: TokenCounter,
        *,
        minimum_output_tokens: int = 0,
        desired_output_tokens: int | None = None,
        context_limit: int | None = None,
        context_limit_source: str = "configured",
    ) -> ContextCompilation:
        serialized_components = [
            component for component in assembly.components if component.include_in_context
        ]
        reconstructed_prompt = "".join(component.text for component in serialized_components)
        if reconstructed_prompt != assembly.prompt_text:
            raise ValueError(
                "PromptAssembly prompt_text contains serialized material that is not exactly represented by include_in_context components"
            )

        effective_context_limit = max(
            int(self.config.model.context_limit if context_limit is None else context_limit),
            1,
        )
        plan = compute_call_budget(
            self.config, call_kind=assembly.kind, context_limit=effective_context_limit
        )
        structured_floor = structured_output_token_floor(
            contract,
            config=self.config,
            counter=counter,
            call_kind=assembly.kind,
        )
        minimum_required_output = max(
            int(minimum_output_tokens),
            int(structured_floor),
        )
        desired_output = max(
            minimum_required_output,
            int(plan.output_tokens if desired_output_tokens is None else desired_output_tokens),
        )
        components = [
            *assembly.components,
            PromptComponent(
                name="constraint_schema",
                category="constraint_schema",
                text=stable_json_dumps(contract.json_schema or {}, indent=None),
                include_in_context=False,
            ),
        ]

        # First measure the richest candidate without using a percentage-based
        # output reservation as an admission gate. If tokenizer accounting is
        # exact, only the explicit fixed safety margin is needed; proportional
        # safety is reserved for conservative/estimated counting.
        measured = build_budget(
            counter,
            components,
            self.config.context,
            effective_context_limit,
            reserved_response_tokens=minimum_required_output,
            safety_margin_tokens=0,
        )
        safety_margin = (
            int(self.config.context.safety_margin_tokens)
            if measured.exact
            else int(plan.safety_margin_tokens)
        )
        available_output = max(
            0, effective_context_limit - measured.input_tokens - safety_margin
        )
        if available_output >= minimum_required_output:
            # The configured per-kind output plan is a desired maximum. It does
            # not force semantic input loss. When less room remains, the call
            # receives all remaining safe headroom as long as its minimum valid
            # output requirement still fits.
            reserved = min(
                desired_output,
                available_output,
            )
        else:
            reserved = minimum_required_output

        report = build_budget(
            counter,
            components,
            self.config.context,
            effective_context_limit,
            reserved_response_tokens=reserved,
            safety_margin_tokens=safety_margin,
        )
        prompt_count = counter.count_text(assembly.prompt_text)
        if prompt_count.tokens != report.input_tokens:
            raise ValueError(
                "Context accounting does not exactly reproduce serialized prompt token count"
            )
        component_rows: list[dict[str, Any]] = []
        for index, (component, item) in enumerate(zip(components, report.breakdown, strict=True)):
            row = asdict(item)
            row.update(
                {
                    "index": index,
                    "chars": len(component.text),
                    "sha256": sha256_text(component.text),
                }
            )
            component_rows.append(row)
        return ContextCompilation(
            report=report,
            plan=plan,
            structured_output_floor_tokens=structured_floor,
            minimum_output_tokens=int(minimum_output_tokens),
            desired_output_tokens=desired_output,
            context_limit_source=str(context_limit_source),
            serialized_prompt_chars=len(assembly.prompt_text),
            serialized_prompt_sha256=sha256_text(assembly.prompt_text),
            serialized_components=component_rows,
        )
