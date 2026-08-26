from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from swaag.budgeting import CallBudgetPlan, compute_call_budget, structured_output_token_floor
from swaag.config import AgentConfig
from swaag.tokens import TokenCounter, build_budget
from swaag.types import BudgetReport, ContractSpec, PromptAssembly, PromptComponent
from swaag.utils import stable_json_dumps


@dataclass(slots=True)
class ContextCompilation:
    report: BudgetReport
    plan: CallBudgetPlan
    structured_output_floor_tokens: int
    minimum_output_tokens: int

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
            "policy": asdict(self.plan),
            "components": [asdict(item) for item in self.report.breakdown],
        }


class ContextCompiler:
    """Mechanical compiler for one model request.

    It owns exact/conservative accounting and hard admission. It deliberately
    does not decide semantic relevance. Candidate semantic material must be
    selected or reduced by an LLM before being handed to this compiler.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def plan(self, *, call_kind: str) -> CallBudgetPlan:
        return compute_call_budget(self.config, call_kind=call_kind)

    def compile(
        self,
        assembly: PromptAssembly,
        contract: ContractSpec,
        counter: TokenCounter,
        *,
        minimum_output_tokens: int = 0,
    ) -> ContextCompilation:
        plan = compute_call_budget(self.config, call_kind=assembly.kind)
        structured_floor = structured_output_token_floor(
            contract,
            config=self.config,
            counter=counter,
            call_kind=assembly.kind,
        )
        reserved = max(
            int(minimum_output_tokens),
            int(plan.output_tokens),
            int(structured_floor),
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
        report = build_budget(
            counter,
            components,
            self.config.context,
            self.config.model.context_limit,
            reserved_response_tokens=reserved,
            safety_margin_tokens=plan.safety_margin_tokens,
        )
        return ContextCompilation(
            report=report,
            plan=plan,
            structured_output_floor_tokens=structured_floor,
            minimum_output_tokens=int(minimum_output_tokens),
        )
