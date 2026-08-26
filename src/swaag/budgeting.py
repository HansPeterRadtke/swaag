from __future__ import annotations

"""Dynamic, scale-free prompt budgeting.

The agent previously relied on fixed absolute token reserves and tiny
per-section caps. That worked only for one context size and silently
distorted behavior at other sizes. This module defines a call-type-aware
budget policy:

1. derive an output reserve from the call kind and context size,
2. reserve fixed overhead and a safety margin,
3. compute the maximum safe input budget,
4. reserve enough output for the constrained JSON schema.

The runtime performs the final exact fit check with ``build_budget``.
"""

from dataclasses import dataclass
from typing import Any
from typing import Literal

from swaag.config import AgentConfig
from swaag.tokens import TokenCounter
from swaag.types import ContractSpec
from swaag.utils import stable_json_dumps

CallBudgetClass = Literal["tiny", "small", "medium", "large"]


@dataclass(slots=True)
class CallBudgetPlan:
    call_kind: str
    budget_class: CallBudgetClass
    context_limit: int
    output_tokens: int
    safety_margin_tokens: int
    fixed_overhead_tokens: int
    safe_input_budget: int


def classify_call_budget(call_kind: str) -> CallBudgetClass:
    del call_kind
    return "small"


def compute_call_budget(
    config: AgentConfig,
    *,
    call_kind: str,
    context_limit: int | None = None,
) -> CallBudgetPlan:
    budget_class = str(config.budget_policy.call_classes.get(call_kind, "small"))
    context_limit = max(int(config.model.context_limit if context_limit is None else context_limit), 1)
    output_ratio = float(
        config.budget_policy.output_ratio_by_kind.get(
            call_kind,
            config.budget_policy.output_ratio[budget_class],
        )
    )
    output_floor_ratio = float(
        config.budget_policy.output_floor_ratio_by_kind.get(
            call_kind,
            config.budget_policy.output_floor_ratio[budget_class],
        )
    )

    output_tokens = max(
        int(round(context_limit * output_ratio)),
        int(round(context_limit * output_floor_ratio)),
    )
    safety_margin = max(
        int(round(context_limit * float(config.budget_policy.safety_ratio[budget_class]))),
        int(config.context.safety_margin_tokens),
    )
    fixed_overhead = max(
        int(round(context_limit * float(config.budget_policy.fixed_overhead_ratio[budget_class]))),
        int(config.budget_policy.fixed_overhead_min_tokens),
    )
    safe_input_budget = max(context_limit - output_tokens - safety_margin - fixed_overhead, int(config.budget_policy.safe_input_floor_tokens))
    return CallBudgetPlan(
        call_kind=call_kind,
        budget_class=budget_class,
        context_limit=context_limit,
        output_tokens=output_tokens,
        safety_margin_tokens=safety_margin,
        fixed_overhead_tokens=fixed_overhead,
        safe_input_budget=safe_input_budget,
    )


def _schema_minimum_instance(schema: dict[str, Any]) -> Any:
    """Build a smallest structurally valid value for a portable JSON schema.

    Unbounded strings and arrays have no honest deterministic upper bound.  The
    caller supplies the operation-specific useful-output minimum separately;
    this value only measures the syntax needed to satisfy the contract.
    """
    if not isinstance(schema, dict):
        return ""
    variants = schema.get("oneOf") or schema.get("anyOf")
    if isinstance(variants, list) and variants:
        candidates = [_schema_minimum_instance(item) for item in variants if isinstance(item, dict)]
        if candidates:
            return min(candidates, key=lambda item: len(stable_json_dumps(item)))
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        ordered = [item for item in schema_type if item != "null"]
        schema_type = ordered[0] if ordered else schema_type[0]
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        return min(schema["enum"], key=lambda item: len(stable_json_dumps(item)))
    if schema_type == "object":
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return {}
        required = schema.get("required")
        required_keys = [str(key) for key in required] if isinstance(required, list) and required else list(properties)
        return {
            str(key): _schema_minimum_instance(properties[key])
            for key in required_keys
            if key in properties and isinstance(properties[key], dict)
        }
    if schema_type == "array":
        return []
    if schema_type == "string":
        return ""
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0
    if schema_type == "boolean":
        return True
    if schema_type == "null":
        return None
    return ""


def structured_output_token_floor(
    contract: ContractSpec,
    *,
    config: AgentConfig,
    counter: TokenCounter,
    call_kind: str,
) -> int:
    if contract.json_schema:
        sample_instance = _schema_minimum_instance(contract.json_schema)
        instance_tokens = max(counter.count_text(stable_json_dumps(sample_instance)).tokens, 1)
        factor = config.budget_policy.structured_output_json_factor_by_contract.get(
            contract.name,
            config.budget_policy.structured_output_json_factor_by_contract.get(
                call_kind,
                config.budget_policy.structured_output_json_factor_default,
            ),
        )
        bounded_tokens = int(round(instance_tokens * factor))
        return max(int(config.budget_policy.structured_output_json_floor_tokens), bounded_tokens)
    return 0
