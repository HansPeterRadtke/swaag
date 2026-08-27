from __future__ import annotations

"""Mechanical output planning for exactly measured model requests.

Per-kind ratios express desired output headroom only. They never reserve
context ahead of useful input. The context compiler accounts for the actual
serialized request, applies an operation/schema minimum, and uses
proportional safety only when exact tokenizer accounting is unavailable.
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
    return CallBudgetPlan(
        call_kind=call_kind,
        budget_class=budget_class,
        context_limit=context_limit,
        output_tokens=output_tokens,
        safety_margin_tokens=safety_margin,
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
