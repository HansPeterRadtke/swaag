from __future__ import annotations

"""Dynamic, scale-free prompt budgeting.

The agent previously relied on fixed absolute token reserves and tiny
per-section caps. That worked only for one context size and silently
distorted behavior at other sizes. This module defines a call-type-aware
budget policy:

1. derive a dynamic output ceiling from the call kind,
2. derive a dynamic safety margin from context size + call kind,
3. compute a safe input budget,
4. distribute that budget across optional context sections by priority.

Deterministic exact budgeting still happens later via ``build_budget``; this
module only decides how much *candidate context* should be built before the
final exact fit check.
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


@dataclass(slots=True)
class SectionBudgets:
    history_tokens: int
    environment_files_tokens: int
    semantic_tokens: int
    guidance_tokens: int
    skills_tokens: int
    notes_tokens: int
    available_input_tokens: int

    def for_section(self, name: str) -> int:
        return {
            "history": self.history_tokens,
            "environment_files": self.environment_files_tokens,
            "semantic": self.semantic_tokens,
            "guidance": self.guidance_tokens,
            "skills": self.skills_tokens,
            "notes": self.notes_tokens,
        }[name]


def classify_call_budget(call_kind: str) -> CallBudgetClass:
    del call_kind
    return "small"


def compute_call_budget(
    config: AgentConfig,
    *,
    call_kind: str,
) -> CallBudgetPlan:
    budget_class = str(config.budget_policy.call_classes.get(call_kind, "small"))
    context_limit = max(int(config.model.context_limit), 256)
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


def compute_section_budgets(
    config: AgentConfig,
    *,
    call_kind: str,
    safe_input_budget: int,
) -> SectionBudgets:
    weights = dict(config.budget_policy.section_priorities.get(call_kind, config.budget_policy.section_priorities["default"]))
    total_weight = sum(max(value, 0.0) for value in weights.values()) or float(len(weights))
    floors = {
        key: max(
            int(config.budget_policy.section_floor_min_tokens[key]),
            int(round(safe_input_budget * float(config.budget_policy.section_floor_ratio[key]))),
        )
        for key in ("history", "environment_files", "semantic", "guidance", "skills", "notes")
    }

    allocations: dict[str, int] = {}
    remaining = safe_input_budget
    for name in ("history", "environment_files", "semantic", "guidance", "skills", "notes"):
        share = int(round(safe_input_budget * (weights.get(name, 0.0) / total_weight)))
        share = max(share, floors[name])
        allocations[name] = share
        remaining -= share

    if remaining < 0:
        deficit = -remaining
        total_alloc = sum(allocations.values()) or 1
        for name in list(allocations):
            reduction = int(round(deficit * (allocations[name] / total_alloc)))
            allocations[name] = max(floors[name], allocations[name] - reduction)
        total = sum(allocations.values())
        if total > safe_input_budget:
            ordered = sorted(allocations, key=lambda item: allocations[item], reverse=True)
            index = 0
            while total > safe_input_budget and ordered:
                name = ordered[index % len(ordered)]
                if allocations[name] > floors[name]:
                    allocations[name] -= 1
                    total -= 1
                index += 1

    return SectionBudgets(
        history_tokens=allocations["history"],
        environment_files_tokens=allocations["environment_files"],
        semantic_tokens=allocations["semantic"],
        guidance_tokens=allocations["guidance"],
        skills_tokens=allocations["skills"],
        notes_tokens=allocations["notes"],
        available_input_tokens=safe_input_budget,
    )


def _schema_upper_bound_instance(schema: dict[str, Any], *, depth: int = 0) -> Any:
    if not isinstance(schema, dict):
        return ""
    variants = schema.get("oneOf") or schema.get("anyOf")
    if isinstance(variants, list) and variants:
        candidates = [_schema_upper_bound_instance(item, depth=depth) for item in variants if isinstance(item, dict)]
        if candidates:
            return max(candidates, key=lambda item: len(stable_json_dumps(item)))
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        ordered = [item for item in schema_type if item != "null"]
        schema_type = ordered[0] if ordered else schema_type[0]
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        return max(schema["enum"], key=lambda item: len(stable_json_dumps(item)))
    if schema_type == "object":
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return {}
        required = schema.get("required")
        required_keys = [str(key) for key in required] if isinstance(required, list) and required else list(properties)
        return {
            str(key): _schema_upper_bound_instance(properties[key], depth=depth + 1)
            for key in required_keys
            if key in properties and isinstance(properties[key], dict)
        }
    if schema_type == "array":
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return []
        max_items = schema.get("maxItems", schema.get("minItems", 1))
        try:
            count = int(max_items)
        except (TypeError, ValueError):
            count = 1
        count = max(1, min(count, 8 if depth == 0 else 3))
        item_value = _schema_upper_bound_instance(item_schema, depth=depth + 1)
        return [item_value for _ in range(count)]
    if schema_type == "string":
        max_length = schema.get("maxLength", schema.get("minLength", 16))
        try:
            length = int(max_length)
        except (TypeError, ValueError):
            length = 16
        length = max(1, min(length, 160))
        # Use a token-dense placeholder rather than a repeated character. A
        # repeated "xxxx..." string badly underestimates worst-case token
        # usage on the active tokenizer because it compresses into very few
        # tokens. Short space-separated atoms give a safer bounded estimate.
        dense = " ".join("a" for _ in range(length))
        return dense[:length]
    if schema_type == "integer":
        if isinstance(schema.get("maximum"), int):
            return int(schema["maximum"])
        if isinstance(schema.get("minimum"), int):
            return int(schema["minimum"])
        return 999999
    if schema_type == "number":
        if isinstance(schema.get("maximum"), (int, float)):
            return float(schema["maximum"])
        if isinstance(schema.get("minimum"), (int, float)):
            return float(schema["minimum"])
        return 999999.0
    if schema_type == "boolean":
        return True
    if schema_type == "null":
        return None
    return "x" * 16


def structured_output_token_floor(
    contract: ContractSpec,
    *,
    config: AgentConfig,
    counter: TokenCounter,
    call_kind: str,
) -> int:
    if contract.mode == "plain":
        return 0
    if contract.json_schema:
        sample_instance = _schema_upper_bound_instance(contract.json_schema)
        instance_tokens = max(counter.count_text(stable_json_dumps(sample_instance)).tokens, 1)
        schema_tokens = max(counter.count_text(stable_json_dumps(contract.json_schema)).tokens, 1)
        factor = config.budget_policy.structured_output_json_factor_by_contract.get(
            contract.name,
            config.budget_policy.structured_output_json_factor_by_contract.get(
                call_kind,
                config.budget_policy.structured_output_json_factor_default,
            ),
        )
        bounded_tokens = int(round(instance_tokens * factor))
        return max(int(config.budget_policy.structured_output_json_floor_tokens), min(bounded_tokens, schema_tokens))
    if contract.mode == "gbnf" and contract.grammar:
        grammar_tokens = max(counter.count_text(contract.grammar).tokens, 1)
        return max(
            int(config.budget_policy.structured_output_grammar_floor_tokens),
            int(round(grammar_tokens * float(config.budget_policy.structured_output_grammar_factor))),
        )
    return 0
