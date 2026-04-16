from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from swaag.config import ContextConfig
from swaag.types import BudgetComponentReport, BudgetReport, PromptComponent


@dataclass(slots=True)
class CountResult:
    tokens: int
    exact: bool
    strategy: str


class TokenCounter(Protocol):
    def count_text(self, text: str) -> CountResult: ...


class ExactTokenCounter:
    def __init__(self, tokenize_func):
        self._tokenize_func = tokenize_func
        self._cache: dict[str, int] = {}

    def count_text(self, text: str) -> CountResult:
        if text not in self._cache:
            self._cache[text] = int(self._tokenize_func(text))
        return CountResult(tokens=self._cache[text], exact=True, strategy="llama_cpp_server")


class ConservativeEstimator:
    def __init__(self, chars_per_token: float = 3.0):
        self._chars_per_token = chars_per_token

    def count_text(self, text: str) -> CountResult:
        tokens = max(1, math.ceil(len(text) / self._chars_per_token)) if text else 0
        return CountResult(tokens=tokens, exact=False, strategy="chars_per_token")



def count_text(counter: TokenCounter, text: str) -> CountResult:
    return counter.count_text(text)



def count_message_list(counter: TokenCounter, rendered_messages: str) -> CountResult:
    return counter.count_text(rendered_messages)



def count_tool_descriptions(counter: TokenCounter, rendered_tools: str) -> CountResult:
    return counter.count_text(rendered_tools)



def count_grammar(counter: TokenCounter, rendered_contract: str) -> CountResult:
    return counter.count_text(rendered_contract)



def reserve_for_response(context: ContextConfig) -> int:
    return context.reserved_response_tokens



def build_budget(
    counter: TokenCounter,
    components: list[PromptComponent],
    context: ContextConfig,
    context_limit: int,
    *,
    reserved_response_tokens: int | None = None,
    safety_margin_tokens: int | None = None,
) -> BudgetReport:
    exact = True
    context_prefix = ""
    context_prefix_tokens = 0
    input_tokens = 0
    non_context_tokens = 0
    breakdown: list[BudgetComponentReport] = []
    response_tokens = context.reserved_response_tokens if reserved_response_tokens is None else int(reserved_response_tokens)
    safety_tokens = context.safety_margin_tokens if safety_margin_tokens is None else int(safety_margin_tokens)

    for component in components:
        if component.include_in_context:
            context_prefix += component.text
            counted = counter.count_text(context_prefix)
            delta = counted.tokens - context_prefix_tokens
            context_prefix_tokens = counted.tokens
            input_tokens += delta
        else:
            counted = counter.count_text(component.text)
            delta = counted.tokens
            non_context_tokens += delta
        exact = exact and counted.exact
        breakdown.append(
            BudgetComponentReport(
                name=component.name,
                category=component.category or component.name,
                tokens=delta,
                exact=counted.exact,
                include_in_context=component.include_in_context,
                optional=component.optional,
            )
        )

    required_tokens = input_tokens + response_tokens + safety_tokens
    return BudgetReport(
        context_limit=context_limit,
        input_tokens=input_tokens,
        reserved_response_tokens=response_tokens,
        safety_margin_tokens=safety_tokens,
        required_tokens=required_tokens,
        non_context_tokens=non_context_tokens,
        fits=required_tokens <= context_limit,
        exact=exact,
        breakdown=breakdown,
    )



def admission_check(report: BudgetReport) -> bool:
    return report.fits
