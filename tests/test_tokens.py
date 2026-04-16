from __future__ import annotations

from swaag.tokens import (
    ConservativeEstimator,
    ExactTokenCounter,
    admission_check,
    build_budget,
    count_grammar,
    count_message_list,
    count_text,
    count_tool_descriptions,
    reserve_for_response,
)
from swaag.types import PromptComponent


def test_exact_counter_counts_text() -> None:
    counter = ExactTokenCounter(lambda text: len(text.split()) if text.strip() else 0)
    result = count_text(counter, "one two three")
    assert result.tokens == 3
    assert result.exact is True


def test_conservative_counter_counts_empty_text() -> None:
    counter = ConservativeEstimator(chars_per_token=4.0)
    result = count_text(counter, "")
    assert result.tokens == 0
    assert result.exact is False


def test_message_tool_and_grammar_counting(make_config) -> None:
    counter = ExactTokenCounter(lambda text: len(text.split()) if text.strip() else 0)
    assert count_message_list(counter, "alpha beta").tokens == 2
    assert count_tool_descriptions(counter, "tool schema text").tokens == 3
    assert count_grammar(counter, 'root ::= "yes"').tokens == 3
    assert reserve_for_response(make_config().context) == make_config().context.reserved_response_tokens


def test_budget_exact_boundary_fit(make_config) -> None:
    config = make_config(model__context_limit=16, context__reserved_response_tokens=4, context__safety_margin_tokens=2)
    counter = ExactTokenCounter(lambda text: len(text.split()) if text.strip() else 0)
    report = build_budget(
        counter,
        [PromptComponent(name="body", text="one two three four five six", category="history")],
        config.context,
        config.model.context_limit,
    )
    assert report.input_tokens == 6
    assert report.required_tokens == 12
    assert report.fits is True
    assert admission_check(report) is True


def test_budget_one_token_overflow(make_config) -> None:
    config = make_config(model__context_limit=11, context__reserved_response_tokens=4, context__safety_margin_tokens=2)
    counter = ExactTokenCounter(lambda text: len(text.split()) if text.strip() else 0)
    report = build_budget(
        counter,
        [PromptComponent(name="body", text="one two three four five six", category="history")],
        config.context,
        config.model.context_limit,
    )
    assert report.required_tokens == 12
    assert report.fits is False
    assert admission_check(report) is False


def test_budget_tracks_wrapper_and_non_context_components(make_config) -> None:
    config = make_config(model__context_limit=64)
    counter = ConservativeEstimator(chars_per_token=2.0)
    report = build_budget(
        counter,
        [
            PromptComponent(name="wrapper", text="abcd", category="wrapper"),
            PromptComponent(name="schema", text="abcdefgh", category="grammar", include_in_context=False),
        ],
        config.context,
        config.model.context_limit,
        reserved_response_tokens=10,
    )
    assert report.input_tokens == 2
    assert report.non_context_tokens == 4
    assert report.reserved_response_tokens == 10


def test_budget_invariant_if_admitted_then_within_limit(make_config) -> None:
    config = make_config(model__context_limit=256, context__reserved_response_tokens=32, context__safety_margin_tokens=16)
    counter = ExactTokenCounter(lambda text: len(text.split()) if text.strip() else 0)
    report = build_budget(
        counter,
        [
            PromptComponent(name="system", text="alpha beta", category="system_prompt"),
            PromptComponent(name="user", text="one two three", category="current_user"),
        ],
        config.context,
        config.model.context_limit,
    )
    assert report.fits is True
    assert report.input_tokens + report.reserved_response_tokens + report.safety_margin_tokens <= report.context_limit


def test_budget_handles_huge_user_prompt(make_config) -> None:
    config = make_config(model__context_limit=50, context__reserved_response_tokens=10, context__safety_margin_tokens=5)
    counter = ExactTokenCounter(lambda text: len(text.split()) if text.strip() else 0)
    huge = "word " * 100
    report = build_budget(counter, [PromptComponent(name="user", text=huge, category="current_user")], config.context, config.model.context_limit)
    assert report.fits is False
