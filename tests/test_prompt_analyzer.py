from __future__ import annotations

import pytest

from swaag.prompt_analyzer import (
    PromptAnalysisValidationError,
    analysis_from_payload,
    analyze_prompt_emergency_fallback,
)


def test_prompt_analyzer_emergency_fallback_returns_neutral_unstructured_analysis() -> None:
    """Emergency fallback never claims semantic understanding."""

    analysis = analyze_prompt_emergency_fallback("Build a Python tool that reads src/app.py and writes tests for it.")

    assert analysis.task_type == "unstructured"
    assert analysis.completeness == "partial"
    assert analysis.requires_expansion is True
    assert analysis.requires_decomposition is True
    assert any(token.endswith(".py") for token in analysis.detected_entities)


def test_prompt_analyzer_emergency_fallback_handles_empty_prompt_as_incomplete() -> None:
    analysis = analyze_prompt_emergency_fallback("   ")

    assert analysis.task_type == "incomplete"
    assert analysis.completeness == "incomplete"
    assert analysis.requires_expansion is True


def test_prompt_analyzer_emergency_fallback_is_deterministic() -> None:
    prompt = "Read app.py and update tests"
    first = analyze_prompt_emergency_fallback(prompt)
    second = analyze_prompt_emergency_fallback(prompt)

    assert first == second


def test_prompt_analysis_from_payload_validates_bounds() -> None:
    analysis = analysis_from_payload(
        {
            "task_type": "structured",
            "completeness": "complete",
            "requires_expansion": False,
            "requires_decomposition": True,
            "confidence": 0.9,
            "detected_entities": ["src/app.py"],
            "detected_goals": ["build"],
        }
    )

    assert analysis.task_type == "structured"
    assert analysis.confidence == 0.9


def test_prompt_analysis_rejects_invalid_confidence() -> None:
    with pytest.raises(PromptAnalysisValidationError):
        analysis_from_payload(
            {
                "task_type": "structured",
                "completeness": "complete",
                "requires_expansion": False,
                "requires_decomposition": True,
                "confidence": 1.5,
                "detected_entities": [],
                "detected_goals": [],
            }
        )


def test_prompt_analysis_rejects_unknown_task_type() -> None:
    with pytest.raises(PromptAnalysisValidationError):
        analysis_from_payload(
            {
                "task_type": "wild_guess",
                "completeness": "complete",
                "requires_expansion": False,
                "requires_decomposition": False,
                "confidence": 0.5,
                "detected_entities": [],
                "detected_goals": [],
            }
        )


def test_prompt_analysis_rejects_vague_without_expansion() -> None:
    with pytest.raises(PromptAnalysisValidationError):
        analysis_from_payload(
            {
                "task_type": "vague",
                "completeness": "incomplete",
                "requires_expansion": False,
                "requires_decomposition": False,
                "confidence": 0.5,
                "detected_entities": [],
                "detected_goals": [],
            }
        )
