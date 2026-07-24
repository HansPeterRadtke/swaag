from __future__ import annotations

import pytest

from swaag.prompt_analyzer import (
    PromptAnalysisValidationError,
    analysis_from_payload,
)


def test_prompt_analysis_from_payload_validates_bounds() -> None:
    analysis = analysis_from_payload(
        {
            "task_type": "structured",
            "completeness": "complete",
            "requires_expansion": False,
            "requires_decomposition": True,
            "missing_required_information": False,
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


def test_prompt_analysis_preserves_model_declared_expansion_flag() -> None:
    analysis = analysis_from_payload(
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

    assert analysis.requires_expansion is False


def test_prompt_analysis_missing_information_is_independent_of_shape_labels() -> None:
    analysis = analysis_from_payload({
        "task_type": "unstructured", "completeness": "partial",
        "requires_expansion": False, "requires_decomposition": False,
        "missing_required_information": True, "confidence": 0.9,
        "detected_entities": ["request.txt", "context.txt"],
        "detected_goals": ["read evidence and clarify"],
    })

    assert analysis.task_type == "unstructured"
    assert analysis.completeness == "partial"
    assert analysis.missing_required_information is True


def test_prompt_analysis_allows_missing_information_with_expansion_when_model_declares_both() -> None:
    analysis = analysis_from_payload({
        "task_type": "vague", "completeness": "partial",
        "requires_expansion": True, "requires_decomposition": False,
        "missing_required_information": True, "confidence": 0.8,
        "detected_entities": [], "detected_goals": ["clarify and expand"],
    })

    assert analysis.requires_expansion is True
    assert analysis.missing_required_information is True
