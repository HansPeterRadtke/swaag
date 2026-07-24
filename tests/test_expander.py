from __future__ import annotations

import pytest

from swaag.expander import ExpansionValidationError, expanded_task_from_payload


def test_expanded_task_from_payload_preserves_model_fields() -> None:
    expanded = expanded_task_from_payload(
        {
            "original_goal": "Construire un outil local",
            "expanded_goal": "Construire un outil local avec validation explicite.",
            "scope": ["scope choisi par le modele"],
            "constraints": ["contrainte choisie par le modele"],
            "expected_outputs": ["sortie choisie par le modele"],
            "assumptions": ["hypothese choisie par le modele"],
        },
        original_goal="fallback original",
    )

    assert expanded.original_goal == "Construire un outil local"
    assert expanded.expanded_goal == "Construire un outil local avec validation explicite."
    assert expanded.scope == ["scope choisi par le modele"]


def test_expanded_task_from_payload_rejects_empty_goal() -> None:
    with pytest.raises(ExpansionValidationError):
        expanded_task_from_payload(
            {
                "original_goal": "",
                "expanded_goal": "",
                "scope": [],
                "constraints": [],
                "expected_outputs": [],
                "assumptions": [],
            },
            original_goal="make a game",
        )
