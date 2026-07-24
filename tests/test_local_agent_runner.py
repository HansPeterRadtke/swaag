from __future__ import annotations

import json
from pathlib import Path

import pytest

from swaag.benchmark import local_agent_runner
from swaag.types import CompletionResult


class FakeClient:
    def __init__(self, payloads: list[object]):
        self._payloads = list(payloads)
        self.prompts: list[str] = []

    def complete(self, prompt: str, **_kwargs) -> CompletionResult:
        self.prompts.append(prompt)
        payload = self._payloads.pop(0)
        return CompletionResult(
            text=payload if isinstance(payload, str) else json.dumps(payload),
            raw_request={},
            raw_response={},
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
        )


def test_local_agent_runner_applies_structured_edit_to_candidate_file(tmp_path: Path) -> None:
    workspace = tmp_path
    target = workspace / "maths.py"
    target.write_text(
        "def separability_matrix(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    client = FakeClient(
        [
            {"paths": ["maths.py"], "reason": "Inspect the model-selected file."},
            {
                "summary": "Adjust the helper implementation.",
                "path": "maths.py",
                "find": "return a + b",
                "replace": "return a - b",
            }
        ]
    )

    payload = local_agent_runner._solve_with_structured_edit(
        workspace,
        "Fix `separability_matrix` so nested models work correctly.",
        client=client,
    )

    assert payload["edited_path"] == "maths.py"
    assert "return a - b" in target.read_text(encoding="utf-8")
    assert len(client.prompts) == 2
    assert "maths.py" in client.prompts[0]
    assert "maths.py" in client.prompts[1]


def test_local_agent_runner_retries_when_selected_snippet_does_not_apply(tmp_path: Path) -> None:
    workspace = tmp_path
    target = workspace / "worker.py"
    target.write_text(
        "def handle(value):\n    return value + 1\n",
        encoding="utf-8",
    )
    client = FakeClient(
        [
            {"paths": ["worker.py"], "reason": "Inspect the model-selected file."},
            {
                "summary": "Broken first attempt.",
                "path": "worker.py",
                "find": "return value + 2",
                "replace": "return value - 1",
            },
            {
                "summary": "Valid second attempt.",
                "path": "worker.py",
                "find": "return value + 1",
                "replace": "return value - 1",
            },
        ]
    )

    payload = local_agent_runner._solve_with_structured_edit(
        workspace,
        "Repair `handle` in worker.py.",
        client=client,
    )

    assert payload["edited_path"] == "worker.py"
    assert "return value - 1" in target.read_text(encoding="utf-8")
    assert len(client.prompts) == 3
    assert "Previous attempt failed" in client.prompts[2]


def test_local_agent_runner_uses_model_selected_files_for_context(tmp_path: Path) -> None:
    (tmp_path / "CONTRIBUTING.md").write_text("pow support docs\n", encoding="utf-8")
    source = tmp_path / "module.py"
    source.write_text("def pow_fix(value):\n    return value\n", encoding="utf-8")
    client = FakeClient([{"paths": ["module.py"], "reason": "The model selected the implementation file."}])

    candidates = local_agent_runner._select_candidate_files(
        tmp_path,
        "Choose the relevant file.",
        client=client,
        policy=local_agent_runner.LocalRunnerPolicy(candidate_file_limit=1),
    )

    assert candidates == ["module.py"]
    assert "CONTRIBUTING.md" in client.prompts[0]
    assert "module.py" in client.prompts[0]


def test_local_agent_runner_retries_when_model_returns_invalid_json(tmp_path: Path) -> None:
    workspace = tmp_path
    target = workspace / "worker.py"
    target.write_text(
        "def handle(value):\n    return value + 1\n",
        encoding="utf-8",
    )
    client = FakeClient(
        [
            {"paths": ["worker.py"], "reason": "Inspect the model-selected file."},
            '{"summary": "oops", "path": "worker.py", "find": "return value + 1", "replace": ',
            {
                "summary": "Valid second attempt.",
                "path": "worker.py",
                "find": "return value + 1",
                "replace": "return value - 1",
            },
        ]
    )

    payload = local_agent_runner._solve_with_structured_edit(
        workspace,
        "Repair `handle` in worker.py.",
        client=client,
    )

    assert payload["edited_path"] == "worker.py"
    assert "return value - 1" in target.read_text(encoding="utf-8")
    assert len(client.prompts) == 3
    assert "Previous attempt failed" in client.prompts[2]


def test_local_agent_runner_solver_prompt_names_structured_edit_keys() -> None:
    prompt = local_agent_runner._build_solver_prompt(
        "Fix worker.py.",
        [("worker.py", "def handle(value):\n    return value + 1\n")],
        policy=local_agent_runner.LocalRunnerPolicy(),
    )

    assert "keys summary, path, find, and replace" in prompt
    assert "summary is one short description" in prompt
    assert "path is the single file to edit" in prompt
    assert "find is the exact text snippet" in prompt
    assert "replace is the exact replacement text" in prompt


def test_local_agent_runner_rejects_missing_find_text(tmp_path: Path) -> None:
    workspace = tmp_path
    target = workspace / "worker.py"
    target.write_text(
        "def handle(value):\n    total = value + 1\n    return total\n",
        encoding="utf-8",
    )

    with pytest.raises(local_agent_runner.LocalAgentRunnerError, match="did not appear exactly"):
        local_agent_runner._apply_edit(
            workspace,
            relative_path="worker.py",
            find="if missing:\n    total = value + 1\n    return total",
            replace="    total = value - 1\n",
        )
    assert target.read_text(encoding="utf-8") == "def handle(value):\n    total = value + 1\n    return total\n"


def test_local_agent_runner_rejects_missing_snippet_without_appending(tmp_path: Path) -> None:
    workspace = tmp_path
    target = workspace / "worker.py"
    target.write_text(
        "def handle(value):\n    return value + 1\n",
        encoding="utf-8",
    )

    with pytest.raises(local_agent_runner.LocalAgentRunnerError, match="did not appear exactly"):
        local_agent_runner._apply_edit(
            workspace,
            relative_path="worker.py",
            find="totally missing snippet",
            replace="def fallback():\n    return 1",
        )
    assert target.read_text(encoding="utf-8") == "def handle(value):\n    return value + 1\n"


def test_local_agent_runner_rejects_noop_edit(tmp_path: Path) -> None:
    workspace = tmp_path
    target = workspace / "worker.py"
    target.write_text(
        "def handle(value):\n    return value + 1\n",
        encoding="utf-8",
    )

    with pytest.raises(local_agent_runner.LocalAgentRunnerError, match="would not change"):
        local_agent_runner._apply_edit(
            workspace,
            relative_path="worker.py",
            find="return value + 1",
            replace="return value + 1",
        )
    assert target.read_text(encoding="utf-8").count("return value + 1") == 1
