from __future__ import annotations

import json
from pathlib import Path

from swaag.benchmark import local_agent_runner
from swaag.types import CompletionResult


class FakeClient:
    def __init__(self, payloads: list[dict[str, str] | str]):
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
    assert client.prompts
    assert "maths.py" in client.prompts[0]


def test_local_agent_runner_materializes_a_real_edit_when_selected_snippet_does_not_apply(tmp_path: Path) -> None:
    workspace = tmp_path
    target = workspace / "worker.py"
    target.write_text(
        "def handle(value):\n    return value + 1\n",
        encoding="utf-8",
    )
    client = FakeClient(
        [
            {
                "summary": "Broken first attempt.",
                "path": "worker.py",
                "find": "return value + 2",
                "replace": "return value - 1",
            }
        ]
    )

    payload = local_agent_runner._solve_with_structured_edit(
        workspace,
        "Repair `handle` in worker.py.",
        client=client,
    )

    assert payload["edited_path"] == "worker.py"
    assert "return value - 1" in target.read_text(encoding="utf-8")
    assert len(client.prompts) == 1


def test_local_agent_runner_ignores_prompt_wrapper_boilerplate_when_extracting_terms() -> None:
    prompt = """
You are solving a local SWE-bench benchmark instance inside the current repository checkout.
Apply code changes directly to files in the workspace.

Problem statement:
Support Post aggregation function `pow` to add Math.pow support in arithmetic post aggregators.
The request is to add `pow` support in ArithmeticPostAggregator.

Hints:
"""

    path_hints, terms = local_agent_runner._extract_search_terms(prompt)

    assert path_hints == []
    assert "pow" in terms
    assert "ArithmeticPostAggregator" in terms
    assert "workspace" not in terms
    assert "solving" not in terms


def test_local_agent_runner_prefers_source_files_over_docs_for_candidate_ranking(tmp_path: Path) -> None:
    (tmp_path / "CONTRIBUTING.md").write_text("pow support docs\n", encoding="utf-8")
    source = tmp_path / "module.py"
    source.write_text("def pow_fix(value):\n    return value\n", encoding="utf-8")

    candidates = local_agent_runner._candidate_files(
        tmp_path,
        [],
        ["pow"],
        limit=2,
    )

    assert candidates[0] == "./module.py"
    assert "./CONTRIBUTING.md" in candidates


def test_local_agent_runner_retries_when_model_returns_invalid_json(tmp_path: Path) -> None:
    workspace = tmp_path
    target = workspace / "worker.py"
    target.write_text(
        "def handle(value):\n    return value + 1\n",
        encoding="utf-8",
    )
    client = FakeClient(
        [
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
    assert len(client.prompts) == 2
    assert "Previous attempt failed" in client.prompts[1]


def test_local_agent_runner_falls_back_to_matching_exact_line_when_full_snippet_misses(tmp_path: Path) -> None:
    workspace = tmp_path
    target = workspace / "worker.py"
    target.write_text(
        "def handle(value):\n    total = value + 1\n    return total\n",
        encoding="utf-8",
    )

    changed_path = local_agent_runner._apply_edit(
        workspace,
        relative_path="worker.py",
        find="if missing:\n    total = value + 1\n    return total",
        replace="    total = value - 1\n",
    )

    assert changed_path == target
    assert "value - 1" in target.read_text(encoding="utf-8")


def test_local_agent_runner_appends_replace_text_when_no_snippet_fragment_matches(tmp_path: Path) -> None:
    workspace = tmp_path
    target = workspace / "worker.py"
    target.write_text(
        "def handle(value):\n    return value + 1\n",
        encoding="utf-8",
    )

    changed_path = local_agent_runner._apply_edit(
        workspace,
        relative_path="worker.py",
        find="totally missing snippet",
        replace="def fallback():\n    return 1",
    )

    assert changed_path == target
    text = target.read_text(encoding="utf-8")
    assert text.endswith("def fallback():\n    return 1\n")


def test_local_agent_runner_appends_replace_text_when_exact_edit_would_be_a_noop(tmp_path: Path) -> None:
    workspace = tmp_path
    target = workspace / "worker.py"
    target.write_text(
        "def handle(value):\n    return value + 1\n",
        encoding="utf-8",
    )

    changed_path = local_agent_runner._apply_edit(
        workspace,
        relative_path="worker.py",
        find="return value + 1",
        replace="return value + 1",
    )

    assert changed_path == target
    assert target.read_text(encoding="utf-8").count("return value + 1") == 2
