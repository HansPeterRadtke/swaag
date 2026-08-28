from __future__ import annotations

from pathlib import Path

import pytest

from swaag.history import HistoryStore
from swaag.tools.base import ToolValidationError
from swaag.tools.registry import ToolRegistry


def _state(config, session_id: str):
    return HistoryStore(config.sessions.root).create(
        config_fingerprint=config.config_fingerprint(),
        model_base_url=config.model.base_url,
        session_id=session_id,
    )


def _search_input(**overrides):
    return {
        "path": "matches.txt",
        "pattern": "needle",
        "regex": False,
        "ignore_case": False,
        "start_index": 0,
        "max_matches": 2,
        **overrides,
    }


def test_file_search_pages_every_exact_match_without_hidden_truncation(
    make_config, tmp_path: Path
) -> None:
    (tmp_path / "matches.txt").write_text(
        "needle one\nneedle two\nneedle three\nneedle four\nneedle five\n",
        encoding="utf-8",
    )
    config = make_config()
    state = _state(config, "session_file_search_pages")
    registry = ToolRegistry()

    _, first = registry.dispatch(
        "search_in_file", _search_input(), config, state
    )
    _, second = registry.dispatch(
        "search_in_file",
        _search_input(start_index=first.output["next_index"]),
        config,
        state,
    )
    _, third = registry.dispatch(
        "search_in_file",
        _search_input(start_index=second.output["next_index"]),
        config,
        state,
    )

    assert first.output["finished"] is False
    assert first.output["truncated"] is True
    assert second.output["finished"] is False
    assert third.output["finished"] is True
    assert third.output["truncated"] is False
    assert third.output["next_index"] == 5
    assert [
        item["line_number"]
        for result in (first, second, third)
        for item in result.output["matches"]
    ] == [1, 2, 3, 4, 5]
    assert first.generated_events[-1].payload["finished"] is False


def test_repo_search_pagination_is_stable_across_file_boundaries(
    make_config, tmp_path: Path
) -> None:
    (tmp_path / "a.txt").write_text("needle a1\nneedle a2\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("needle b1\nneedle b2\n", encoding="utf-8")
    config = make_config()
    state = _state(config, "session_repo_search_pages")
    registry = ToolRegistry()
    base = {
        "path": ".",
        "pattern": "needle",
        "regex": False,
        "ignore_case": False,
        "start_index": 0,
        "max_matches": 3,
    }

    _, first = registry.dispatch("search_repo", base, config, state)
    _, second = registry.dispatch(
        "search_repo",
        base | {"start_index": first.output["next_index"]},
        config,
        state,
    )

    assert first.output["finished"] is False
    assert first.output["matched_files"] == ["a.txt", "b.txt"]
    assert [item["relative_path"] for item in first.output["matches"]] == [
        "a.txt",
        "a.txt",
        "b.txt",
    ]
    assert second.output["finished"] is True
    assert second.output["start_index"] == 3
    assert second.output["next_index"] == 4
    assert [item["line_text"] for item in second.output["matches"]] == [
        "needle b2"
    ]


def test_search_start_index_rejects_boolean_and_negative_values(make_config) -> None:
    tool = ToolRegistry().get("search_in_file")

    with pytest.raises(ToolValidationError, match="start_index"):
        tool.validate(_search_input(start_index=True))
    with pytest.raises(ToolValidationError, match="start_index"):
        tool.validate(_search_input(start_index=-1))
