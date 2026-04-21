from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from swaag.config import AgentConfig, load_config
from swaag.test_categories import (
    AGENT_TEST_FILES,
    CODE_CORRECTNESS_TEST_FILES,
    _DEVCHECK_SYSTEM_PROFILE_FILES,
    project_root,
)


@pytest.fixture(autouse=True)
def _fast_benchmark_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWAAG_BENCHMARK_TEST_RETRIEVAL_BACKEND", "degraded_lexical")


@pytest.fixture()
def make_config(tmp_path: Path):
    def _make(**overrides: Any) -> AgentConfig:
        env = {
            "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
            "SWAAG__TOOLS__READ_ROOTS": f'["{tmp_path}"]',
            "SWAAG__MODEL__BASE_URL": "http://127.0.0.1:9999",
            "SWAAG__RETRIEVAL__BACKEND": "degraded_lexical",
        }
        config = load_config(env=env)
        for path, value in overrides.items():
            target: Any = config
            parts = path.split("__")
            for part in parts[:-1]:
                target = getattr(target, part)
            setattr(target, parts[-1], value)
        return config

    return _make


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    root = project_root()
    for item in items:
        path = Path(str(item.fspath)).resolve()
        try:
            relative = str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            continue
        # Primary two-category markers: code_correctness or agent_test.
        if relative in AGENT_TEST_FILES:
            item.add_marker(pytest.mark.agent_test)
        elif relative in CODE_CORRECTNESS_TEST_FILES:
            item.add_marker(pytest.mark.code_correctness)
        # Devcheck subset markers (internal routing only — not for deselection).
        if relative in _DEVCHECK_SYSTEM_PROFILE_FILES or relative in AGENT_TEST_FILES:
            item.add_marker(pytest.mark.system)
        else:
            item.add_marker(pytest.mark.fast)
