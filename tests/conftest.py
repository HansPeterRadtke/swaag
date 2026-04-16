from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from swaag.config import AgentConfig, load_config
from swaag.testlanes import BENCHMARK_HEAVY_TEST_FILES, INTEGRATION_TEST_FILES, LIVE_TEST_FILES, SYSTEM_TEST_FILES, project_root


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
        if item.get_closest_marker("benchmark_heavy"):
            item.add_marker(pytest.mark.benchmark_heavy)
            continue
        if item.get_closest_marker("live"):
            item.add_marker(pytest.mark.live)
            continue
        if item.get_closest_marker("integration"):
            item.add_marker(pytest.mark.integration)
            continue
        if relative in BENCHMARK_HEAVY_TEST_FILES:
            item.add_marker(pytest.mark.benchmark_heavy)
            continue
        if relative in LIVE_TEST_FILES:
            item.add_marker(pytest.mark.live)
            continue
        if relative in INTEGRATION_TEST_FILES:
            item.add_marker(pytest.mark.integration)
            continue
        if relative in SYSTEM_TEST_FILES:
            item.add_marker(pytest.mark.system)
        else:
            item.add_marker(pytest.mark.fast)
