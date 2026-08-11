from __future__ import annotations

from pathlib import Path

from swaag.environment.environment import AgentEnvironment
from swaag.history import HistoryStore
from swaag.runtime import AgentRuntime


def test_initial_environment_persists_bounded_workspace_manifest(make_config, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ["a.py", "b.py", "c.py", "d.py"]:
        (workspace / name).write_text(name, encoding="utf-8")
    config = make_config()
    config.tools.read_roots = [workspace]
    config.sessions.root = tmp_path / "sessions"
    config.context.workspace_manifest_max_files = 3
    state = HistoryStore(config.sessions.root).create(config_fingerprint="cfg", model_base_url="http://model")
    env = AgentEnvironment(config, state)
    events = env.initialize_events()
    payload = events[0].payload
    assert payload["listed_files"] == ["a.py", "b.py", "c.py"]
    assert payload["listing_truncated"] is True


def test_environment_rebuild_restores_initial_workspace_manifest(make_config, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test_actual.py").write_text("x=1\n", encoding="utf-8")
    config = make_config()
    config.tools.read_roots = [workspace]
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()
    rebuilt = runtime.history.rebuild_from_history(state.session_id)
    assert "test_actual.py" in rebuilt.environment.workspace.listed_files
    assert rebuilt.environment.workspace.listing_truncated is False


def test_action_context_includes_workspace_manifest(make_config, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "module.py").write_text("x=1\n", encoding="utf-8")
    (workspace / "test_module.py").write_text("assert True\n", encoding="utf-8")
    config = make_config()
    config.tools.read_roots = [workspace]
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()
    components = runtime._runtime_context_components(state, runtime._counter(state))
    environment = next(item.text for item in components if item.name == "environment_state")
    assert '"workspace_files"' in environment
    assert "module.py" in environment
    assert "test_module.py" in environment
    assert '"workspace_listing_truncated":false' in environment.replace(" ", "")


def test_default_manifest_bound_is_positive(make_config) -> None:
    assert make_config().context.workspace_manifest_max_files > 0
