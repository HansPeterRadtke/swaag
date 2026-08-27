from __future__ import annotations

from pathlib import Path

from swaag.environment.environment import AgentEnvironment
from swaag.grammar import yes_no_contract
from swaag.history import HistoryStore
from swaag.runtime import AgentRuntime, RuntimeContextProjection
from swaag.tokens import ExactTokenCounter


def test_initial_environment_defers_live_manifest_to_context_compilation(make_config, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ["a.py", "b.py", "c.py", "d.py"]:
        (workspace / name).write_text(name, encoding="utf-8")
    config = make_config()
    config.tools.read_roots = [workspace]
    config.sessions.root = tmp_path / "sessions"
    state = HistoryStore(config.sessions.root).create(config_fingerprint="cfg", model_base_url="http://model")
    env = AgentEnvironment(config, state)
    events = env.initialize_events()
    payload = events[0].payload
    assert payload["listed_files"] == []
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
    assert rebuilt.environment.workspace.listed_files == []
    assert rebuilt.environment.workspace.listing_truncated is True
    manifest = next(
        item.text
        for item in runtime._runtime_context_components(rebuilt, runtime._counter(rebuilt))
        if item.name == "workspace_file_manifest"
    )
    assert "test_actual.py" in manifest


def test_action_context_includes_workspace_manifest(make_config, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    nested = workspace / "nested"
    nested.mkdir()
    (workspace / "module.py").write_text("x=1\n", encoding="utf-8")
    (nested / "test_module.py").write_text("assert True\n", encoding="utf-8")
    config = make_config()
    config.tools.read_roots = [workspace]
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()
    state.environment.workspace.cwd = str(nested)
    components = runtime._runtime_context_components(state, runtime._counter(state))
    manifest = next(item.text for item in components if item.name == "workspace_file_manifest")
    assert "module.py" in manifest
    assert "test_module.py" in manifest
    assert '"count":2' in manifest.replace(" ", "")


def test_workspace_manifest_excludes_runtime_state_and_escaping_symlinks(
    make_config,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    (workspace / "outside-link.txt").symlink_to(outside)
    (workspace / "visible.txt").write_text("visible\n", encoding="utf-8")
    config = make_config()
    config.tools.read_roots = [workspace]
    config.sessions.root = workspace / ".runtime" / "sessions"
    runtime = AgentRuntime(config)
    state = runtime.create_or_load_session()

    filesystem = AgentEnvironment(config, state).filesystem
    assert filesystem.list_files(".") == ["visible.txt"]
    assert filesystem.snapshot() == {"visible.txt": "visible\n"}


def test_fixed_manifest_bound_is_not_a_runtime_context_policy(make_config) -> None:
    assert not hasattr(make_config().context, "workspace_manifest_max_files")


def test_measured_workspace_overflow_uses_semantic_projection_with_raw_recovery(
    make_config,
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for index in range(160):
        (workspace / f"module_{index:03d}.py").write_text("x = 1\n", encoding="utf-8")
    config = make_config(model__context_limit=350)
    config.tools.read_roots = [workspace]
    config.sessions.root = tmp_path / "sessions"
    runtime = AgentRuntime(
        config,
        model_client=object(),
        token_counter=ExactTokenCounter(lambda text: len(text.split()) if text.strip() else 0),
    )
    state = runtime.create_or_load_session()
    components = runtime._runtime_context_components(state, runtime._counter(state))
    assembly = runtime.prompts.build_semantic_operation_prompt(
        kind="history_analysis",
        system_instruction="Inspect the runtime context.",
        components=components,
    )
    compilation = runtime._compile_context(
        state,
        assembly,
        yes_no_contract(),
        minimum_output_tokens=64,
        context_limit_resolution=(350, "test"),
    )
    assert compilation.overflow_tokens > 0

    def reduce(_state, **kwargs):
        assert "module_159.py" in kwargs["source_text"]
        assert kwargs["target_tokens"] > 0
        return "Relevant workspace files can be recovered with list_files.", compilation.report

    monkeypatch.setattr(runtime, "_reduce_text_hierarchically", reduce)
    projected = runtime._project_runtime_context_for_overflow(
        state,
        original_request="inspect relevant modules",
        compilation=compilation,
        existing_projections={},
        remaining_calls=[8],
    )

    assert projected is not None
    assert projected[0] == "workspace_file_manifest"
    event = runtime.history.read_history(state.session_id)[-1]
    assert event.event_type == "runtime_context_projected"
    assert event.payload["source_sha256"]
    assert event.payload["source_locator"]["recovery_tool"] == "list_files"
    assert event.payload["source_locator"]["recovery_arguments"] == {
        "path": str(workspace)
    }

    projection_event_sequence = event.sequence

    def must_not_reduce(*_args, **_kwargs):
        raise AssertionError("durable projection should be reused")

    monkeypatch.setattr(
        runtime,
        "_reduce_text_hierarchically",
        must_not_reduce,
    )
    reused = runtime._project_runtime_context_for_overflow(
        state,
        original_request="inspect relevant modules",
        compilation=compilation,
        existing_projections={},
        remaining_calls=[8],
    )
    assert reused is not None
    reuse_event = runtime.history.read_history(state.session_id)[-1]
    assert reuse_event.event_type == "runtime_context_projection_reused"
    assert reuse_event.payload["projection_event_sequence"] == projection_event_sequence

    source_name, projection = reused
    assert isinstance(projection, RuntimeContextProjection)
    projected_components = runtime._runtime_context_components(
        state,
        runtime._counter(state),
        projections={source_name: projection},
    )
    projected_manifest = next(
        item.text
        for item in projected_components
        if item.name == "workspace_file_manifest"
    )
    assert "SEMANTIC PROJECTION" in projected_manifest

    (workspace / "added_after_projection.py").write_text("x = 2\n", encoding="utf-8")
    refreshed_components = runtime._runtime_context_components(
        state,
        runtime._counter(state),
        projections={source_name: projection},
    )
    refreshed_manifest = next(
        item.text
        for item in refreshed_components
        if item.name == "workspace_file_manifest"
    )
    assert "added_after_projection.py" in refreshed_manifest
    assert "SEMANTIC PROJECTION" not in refreshed_manifest
