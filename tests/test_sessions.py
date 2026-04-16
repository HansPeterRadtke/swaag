from __future__ import annotations

from swaag.runtime import AgentRuntime
from swaag.history import HistoryStore


def test_history_store_create_and_list(make_config) -> None:
    config = make_config()
    store = HistoryStore(config.sessions.root)
    state = store.create(config_fingerprint="abc", model_base_url="http://example.test")
    rebuilt = store.rebuild_from_history(state.session_id)

    assert rebuilt.session_id == state.session_id
    assert state.session_id in store.list_sessions()
    assert store.history_path(state.session_id).exists()


def test_history_store_resolves_latest_and_named_sessions(make_config) -> None:
    config = make_config()
    store = HistoryStore(config.sessions.root)
    first = store.create(config_fingerprint="abc", model_base_url="http://example.test", session_name="alpha", session_name_source="explicit")
    second = store.create(config_fingerprint="abc", model_base_url="http://example.test", session_name="beta", session_name_source="explicit")

    assert store.resolve_session_ref("alpha") == first.session_id
    assert store.resolve_session_ref("beta") == second.session_id
    assert store.resolve_session_ref(None, latest_if_none=True) == second.session_id


def test_history_store_create_or_load_user_session_resumes_or_creates_by_name(make_config) -> None:
    config = make_config()
    store = HistoryStore(config.sessions.root)

    created = store.create_or_load_user_session(
        config_fingerprint="abc",
        model_base_url="http://example.test",
        session_ref="build-fixes",
        prefer_latest=True,
    )
    resumed = store.create_or_load_user_session(
        config_fingerprint="abc",
        model_base_url="http://example.test",
        session_ref="build-fixes",
        prefer_latest=True,
    )

    assert created.session_name == "build-fixes"
    assert resumed.session_id == created.session_id


def test_history_store_can_upgrade_placeholder_session_name_from_prompt(make_config) -> None:
    config = make_config()
    store = HistoryStore(config.sessions.root)
    state = store.create(config_fingerprint="abc", model_base_url="http://example.test")

    session_name = store.ensure_human_readable_name(state, "Fix parser bug in CLI")
    rebuilt = store.rebuild_from_history(state.session_id)

    assert session_name == rebuilt.session_name
    assert rebuilt.session_name.startswith("fix-parser-bug")
    assert rebuilt.session_name_source != "placeholder"


def test_runtime_create_or_load_user_session_defaults_to_latest(make_config) -> None:
    config = make_config()
    store = HistoryStore(config.sessions.root)
    first = store.create(config_fingerprint="abc", model_base_url="http://example.test", session_name="alpha", session_name_source="explicit")
    second = store.create(config_fingerprint="abc", model_base_url="http://example.test", session_name="beta", session_name_source="explicit")
    runtime = AgentRuntime(config, history_store=store)

    state = runtime.create_or_load_user_session()

    assert state.session_id == second.session_id
    assert state.session_name == "beta"
    assert state.session_id != first.session_id
