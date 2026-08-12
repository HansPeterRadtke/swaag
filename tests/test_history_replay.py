

def test_history_replay_accepts_wait_completed(make_config, tmp_path) -> None:
    from swaag.history import HistoryStore

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    history = HistoryStore(config.sessions.root)
    state = history.create(config_fingerprint="cfg", model_base_url="http://model")
    history.record_event(state, "wait_entered", {"reason": "wait_seconds:0", "process_ids": []})
    history.record_event(state, "wait_resumed", {"reason": "wait_seconds:0", "process_ids": []})
    history.record_event(
        state,
        "wait_completed",
        {"reason": "wait_seconds:0", "requested_seconds": 0.0, "requested_duration": "0 ms", "elapsed_seconds": 0.0},
    )
    rebuilt = history.rebuild_from_history(state.session_id)
    assert rebuilt.environment.waiting is False
    assert rebuilt.environment.waiting_reason == ""
    assert rebuilt.environment.waiting_process_ids == []
    assert any(event.event_type == "wait_completed" for event in history.read_history(state.session_id))
