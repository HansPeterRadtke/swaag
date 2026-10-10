

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


def test_history_compression_legacy_prefix_replay_remains_compatible(make_config, tmp_path) -> None:
    from dataclasses import asdict
    from swaag.compression import summary_message_payload
    from swaag.history import HistoryStore
    from swaag.types import Message

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    history = HistoryStore(config.sessions.root)
    state = history.create(config_fingerprint="cfg", model_base_url="http://model")
    for index in range(4):
        history.record_event(
            state,
            "message_added",
            {"message": asdict(Message(role="user", content=f"m{index}", created_at=f"t{index}"))},
        )
    summary = summary_message_payload(
        "legacy prefix summary",
        source_message_count=2,
        created_at="ts",
    )
    # Deliberately omit source_message_start to model an old durable event.
    history.record_event(
        state,
        "history_compressed",
        {
            "source_message_count": 2,
            "summary_message": summary,
            "summary_budget_report": {},
        },
    )
    assert [message.content for message in state.messages] == [
        "legacy prefix summary",
        "m2",
        "m3",
    ]
    rebuilt = history.rebuild_from_history(state.session_id)
    assert [message.content for message in rebuilt.messages] == [
        "legacy prefix summary",
        "m2",
        "m3",
    ]


def test_history_compression_replays_selected_middle_span(make_config, tmp_path) -> None:
    from dataclasses import asdict
    from swaag.compression import summary_message_payload
    from swaag.history import HistoryStore
    from swaag.types import Message

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    history = HistoryStore(config.sessions.root)
    state = history.create(config_fingerprint="cfg", model_base_url="http://model")
    for index in range(4):
        history.record_event(
            state,
            "message_added",
            {"message": asdict(Message(role="user", content=f"m{index}", created_at=f"t{index}"))},
        )
    summary = summary_message_payload(
        "middle summary",
        source_message_start=1,
        source_message_count=2,
        created_at="ts",
    )
    history.record_event(
        state,
        "history_compressed",
        {
            "source_message_start": 1,
            "source_message_count": 2,
            "summary_message": summary,
            "summary_budget_report": {},
        },
    )
    assert [message.content for message in state.messages] == [
        "m0",
        "middle summary",
        "m3",
    ]
    rebuilt = history.rebuild_from_history(state.session_id)
    assert [message.content for message in rebuilt.messages] == [
        "m0",
        "middle summary",
        "m3",
    ]

def test_repeated_appends_do_not_rescan_same_large_verified_jsonl(make_config, tmp_path) -> None:
    from unittest.mock import patch
    from swaag.history import HistoryStore

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    history = HistoryStore(config.sessions.root)
    state = history.create(config_fingerprint="cfg", model_base_url="http://model")
    with patch.object(history, "_iter_history_jsonl", wraps=history._iter_history_jsonl) as scan:
        for n in range(15):
            history.record_event(state, "error", {
                "operation": "regression_test", "error": f"step_{n}", "error_type": "Test",
            })
        assert len(history.read_history(state.session_id)) == 16
        assert scan.call_count == 0, "a validated append must not rehash full JSONL"

        path = history.history_path(state.session_id)
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        history.record_event(state, "error", {
            "operation": "post_external_change", "error": "still_valid", "error_type": "Test",
        })
        assert scan.call_count == 1, "an external file edit must trigger full verification"
        history.record_event(state, "error", {
            "operation": "later_append", "error": "still_fast", "error_type": "Test",
        })
        assert scan.call_count == 1
    assert history.rebuild_from_history(state.session_id).event_count == state.event_count


def test_external_history_tampering_is_still_detected_with_file_stat_cache(make_config, tmp_path) -> None:
    import pytest
    from swaag.history import HistoryStore
    from swaag.history import HistoryCorruptionError

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    history = HistoryStore(config.sessions.root)
    state = history.create(config_fingerprint="cfg", model_base_url="http://model")
    history.record_event(state, "error", {
        "operation": "tamper_check", "error": "original_token", "error_type": "Test",
    })
    path = history.history_path(state.session_id)
    source = path.read_text(encoding="utf-8")
    assert "original_token" in source
    path.write_text(source.replace("original_token", "modified_token", 1) + "\n", encoding="utf-8")
    with pytest.raises(HistoryCorruptionError):
        list(history.iter_history(state.session_id))
