from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from swaag.history import HistoryStore
from swaag.scheduler import WakeupStore, parse_duration
from swaag.wakeup_dispatcher import due_session_ids


def test_duration_parser_supports_subsecond_and_long_units() -> None:
    assert parse_duration("1 millisecond").total_seconds() == 0.001
    assert parse_duration("250 ms").total_seconds() == 0.25
    assert parse_duration("1 week").total_seconds() == 604800
    assert parse_duration("2 months").total_seconds() == 2 * 2629800
    assert parse_duration("3 years").total_seconds() == 3 * 31557600


def test_claim_and_delivery_are_separate_and_idempotent(tmp_path: Path) -> None:
    now = datetime(2026, 8, 8, 10, 0, tzinfo=timezone.utc)
    store = WakeupStore(tmp_path)
    wakeup = store.schedule(session_id="s", reason="resume", duration="1 second", now=now)
    claimed = store.claim_due(session_id="s", now=now + timedelta(seconds=2))
    assert [item.status for item in claimed] == ["claimed"]
    assert store.claim_due(session_id="s", now=now + timedelta(seconds=3)) == []
    delivered = store.mark_delivered(wakeup_id=wakeup.wakeup_id, now=now + timedelta(seconds=3))
    assert delivered.status == "delivered"
    assert store.mark_delivered(wakeup_id=wakeup.wakeup_id) == store.mark_delivered(wakeup_id=wakeup.wakeup_id)


def test_stale_claim_is_reclaimable_after_lease(tmp_path: Path) -> None:
    now = datetime(2026, 8, 8, 10, 0, tzinfo=timezone.utc)
    store = WakeupStore(tmp_path)
    wakeup = store.schedule(session_id="s", reason="resume", duration="1 second", now=now)
    first = store.claim_due(session_id="s", now=now + timedelta(seconds=2), claim_lease_seconds=60)
    assert first[0].wakeup_id == wakeup.wakeup_id
    assert store.claim_due(session_id="s", now=now + timedelta(seconds=30), claim_lease_seconds=60) == []
    reclaimed = store.claim_due(session_id="s", now=now + timedelta(seconds=63), claim_lease_seconds=60)
    assert reclaimed[0].wakeup_id == wakeup.wakeup_id
    assert reclaimed[0].status == "claimed"


def test_wakeup_store_concurrent_schedules_do_not_corrupt_json(tmp_path: Path) -> None:
    store = WakeupStore(tmp_path)
    failures: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            store.schedule(session_id=f"s{index % 3}", reason=f"r{index}", duration="1 hour")
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(30)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert len(store.list_all()) == 30


def test_control_enqueue_with_explicit_id_is_idempotent(tmp_path: Path) -> None:
    history = HistoryStore(tmp_path / "sessions")
    history.create(config_fingerprint="cfg", model_base_url="http://model", session_id="s")
    first = history.enqueue_control_message("s", "wake", source="scheduler", control_id="wakeup_fixed")
    second = history.enqueue_control_message("s", "different ignored duplicate", source="scheduler", control_id="wakeup_fixed")
    assert first == second
    pending = history.list_pending_control_messages("s")
    assert len(pending) == 1
    assert pending[0]["control_id"] == "wakeup_fixed"
    history.mark_control_message_processed("s", "wakeup_fixed")
    third = history.enqueue_control_message("s", "still duplicate", source="scheduler", control_id="wakeup_fixed")
    assert third["message"] == "wake"
    assert history.list_pending_control_messages("s") == []


def test_due_session_ids_discovers_only_due_sessions(make_config, tmp_path: Path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    now = datetime(2026, 8, 8, 10, 0, tzinfo=timezone.utc)
    store = WakeupStore(config.sessions.root)
    store.schedule(session_id="due", reason="now", duration="1 second", now=now)
    store.schedule(session_id="later", reason="later", duration="1 hour", now=now)
    assert due_session_ids(config, now=now + timedelta(seconds=2)) == ["due"]


def test_wait_seconds_accepts_human_duration(make_config, tmp_path: Path) -> None:
    from swaag.tools.registry import ToolRegistry

    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    state = HistoryStore(config.sessions.root).create(config_fingerprint="cfg", model_base_url="http://model")
    started = time.monotonic()
    _, result = ToolRegistry().dispatch("wait_seconds", {"seconds": None, "duration": "5 ms"}, config, state)
    elapsed = time.monotonic() - started
    assert result.output["requested_seconds"] == 0.005
    assert result.output["requested_duration"] == "5 ms"
    assert elapsed < 0.5


def test_dispatcher_resumes_due_session_without_duplicate_user_message(make_config, tmp_path: Path) -> None:
    import json

    from swaag.runtime import AgentRuntime
    from tests.test_agent_action_loop import FakeModelClient
    from swaag.wakeup_dispatcher import dispatch_once

    config = make_config(model__context_limit=32000)
    config.sessions.root = tmp_path / "sessions"

    def wake_response(payload):
        prompt = str(payload["prompt"])
        assert "Keep watching the deployment." in prompt
        assert "Scheduled wakeup is due: inspect deployment" in prompt
        return json.dumps({"assistant_message": "Wakeup handled.", "tool_calls": [], "continue_loop": False})

    client = FakeModelClient(
        responses=[
            json.dumps({"assistant_message": "Monitoring started.", "tool_calls": [], "continue_loop": False}),
            wake_response,
        ]
    )
    runtime = AgentRuntime(config, model_client=client)
    first = runtime.run_turn("Keep watching the deployment.")
    state = runtime.history.rebuild_from_history(first.session_id)
    assert [m.content for m in state.messages if m.role == "user"] == ["Keep watching the deployment."]

    WakeupStore(config.sessions.root).schedule(
        session_id=first.session_id,
        reason="inspect deployment",
        duration="1 ms",
    )
    time.sleep(0.01)

    assert dispatch_once(config, runtime=runtime) == [first.session_id]
    rebuilt = runtime.history.rebuild_from_history(first.session_id)
    assert [m.content for m in rebuilt.messages if m.role == "user"] == ["Keep watching the deployment."]
    assert any(m.content == "Wakeup handled." for m in rebuilt.messages if m.role == "assistant")
    assert WakeupStore(config.sessions.root).list(session_id=first.session_id)[0].status == "delivered"
    assert runtime.history.list_pending_control_messages(first.session_id) == []
    assert dispatch_once(config, runtime=runtime) == []


def test_processed_control_is_durable_without_becoming_user_message(make_config, tmp_path: Path) -> None:
    import json

    from swaag.runtime import AgentRuntime
    from tests.test_agent_action_loop import FakeModelClient

    config = make_config(model__context_limit=32000)
    config.sessions.root = tmp_path / "sessions"
    client = FakeModelClient(responses=[
        json.dumps({"assistant_message": "initial", "tool_calls": [], "continue_loop": False}),
        json.dumps({"assistant_message": "control handled", "tool_calls": [], "continue_loop": False}),
    ])
    runtime = AgentRuntime(config, model_client=client)
    first = runtime.run_turn("Original user request.")
    runtime.history.enqueue_control_message(first.session_id, "Internal scheduler control.", source="scheduler", control_id="control_fixed")
    state = runtime.history.rebuild_from_history(first.session_id)
    runtime.run_pending_controls_in_session(state)

    rebuilt = runtime.history.rebuild_from_history(first.session_id)
    assert [m.content for m in rebuilt.messages if m.role == "user"] == ["Original user request."]
    processed = [e for e in runtime.history.read_history(first.session_id) if e.event_type == "control_message_processed"]
    assert processed[-1].payload["message"] == "Internal scheduler control."
    query = runtime.history.query_history_details(first.session_id, "scheduler control", max_results=5)
    assert any("Internal scheduler control." in item["preview"] for item in query["matches"])
