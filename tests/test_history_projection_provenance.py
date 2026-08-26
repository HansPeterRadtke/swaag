from __future__ import annotations

from swaag.compression import message_source_event_references, summary_message_payload
from swaag.history import HistoryStore
from swaag.prompts import PromptBuilder
from swaag.runtime import AgentRuntime
from swaag.types import Message


def _message_payload(role: str, content: str, *, metadata: dict | None = None) -> dict:
    return {
        "role": role,
        "content": content,
        "created_at": "2026-08-26T00:00:00+00:00",
        "name": None,
        "metadata": dict(metadata or {}),
    }


def test_summary_keeps_transitive_exact_event_lineage_and_can_reexpand(
    make_config, tmp_path
) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store = HistoryStore(config.sessions.root)
    state = store.create(config_fingerprint="cfg", model_base_url="http://model")

    user_event = store.record_event(
        state,
        "message_added",
        {"message": _message_payload("user", "Remember archive-marker-712")},
    )
    tool_event = store.record_event(
        state,
        "tool_result",
        {
            "tool_name": "echo",
            "raw_input": {"text": "archive-marker-712"},
            "validated_input": {"text": "archive-marker-712"},
            "output": {"text": "archive-marker-712"},
        },
    )
    tool_message_event = store.record_event(
        state,
        "message_added",
        {
            "message": {
                **_message_payload(
                    "tool",
                    "archive-marker-712",
                    metadata={
                        "source_event_sequence": tool_event.sequence,
                        "source_event_hash": tool_event.hash,
                        "source_event_type": tool_event.event_type,
                    },
                ),
                "name": "echo",
            }
        },
    )

    references = message_source_event_references(state.messages)
    summary = summary_message_payload(
        "The user asked to retain a marker and echo returned it.",
        source_message_count=2,
        created_at="2026-08-26T00:01:00+00:00",
        source_event_references=references,
    )
    event_payload = {
        "source_message_count": 2,
        "source_event_references": references,
        "source_event_ranges": summary["metadata"]["source_event_ranges"],
        "summary_message": summary,
        "summary_budget_report": {},
    }
    store.record_event(state, "summary_created", event_payload)
    projection_event = store.record_event(state, "history_compressed", event_payload)

    assert len(state.messages) == 1
    projected = state.messages[0]
    assert projected.role == "summary"
    assert projected.metadata["projection_event_sequence"] == projection_event.sequence
    assert projected.metadata["source_event_ranges"] == [
        {
            "session_id": state.session_id,
            "start_sequence": user_event.sequence,
            "end_sequence": tool_message_event.sequence,
        }
    ]
    assert {
        (item["sequence"], item["relationship"])
        for item in projected.metadata["source_event_references"]
    } >= {
        (user_event.sequence, "message"),
        (tool_event.sequence, "authoritative_payload"),
        (tool_message_event.sequence, "message"),
    }

    prompt = PromptBuilder(config).build_agent_action_prompt(
        state.messages,
        [],
        original_request="Recall the marker",
        pending_user_messages=[],
        prompt_mode="standard",
    ).prompt_text
    assert "DERIVED HISTORY SUMMARY" in prompt
    assert "use history_window to re-expand exact events" in prompt
    assert (
        f"source_event_ranges={state.session_id}:"
        f"{user_event.sequence}-{tool_message_event.sequence}"
    ) in prompt

    rebuilt = store.rebuild_from_history(state.session_id, prefer_checkpoint=False)
    assert rebuilt.messages[0].metadata == projected.metadata

    archived = store.archive_session(state.session_id, remove_active=True)
    assert archived["event_count"] >= projection_event.sequence
    exact = store.read_history_window(
        state.session_id,
        start_sequence=user_event.sequence,
        limit=2,
    )
    assert exact[0].payload["message"]["content"] == "Remember archive-marker-712"
    assert exact[1].payload["output"]["text"] == "archive-marker-712"


def test_nested_summary_lineage_includes_raw_sources_and_prior_projection(
    make_config, tmp_path
) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store = HistoryStore(config.sessions.root)
    state = store.create(config_fingerprint="cfg", model_base_url="http://model")
    store.record_event(
        state,
        "message_added",
        {"message": _message_payload("user", "first fact")},
    )
    first_refs = message_source_event_references(state.messages)
    first_summary = summary_message_payload(
        "first summary",
        source_message_count=1,
        created_at="t1",
        source_event_references=first_refs,
    )
    first_event = store.record_event(
        state,
        "history_compressed",
        {
            "source_message_count": 1,
            "summary_message": first_summary,
            "summary_budget_report": {},
        },
    )
    store.record_event(
        state,
        "message_added",
        {"message": _message_payload("assistant", "later fact")},
    )

    nested = message_source_event_references(state.messages)

    assert any(item["sequence"] == first_refs[0]["sequence"] for item in nested)
    assert any(
        item["sequence"] == first_event.sequence
        and item["relationship"] == "derived_projection"
        for item in nested
    )


def test_legacy_summary_source_count_replays_into_metadata(make_config, tmp_path) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store = HistoryStore(config.sessions.root)
    state = store.create(config_fingerprint="cfg", model_base_url="http://model")
    store.record_event(
        state,
        "message_added",
        {"message": _message_payload("user", "legacy fact")},
    )
    store.record_event(
        state,
        "history_compressed",
        {
            "source_message_count": 1,
            "summary_message": {
                "role": "summary",
                "content": "legacy summary",
                "created_at": "t1",
                "source_message_count": 1,
            },
            "summary_budget_report": {},
        },
    )

    rebuilt = store.rebuild_from_history(state.session_id, prefer_checkpoint=False)

    assert rebuilt.messages[0].metadata["source_message_count"] == 1


def test_archived_history_retrieval_lineage_survives_runtime_tool_message(
    make_config, tmp_path
) -> None:
    config = make_config()
    config.sessions.root = tmp_path / "sessions"
    store = HistoryStore(config.sessions.root)
    archived_state = store.create(
        config_fingerprint="cfg",
        model_base_url="http://model",
        session_id="session_archived_source",
    )
    source_event = store.record_event(
        archived_state,
        "message_added",
        {"message": _message_payload("user", "archived exact fact")},
    )
    store.archive_session(archived_state.session_id, remove_active=True)

    runtime = AgentRuntime(config, model_client=object())
    result = runtime.execute_tool_once(
        "history_window",
        {
            "session_ref": archived_state.session_id,
            "start_sequence": source_event.sequence,
            "limit": 1,
        },
    )
    state = runtime.history.create_or_load(
        config_fingerprint=config.config_fingerprint(),
        model_base_url=config.model.base_url,
        session_id=result.session_id,
    )
    references = message_source_event_references(state.messages)

    assert any(
        item["session_id"] == archived_state.session_id
        and item["sequence"] == source_event.sequence
        and item["hash"] == source_event.hash
        and item["relationship"] == "retrieved_history_event"
        for item in references
    )
    projected = summary_message_payload(
        "An exact archived fact was retrieved.",
        source_message_count=len(state.messages),
        created_at="t2",
        source_event_references=references,
    )
    assert any(
        item["session_id"] == archived_state.session_id
        and item["start_sequence"] == source_event.sequence
        for item in projected["metadata"]["source_event_ranges"]
    )
