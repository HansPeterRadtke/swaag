from __future__ import annotations

import ast
import inspect
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from swaag.events import EventSchemaError
from swaag.environment.environment import AgentEnvironment
from swaag.history import HistoryCorruptionError, _IGNORED_REBUILD_EVENT_TYPES, _STATEFUL_REBUILD_EVENT_TYPES, replay_history
from swaag.runtime import AgentRuntime

from tests.helpers import FakeModelClient, plan_response, plan_step

def _build_scenario_runtime(make_config, tmp_path: Path) -> tuple[AgentRuntime, Path, str]:
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world\n", encoding="utf-8")
    goal = f"Add a note, read {sample.name}, edit {sample.name}, compute 6 * 7, and answer."
    config = make_config(tools__allow_side_effect_tools=True, runtime__max_tool_steps=5)
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_note", "Add a note", "note", expected_tool="notes", expected_output="Stored note", success_criteria="The note is saved."),
                    plan_step(
                        "step_read",
                        "Read the file",
                        "read",
                        expected_tool="read_text",
                        expected_output="File contents",
                        success_criteria="The file contents are available.",
                        depends_on=["step_note"],
                    ),
                    plan_step(
                        "step_edit",
                        "Edit the file",
                        "write",
                        expected_tool="edit_text",
                        expected_output="Updated file",
                        success_criteria="The file text is updated.",
                        depends_on=["step_read"],
                    ),
                    plan_step(
                        "step_calc",
                        "Compute a number",
                        "tool",
                        expected_tool="calculator",
                        expected_output="Calculated value",
                        success_criteria="The expression is evaluated.",
                        depends_on=["step_edit"],
                    ),
                    plan_step(
                        "step_answer",
                        "Answer the user",
                        "respond",
                        expected_output="Final answer",
                        success_criteria="The user receives the result.",
                        depends_on=["step_calc"],
                    ),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "notes", "tool_input": {"action": "add", "title": "Todo", "content": "Remember 42"}}),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "read_text", "tool_input": {"path": str(sample)}}),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "edit_text", "tool_input": {"path": str(sample), "operation": "replace_pattern_all", "pattern": "hello", "replacement": "hi"}}),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "6 * 7"}}),
            "done",
        ]
    )
    return AgentRuntime(config, model_client=fake_client), sample, goal


def test_rebuild_from_history_matches_original_state(make_config, tmp_path: Path) -> None:
    runtime, sample, goal = _build_scenario_runtime(make_config, tmp_path)
    result = runtime.run_turn(goal)
    state_before = runtime.history.rebuild_from_history(result.session_id)

    session_dir = runtime.history.history_path(result.session_id).parent
    for path in session_dir.iterdir():
        if path.name != "complete_history.jsonl":
            path.unlink()

    rebuilt = runtime.history.rebuild_from_history(result.session_id)

    assert asdict(rebuilt) == asdict(state_before)
    assert sample.read_text(encoding="utf-8") == "hi world\n"


def test_history_contains_required_event_types(make_config, tmp_path: Path) -> None:
    runtime, _, goal = _build_scenario_runtime(make_config, tmp_path)
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    event_types = {event.event_type for event in events}

    assert {
        "session_created",
        "message_added",
        "turn_started",
        "prompt_built",
        "budget_checked",
        "model_request_sent",
        "model_response_received",
        "decision_parsed",
        "tool_called",
        "tool_result",
        "verification_started",
        "verification_completed",
        "verification_type_used",
        "verification_passed",
        "note_added",
        "reader_opened",
        "reader_chunk_read",
        "file_chunk_read",
        "edit_applied",
        "file_write_applied",
        "turn_finished",
    }.issubset(event_types)


def test_corrupted_projections_do_not_block_recovery(make_config, tmp_path: Path) -> None:
    runtime, _, goal = _build_scenario_runtime(make_config, tmp_path)
    result = runtime.run_turn(goal)

    runtime.history.current_state_path(result.session_id).write_text("not-json", encoding="utf-8")
    runtime.history.notes_path(result.session_id).write_text("not-json", encoding="utf-8")
    runtime.history.reader_state_path(result.session_id).write_text("not-json", encoding="utf-8")

    rebuilt = runtime.history.rebuild_from_history(result.session_id)
    assert rebuilt.turn_count == 1
    assert rebuilt.notes
    assert rebuilt.reader_states


def test_only_history_module_writes_files(monkeypatch, make_config, tmp_path: Path) -> None:
    runtime, _, goal = _build_scenario_runtime(make_config, tmp_path)
    original_open = Path.open
    original_write_text = Path.write_text
    original_mkdir = Path.mkdir
    original_replace = os.replace

    def _allowed_stack() -> bool:
        for frame in inspect.stack()[2:]:
            if frame.filename.endswith("/src/swaag/history.py"):
                return True
        return False

    def guarded_open(self: Path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if any(flag in mode for flag in ("w", "a", "+", "x")) and not _allowed_stack():
            raise AssertionError(f"Write outside central history logger: {self}")
        return original_open(self, *args, **kwargs)

    def guarded_write_text(self: Path, *args, **kwargs):
        if not _allowed_stack():
            raise AssertionError(f"write_text outside central history logger: {self}")
        return original_write_text(self, *args, **kwargs)

    def guarded_mkdir(self: Path, *args, **kwargs):
        if not _allowed_stack():
            raise AssertionError(f"mkdir outside central history logger: {self}")
        return original_mkdir(self, *args, **kwargs)

    def guarded_replace(src, dst, *args, **kwargs):
        if not _allowed_stack():
            raise AssertionError(f"os.replace outside central history logger: {src} -> {dst}")
        return original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "write_text", guarded_write_text)
    monkeypatch.setattr(Path, "mkdir", guarded_mkdir)
    monkeypatch.setattr(os, "replace", guarded_replace)

    runtime.run_turn(goal)


def test_history_replay_recovers_same_final_answer(make_config, tmp_path: Path) -> None:
    runtime, _, goal = _build_scenario_runtime(make_config, tmp_path)
    result = runtime.run_turn(goal)
    rebuilt = runtime.history.rebuild_from_history(result.session_id)
    assert rebuilt.messages[-1].role == "assistant"
    assert rebuilt.messages[-1].content == result.assistant_text


def test_replay_history_function_uses_history_file_only(make_config, tmp_path: Path) -> None:
    runtime, _, goal = _build_scenario_runtime(make_config, tmp_path)
    result = runtime.run_turn(goal)
    state_before = runtime.history.rebuild_from_history(result.session_id)
    history_file = runtime.history.history_path(result.session_id)
    rebuilt = replay_history(history_file)
    assert asdict(rebuilt) == asdict(state_before)


def test_rebuild_ignores_action_selection_resolution_event(make_config) -> None:
    config = make_config()
    runtime = AgentRuntime(config, model_client=FakeModelClient(responses=[]))
    state = runtime.create_or_load_session()

    runtime.history.record_event(
        state,
        "action_selection_resolved",
        {
            "selected_action": "execute_step",
            "candidates": ["retry_step", "execute_step"],
            "source": "model",
        },
    )

    rebuilt = runtime.history.rebuild_from_history(state.session_id)

    assert rebuilt.event_count == state.event_count
    assert rebuilt.session_id == state.session_id


def test_rebuild_event_sets_cover_allowed_event_types() -> None:
    from swaag.events import ALLOWED_EVENT_TYPES

    assert _STATEFUL_REBUILD_EVENT_TYPES.isdisjoint(_IGNORED_REBUILD_EVENT_TYPES)
    assert _STATEFUL_REBUILD_EVENT_TYPES | _IGNORED_REBUILD_EVENT_TYPES == ALLOWED_EVENT_TYPES


def test_history_store_does_not_create_root_until_first_event(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    from swaag.history import HistoryStore

    store = HistoryStore(root)

    assert store.root == root
    assert not root.exists()


def test_event_factory_rejects_unknown_event_type(make_config) -> None:
    config = make_config()
    from swaag.history import HistoryStore

    store = HistoryStore(config.sessions.root)
    state = store.create(config_fingerprint="abc", model_base_url="http://example.test")
    with pytest.raises(EventSchemaError):
        store.record_event(state, "not_allowed", {})


def test_event_factory_rejects_missing_required_payload(make_config) -> None:
    config = make_config()
    from swaag.history import HistoryStore

    store = HistoryStore(config.sessions.root)
    state = store.create(config_fingerprint="abc", model_base_url="http://example.test")
    with pytest.raises(EventSchemaError):
        store.record_event(state, "tool_called", {"tool_name": "calculator"})


def test_rebuild_cannot_write_projections_directly(make_config) -> None:
    config = make_config()
    from swaag.history import HistoryInvariantError, HistoryStore

    store = HistoryStore(config.sessions.root)
    state = store.create(config_fingerprint="abc", model_base_url="http://example.test")
    with pytest.raises(HistoryInvariantError):
        store.rebuild_from_history(state.session_id, write_projections=True)


def test_history_index_is_derived_and_matches_state(make_config, tmp_path: Path) -> None:
    runtime, _, goal = _build_scenario_runtime(make_config, tmp_path)
    result = runtime.run_turn(goal)
    state = runtime.history.rebuild_from_history(result.session_id)
    index_payload = json.loads(runtime.history.history_index_path(result.session_id).read_text(encoding="utf-8"))

    assert index_payload["session_id"] == result.session_id
    assert index_payload["event_count"] == state.event_count
    assert index_payload["last_event_hash"] == state.last_event_hash
    assert index_payload["checkpoint_event_count"] == state.event_count


def test_checkpoint_is_derived_and_rebuilds_same_state(make_config, tmp_path: Path) -> None:
    runtime, _, goal = _build_scenario_runtime(make_config, tmp_path)
    result = runtime.run_turn(goal)
    checkpoint_path = runtime.history.checkpoint_path(result.session_id)
    assert checkpoint_path.exists()

    rebuilt = runtime.history.rebuild_from_history(result.session_id, prefer_checkpoint=True)
    replayed = replay_history(runtime.history.history_path(result.session_id))

    assert asdict(rebuilt) == asdict(replayed)


def test_corrupted_checkpoint_falls_back_to_history(make_config, tmp_path: Path) -> None:
    runtime, _, goal = _build_scenario_runtime(make_config, tmp_path)
    result = runtime.run_turn(goal)
    runtime.history.checkpoint_path(result.session_id).write_text("not-json", encoding="utf-8")

    rebuilt = runtime.history.rebuild_from_history(result.session_id, prefer_checkpoint=True)

    assert rebuilt.turn_count == 1


def test_read_history_window_returns_ordered_subset(make_config, tmp_path: Path) -> None:
    runtime, _, goal = _build_scenario_runtime(make_config, tmp_path)
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    window = runtime.history.read_history_window(result.session_id, start_sequence=3, limit=5)

    assert [event.sequence for event in window] == [3, 4, 5, 6, 7]
    assert [event.id for event in window] == [event.id for event in events[2:7]]


def test_iter_history_chunks_preserves_event_order(make_config, tmp_path: Path) -> None:
    runtime, _, goal = _build_scenario_runtime(make_config, tmp_path)
    result = runtime.run_turn(goal)
    all_events = runtime.history.read_history(result.session_id)
    chunks = list(runtime.history.iter_history_chunks(result.session_id, chunk_size=4))

    assert chunks
    assert all(1 <= len(chunk) <= 4 for chunk in chunks)
    flattened = [event.id for chunk in chunks for event in chunk]
    assert flattened == [event.id for event in all_events]


def test_replay_window_rebuilds_prefix_state(make_config, tmp_path: Path) -> None:
    runtime, _, goal = _build_scenario_runtime(make_config, tmp_path)
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    turn_finished = next(event for event in events if event.event_type == "turn_finished")

    prefix = runtime.history.replay_window(result.session_id, end_sequence=turn_finished.sequence - 1, chunk_size=5)

    assert prefix.turn_count == 0
    assert prefix.messages[-1].role == "assistant"
    assert prefix.messages[-1].content == result.assistant_text
    assert prefix.event_count == turn_finished.sequence - 1


def test_full_system_replay_after_replan_matches_state(make_config, tmp_path: Path) -> None:
    sample = tmp_path / "replan.txt"
    sample.write_text("hello world\n", encoding="utf-8")
    config = make_config(tools__allow_side_effect_tools=True, runtime__max_tool_steps=6, planner__max_replans=1)
    goal = "Read the file, compute 6 * 7, edit the file, and answer."
    # NOTE: this test exercises history replay after replan. We deliberately
    # avoid putting an edit_text/write_file step *before* the failing step,
    # because the runtime takes a specialised tool_input routing for those
    # tools that consumes an extra fake response and complicates the queue.
    # The failing step (step_calc) uses the calculator tool, which goes
    # through the standard tool_decision contract.
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_read", "Read the file", "read", expected_tool="read_text", expected_output="File contents", success_criteria="The file contents are available."),
                    plan_step("step_calc", "Compute a number", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The expression is evaluated.", depends_on=["step_read"]),
                    plan_step("step_edit", "Edit the file", "write", expected_tool="edit_text", expected_output="Updated file", success_criteria="The file text is updated.", depends_on=["step_calc"]),
                    plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user receives the result.", depends_on=["step_edit"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "read_text", "tool_input": {"path": str(sample)}}),
            # Wrong tool for step_calc (echo instead of calculator) — triggers verification failure → replan.
            json.dumps({"action": "call_tool", "response": "", "tool_name": "echo", "tool_input": {"text": "wrong"}}),
            plan_response(
                goal=goal,
                steps=[
                    plan_step("step_calc_replan", "Compute a number", "tool", expected_tool="calculator", expected_output="Calculated value", success_criteria="The expression is evaluated."),
                    plan_step("step_edit_replan", "Edit the file", "write", expected_tool="edit_text", expected_output="Updated file", success_criteria="The file text is updated.", depends_on=["step_calc_replan"]),
                    plan_step("step_answer_replan", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user receives the result.", depends_on=["step_edit_replan"]),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "6 * 7"}}),
            # tool_input:edit_text — runtime extracts the tool_input dict from this wrapped call.
            json.dumps({"action": "call_tool", "response": "", "tool_name": "edit_text", "tool_input": {"path": str(sample), "operation": "replace_pattern_all", "pattern": "hello", "replacement": "hi"}}),
            "done",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    original = runtime.history.rebuild_from_history(result.session_id)
    session_dir = runtime.history.history_path(result.session_id).parent
    for path in session_dir.iterdir():
        if path.name != "complete_history.jsonl":
            path.unlink()

    replayed = replay_history(runtime.history.history_path(result.session_id))

    assert asdict(replayed) == asdict(original)


def test_source_tree_has_no_direct_write_calls_outside_history_module() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "swaag"
    forbidden_attrs = {"write_text", "write_bytes", "mkdir", "unlink", "rename"}
    allowed_paths = {
        root / "history.py",
        root / "environment" / "environment.py",
        root / "environment" / "process.py",
    }

    for path in root.rglob("*.py"):
        if path in allowed_paths:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr in forbidden_attrs:
                    raise AssertionError(f"{path} uses forbidden direct filesystem write call: {func.attr}")
                if func.attr == "open":
                    mode_arg = None
                    if len(node.args) >= 1:
                        mode_arg = node.args[0]
                    for keyword in node.keywords:
                        if keyword.arg == "mode":
                            mode_arg = keyword.value
                    if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str) and any(flag in mode_arg.value for flag in "wax+"):
                        raise AssertionError(f"{path} opens files for writing outside history.py")
                if isinstance(func.value, ast.Name) and func.value.id == "os" and func.attr == "replace":
                    raise AssertionError(f"{path} uses os.replace outside history.py")
            elif isinstance(func, ast.Name) and func.id == "open":
                mode_arg = None
                if len(node.args) >= 1:
                    mode_arg = node.args[0]
                for keyword in node.keywords:
                    if keyword.arg == "mode":
                        mode_arg = keyword.value
                if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str) and any(flag in mode_arg.value for flag in "wax+"):
                    raise AssertionError(f"{path} uses open(..., mode={mode_arg.value!r}) outside history.py")


def test_source_tree_creates_history_events_only_via_event_factory() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "swaag"
    allowed_files = {"events.py", "history.py"}
    for path in root.rglob("*.py"):
        if path.name in allowed_files:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "HistoryEvent":
                raise AssertionError(f"{path} constructs HistoryEvent directly instead of using create_event")


def test_history_ids_sequences_and_hashes_are_strict(make_config, tmp_path: Path) -> None:
    runtime, _, goal = _build_scenario_runtime(make_config, tmp_path)
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)

    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert len({event.id for event in events}) == len(events)
    assert all(event.version == 1 for event in events)
    assert events[0].prev_hash is None
    assert all(event.hash for event in events)
    for previous, current in zip(events, events[1:]):
        assert current.prev_hash == previous.hash


def test_history_tamper_detection_fails_rebuild(make_config, tmp_path: Path) -> None:
    runtime, _, goal = _build_scenario_runtime(make_config, tmp_path)
    result = runtime.run_turn(goal)
    history_file = runtime.history.history_path(result.session_id)
    lines = history_file.read_text(encoding="utf-8").splitlines()
    line_index = next(
        index
        for index, raw in enumerate(lines)
        if json.loads(raw)["type"] == "message_added"
    )
    corrupted = json.loads(lines[line_index])
    corrupted["payload"]["message"]["content"] = "tampered"
    lines[line_index] = json.dumps(corrupted, sort_keys=True)
    history_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(HistoryCorruptionError):
        runtime.history.rebuild_from_history(result.session_id)


def test_event_completeness_matches_observed_operations(make_config, tmp_path: Path) -> None:
    runtime, sample, goal = _build_scenario_runtime(make_config, tmp_path)
    result = runtime.run_turn(goal)
    events = runtime.history.read_history(result.session_id)
    event_types = [event.event_type for event in events]

    assert event_types.count("model_request_sent") == len(runtime.client.requests)
    assert event_types.count("model_response_received") == len(runtime.client.requests)
    assert event_types.count("model_tokenize_requested") == len(runtime.client.tokenize_requests)
    assert event_types.count("model_tokenize_result") == len(runtime.client.tokenize_requests)
    assert event_types.count("tool_called") == 4
    assert event_types.count("tool_result") == 4
    assert event_types.count("tool_graph_planned") == 4
    assert event_types.count("verification_started") == event_types.count("verification_completed")
    assert event_types.count("verification_completed") == event_types.count("verification_type_used")
    assert event_types.count("verification_completed") == event_types.count("verification_passed")
    assert event_types.count("file_read_requested") >= 2
    assert event_types.count("file_write_applied") == 1
    assert any(event.event_type == "file_chunk_read" and event.payload["source_ref"] == str(sample) for event in events)
    assert any(event.event_type == "edit_applied" and event.payload["path"] == str(sample) for event in events)


def test_negative_write_bypass_guard_detects_direct_write(monkeypatch, tmp_path: Path) -> None:
    original_write_text = Path.write_text

    def guarded_write_text(self: Path, *args, **kwargs):
        for frame in inspect.stack()[1:]:
            if frame.filename.endswith("/src/swaag/history.py"):
                return original_write_text(self, *args, **kwargs)
        raise AssertionError(f"write_text outside central history logger: {self}")

    monkeypatch.setattr(Path, "write_text", guarded_write_text)

    with pytest.raises(AssertionError):
        (tmp_path / "bypass.txt").write_text("bad", encoding="utf-8")


def test_replay_restores_prompt_analysis_and_strategy(make_config) -> None:
    config = make_config(runtime__max_tool_steps=2)
    goal = "make a game"
    expanded_goal = goal + " Build a small arcade prototype with one core mechanic and a playable loop."
    fake_client = FakeModelClient(
        contract_responses={
            "prompt_analysis": [
                json.dumps(
                    {
                        "task_type": "vague",
                        "completeness": "incomplete",
                        "requires_expansion": True,
                        "requires_decomposition": False,
                        "confidence": 0.9,
                        "detected_entities": [],
                        "detected_goals": ["make a game"],
                    }
                )
            ],
            "task_decision": [
                json.dumps(
                    {
                        "split_task": False,
                        "expand_task": True,
                        "ask_user": False,
                        "assume_missing": True,
                        "generate_ideas": True,
                        "confidence": 0.9,
                        "reason": "prompt_is_vague",
                    }
                )
            ],
            "task_expansion": [
                json.dumps(
                    {
                        "original_goal": goal,
                        "expanded_goal": expanded_goal,
                        "scope": ["single playable loop"],
                        "constraints": ["small scope"],
                        "expected_outputs": ["prototype"],
                        "assumptions": ["arcade"],
                    }
                )
            ],
        },
        responses=[
            plan_response(
                goal=expanded_goal,
                steps=[
                    plan_step("step_answer", "Answer the user", "respond", expected_output="Final answer", success_criteria="The user gets a response."),
                ],
            ),
            "done",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)
    result = runtime.run_turn(goal)
    replayed = replay_history(runtime.history.history_path(result.session_id))

    assert replayed.prompt_analysis is not None
    assert replayed.latest_decision is not None
    assert replayed.expanded_task is not None
    assert replayed.active_strategy is not None


def test_history_rebuild_tracks_wait_events_and_background_process_completion(make_config) -> None:
    config = make_config(
        tools__allow_side_effect_tools=True,
        tools__allow_stateful_tools=True,
        runtime__background_poll_seconds=0.0,
        runtime__tool_timeout_seconds=2,
    )
    runtime = AgentRuntime(config, model_client=FakeModelClient())
    state = runtime.create_or_load_session()
    environment = AgentEnvironment(config, state)

    started = environment.run_shell_command("sleep 0.05; printf ready", background=True)
    for event in started.generated_events:
        runtime.history.record_event(
            state,
            event.event_type,
            event.payload,
            metadata=event.metadata,
            derived_writes=event.derived_writes,
        )
    process_id = str(started.output["process_id"])
    runtime.history.record_event(
        state,
        "wait_entered",
        {"reason": "background_jobs_running", "process_ids": [process_id]},
    )

    rebuilt_waiting = runtime.history.rebuild_from_history(state.session_id)
    assert rebuilt_waiting.environment.waiting is True
    assert rebuilt_waiting.environment.waiting_process_ids == [process_id]
    assert rebuilt_waiting.environment.processes[process_id].status == "running"

    resumed_environment = AgentEnvironment(config, rebuilt_waiting)
    deadline = time.monotonic() + 2.0
    while True:
        update = resumed_environment.poll_background_process(process_id)
        for event in update.generated_events:
            runtime.history.record_event(
                rebuilt_waiting,
                event.event_type,
                event.payload,
                metadata=event.metadata,
                derived_writes=event.derived_writes,
            )
        if update.completed:
            break
        if time.monotonic() >= deadline:
            raise AssertionError("background process did not complete in time")
        time.sleep(0.01)
    runtime.history.record_event(
        rebuilt_waiting,
        "wait_resumed",
        {"reason": "background_progress", "process_ids": [process_id]},
    )

    rebuilt_final = runtime.history.rebuild_from_history(state.session_id)
    assert rebuilt_final.environment.waiting is False
    assert rebuilt_final.environment.waiting_process_ids == []
    assert rebuilt_final.environment.processes[process_id].status == "completed"
