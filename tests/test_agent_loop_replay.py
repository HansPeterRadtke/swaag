from __future__ import annotations

import json
from pathlib import Path

from swaag.runtime import AgentRuntime
from swaag.testing.llm_record_replay import RecordReplayModelClient
from tests.helpers import FakeModelClient, plan_response, plan_step


def test_record_replay_client_replays_runtime_tool_flow(make_config, tmp_path: Path) -> None:
    cassette_path = tmp_path / "runtime_tool_flow.json"
    config = make_config(
        tools__allow_stateful_tools=True,
        runtime__max_tool_steps=6,
        runtime__max_reasoning_steps=8,
        runtime__max_total_actions=10,
    )
    prompt = "Use the calculator tool to compute 6 * 7. Reply with only the integer."
    recording_delegate = FakeModelClient(
        responses=[
            plan_response(
                goal=prompt,
                steps=[
                    plan_step(
                        "step_calc",
                        "Compute the value",
                        "tool",
                        expected_tool="calculator",
                        expected_output="Calculated result",
                        success_criteria="The expression is evaluated.",
                    ),
                    plan_step(
                        "step_answer",
                        "Answer the user",
                        "respond",
                        expected_output="Final answer",
                        success_criteria="Return only the computed integer.",
                        depends_on=["step_calc"],
                    ),
                ],
            ),
            json.dumps({"action": "call_tool", "response": "", "tool_name": "calculator", "tool_input": {"expression": "6 * 7"}}),
            "42",
        ]
    )
    recording_client = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="record",
        delegate=recording_delegate,
        request_metadata={"test_category": "agent_behavior_replay", "fixture": "runtime_tool_flow"},
    )
    recording_runtime = AgentRuntime(config, model_client=recording_client)
    recorded_turn = recording_runtime.run_turn(prompt)

    assert recorded_turn.assistant_text.strip() == "42"
    assert cassette_path.exists()

    replay_client = RecordReplayModelClient(
        cassette_path=cassette_path,
        mode="replay",
        delegate=FakeModelClient(),
        request_metadata={"test_category": "agent_behavior_replay", "fixture": "runtime_tool_flow"},
    )
    replay_runtime = AgentRuntime(config, model_client=replay_client)
    replayed_turn = replay_runtime.run_turn(prompt)

    assert replayed_turn.assistant_text.strip() == "42"
    replayed_events = replay_runtime.history.read_history(replayed_turn.session_id)
    assert any(event.event_type == "tool_called" and event.payload.get("tool_name") == "calculator" for event in replayed_events)
    assert any(event.event_type == "verification_passed" for event in replayed_events)
